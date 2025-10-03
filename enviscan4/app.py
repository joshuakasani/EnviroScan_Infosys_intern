import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import os
from datetime import timedelta

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EnviroScan: Pollution Source Identifier",
    page_icon="💨",
    layout="wide"
)

# --- 2. CACHED DATA LOADING ---
@st.cache_data
def load_data():
    """
    Loads all necessary data and models.
    Uses caching to prevent reloading on every interaction.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(script_dir, 'models', 'pollution_source_model.joblib')
    TRAIN_PATH = os.path.join(script_dir, 'data', 'train.csv')
    APP_DATA_PATH = os.path.join(script_dir, 'data', 'app_daily_data.csv')

    try:
        model = joblib.load(MODEL_PATH)
        train_df = pd.read_csv(TRAIN_PATH)
        # --- FIX for Timezone Error ---
        # When loading, convert timestamp to timezone-naive to match Streamlit's date_input
        app_df = pd.read_csv(APP_DATA_PATH, parse_dates=['timestamp'])
        app_df['timestamp'] = app_df['timestamp'].dt.tz_localize(None)
        # --- END OF FIX ---
    except FileNotFoundError as e:
        st.error(f"ERROR: A required data file was not found: {e.filename}")
        st.error("Please run the `scripts/preprocess_for_app.py` script first to generate the app data.")
        st.stop()

    unique_labels = sorted(np.unique(train_df['pollution_source']))
    inverse_class_mapping = {i: label for i, label in enumerate(unique_labels)}

    numerical_cols = [
        'co', 'no2', 'o3', 'pm10', 'pm25', 'so2', 'temperature', 'humidity',
        'wind_speed', 'wind_direction', 'distance_to_nearest_industrial_m',
        'distance_to_nearest_major_roads_m', 'distance_to_nearest_dump_site_m',
        'distance_to_nearest_agricultural_m'
    ]
    
    scaler = StandardScaler()
    valid_cols_for_fit = [col for col in numerical_cols if col in train_df.columns]
    scaler.fit(train_df[valid_cols_for_fit])
    
    model_columns = train_df.drop('pollution_source', axis=1).columns

    return model, app_df, scaler, inverse_class_mapping, model_columns, numerical_cols

# --- 3. MAP GENERATION FUNCTION ---
def create_map(df):
    """Creates a Folium map with markers based on the input dataframe."""
    if df.empty:
        return folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    map_center = [df['latitude'].iloc[0], df['longitude'].iloc[0]]
    zoom_start = 8 if len(df['location_name'].unique()) == 1 else 5
    
    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="CartoDB positron")

    source_styles = {
        'Vehicular': {'color': 'blue', 'icon': 'car'},
        'Industrial': {'color': 'gray', 'icon': 'industry'},
        'Agricultural_Burning': {'color': 'orange', 'icon': 'fire'},
        'Background_Mixed': {'color': 'green', 'icon': 'leaf'}
    }
    
    if len(df) > 1000:
        df_sample = df.sample(n=1000, random_state=42)
    else:
        df_sample = df

    for _, row in df_sample.iterrows():
        source = row.get('predicted_source', 'N/A')
        confidence = row.get('confidence', 0)
        style = source_styles.get(source, {'color': 'black', 'icon': 'question-sign'})
        popup_html = f"""
        <b>City:</b> {row['location_name']}<br>
        <b>Predicted Source:</b> {source}<br>
        <b>Confidence:</b> {confidence:.2%}<br>
        <hr>
        <b>PM2.5:</b> {row['pm25']:.2f} µg/m³<br>
        <b>Date:</b> {row['timestamp'].strftime('%Y-%m-%d')}
        """
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=style['color'], icon=style['icon'], prefix='fa')
        ).add_to(m)

    return m

# --- 4. MAIN APPLICATION ---
model, app_df, scaler, inverse_class_mapping, model_columns, numerical_cols = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.title("Filters")

# --- FIX for KeyError ---
# This check ensures the app doesn't crash if the pre-processing script failed
if 'location_name' not in app_df.columns:
    st.error("The 'location_name' column is missing from `app_daily_data.csv`.")
    st.error("Please re-run the `scripts/preprocess_for_app.py` script to fix the data file.")
    st.stop()
# --- END OF FIX ---

city_list = ["All Cities"] + sorted(app_df['location_name'].unique().tolist())
selected_city = st.sidebar.selectbox("Select City", options=city_list)

min_date = app_df['timestamp'].min().date()
max_date = app_df['timestamp'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(max_date - timedelta(days=30), max_date),
    min_value=min_date,
    max_value=max_date,
    format="YYYY-MM-DD"
)

# --- DATA FILTERING LOGIC ---
filtered_df = app_df.copy()
if selected_city != "All Cities":
    filtered_df = filtered_df[filtered_df['location_name'] == selected_city]

if len(date_range) == 2:
    # --- FIX for Timezone Error ---
    # Both sides of the comparison are now timezone-naive
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    filtered_df = filtered_df[
        (filtered_df['timestamp'].dt.date >= start_date.date()) & 
        (filtered_df['timestamp'].dt.date <= end_date.date())
    ]
    # --- END OF FIX ---

# --- MAIN PAGE LAYOUT ---
st.title("EnviroScan: AI-Powered Pollution Source Identifier")
st.markdown("This dashboard visualizes daily average pollution data and predicts the likely source using a machine learning model.")

# --- PREDICTIONS ---
if not filtered_df.empty:
    df_to_predict = filtered_df.copy()

    df_to_predict['is_weekend'] = (df_to_predict['timestamp'].dt.dayofweek >= 5).astype(int)
    df_to_predict['month_sin'] = np.sin(2 * np.pi * df_to_predict['timestamp'].dt.month / 12.0)
    df_to_predict['month_cos'] = np.cos(2 * np.pi * df_to_predict['timestamp'].dt.month / 12.0)
    
    df_to_predict = pd.get_dummies(df_to_predict, columns=['location_name'], prefix='location')
    df_to_predict = df_to_predict.reindex(columns=model_columns, fill_value=0)

    valid_numerical_cols = [col for col in numerical_cols if col in df_to_predict.columns]
    
    if not df_to_predict.empty:
        df_to_predict[valid_numerical_cols] = scaler.transform(df_to_predict[valid_numerical_cols])

        predictions = model.predict(df_to_predict)
        probabilities = model.predict_proba(df_to_predict)
        
        filtered_df['predicted_source'] = pd.Series(predictions, index=filtered_df.index).map(inverse_class_mapping)
        filtered_df['confidence'] = probabilities.max(axis=1)
        
        st.header(f"Analysis for: {selected_city}")
        latest_data = filtered_df.sort_values('timestamp').iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Most Recent Predicted Source", latest_data['predicted_source'])
        col2.metric("Model Confidence", f"{latest_data['confidence']:.2%}")
        col3.metric("Latest Daily PM2.5", f"{latest_data['pm25']:.2f} µg/m³")

        SAFE_PM25_THRESHOLD = 60
        if latest_data['pm25'] > SAFE_PM25_THRESHOLD:
            st.error(f"🚨 ALERT: Latest daily average PM2.5 ({latest_data['pm25']:.2f} µg/m³) exceeds the safe threshold!", icon="🚨")

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Predicted Source Distribution")
            source_counts = filtered_df['predicted_source'].value_counts()
            fig_pie = px.pie(values=source_counts.values, names=source_counts.index, title="Source Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')
            
            csv = convert_df_to_csv(filtered_df)
            st.download_button(
                label="Download Report as CSV",
                data=csv,
                file_name=f"{selected_city.replace(' ', '_')}_pollution_report.csv",
                mime='text/csv',
            )

        with col2:
            st.subheader("Pollutant Trends Over Time")
            if selected_city != "All Cities":
                plot_df = filtered_df
                title_text = f"Daily Average Pollutant Levels in {selected_city}"
            else:
                plot_df = filtered_df.groupby('timestamp').mean(numeric_only=True).reset_index()
                title_text = "Daily Average Pollutant Levels (All Cities)"
            
            fig_line = px.line(plot_df, x='timestamp', y=['pm25', 'no2', 'so2'], title=title_text)
            st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("Geospatial Analysis Map")
        map_to_display = create_map(filtered_df)
        st_folium(map_to_display, width='100%', height=500, returned_objects=[])

    else:
        st.warning("No data available for the selected filters. Please select a different date range or city.")
else:
    st.warning("No data available for the selected filters. Please select a different date range or city.")