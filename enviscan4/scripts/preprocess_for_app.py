# scripts/preprocess_for_app.py

import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

print("Starting pre-processing for the Streamlit app...")

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(script_dir, '..', 'data', 'consolidated_enviro_data.csv')
OUTPUT_PATH = os.path.join(script_dir, '..', 'data', 'app_daily_data.csv')

# Use memory-efficient dtypes
dtype_spec = {
    'latitude': 'float32', 'longitude': 'float32', 'value': 'float32',
    'temperature': 'float32', 'humidity': 'float32', 'wind_speed': 'float32',
    'wind_direction': 'float32', 'distance_to_nearest_industrial_m': 'float32',
    'distance_to_nearest_major_roads_m': 'float32', 'distance_to_nearest_dump_site_m': 'float32',
    'distance_to_nearest_agricultural_m': 'float32'
}

try:
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=['timestamp'], dtype=dtype_spec)

    print("Pivoting data...")
    # Pivot to get pollutants as columns
    # Using groupby/unstack is more robust for large data
    index_cols = ['location_name', 'latitude', 'longitude', 'timestamp']
    pivoted_df = df.set_index(index_cols + ['pollutant'])['value'].unstack(level='pollutant').reset_index()
    pivoted_df.columns.name = None

    print("Merging metadata...")
    # Get metadata columns (excluding the pollutants and value)
    metadata_cols = [col for col in df.columns if col not in ['pollutant', 'value', 'unit']]
    metadata = df[metadata_cols].drop_duplicates(subset=['location_name', 'timestamp'])
    
    # Merge back
    merged_df = pd.merge(pivoted_df, metadata, on=['location_name', 'latitude', 'longitude', 'timestamp'], how='left')

    print("Aggregating to daily averages...")
    # Set timestamp as index for resampling
    merged_df.set_index('timestamp', inplace=True)

    # Define columns
    numeric_cols_to_agg = [c for c in ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2', 'temperature', 'humidity', 'wind_speed', 'wind_direction'] if c in merged_df.columns]
    static_cols = ['latitude', 'longitude', 'distance_to_nearest_industrial_m', 'distance_to_nearest_major_roads_m', 'distance_to_nearest_dump_site_m', 'distance_to_nearest_agricultural_m']
    valid_static_cols = [col for col in static_cols if col in merged_df.columns]

    # --- THE FIX IS HERE ---
    # Group by location and resample
    grouped = merged_df.groupby('location_name').resample('D')
    
    # Aggregate
    daily_df = grouped.agg(
        {**{col: 'mean' for col in numeric_cols_to_agg}, 
         **{col: 'first' for col in valid_static_cols}}
    )

    # Fill gaps while still grouped (preserves location_name as index)
    daily_df = daily_df.ffill().bfill()

    # FINAL RESET to make location_name and timestamp into columns
    daily_df = daily_df.reset_index()
    # -----------------------

    print(f"Saving pre-processed data to {OUTPUT_PATH}...")
    # Verify location_name exists before saving
    if 'location_name' not in daily_df.columns:
        raise KeyError("Failed to create 'location_name' column during preprocessing.")
        
    daily_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Pre-processing complete! File saved with {len(daily_df)} rows.")
    print(f"Columns: {daily_df.columns.tolist()}")

except FileNotFoundError:
    print(f"❌ ERROR: Raw data file not found at {RAW_DATA_PATH}")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")