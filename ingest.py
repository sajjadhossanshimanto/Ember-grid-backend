import argparse
import glob
import pandas as pd
import numpy as np

def load_modis_files(pattern: str = "modis_*.csv") -> pd.DataFrame:
    files = glob.glob(pattern)
    if not files:
        print(f"Warning: No files found matching {pattern}")
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        print(f"Loading MODIS: {f}")
        df = pd.read_csv(f, low_memory=False)
        # 1. Keep only presumed vegetation fires (type == 0) if 'type' column exists
        if 'type' in df.columns:
            df = df[df['type'] == 0]
        
        # 2. Filter out low confidence (< 30)
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(0)
        df = df[df['confidence'] >= 30].copy()
        df['conf_level'] = np.where(df['confidence'] >= 80, 'high', 'nominal')
        
        # 3. Standardize schema
        df = df.rename(columns={'brightness': 'temp_mir', 'bright_t31': 'temp_tir'})
        df['instrument'] = 'MODIS'
        dfs.append(df[['latitude', 'longitude', 'acq_date', 'acq_time', 'instrument', 
                       'conf_level', 'temp_mir', 'temp_tir', 'frp', 'daynight']])
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_viirs_files(pattern: str = "viirs*.csv") -> pd.DataFrame:
    files = glob.glob(pattern)
    if not files:
        print(f"Warning: No files found matching {pattern}")
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        print(f"Loading VIIRS: {f}")
        df = pd.read_csv(f, low_memory=False)
        # 1. Keep only presumed vegetation fires (type == 0)
        if 'type' in df.columns:
            df = df[df['type'] == 0]
            
        # 2. Filter out low confidence ('l')
        df['confidence'] = df['confidence'].astype(str).str.lower()
        df = df[df['confidence'].isin(['n', 'h', 'nominal', 'high'])].copy()
        df['conf_level'] = np.where(df['confidence'].str.startswith('h'), 'high', 'nominal')
        
        # 3. Standardize schema
        df = df.rename(columns={'bright_ti4': 'temp_mir', 'bright_ti5': 'temp_tir'})
        df['instrument'] = 'VIIRS'
        dfs.append(df[['latitude', 'longitude', 'acq_date', 'acq_time', 'instrument', 
                       'conf_level', 'temp_mir', 'temp_tir', 'frp', 'daynight']])
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_harmonized_parquet(
    output_path: str = "harmonized_fires.parquet",
    grid_deg: float = 0.01,
    modis_pattern: str = "modis_*.csv",
    viirs_pattern: str = "viirs*.csv",
):
    modis_df = load_modis_files(modis_pattern)
    viirs_df = load_viirs_files(viirs_pattern)
    
    if modis_df.empty and viirs_df.empty:
        raise FileNotFoundError("No MODIS or VIIRS CSV files found at the supplied paths.")

    combined = pd.concat([modis_df, viirs_df], ignore_index=True)
    
    # Ensure acq_date is parsed for year/month extraction
    combined['acq_date'] = pd.to_datetime(combined['acq_date'])
    combined['year'] = combined['acq_date'].dt.year
    combined['month'] = combined['acq_date'].dt.month
    combined['day_of_year'] = combined['acq_date'].dt.dayofyear
    combined['acq_date_str'] = combined['acq_date'].dt.strftime('%Y-%m-%d')

    # Spatial Harmonization: Snap coordinates to ~1km grid (0.01 deg ≈ 1.11 km)
    combined['grid_lat'] = ((combined['latitude'] / grid_deg).round() * grid_deg).round(4)
    combined['grid_lon'] = ((combined['longitude'] / grid_deg).round() * grid_deg).round(4)

    # Pre-calculate boolean flags for fast aggregation
    combined['is_modis'] = (combined['instrument'] == 'MODIS').astype(int)
    combined['is_viirs'] = (combined['instrument'] == 'VIIRS').astype(int)
    # Sub-MODIS sensitivity filter: VIIRS fires with FRP >= 1.5 MW closely match MODIS detection limits
    combined['is_modis_equiv'] = ((combined['instrument'] == 'MODIS') | (combined['frp'] >= 1.5)).astype(int)

    print("Aggregating into 1km daily harmonized grid cells...")
    harmonized = (
        combined.groupby(['acq_date_str', 'year', 'month', 'day_of_year', 'grid_lat', 'grid_lon'], as_index=False)
        .agg(
            raw_total_detections=('frp', 'count'),
            raw_modis_detections=('is_modis', 'sum'),
            raw_viirs_detections=('is_viirs', 'sum'),
            modis_equiv_hits=('is_modis_equiv', 'sum'),
            mean_frp=('frp', 'mean'),
            max_frp=('frp', 'max'),
            max_temp_mir=('temp_mir', 'max'),
            max_temp_tir=('temp_tir', 'max'),
            daynight=('daynight', 'first')
        )
    )

    # Determine sensor source label for each 1km cell-day
    conditions = [
        (harmonized['raw_modis_detections'] > 0) & (harmonized['raw_viirs_detections'] > 0),
        (harmonized['raw_modis_detections'] > 0),
        (harmonized['raw_viirs_detections'] > 0)
    ]
    choices = ['MODIS+VIIRS', 'MODIS_ONLY', 'VIIRS_ONLY']
    harmonized['sensor_source'] = np.select(conditions, choices, default='UNKNOWN')

    # Harmonized Fire Unit:
    # 1.0 = Confirmed 1km active fire cell (and passes sensitivity threshold if VIIRS-only)
    # 0.35 = Small/low-intensity VIIRS fire (<1.5 MW FRP) down-weighted so post-2012 counts don't artificially spike
    harmonized['harmonized_fire_score'] = np.where(harmonized['modis_equiv_hits'] > 0, 1.0, 0.35)

    harmonized = harmonized.rename(columns={'acq_date_str': 'acq_date'})
    harmonized.to_parquet(output_path, index=False, compression='snappy')
    print(f"Success! Saved {len(harmonized):,} harmonized 1km cell-days to {output_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Build a harmonized fire dataset from MODIS and VIIRS CSV files."
#     )
#     parser.add_argument("modis_*.csv", help="Path to the MODIS CSV file")
#     parser.add_argument("viirs_*.csv", help="Path to the VIIRS CSV file")
#     args = parser.parse_args()
#     build_harmonized_parquet(modis_pattern=args.modis_csv, viirs_pattern=args.viirs_csv)
build_harmonized_parquet()