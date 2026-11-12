from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
from datetime import datetime


app = FastAPI(title="Hotspot Harmonization API")

def harmonize_hotspots(modis_file, viirs_file):
    modis = pd.read_csv(modis_file)
    viirs = pd.read_csv(viirs_file)
    
    # 1. Standardize Brightness (Removed inplace=True)
    modis = modis.rename(columns={'brightness': 'bright_mir', 'bright_t31': 'bright_tir'})
    viirs = viirs.rename(columns={'bright_ti4': 'bright_mir', 'bright_ti5': 'bright_tir'})
    
    # 2. Harmonize Confidence Levels
    conditions = [
        (modis['confidence'] < 30),
        (modis['confidence'] >= 30) & (modis['confidence'] < 80),
        (modis['confidence'] >= 80)
    ]
    modis['confidence_cat'] = np.select(conditions, ['l', 'n', 'h'], default='n')
    viirs['confidence_cat'] = viirs['confidence']
    
    # Drop original confidence to prevent type conflicts during concatenation (Removed inplace=True)
    modis = modis.drop(columns=['confidence'])
    viirs = viirs.drop(columns=['confidence'])

    # 3. Format Date and Time
    modis['acq_time_str'] = modis['acq_time'].astype(str).str.zfill(4)
    viirs['acq_time_str'] = viirs['acq_time'].astype(str).str.zfill(4)
    
    modis['acq_datetime'] = pd.to_datetime(modis['acq_date'] + ' ' + modis['acq_time_str'], format='%Y-%m-%d %H%M')
    viirs['acq_datetime'] = pd.to_datetime(viirs['acq_date'] + ' ' + viirs['acq_time_str'], format='%Y-%m-%d %H%M')
    
    # 4. Calculate spatial footprint (scan * track) to handle 1km vs 375m difference
    modis['pixel_area_km2'] = modis['scan'] * modis['track']
    viirs['pixel_area_km2'] = viirs['scan'] * viirs['track']
    
    # Combine datasets
    combined = pd.concat([modis, viirs], ignore_index=True)
    
    # Select final harmonized schema
    final_cols = [
        'instrument', 'satellite', 'acq_datetime', 'latitude', 'longitude',
        'bright_mir', 'bright_tir', 'frp', 'confidence_cat', 'pixel_area_km2',
        'daynight', 'type'
    ]
    return combined[final_cols]

# Load and prepare data on startup (For production, consider a database like PostGIS)
DATA = harmonize_hotspots(r"data\modis_2025_United_States.csv", r"data\viirs-jpss1_2025_United_States.csv")

@app.get("/api/hotspots")
def get_hotspots(
    start_date: str = Query(None, description="Start date YYYY-MM-DD"),
    end_date: str = Query(None, description="End date YYYY-MM-DD"),
    min_frp: float = Query(None, description="Minimum Fire Radiative Power"),
    bbox: str = Query(None, description="Bounding box 'min_lon,min_lat,max_lon,max_lat'")
):
    """Provides raw harmonized hotspot data for frontend map plotting."""
    df = DATA.copy()
    
    if start_date:
        df = df[df['acq_datetime'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['acq_datetime'] <= pd.to_datetime(end_date)]
    if min_frp is not None:
        df = df[df['frp'] >= min_frp]
    if bbox:
        try:
            min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(','))
            df = df[
                (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon) &
                (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat)
            ]
        except ValueError:
            return {"error": "Invalid bbox format. Use min_lon,min_lat,max_lon,max_lat"}
            
    df['acq_datetime'] = df['acq_datetime'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    return {"count": len(df), "data": df.to_dict(orient="records")}

@app.get("/api/calendar-stats")
def get_calendar_stats(
    year: int = Query(2025, description="Year to aggregate"),
    bbox: str = Query(None, description="Bounding box 'min_lon,min_lat,max_lon,max_lat'")
):
    """Provides aggregated statistics to power the Burn Activity Calendar."""
    df = DATA.copy()
    
    if bbox:
        min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(','))
        df = df[
            (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon) &
            (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat)
        ]
        
    df = df[df['acq_datetime'].dt.year == year]
    df['month'] = df['acq_datetime'].dt.month
    
    # Aggregate stats by month
    monthly_stats = df.groupby('month').agg(
        total_hotspots=('latitude', 'count'),
        total_frp=('frp', 'sum'),
        total_burned_area_km2=('pixel_area_km2', 'sum'),
        modis_count=('instrument', lambda x: (x == 'MODIS').sum()),
        viirs_count=('instrument', lambda x: (x == 'VIIRS').sum())
    ).reset_index()
    
    # Ensure all 12 months are returned for the calendar even if no fires occurred
    all_months = pd.DataFrame({'month': range(1, 13)})
    monthly_stats = pd.merge(all_months, monthly_stats, on='month', how='left').fillna(0)
    
    return monthly_stats.to_dict(orient="records")