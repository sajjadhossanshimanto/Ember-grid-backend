import io
import os
import duckdb
from typing import Optional, Literal
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(
    title="FIRMS Harmonized Burn Activity Calendar API",
    description="Harmonizes MODIS (1km) and VIIRS (375m) active fire hotspots into a consistent time series.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PARQUET_PATH = r"data\harmonized_fires.parquet"

def get_db():
    if not os.path.exists(PARQUET_PATH):
        raise HTTPException(
            status_code=503, 
            detail=f"Dataset '{PARQUET_PATH}' not found. Run 'python ingest.py <modis.csv> <viirs.csv>' first."
        )
    return duckdb.connect(database=":memory:")


# -------------------------------------------------------------------------
# ENDPOINT 1: Burning Activity Calendar Heatmap & Historical Anomalies
# -------------------------------------------------------------------------
@app.get("/api/v1/calendar-heatmap")
def get_calendar_heatmap(
    min_lat: float = Query(24.0, description="Minimum Latitude (Default: Contiguous US South)"),
    max_lat: float = Query(50.0, description="Maximum Latitude (Default: Contiguous US North)"),
    min_lon: float = Query(-125.0, description="Minimum Longitude (Default: Contiguous US West)"),
    max_lon: float = Query(-66.0, description="Maximum Longitude (Default: Contiguous US East)"),
    start_year: int = Query(2000, description="Start Year"),
    end_year: int = Query(2026, description="End Year"),
    metric: Literal["harmonized_score", "cell_count", "total_frp"] = Query("harmonized_score")
):
    """
    Returns a Year x Month matrix of harmonized burning activity, along with
    historical monthly medians, anomaly percentages, and colorblind-safe severity levels.
    """
    con = get_db()
    
    metric_sql = {
        "harmonized_score": "ROUND(SUM(harmonized_fire_score), 1)",
        "cell_count": "COUNT(*)",
        "total_frp": "ROUND(SUM(mean_frp), 1)"
    }[metric]

    query = f"""
    WITH monthly_agg AS (
        SELECT 
            year,
            month,
            {metric_sql} AS activity_value,
            COUNT(*) AS active_1km_cells,
            SUM(raw_modis_detections) AS raw_modis,
            SUM(raw_viirs_detections) AS raw_viirs,
            ROUND(AVG(mean_frp), 2) AS avg_frp
        FROM '{PARQUET_PATH}'
        WHERE grid_lat BETWEEN ? AND ?
          AND grid_lon BETWEEN ? AND ?
          AND year BETWEEN ? AND ?
        GROUP BY year, month
    ),
    climatology AS (
        SELECT 
            month,
            ROUND(MEDIAN(activity_value), 1) AS historical_median,
            ROUND(AVG(activity_value), 1) AS historical_mean,
            ROUND(STDDEV_POP(activity_value), 1) AS historical_std,
            ROUND(QUANTILE_CONT(activity_value, 0.25), 1) AS p25,
            ROUND(QUANTILE_CONT(activity_value, 0.75), 1) AS p75,
            ROUND(MAX(activity_value), 1) AS historical_max
        FROM monthly_agg
        GROUP BY month
    )
    SELECT 
        m.year,
        m.month,
        m.activity_value,
        m.active_1km_cells,
        m.raw_modis,
        m.raw_viirs,
        m.avg_frp,
        c.historical_median,
        c.historical_mean,
        c.p25,
        c.p75,
        c.historical_max,
        CASE 
            WHEN c.historical_median > 0 
            THEN ROUND(((m.activity_value - c.historical_median) / c.historical_median) * 100.0, 1)
            ELSE 0.0 
        END AS anomaly_pct
    FROM monthly_agg m
    JOIN climatology c ON m.month = c.month
    ORDER BY m.year DESC, m.month ASC
    """
    
    df = con.execute(query, [min_lat, max_lat, min_lon, max_lon, start_year, end_year]).df()
    
    if df.empty:
        return {"matrix": [], "monthly_climatology": [], "color_ramp_guide": {}}

    # Assign accessible severity level & Viridis/YlOrRd colorblind-safe hex codes
    def classify_severity(row):
        val = row['activity_value']
        med = row['historical_median']
        p75 = row['p75']
        if val == 0:
            return ("none", "#f7f7f7")
        elif val < med * 0.75:
            return ("low", "#fed976")          # Soft Yellow
        elif val <= p75 or val <= med * 1.25:
            return ("normal", "#fd8d3c")       # Orange
        elif val <= med * 1.75:
            return ("high", "#e31a1c")         # Bright Red
        else:
            return ("extreme_anomaly", "#800026") # Deep Crimson

    severities = df.apply(classify_severity, axis=1)
    df['severity_level'] = [s[0] for s in severities]
    df['color_hex'] = [s[1] for s in severities]

    # Extract distinct monthly climatology for baseline reference charts
    clim_df = (
        df[['month', 'historical_median', 'historical_mean', 'p25', 'p75', 'historical_max']]
        .drop_duplicates(subset=['month'])
        .sort_values('month')
    )

    return {
        "bbox": {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon},
        "metric_used": metric,
        "matrix": df.to_dict(orient="records"),
        "monthly_climatology": clim_df.to_dict(orient="records")
    }


# -------------------------------------------------------------------------
# ENDPOINT 2: Harmonization Proof & Sensor Calibration Stats
# -------------------------------------------------------------------------
@app.get("/api/v1/harmonization-stats")
def get_harmonization_stats(
    min_lat: float = Query(24.0),
    max_lat: float = Query(50.0),
    min_lon: float = Query(-125.0),
    max_lon: float = Query(-66.0),
    year: Optional[int] = Query(None, description="Optional specific year (e.g. 2025)")
):
    """
    Demonstrates the impact of harmonization: compares raw unharmonized MODIS + VIIRS 
    counts against spatially gridded (1km) and sensitivity-calibrated counts.
    """
    con = get_db()
    year_filter = f"AND year = {int(year)}" if year else ""
    
    query = f"""
    SELECT 
        year,
        month,
        SUM(raw_modis_detections) AS raw_modis_hotspots,
        SUM(raw_viirs_detections) AS raw_viirs_hotspots,
        SUM(raw_total_detections) AS naive_combined_hotspots,
        COUNT(*) AS gridded_1km_fire_cells,
        ROUND(SUM(harmonized_fire_score), 1) AS harmonized_burning_index,
        SUM(CASE WHEN sensor_source = 'MODIS+VIIRS' THEN 1 ELSE 0 END) AS dual_sensor_confirmed_cells
    FROM '{PARQUET_PATH}'
    WHERE grid_lat BETWEEN ? AND ?
      AND grid_lon BETWEEN ? AND ?
      {year_filter}
    GROUP BY year, month
    ORDER BY year, month
    """
    df = con.execute(query, [min_lat, max_lat, min_lon, max_lon]).df()
    
    if df.empty:
        return {"summary": {}, "timeline": []}

    total_modis = int(df['raw_modis_hotspots'].sum())
    total_viirs = int(df['raw_viirs_hotspots'].sum())
    total_naive = int(df['naive_combined_hotspots'].sum())
    total_harmonized = float(df['harmonized_burning_index'].sum())

    return {
        "summary": {
            "raw_modis_total": total_modis,
            "raw_viirs_total": total_viirs,
            "naive_sum_total": total_naive,
            "harmonized_index_total": round(total_harmonized, 1),
            "overcounting_reduction_pct": round((1 - (total_harmonized / max(total_naive, 1))) * 100, 1),
            "viirs_to_modis_raw_ratio": round(total_viirs / max(total_modis, 1), 2)
        },
        "timeline": df.to_dict(orient="records")
    }


# -------------------------------------------------------------------------
# ENDPOINT 3: Interactive Map Hotspots (GeoJSON)
# -------------------------------------------------------------------------
@app.get("/api/v1/hotspots/geojson")
def get_map_hotspots(
    min_lat: float = Query(24.0),
    max_lat: float = Query(50.0),
    min_lon: float = Query(-125.0),
    max_lon: float = Query(-66.0),
    start_date: str = Query("2025-01-01", description="YYYY-MM-DD"),
    end_date: str = Query("2025-01-31", description="YYYY-MM-DD"),
    limit: int = Query(5000, le=25000, description="Max points to render on map")
):
    """
    Returns harmonized 1km fire cells as GeoJSON FeatureCollection for Mapbox/Leaflet.
    """
    con = get_db()
    query = f"""
    SELECT 
        grid_lat,
        grid_lon,
        acq_date,
        sensor_source,
        raw_total_detections,
        ROUND(mean_frp, 2) AS mean_frp,
        ROUND(max_frp, 2) AS max_frp,
        ROUND(max_temp_mir, 1) AS max_temp_mir,
        harmonized_fire_score,
        daynight
    FROM '{PARQUET_PATH}'
    WHERE grid_lat BETWEEN ? AND ?
      AND grid_lon BETWEEN ? AND ?
      AND acq_date BETWEEN ? AND ?
    ORDER BY max_frp DESC
    LIMIT ?
    """
    df = con.execute(query, [min_lat, max_lat, min_lon, max_lon, start_date, end_date, limit]).df()

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["grid_lon"], row["grid_lat"]]
            },
            "properties": {
                "acq_date": row["acq_date"],
                "sensor_source": row["sensor_source"],
                "raw_detections_in_1km_cell": int(row["raw_total_detections"]),
                "mean_frp_mw": float(row["mean_frp"]),
                "max_frp_mw": float(row["max_frp"]),
                "max_temp_kelvin": float(row["max_temp_mir"]),
                "harmonized_score": float(row["harmonized_fire_score"]),
                "daynight": row["daynight"]
            }
        }
        for _, row in df.iterrows()
    ]

    return {
        "type": "FeatureCollection",
        "count": len(features),
        "features": features
    }


# -------------------------------------------------------------------------
# ENDPOINT 4: Downloadable CSV Export
# -------------------------------------------------------------------------
@app.get("/api/v1/export-csv")
def export_harmonized_csv(
    min_lat: float = Query(24.0),
    max_lat: float = Query(50.0),
    min_lon: float = Query(-125.0),
    max_lon: float = Query(-66.0),
    start_date: str = Query("2025-01-01"),
    end_date: str = Query("2025-12-31")
):
    """
    Streams the filtered harmonized dataset directly as a downloadable CSV file.
    Fulfills the challenge objective: 'A downloadable CSV File of hot spots detected during a search'.
    """
    con = get_db()
    query = f"""
    SELECT 
        acq_date,
        grid_lat AS latitude_1km,
        grid_lon AS longitude_1km,
        sensor_source,
        harmonized_fire_score,
        raw_modis_detections,
        raw_viirs_detections,
        ROUND(mean_frp, 2) AS mean_frp_mw,
        ROUND(max_frp, 2) AS max_frp_mw,
        ROUND(max_temp_mir, 2) AS max_brightness_kelvin,
        daynight
    FROM '{PARQUET_PATH}'
    WHERE grid_lat BETWEEN ? AND ?
      AND grid_lon BETWEEN ? AND ?
      AND acq_date BETWEEN ? AND ?
    ORDER BY acq_date ASC, max_frp_mw DESC
    """
    df = con.execute(query, [min_lat, max_lat, min_lon, max_lon, start_date, end_date]).df()

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    filename = f"harmonized_hotspots_{start_date}_to_{end_date}.csv"
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )