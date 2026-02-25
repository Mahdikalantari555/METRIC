#!/usr/bin/env python3
"""
Calculate ET for a given ROI and date range using Planetary Computer data.

This script demonstrates the complete workflow:
1. Fetch Landsat data from Planetary Computer
2. Load meteorological data (from Open-Meteo or CSV)
3. Run METRIC pipeline
4. Save results

This version is configured to run directly without command-line arguments.
Edit the CONFIGURATION section below to customize the parameters.
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from shapely.geometry import shape

# Remove the PROJ_LIB variable from the environment for this script's session
if 'PROJ_LIB' in os.environ:
    del os.environ['PROJ_LIB']


# Import METRIC components
from metric_et import METRICPipeline
from metric_et.io.planetary_computer_fetcher import PlanetaryComputerLandsatFetcher

# Optional: Use Open-Meteo for weather data
try:
    from metric_et.io.dynamic_weather_fetcher import DynamicWeatherFetcher
    WEATHER_FETCHER_AVAILABLE = True
except ImportError:
    WEATHER_FETCHER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration - set via function arguments
ROI_PATH = 'testutm.geojson'
OUTPUT_DIR = 'planetary_output'
MAX_CLOUD_COVER = 50.0
RESOLUTION = 30.0
START_DATE = '2023-04-11'
END_DATE = '2023-04-20'
SOURCE_CRS = 'EPSG:32639'  # CRS of input ROI. Use UTM zone (e.g., 'EPSG:32639') if ROI is in UTM

# ============================================================================


def load_roi(roi_path: str) -> dict:
    """Load ROI geometry from GeoJSON file."""
    with open(roi_path, 'r') as f:
        geojson = json.load(f)
    
    # Extract geometry
    if 'features' in geojson:
        # FeatureCollection - use first feature
        geometry = geojson['features'][0]['geometry']
    elif 'geometry' in geojson:
        # Single feature
        geometry = geojson['geometry']
    else:
        # Assume it's just the geometry
        geometry = geojson
    
    # Validate geometry
    roi_shapely = shape(geometry)
    if not roi_shapely.is_valid:
        raise ValueError(f"Invalid ROI geometry in {roi_path}")
    
    logger.info(f"Loaded ROI: {roi_shapely.area:.6f} sq degrees (~{roi_shapely.area * 111**2:.1f} km²)")
    return geometry


def fetch_all_scenes(
    roi_geometry: dict,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 50.0,
    output_dir: str = None,
    source_crs: str = "EPSG:4326"
) -> list:
    """
    Fetch ALL Landsat scenes from Planetary Computer for the entire date range.
    Only fetches Landsat 8 and Landsat 9 scenes.
    
    Args:
        roi_geometry: ROI geometry (GeoJSON dict or shapely geometry)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_cloud_cover: Maximum cloud cover percentage
        output_dir: Output directory for downloaded scenes
        source_crs: CRS of input ROI geometry. If not WGS84 (e.g., UTM), 
                    it will be transformed to WGS84 for STAC queries.
                    Examples: "EPSG:4326" (default), "EPSG:32639" (UTM Zone 39N)
    
    Returns:
        List of scene info dictionaries with file paths
    """
    import warnings
    logger.info(f"Fetching Landsat 8/9 data from {start_date} to {end_date} from Planetary Computer")
    logger.info(f"Source CRS: {source_crs}")
    
    # Initialize fetcher with source CRS for CRS transformation
    fetcher = PlanetaryComputerLandsatFetcher(
        max_cloud_cover=max_cloud_cover,
        source_crs=source_crs
    )
    
    # Get ROI bbox (will be transformed to WGS84 by fetcher if needed)
    from shapely.geometry import shape
    if isinstance(roi_geometry, dict):
        roi_shapely = shape(roi_geometry)
    else:
        roi_shapely = roi_geometry
    
    # Normalize geometry (transform to WGS84 if source_crs is not WGS84)
    roi_geometry_normalized = fetcher._normalize_geometry(roi_geometry)
    roi_bbox = list(roi_geometry_normalized.bounds)
    
    # Normalize dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Search with platform filter for Landsat 8 and 9
    # Use pystac_client to search with platform filter
    search_params = {
        "collections": [fetcher.collection],
        "bbox": roi_bbox,
        "datetime": f"{start.isoformat()}/{end.isoformat()}",
        "limit": 100,
    }
    
    # Add platform filter for Landsat 8 and 9
    # Platforms: landsat-8, landsat-9
    search_params["query"] = {
        "platform": {
            "in": ["landsat-8", "landsat-9"]
        }
    }
    
    try:
        search = fetcher.client.search(**search_params)
        items = list(search.items())
        logger.info(f"Found {len(items)} Landsat 8/9 scene(s) matching criteria")
    except Exception as e:
        logger.warning(f"STAC search with platform filter failed: {e}, trying without filter")
        # Fallback: search without platform filter
        del search_params["query"]
        search = fetcher.client.search(**search_params)
        items = list(search.items())
    
    # Filter by cloud cover using eo:cloud_cover (STAC standard property)
    filtered_items = []
    for item in items:
        # STAC uses 'eo:cloud_cover' for Landsat cloud cover percentage
        cloud_cover = item.properties.get('eo:cloud_cover') or item.properties.get('cloud_cover', 0.0)
        if cloud_cover is None:
            cloud_cover = item.properties.get('cloud_cover', 0.0)
        if cloud_cover <= max_cloud_cover:
            filtered_items.append(item)
        else:
            logger.info(f"Filtered out {item.id}: cloud cover {cloud_cover}% > {max_cloud_cover}%")
    
    # Sort by date
    filtered_items.sort(key=lambda item: item.datetime)
    
    logger.info(f"Found {len(filtered_items)} scenes with cloud cover <= {max_cloud_cover}%")
    
    if not filtered_items:
        logger.warning("No Landsat 8/9 scenes found matching criteria")
        return []
    
    # Download each scene
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    for item in filtered_items:
        scene_date = item.properties["datetime"][:10]
        path = item.properties.get('landsat:wrs_path')
        row = item.properties.get('landsat:wrs_row')
        
        # Create scene directory
        scene_dir = output_path / f"landsat_{scene_date.replace('-', '')}_{path}_{row}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and clip bands
        try:
            downloaded_files = fetcher.download_and_clip_bands(
                item, roi_bbox, scene_dir, RESOLUTION
            )
            
            # Create MTL.json
            mtl_path = fetcher._create_mtl_metadata(item, scene_dir)
            
            results.append({
                "scene_id": item.id,
                "date": scene_date,
                "cloud_cover": item.properties.get('eo:cloud_cover'),
                "path": path,
                "row": row,
                "directory": str(scene_dir),
                "bands_downloaded": len(downloaded_files),
                "mtl_file": str(mtl_path),
                "band_files": downloaded_files
            })
            logger.info(f"Downloaded scene: {item.id} for {scene_date}")
        except Exception as e:
            logger.warning(f"Failed to download scene {item.id}: {e}")
            continue
    
    logger.info(f"Successfully fetched {len(results)} scene(s)")
    return results


def fetch_weather_data_range(
    roi_center: tuple,
    start_date: str,
    end_date: str,
    method: str = 'open-meteo'
) -> dict:
    """
    Fetch weather data for the entire date range (simple point-based).
    
    Note: For more accurate weather, use fetch_weather_for_scene() which 
    interpolates over the scene area.
    
    Args:
        roi_center: (lon, lat) of ROI center
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        method: 'open-meteo' or 'csv'
    
    Returns:
        Weather data dictionary with lists for each variable
    """
    import requests
    
    logger.info(f"Fetching weather data from {start_date} to {end_date} at {roi_center}")
    
    if method == 'open-meteo':
        # Use Open-Meteo API directly for simple point-based weather
        try:
            lat, lon = roi_center[1], roi_center[0]
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,shortwave_radiation",
                "daily": "et0_fao_evapotranspiration",
                "timezone": "auto"
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Parse the response
            dates = []
            temps = []
            humidities = []
            winds = []
            pressures = []
            radiations = []
            
            if 'hourly' in data:
                hourly = data['hourly']
                times = hourly.get('time', [])
                temps = hourly.get('temperature_2m', [])
                humidities = hourly.get('relative_humidity_2m', [])
                winds = hourly.get('wind_speed_10m', [])
                pressures = hourly.get('surface_pressure', [])
                radiations = hourly.get('shortwave_radiation', [])
                
                # Take values at 10:30 (index 10 in hourly data)
                dates = [t[:10] for t in times[::24]]  # Daily dates
                temps = temps[10::24] if temps else []
                humidities = humidities[10::24] if humidities else []
                winds = winds[10::24] if winds else []
                pressures = pressures[10::24] if pressures else []
                radiations = radiations[10::24] if radiations else []
            
            weather_data = {
                'dates': [datetime.strptime(d, '%Y-%m-%d') for d in dates],
                'temperature_2m': [t if t is not None else 25.0 for t in temps],
                'relative_humidity': [h if h is not None else 50.0 for h in humidities],
                'wind_speed': [w if w is not None else 3.0 for w in winds],
                'pressure': [p if p is not None else 1013.25 for p in pressures],
                'solar_radiation': [r if r is not None else 800.0 for r in radiations],
            }
            logger.info(f"Weather data fetched from Open-Meteo for {len(dates)} days")
            return weather_data
        except Exception as e:
            logger.warning(f"Failed to fetch from Open-Meteo: {e}")
    
    # Fallback to default/synthetic weather data
    logger.warning("Using default weather data - replace with actual data!")
    # Generate dates
    dates = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    
    weather_data = {
        'dates': dates,
        'temperature_2m': [25.0] * len(dates),  # °C
        'relative_humidity': [50.0] * len(dates),  # %
        'wind_speed': [3.0] * len(dates),  # m/s
        'pressure': [1013.25] * len(dates),  # hPa
        'solar_radiation': [800.0] * len(dates),  # W/m²
    }
    return weather_data


def fetch_weather_data(
    roi_center: tuple,
    target_date: str,
    method: str = 'open-meteo'
) -> dict:
    """
    Fetch weather data for the target date and location.
    
    Args:
        roi_center: (lon, lat) of ROI center
        target_date: Date string (YYYY-MM-DD)
        method: 'open-meteo' or 'csv'
    
    Returns:
        Weather data dictionary
    """
    logger.info(f"Fetching weather data for {target_date} at {roi_center}")
    
    if method == 'open-meteo' and WEATHER_FETCHER_AVAILABLE:
        # Use dynamic weather fetcher
        fetcher = DynamicWeatherFetcher()
        weather_data = fetcher.fetch_weather(
            latitude=roi_center[1],
            longitude=roi_center[0],
            start_date=target_date,
            end_date=target_date
        )
        logger.info("Weather data fetched from Open-Meteo")
        return weather_data
    else:
        # Use default/synthetic weather data
        # In production, you would load from CSV or another source
        logger.warning("Using default weather data - replace with actual data!")
        weather_data = {
            'dates': [datetime.strptime(target_date, '%Y-%m-%d')],
            'temperature_2m': [25.0],  # °C
            'relative_humidity': [50.0],  # %
            'wind_speed': [3.0],  # m/s
            'pressure': [1013.25],  # hPa
            'solar_radiation': [800.0],  # W/m²
        }
        return weather_data


def calculate_roi_center(roi_geometry: dict) -> tuple:
    """Calculate the center of the ROI (lon, lat)."""
    roi_shapely = shape(roi_geometry)
    centroid = roi_shapely.centroid
    return (centroid.x, centroid.y)


def run_metric_pipeline(
    cube: xr.Dataset,
    weather_data: dict,
    output_dir: str,
    config: dict = None
) -> dict:
    """
    Run the METRIC pipeline on the fetched data.
    
    Args:
        cube: DataCube with Landsat data
        weather_data: Weather data dictionary
        output_dir: Output directory for results
        config: Optional configuration dictionary
    
    Returns:
        Results dictionary
    """
    logger.info("Starting METRIC pipeline")
    
    # Initialize pipeline
    pipeline = METRICPipeline(config=config)
    
    # Load data (we already have a DataCube)
    # Note: The pipeline expects to load data itself, so we need to adapt
    # For now, we'll use the pipeline's internal methods
    pipeline.data = cube
    
    # Preprocess
    logger.info("Preprocessing data")
    pipeline.preprocess()
    
    # Calculate surface properties
    logger.info("Calculating surface properties")
    pipeline.calculate_surface_properties()
    
    # Calculate radiation balance
    logger.info("Calculating radiation balance")
    pipeline.calculate_radiation_balance()
    
    # Calculate energy balance
    logger.info("Calculating energy balance")
    pipeline.calculate_energy_balance()
    
    # Calibrate
    logger.info("Calibrating")
    pipeline.calibrate()
    
    # Calculate ET
    logger.info("Calculating ET")
    pipeline.calculate_et()
    
    # Save results
    logger.info(f"Saving results to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    pipeline.save_results(output_dir)
    
    # Get results
    results = pipeline.get_results()
    
    logger.info("METRIC pipeline completed successfully")
    return results


def process_single_scene(
    scene: dict,
    weather_data: dict,
    output_dir: str,
    config: dict = None
) -> dict:
    """
    Process a single downloaded scene: run METRIC pipeline.
    
    Args:
        scene: Scene info dictionary from fetch_all_scenes
        weather_data: Weather data dictionary (single date or range)
        output_dir: Output directory for results
        config: Optional METRIC configuration
    
    Returns:
        Results dictionary
    """
    scene_date = scene['date']
    scene_dir = scene['directory']
    scene_id = scene['scene_id']
    
    logger.info(f"Processing scene: {scene_id} for date {scene_date}")
    logger.info(f"Cloud cover: {scene['cloud_cover']}%")
    logger.info(f"Scene directory: {scene_dir}")
    
    # Extract weather for this specific date if weather data is a range
    weather_for_scene = None
    if weather_data and len(weather_data.get('dates', [])) > 1:
        # Find the index for this scene's date
        scene_idx = None
        dates = weather_data.get('dates', [])
        for i, d in enumerate(dates):
            if isinstance(d, datetime):
                d_str = d.strftime('%Y-%m-%d')
            else:
                d_str = str(d)[:10]
            if d_str == scene_date:
                scene_idx = i
                break
        
        if scene_idx is not None:
            # Extract single-date weather
            weather_for_scene = {
                'dates': [dates[scene_idx]],
                'temperature_2m': [weather_data['temperature_2m'][scene_idx]],
                'relative_humidity': [weather_data['relative_humidity'][scene_idx]],
                'wind_speed': [weather_data['wind_speed'][scene_idx]],
                'pressure': [weather_data['pressure'][scene_idx]],
                'solar_radiation': [weather_data['solar_radiation'][scene_idx]],
            }
    
    # If no weather data found in range, fetch it dynamically for this scene
    if weather_for_scene is None:
        if WEATHER_FETCHER_AVAILABLE:
            logger.info(f"Fetching weather dynamically for scene {scene_id}")
            # Use DynamicWeatherFetcher which interpolates over the scene area
            # This requires target coordinates which we don't have yet
            # So we'll use a simple approach: fetch from Open-Meteo directly
            import requests
            try:
                # Get scene center from MTL
                mtl_path = os.path.join(scene_dir, 'MTL.json')
                with open(mtl_path, 'r') as f:
                    mtl = json.load(f)
                
                # Get bbox and compute center
                bbox = mtl.get('bbox')
                if bbox:
                    lon = (bbox[0] + bbox[2]) / 2
                    lat = (bbox[1] + bbox[3]) / 2
                    
                    url = "https://archive-api.open-meteo.com/v1/archive"
                    params = {
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": scene_date,
                        "end_date": scene_date,
                        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,shortwave_radiation",
                        "daily": "et0_fao_evapotranspiration",
                        "timezone": "auto"
                    }
                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'hourly' in data:
                        hourly = data['hourly']
                        # Get 10:30 values
                        weather_for_scene = {
                            'dates': [datetime.strptime(scene_date, '%Y-%m-%d')],
                            'temperature_2m': [hourly['temperature_2m'][10] if hourly['temperature_2m'][10] is not None else 25.0],
                            'relative_humidity': [hourly['relative_humidity_2m'][10] if hourly['relative_humidity_2m'][10] is not None else 50.0],
                            'wind_speed': [hourly['wind_speed_10m'][10] if hourly['wind_speed_10m'][10] is not None else 3.0],
                            'pressure': [hourly['surface_pressure'][10] if hourly['surface_pressure'][10] is not None else 1013.25],
                            'solar_radiation': [hourly['shortwave_radiation'][10] if hourly['shortwave_radiation'][10] is not None else 800.0],
                        }
            except Exception as e:
                logger.warning(f"Failed to fetch weather for {scene_date}: {e}")
        
        # Fallback to default weather
        if weather_for_scene is None:
            logger.warning(f"Using default weather for {scene_date}")
            weather_for_scene = {
                'dates': [datetime.strptime(scene_date, '%Y-%m-%d')],
                'temperature_2m': [25.0],
                'relative_humidity': [50.0],
                'wind_speed': [3.0],
                'pressure': [1013.25],
                'solar_radiation': [800.0],
            }
    
    # Run METRIC pipeline using the downloaded scene directory
    # Use the run() method which handles loading and processing
    pipeline = METRICPipeline(config=config)
    
    try:
        results = pipeline.run(
            landsat_dir=scene_dir,
            meteo_data=weather_for_scene,
            output_dir=output_dir,
            roi_path=None
        )
        
        logger.info(f"Processing complete for scene {scene_id}")
        return results
    except Exception as e:
        logger.error(f"METRIC pipeline failed for {scene_id}: {e}")
        raise


def process_single_date(
    roi_geometry: dict,
    target_date: str,
    output_dir: str,
    max_cloud_cover: float = 50.0,
    weather_method: str = 'open-meteo',
    config: dict = None
) -> dict:
    """
    Process a single date: fetch Landsat, fetch weather, run METRIC.
    
    Args:
        roi_geometry: ROI GeoJSON geometry
        target_date: Target date (YYYY-MM-DD)
        output_dir: Output directory
        max_cloud_cover: Maximum cloud cover percentage
        weather_method: Method for weather data ('open-meteo' or 'default')
        config: Optional METRIC configuration
    
    Returns:
        Results dictionary
    """
    logger.info(f"Processing date: {target_date}")
    
    # Step 1: Fetch Landsat data for this specific date (downloads GeoTIFFs and MTL.json)
    # For single date processing, we use the fetcher directly
    fetcher = PlanetaryComputerLandsatFetcher(
        max_cloud_cover=max_cloud_cover
    )
    
    try:
        scenes = fetcher.fetch_scenes(
            roi_geometry=roi_geometry,
            date_range=(target_date, target_date),
            output_dir=output_dir,
            min_cloud_cover=0.0,
            resolution=RESOLUTION,
            sort_by='cloud_cover'
        )
    except Exception as e:
        logger.error(f"Failed to fetch Landsat data: {e}")
        raise
    
    if not scenes:
        raise RuntimeError(f"No Landsat scenes found for {target_date}")
    
    # Use the best scene (first one, sorted by cloud cover)
    scene = scenes[0]
    scene_dir = scene['directory']
    logger.info(f"Using scene: {scene['scene_id']}")
    logger.info(f"Cloud cover: {scene['cloud_cover']}%")
    logger.info(f"Scene directory: {scene_dir}")
    
    # Step 2: Fetch weather data
    roi_center = calculate_roi_center(roi_geometry)
    weather_data = fetch_weather_data(
        roi_center=roi_center,
        target_date=target_date,
        method=weather_method
    )
    
    # Step 3: Run METRIC pipeline using the downloaded scene directory
    # Use the run() method which handles loading and processing
    pipeline = METRICPipeline(config=config)
    
    try:
        results = pipeline.run(
            landsat_dir=scene_dir,
            meteo_data=weather_data,
            output_dir=output_dir,
            roi_path=None
        )
        
        logger.info(f"Processing complete for {target_date}")
        return results
    except Exception as e:
        logger.error(f"METRIC pipeline failed for {target_date}: {e}")
        raise


def main(
    roi_path: str = ROI_PATH,
    output_dir: str = OUTPUT_DIR,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    max_cloud_cover: float = MAX_CLOUD_COVER,
    config_path: str = None
):
    """
    Main execution function.
    
    This version:
    1. Finds ALL scenes in the date range
    2. Downloads them all at once
    3. Passes each scene folder to METRIC pipeline
    
    Args:
        roi_path: Path to GeoJSON file with ROI
        output_dir: Output directory for results
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_cloud_cover: Maximum cloud cover percentage
        config_path: Optional path to METRIC config YAML
    """
    
    # Validate inputs
    if not start_date or not end_date:
        raise ValueError("Must provide both start_date and end_date")
    
    # Parse dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Load ROI
    logger.info(f"Loading ROI from {roi_path}")
    roi_geometry = load_roi(roi_path)
    
    # Load METRIC config if provided
    config = None
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded METRIC configuration from {config_path}")
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a directory for downloaded scenes
    scenes_dir = os.path.join(output_dir, 'scenes')
    os.makedirs(scenes_dir, exist_ok=True)
    
    # Step 1: Fetch ALL Landsat scenes for the entire date range
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching ALL Landsat scenes for date range")
    logger.info("=" * 60)
    
    try:
        scenes = fetch_all_scenes(
            roi_geometry=roi_geometry,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            output_dir=scenes_dir,
            source_crs=SOURCE_CRS
        )
    except Exception as e:
        logger.error(f"Failed to fetch scenes: {e}")
        return {}
    
    if not scenes:
        logger.warning("No scenes found for the specified date range")
        return {}
    
    logger.info(f"Downloaded {len(scenes)} scene(s) to {scenes_dir}")
    
    # Step 2: Fetch weather data for the entire date range
    logger.info("=" * 60)
    logger.info("STEP 2: Fetching weather data for date range")
    logger.info("=" * 60)
    
    roi_center = calculate_roi_center(roi_geometry)
    weather_data = fetch_weather_data_range(
        roi_center=roi_center,
        start_date=start_date,
        end_date=end_date,
        method='open-meteo'
    )
    
    # Step 3: Process each downloaded scene with METRIC
    logger.info("=" * 60)
    logger.info("STEP 3: Processing scenes with METRIC pipeline")
    logger.info("=" * 60)
    
    all_results = {}
    for scene in scenes:
        scene_date = scene['date']
        scene_id = scene['scene_id']
        
        try:
            output_scene_dir = os.path.join(output_dir, f"result_{scene_date}")
            results = process_single_scene(
                scene=scene,
                weather_data=weather_data,
                output_dir=output_scene_dir,
                config=config
            )
            all_results[scene_date] = results
            logger.info(f"Successfully processed {scene_date} ({scene_id})")
        except Exception as e:
            logger.error(f"Failed to process scene {scene_id} for {scene_date}: {e}")
            # Continue with next scene
            continue
    
    # Summary
    logger.info("=" * 60)
    logger.info("Processing Summary")
    logger.info("=" * 60)
    logger.info(f"ROI: {roi_path}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Total scenes downloaded: {len(scenes)}")
    logger.info(f"Successfully processed: {len(all_results)}")
    logger.info(f"Failed: {len(scenes) - len(all_results)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Scenes directory: {scenes_dir}")
    
    if all_results:
        logger.info("\nProcessed dates:")
        for date in all_results.keys():
            logger.info(f"  - {date}")
    
    return all_results


def run_pipeline(roi_path: str, date_range: tuple, output_dir: str = None, **kwargs):
    """
    Simple entry point to run the METRIC pipeline.
    
    Args:
        roi_path: Path to GeoJSON file with ROI geometry
        date_range: Tuple of (start_date, end_date) as strings
        output_dir: Output directory (default: ./metric_output)
        **kwargs: Additional parameters:
            - max_cloud_cover: Maximum cloud cover percentage (default: 50.0)
            - resolution: Output resolution in meters (default: 30.0)
            - config_path: Path to METRIC config YAML
    
    Returns:
        Dictionary of results by date
    
    Example:
        results = run_pipeline(
            roi_path='amirkabir.geojson',
            date_range=('2023-04-01', '2023-04-30'),
            output_dir='my_output'
        )
    """
    start_date, end_date = date_range
    output_dir = output_dir or OUTPUT_DIR
    
    return main(
        roi_path=roi_path,
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=kwargs.get('max_cloud_cover', MAX_CLOUD_COVER),
        config_path=kwargs.get('config_path')
    )


if __name__ == "__main__":
    # Run with default configuration
    main()
