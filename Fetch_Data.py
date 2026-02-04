import os
import geopandas as gpd
import mpcdl
import requests
import pandas as pd
from datetime import datetime, timedelta
import json

# ========================
# Configuration
# ========================
ROI_FILE = "debal.geojson"
OUTPUT_DIR = "debal1401"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target Landsat bands (excluding view angles since we'll use metadata only)
LANDSAT_BANDS = [
    "blue", "green", "red", "nir08", "swir16", "swir22",  # Reflectance
    "lwir11",                                            # Thermal (ST in Kelvin)
    "qa", "qa_pixel"                                     # QA bands
]

# Weather variables for Open-Meteo
HOURLY_WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "surface_pressure", "shortwave_radiation", "et0_fao_evapotranspiration"
]

# Specific path/row to filter
TARGET_PATH = '165'
TARGET_ROW = '039'

def create_metadata_with_sun_angles(item, scene_id, scene_date, cloud_cover, path, row, output_path):
    """Create metadata file with sun elevation and azimuth from STAC properties"""
    mtl_data = {
        "item_id": scene_id,
        "datetime": scene_date,
        "cloud_cover": cloud_cover,
        "path": path,
        "row": row,
        "sun_elevation": item.properties.get("view:sun_elevation", "N/A"),
        "sun_azimuth": item.properties.get("view:sun_azimuth", "N/A"),
        "view_angles_source": "STAC item properties",
        "geometry": item.geometry,
        "bbox": item.bbox,
        "note": "Created from STAC item with sun angles in metadata"
    }
    
    with open(output_path, 'w') as f:
        json.dump(mtl_data, f, indent=2, default=str)
    return True

# def download_weather_for_7am_utc(date, lat, lon, output_dir):
#     """Download weather data specifically for 7:00 UTC (10:30 Iran time)"""
#     try:
#         weather_params = {
#             "latitude": lat,
#             "longitude": lon,
#             "start_date": date,
#             "end_date": date,
#             "hourly": HOURLY_WEATHER_VARS,
#             "models": "best_match",
#             "timezone": "UTC"
#         }
        
#         response = requests.get("https://archive-api.open-meteo.com/v1/archive", 
#                               params=weather_params, timeout=30)
#         response.raise_for_status()
#         weather_data = response.json()
#         hourly = weather_data.get('hourly', {})
        
#         if not hourly or 'time' not in hourly:
#             return None, "No hourly data available"
        
#         df_dict = {'time': pd.to_datetime(hourly['time'])}
#         for var in HOURLY_WEATHER_VARS:
#             if var in hourly:
#                 df_dict[var] = hourly[var]
        
#         weather_df = pd.DataFrame(df_dict)
#         weather_7am = weather_df[weather_df['time'].dt.hour == 7]
        
#         if len(weather_7am) == 0:
#             return None, "No 7:00 UTC data available"
        
#         weather_7am = weather_7am.copy()
#         weather_7am['local_time'] = weather_7am['time'] + pd.Timedelta(hours=3, minutes=30)
#         weather_7am['time_zone'] = 'UTC'
#         weather_7am['local_time_zone'] = 'IRST (UTC+3:30)'
        
#         weather_file = os.path.join(output_dir, f"weather_{date.replace('-', '')}_7am_UTC.csv")
#         weather_7am.to_csv(weather_file, index=False)
        
#         return weather_file, "Success"
        
#     except Exception as e:
#         return None, f"Error: {str(e)}"

# ========================
# Main Pipeline
# ========================
def main():
    print("=" * 70)
    print("LANDSAT & WEATHER DATA DOWNLOAD PIPELINE")
    print("=" * 70)
    print(f"Target Path/Row: {TARGET_PATH}/{TARGET_ROW}")
    print("Features:")
    print("  • Downloads ALL filtered scenes (<30% cloud)")
    print("  • Sun elevation/azimuth in MTL.json metadata only")
    print("  • Weather data for 7:00 UTC (10:30 Iran time)")
    print("=" * 70)
    
    # 1. Load Area of Interest
    print("\n1. Loading Area of Interest (ROI)...")
    roi_gdf = gpd.read_file(ROI_FILE)
    if roi_gdf.crs.to_epsg() != 4326:
        roi_gdf = roi_gdf.to_crs(epsg=4326)
    bbox = list(roi_gdf.total_bounds)  # [min_lon, min_lat, max_lon, max_lat]
    centroid = roi_gdf.centroid.iloc[0]
    lat, lon = centroid.y, centroid.x
    print(f"   ✓ ROI bounds: {bbox}")
    print(f"   ✓ Centroid for weather: ({lat:.4f}, {lon:.4f})")

    # 2. Search for recent, low-cloud Landsat scenes
    print("\n2. Searching Microsoft Planetary Computer for Landsat scenes...")
    end_date = "2022-10-23"
    start_date = "2022-03-21"
    date_range = f"start_date/end_date".replace("start_date", start_date).replace("end_date", end_date)

    items = mpcdl.search_mpc_collection(
        collection="landsat-c2-l2",
        bbox=bbox,
        datetime_range=date_range,
        limit=100  # Increased to find more scenes
    )
    print(f"   ✓ Found {len(items)} total items in date range.")

    # Filter for Landsat 8/9, target path/row AND low cloud cover (<30%)
    low_cloud_items = []
    for item in items:
        scene_id = item.id
        cloud_cover = item.properties.get('eo:cloud_cover', 30)
        item_path = item.properties.get('landsat:wrs_path')
        item_row = item.properties.get('landsat:wrs_row')
        
        # Check if matches Landsat 8/9 (LC08/LC09) AND target path/row AND has low cloud cover
        if ((scene_id.startswith("LC08") or scene_id.startswith("LC09")) and
            str(item_path) == TARGET_PATH and 
            str(item_row) == TARGET_ROW and 
            cloud_cover < 30):
            low_cloud_items.append(item)
    
    print(f"   ✓ {len(low_cloud_items)} scenes with Path/Row {TARGET_PATH}/{TARGET_ROW} and cloud cover < 30%")
    
    if not low_cloud_items:
        # Try to show what scenes were found to help debug
        print("\n   Available scenes in ROI (for debugging):")
        for item in items[:10]:  # Show first 10
            item_path = item.properties.get('landsat:wrs_path', 'N/A')
            item_row = item.properties.get('landsat:wrs_row', 'N/A')
            cloud_cover = item.properties.get('eo:cloud_cover', 'N/A')
            print(f"     Path/Row: {item_path}/{item_row}, Cloud: {cloud_cover}%, ID: {item.id[:30]}...")
        raise RuntimeError(f"No Landsat scenes found with Path/Row {TARGET_PATH}/{TARGET_ROW} and cloud cover < 30%")
    
    # Sort by date (most recent first)
    low_cloud_items.sort(key=lambda x: x.properties["datetime"], reverse=True)
    
    # 3. Download all filtered scenes
    print(f"\n3. Processing {len(low_cloud_items)} filtered scenes...")
    all_downloaded_scenes = []
    
    for idx, item in enumerate(low_cloud_items, 1):
        scene_id = item.id
        scene_date = item.properties["datetime"][:10]  # YYYY-MM-DD
        cloud_cover = item.properties.get('eo:cloud_cover', 'N/A')
        path = item.properties.get('landsat:wrs_path', 'N/A')
        row = item.properties.get('landsat:wrs_row', 'N/A')
        sun_elevation = item.properties.get("view:sun_elevation", "N/A")
        sun_azimuth = item.properties.get("view:sun_azimuth", "N/A")
        
        print(f"\n   [{idx}/{len(low_cloud_items)}] Processing: {scene_id}")
        print(f"      Date: {scene_date}, Cloud: {cloud_cover}%, Path/Row: {path}/{row}")
        print(f"      Sun Angles: Elevation={sun_elevation}, Azimuth={sun_azimuth}")
        
        try:
            # Create scene directory
            scene_output_dir = os.path.join(OUTPUT_DIR, f"landsat_{scene_date.replace('-', '')}_{path}_{row}")
            os.makedirs(scene_output_dir, exist_ok=True)
            
            # 3a. Create MTL.json metadata file with sun angles
            print(f"      a. Creating MTL metadata with sun angles...")
            mtl_file_path = os.path.join(scene_output_dir, "MTL.json")
            
            # Create metadata with sun angles from STAC properties
            create_metadata_with_sun_angles(item, scene_id, scene_date, cloud_cover, path, row, mtl_file_path)
            print(f"         ✓ MTL.json created with sun angles in metadata")
            
            # 3b. Download and clip Landsat bands
            print(f"      b. Downloading {len(LANDSAT_BANDS)} bands...")
            try:
                downloaded_files = mpcdl.download_stac_assets_clipped(
                    stac_items=item,
                    asset_keys=LANDSAT_BANDS,
                    bbox=bbox,
                    output_dir=scene_output_dir,
                    resolution=30.0,
                    sign_items=True,
                    overwrite=False,
                    show_progress=True,
                    stack_bands=False
                )
                
                # Count successful downloads
                successful_bands = len([f for f in downloaded_files.values() if f and os.path.exists(f)])
                print(f"         ✓ {successful_bands}/{len(LANDSAT_BANDS)} bands downloaded")
                
                # Record scene info
                scene_info = {
                    "scene_id": scene_id,
                    "date": scene_date,
                    "cloud_cover": cloud_cover,
                    "path": path,
                    "row": row,
                    "sun_elevation": sun_elevation,
                    "sun_azimuth": sun_azimuth,
                    "directory": scene_output_dir,
                    "bands_downloaded": successful_bands,
                    "mtl_file": mtl_file_path
                }
                all_downloaded_scenes.append(scene_info)
                
            except Exception as e:
                print(f"         ✗ Band download error: {str(e)[:100]}...")
                scene_info = {
                    "scene_id": scene_id,
                    "date": scene_date,
                    "cloud_cover": cloud_cover,
                    "path": path,
                    "row": row,
                    "sun_elevation": sun_elevation,
                    "sun_azimuth": sun_azimuth,
                    "directory": scene_output_dir,
                    "bands_downloaded": 0,
                    "error": str(e)[:200],
                    "mtl_file": mtl_file_path
                }
                all_downloaded_scenes.append(scene_info)
        
        except Exception as e:
            print(f"      ✗ Scene processing failed: {str(e)[:100]}...")
    
    # 4. Download weather data for 7:00 UTC for each unique date
    print(f"\n4. Downloading weather data for 7:00 UTC (10:30 Iran time)...")
    
    # Get unique dates from downloaded scenes
    unique_dates = list(set([scene["date"] for scene in all_downloaded_scenes]))
    print(f"   ✓ Need weather data for {len(unique_dates)} unique dates")
    
    # weather_downloads = {}
    # for date in unique_dates:
    #     print(f"      Fetching weather for {date} at 7:00 UTC...")
    #     weather_file, status = download_weather_for_7am_utc(date, lat, lon, OUTPUT_DIR)
        
        # if weather_file:
        #     weather_downloads[date] = {
        #         "file": weather_file,
        #         "utc_time": "07:00",
        #         "iran_time": "10:30",
        #         "status": status
        #     }
        #     print(f"         ✓ Weather data saved: {os.path.basename(weather_file)}")
        # else:
        #     weather_downloads[date] = {
        #         "file": None,
        #         "utc_time": "07:00",
        #         "iran_time": "10:30",
        #         "status": status
        #     }
        #     print(f"         ✗ {status}")
    
    # 5. Generate summary report
    print(f"\n5. Generating summary report...")
    summary_file = os.path.join(OUTPUT_DIR, "download_summary.json")
    
    successful_scenes = [s for s in all_downloaded_scenes if s.get("bands_downloaded", 0) > 0]
    
    summary = {
        "pipeline_run_date": datetime.now().isoformat(),
        "roi_file": ROI_FILE,
        "roi_bounds": bbox,
        "roi_centroid": {"lat": lat, "lon": lon},
        "target_path": TARGET_PATH,
        "target_row": TARGET_ROW,
        "date_range": f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
        "cloud_threshold": 30,
        "weather_time": "7:00 UTC (10:30 Iran time)",
        "sun_angles_location": "MTL.json metadata only",
        "scenes_found": len(low_cloud_items),
        "scenes_processed": len(all_downloaded_scenes),
        "scenes_successful": len(successful_scenes),
        "scenes_failed": len(all_downloaded_scenes) - len(successful_scenes),
        "total_bands_requested": len(LANDSAT_BANDS) * len(all_downloaded_scenes),
        "total_bands_downloaded": sum([s.get("bands_downloaded", 0) for s in all_downloaded_scenes]),
        "scenes": all_downloaded_scenes,
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"   ✓ Summary saved: {summary_file}")
    
    # 6. Print final statistics
    print(f"\n" + "=" * 70)
    print("PIPELINE COMPLETED - FINAL STATISTICS")
    print("=" * 70)
    
    print(f"\nTarget Path/Row: {TARGET_PATH}/{TARGET_ROW}")
    print(f"Scenes Found: {len(low_cloud_items)}")
    print(f"Scenes Processed: {len(all_downloaded_scenes)}")
    print(f"Successfully Downloaded: {len(successful_scenes)}")
    print(f"Failed: {len(all_downloaded_scenes) - len(successful_scenes)}")
    
    if successful_scenes:
        print(f"\nDownloaded Scenes:")
        for scene in successful_scenes:
            print(f"  • {scene['scene_id']}")
            print(f"    Date: {scene['date']}, Cloud: {scene['cloud_cover']}%, "
                  f"Bands: {scene['bands_downloaded']}")
    
    total_bands = sum([s.get("bands_downloaded", 0) for s in all_downloaded_scenes])
    
    print(f"\nTotal Bands Downloaded: {total_bands}")
    print(f"Sun Angles: Stored in MTL.json metadata")
    
    print(f"\nOutput Directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)

if __name__ == "__main__":
    main()