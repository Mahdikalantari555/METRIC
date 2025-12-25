#!/usr/bin/env python3
"""
METRIC ETa Calculation Script for Data Folder
Processes all Landsat scenes in the data folder and computes ETa using the METRICPipeline.

This script uses the unified METRICPipeline class to automatically handle the complete
METRIC algorithm workflow:
1. Data loading and preprocessing
2. Surface property calculations (NDVI, albedo, emissivity, roughness)
3. Radiation balance calculations
4. Energy balance calculations with METRIC calibration
5. ET calculations (instantaneous and daily)
6. Output generation

The script processes all Landsat scenes found in the data directory and saves
results to the output directory.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

# Suppress xarray casting warnings
warnings.filterwarnings("ignore", message="invalid value encountered in cast")

# Add the metric_et package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metric_et'))

import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask

from metric_et.pipeline import METRICPipeline
from metric_et.io import MeteoReader
from metric_et.output import OutputWriter, write_geotiff
from metric_et.utils import Logger

def process_single_scene(scene_path, weather_data, output_dir):
    """Process a single Landsat scene and compute ETa using METRICPipeline"""

    logger = Logger.get_logger("metric_et")
    logger.info(f"Processing scene: {scene_path}")

    try:
        # Create pipeline instance
        pipeline = METRICPipeline()

        # Run the complete METRIC pipeline
        results = pipeline.run(
            landsat_dir=scene_path,
            meteo_data={},  # Pipeline handles weather data internally
            output_dir=output_dir
        )

        # Get scene information
        scene_id = os.path.basename(scene_path)

        # Access the pipeline's data cube for additional processing
        cube = pipeline.data
        scene_date = cube.acquisition_time.date() if cube.acquisition_time else None

        # Get calibration coefficients
        calibration_result = getattr(pipeline, '_calibration_result', None)
        if calibration_result:
            calibration_a = calibration_result.a_coefficient
            calibration_b = calibration_result.b_coefficient
            logger.info(f"Calibration: a={calibration_a:.6f}, b={calibration_b:.6f}")
            logger.info(f"Calibration result type: {type(calibration_result)}")
            logger.info(f"ts_hot type: {type(calibration_result.ts_hot)}, shape: {getattr(calibration_result.ts_hot, 'shape', 'no shape')}")
            logger.info(f"Hot pixel: Ts={calibration_result.ts_hot!r}, dT_hot={calibration_result.dT_hot!r}")
            logger.info(f"Cold pixel: Ts={calibration_result.ts_cold!r}, dT_cold={calibration_result.dT_cold!r}")
        else:
            calibration_a = calibration_b = None
            logger.warning("No calibration result found")

        # Get statistics for all calculated parameters
        param_stats = {}
        for key, data_array in results.items():
            if data_array is not None:
                try:
                    min_val = float(data_array.min())
                    max_val = float(data_array.max())
                    mean_val = float(data_array.mean())
                    param_stats[f"{key}_min"] = min_val
                    param_stats[f"{key}_max"] = max_val
                    param_stats[f"{key}_mean"] = mean_val
                    logger.debug(f"{key}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}")
                except Exception as e:
                    logger.warning(f"Could not compute stats for {key}: {e}")
                    param_stats[f"{key}_min"] = param_stats[f"{key}_max"] = param_stats[f"{key}_mean"] = None

        # Get weather data values (spatially uniform)
        weather_keys = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure", "shortwave_radiation", "et0_fao_evapotranspiration"]
        for key in weather_keys:
            data = cube.get(key)
            if data is not None:
                try:
                    param_stats[f"{key}_value"] = float(data.mean())  # uniform value
                except:
                    param_stats[f"{key}_value"] = None

        logger.info(f" Scene {scene_id} processed successfully")

        return {
            "scene_id": scene_id,
            "date": scene_date,
            "status": "success",
            "calibration_a": calibration_a,
            "calibration_b": calibration_b,
            **param_stats  # Include all parameter statistics
        }

    except Exception as e:
        logger.error(f"✗ Failed to process {scene_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "scene_id": os.path.basename(scene_path),
            "date": None,
            "status": "failed",
            "error": str(e)
        }

def main():
    """Main function to process all scenes in data folder"""

    # Setup logging
    Logger.setup("metric_et", level="DEBUG")
    logger = Logger.get_logger("metric_et")

    # Define paths
    data_dir = r"C:\Users\ASUS\Desktop\et\METRIC\data"
    output_dir = r"C:\Users\ASUS\Desktop\et\METRIC\output"

    # Create output directory (delete if exists)
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting METRIC ETa calculation for data folder")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Find all Landsat scenes
    scene_pattern = os.path.join(data_dir, "landsat_*")
    scene_paths = glob.glob(scene_pattern)

    if not scene_paths:
        logger.error("No Landsat scenes found in data directory")
        return

    logger.info(f"Found {len(scene_paths)} Landsat scenes to process")

    # Process each scene
    results = []
    for scene_path in scene_paths:
        result = process_single_scene(scene_path, None, output_dir)
        results.append(result)

    # Save summary
    summary_path = os.path.join(output_dir, "processing_summary.csv")
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(summary_path, index=False)

    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Processing complete: {successful}/{len(results)} scenes successful")
    logger.info(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
