#!/usr/bin/env python3
"""
ET Interpolation and Extrapolation Script
===========================================

This script interpolates and extrapolates daily ET values based on the METRIC
ETa calculation results from Calculate_et.py.

The script uses the ETExtrapolator class from metric_et.et.extrapolation to:
1. Interpolate ETa between Landsat scene dates (filling gaps)
2. Extrapolate ETa for future dates beyond the last Landsat scene

Usage:
    python interpolate_et.py

Output:
    - Daily ETa tiff files for interpolated and extrapolated dates
    - Combined NetCDF file with all daily ETa values
    - Summary CSV with statistics
"""

import os
import sys
import glob
import json
import logging
import re
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Add the metric_et package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metric_et'))

from metric_et.et.extrapolation import ETExtrapolator, create_extrapolator

# Try to import Logger, fall back to standard logging if not available
try:
    from metric_et.utils import Logger
except ImportError:
    import logging
    Logger = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ETInterpolator:
    """
    Main class for interpolating and extrapolating ET values.
    
    This class:
    1. Loads ETrF and ETa daily data from METRIC output
    2. Extracts scene dates and ETrF values
    3. Fetches ET0 data from Open-Meteo API
    4. Interpolates ET between scenes
    5. Extrapolates ET beyond the last scene
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        extrapolation_days: int = 14,
        interpolation_method: str = "weighted",
        gap_threshold: int = 30
    ):
        """
        Initialize the ET interpolator.
        
        Args:
            input_dir: Directory containing METRIC output files
            output_dir: Directory to save interpolated/extrapolated results
            extrapolation_days: Number of days to extrapolate beyond last scene
            interpolation_method: "linear" or "weighted"
            gap_threshold: Maximum gap (days) to interpolate
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.extrapolation_days = extrapolation_days
        self.interpolation_method = interpolation_method
        self.gap_threshold = gap_threshold
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize extrapolator
        self.extrapolator = create_extrapolator({
            "extrapolation_days": extrapolation_days,
            "interpolation_method": interpolation_method,
            "min_et_daily": 0.0,
            "max_et_daily": 15.0,  # Max ET in mm/day
            "etrf_min": 0.0,
            "etrf_max": 1.5
        })
        
        # Data storage
        self.scenes: List[Dict] = []
        self.etrf_scenes: List[Tuple[datetime, np.ndarray]] = []
        self.et0_data: Optional[xr.DataArray] = None
        self.bbox: List[float] = []
        
    def load_scenes(self) -> List[Dict]:
        """
        Load all Landsat scenes from the input directory.
        
        Supports two naming conventions:
        1. Old: ETrF_METRIC_YYYYMMDD.tif, ETa_daily_METRIC_YYYYMMDD.tif
        2. New: ETrF_none_L2_30_LC08_YYYYMMDD_amirkabir.tif, ETaDaily_none_L2_30_LC08_YYYYMMDD_amirkabir.tif
        
        Returns:
            List of scene dictionaries with date, etrf, et_daily info
        """
        logger.info(f"Loading scenes from {self.input_dir}")
        
        # Find all ETrF files (two possible naming conventions)
        # Old format: ETrF_METRIC_YYYYMMDD.tif
        # New format: ETrF_none_L2_30_LC08_YYYYMMDD_amirkabir.tif
        etrf_pattern = os.path.join(self.input_dir, "ETrF_*.tif")
        
        etrf_files = sorted(glob.glob(etrf_pattern))
        
        # Filter and parse dates
        scenes = []
        for etrf_file in etrf_files:
            basename = os.path.basename(etrf_file)
            
            # Try to extract date from different naming patterns
            date = None
            date_str = ""
            
            # New format: ETrF_none_L2_30_LC08_YYYYMMDD_amirkabir.tif
            # or: ETrF_none_L2_30_LC09_YYYYMMDD_amirkabir.tif
            if '_LC08_' in basename or '_LC09_' in basename:
                # Extract date after LC08_ or LC09_
                import re
                match = re.search(r'LC0[89]_(\d{8})_', basename)
                if match:
                    date_str = match.group(1)
                    try:
                        date = datetime.strptime(date_str, "%Y%m%d")
                    except ValueError:
                        pass
            
            # Old format: ETrF_METRIC_YYYYMMDD.tif
            if date is None and 'ETrF_METRIC_' in basename:
                date_str = basename.replace("ETrF_METRIC_", "").replace(".tif", "")
                try:
                    date = datetime.strptime(date_str, "%Y%m%d")
                except ValueError:
                    pass
            
            if date is None:
                logger.warning(f"Could not parse date from {basename}")
                continue
            
            # Find corresponding ET daily file
            # New format: ETaDaily_none_L2_30_LC08_YYYYMMDD_amirkabir.tif
            # Old format: ETa_daily_METRIC_YYYYMMDD.tif
            et_daily_file = None
            
            # Try new format first
            if '_LC08_' in basename or '_LC09_' in basename:
                sensor = 'LC08' if '_LC08_' in basename else 'LC09'
                et_daily_pattern = f"ETaDaily_landsat8_L2SP_30_{sensor}_{date_str}_amirkabir.tif"
                et_daily_file = os.path.join(self.input_dir, et_daily_pattern)
            
            # Try old format
            if et_daily_file is None or not os.path.exists(et_daily_file):
                et_daily_pattern = f"ETa_daily_METRIC_{date_str}.tif"
                et_daily_file = os.path.join(self.input_dir, et_daily_pattern)
            
            if et_daily_file is None or not os.path.exists(et_daily_file):
                logger.warning(f"ET daily file not found for {date_str}")
                continue
            
            # Load ETrF raster
            with rasterio.open(etrf_file) as src:
                etrf_data = src.read(1)
                transform = src.transform
                crs = src.crs
                nodata = src.nodata
                
                # Handle nodata values
                if nodata is not None:
                    etrf_data = np.where(etrf_data == nodata, np.nan, etrf_data)
                
                # Get spatial extent
                bounds = src.bounds
                self.bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
                shape = (src.height, src.width)
            
            # Load ET daily raster
            with rasterio.open(et_daily_file) as src:
                et_daily_data = src.read(1)
                nodata_et = src.nodata
                
                if nodata_et is not None:
                    et_daily_data = np.where(et_daily_data == nodata_et, np.nan, et_daily_data)
            
            # Calculate ETrF statistics
            etrf_valid = etrf_data[~np.isnan(etrf_data)]
            if len(etrf_valid) > 0:
                etrf_mean = float(np.nanmean(etrf_valid))
                etrf_min = float(np.nanmin(etrf_valid))
                etrf_max = float(np.nanmax(etrf_valid))
            else:
                etrf_mean = etrf_min = etrf_max = np.nan
            
            scene_info = {
                'date': date,
                'date_str': date.strftime("%Y-%m-%d"),
                'etrf_file': etrf_file,
                'et_daily_file': et_daily_file,
                'etrf_data': etrf_data,
                'et_daily_data': et_daily_data,
                'etrf_mean': etrf_mean,
                'etrf_min': etrf_min,
                'etrf_max': etrf_max,
                'transform': transform,
                'crs': crs,
                'shape': shape
            }
            
            scenes.append(scene_info)
            logger.info(f"Loaded scene: {date_str}, ETrF mean: {etrf_mean:.3f}")
        
        # Sort by date
        scenes.sort(key=lambda x: x['date'])
        
        self.scenes = scenes
        logger.info(f"Loaded {len(scenes)} scenes total")
        
        return scenes
    
    def prepare_etrf_scenes(self) -> List[Tuple[datetime, np.ndarray]]:
        """
        Prepare ETrF scenes for interpolation/extrapolation.
        
        Returns:
            List of (date, etrf_array) tuples
        """
        logger.info("Preparing ETrF scenes for interpolation")
        
        etrf_scenes = []
        
        for scene in self.scenes:
            # Use ETrF data (2D array)
            etrf_data = scene['etrf_data'].copy()
            
            # Replace NaN with reasonable values for interpolation
            # Use the mean ETrF for the scene
            etrf_mean = scene['etrf_mean']
            if np.isnan(etrf_mean):
                etrf_mean = 0.5  # Default ETrF
            
            etrf_data = np.where(np.isnan(etrf_data), etrf_mean, etrf_data)
            
            etrf_scenes.append((scene['date'], etrf_data))
        
        self.etrf_scenes = etrf_scenes
        
        logger.info(f"Prepared {len(etrf_scenes)} ETrF scenes")
        
        return etrf_scenes
    
    def fetch_et0_data(self) -> xr.DataArray:
        """
        Fetch ET0 data for the date range covering all scenes plus extrapolation.
        
        Returns:
            xarray DataArray with ET0 values
        """
        if not self.scenes:
            raise ValueError("No scenes loaded. Call load_scenes() first.")
        
        # Get date range
        first_date = self.scenes[0]['date']
        last_date = self.scenes[-1]['date'] + timedelta(days=self.extrapolation_days)
        
        start_date = first_date.strftime("%Y-%m-%d")
        end_date = last_date.strftime("%Y-%m-%d")
        
        logger.info(f"Fetching ET0 data from {start_date} to {end_date}")
        
        try:
            # Try to fetch historical + forecast ET0
            if self.bbox:
                et0_hist = self.extrapolator.fetch_et0_historical(
                    bbox=self.bbox,
                    start_date=start_date,
                    end_date=last_date.strftime("%Y-%m-%d")
                )
                self.et0_data = et0_hist
                logger.info(f"ET0 data loaded: {len(et0_hist)} days")
            else:
                # Use default bbox if not set
                # Tehran area default
                default_bbox = [51.0, 35.0, 52.0, 36.0]
                et0_hist = self.extrapolator.fetch_et0_historical(
                    bbox=default_bbox,
                    start_date=start_date,
                    end_date=last_date.strftime("%Y-%m-%d")
                )
                self.et0_data = et0_hist
                self.bbox = default_bbox
                logger.info(f"ET0 data loaded with default bbox: {len(et0_hist)} days")
                
        except Exception as e:
            logger.warning(f"Failed to fetch ET0 from API: {e}")
            logger.info("Generating synthetic ET0 data based on seasonal patterns")
            self.et0_data = self._generate_synthetic_et0(first_date, last_date)
        
        return self.et0_data
    
    def _generate_synthetic_et0(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> xr.DataArray:
        """
        Generate synthetic ET0 data based on typical seasonal patterns.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            xarray DataArray with synthetic ET0 values
        """
        logger.info("Generating synthetic ET0 data")
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Generate synthetic ET0 based on day of year
        # Typical pattern for arid regions: higher in summer, lower in winter
        et0_values = np.zeros(n_days)
        
        for i, date in enumerate(dates):
            day_of_year = date.timetuple().tm_yday
            
            # Simplified seasonal pattern (northern hemisphere)
            # Peak around day 172 (June 21), lowest around day 355-365 (December)
            # Range: 2-8 mm/day typical for arid regions
            seasonal_factor = np.sin(2 * np.pi * (day_of_year - 105) / 365)
            et0_values[i] = 5.0 + 2.5 * seasonal_factor
        
        # Create xarray DataArray
        et0_data = xr.DataArray(
            et0_values,
            dims=['time'],
            coords={'time': dates},
            attrs={
                'units': 'mm/day',
                'long_name': 'ET0 FAO Penman-Monteith (synthetic)',
                'source': 'Synthetic (seasonal pattern)',
                'bbox': self.bbox if self.bbox else [51.0, 35.0, 52.0, 36.0]
            }
        )
        
        return et0_data
    
    def interpolate(self) -> Dict:
        """
        Interpolate ETa between Landsat scenes.
        
        Returns:
            Dictionary with interpolation results
        """
        logger.info("Starting interpolation between Landsat scenes")
        
        if not self.etrf_scenes:
            self.prepare_etrf_scenes()
        
        if self.et0_data is None:
            self.fetch_et0_data()
        
        # Perform interpolation
        result = self.extrapolator.interpolate(
            etrf_scenes=self.etrf_scenes,
            et0_data=self.et0_data,
            method=self.interpolation_method,
            gap_threshold=self.gap_threshold
        )
        
        logger.info(f"Interpolation complete: {len(result.get('dates', []))} days")
        
        return result
    
    def extrapolate(self) -> Dict:
        """
        Extrapolate ETa for future dates beyond the last Landsat scene.
        
        Returns:
            Dictionary with extrapolation results
        """
        logger.info("Starting extrapolation beyond last Landsat scene")
        
        if not self.etrf_scenes:
            self.prepare_etrf_scenes()
        
        if self.et0_data is None:
            self.fetch_et0_data()
        
        # Get the last scene's ETrF
        last_scene = self.scenes[-1]
        etrf_last = last_scene['etrf_data'].copy()
        
        # Replace NaN with mean value
        etrf_mean = last_scene['etrf_mean']
        if np.isnan(etrf_mean):
            etrf_mean = 0.5
        etrf_last = np.where(np.isnan(etrf_last), etrf_mean, etrf_last)
        
        # Get ET0 for forecast period (days after last scene)
        last_date = last_scene['date']
        forecast_start = last_date + timedelta(days=1)
        forecast_end = last_date + timedelta(days=self.extrapolation_days)
        
        # Extract ET0 for forecast period
        try:
            et0_forecast = self.et0_data.sel(
                time=slice(forecast_start.strftime("%Y-%m-%d"), 
                          forecast_end.strftime("%Y-%m-%d"))
            )
        except Exception as e:
            logger.warning(f"Could not select forecast ET0: {e}")
            # Generate forecast ET0
            et0_forecast = self._generate_synthetic_et0(forecast_start, forecast_end)
        
        # Perform extrapolation
        result = self.extrapolator.extrapolate(
            etrf_last=etrf_last,
            et0_forecast=et0_forecast
        )
        
        logger.info(f"Extrapolation complete: {len(result.get('dates', []))} days")
        
        return result
    
    def save_results(
        self,
        interpolation_result: Dict,
        extrapolation_result: Dict
    ) -> None:
        """
        Save interpolation and extrapolation results to files.
        
        Args:
            interpolation_result: Results from interpolate()
            interpolation_result: Results from extrapolate()
        """
        logger.info("Saving results to output directory")
        
        # Get spatial reference from first scene
        if not self.scenes:
            logger.error("No scenes available to get spatial reference")
            return
        
        transform = self.scenes[0]['transform']
        crs = self.scenes[0]['crs']
        shape = self.scenes[0]['shape']
        
        # Save interpolated results
        if interpolation_result.get('dates'):
            interp_dates = interpolation_result['dates']
            interp_eta = interpolation_result['ETa_daily']
            
            logger.info(f"Saving {len(interp_dates)} interpolated files")
            
            for i, date in enumerate(interp_dates):
                if i >= len(interp_eta):
                    break
                    
                eta_data = interp_eta[i]
                
                # Handle NaN values
                eta_data = np.where(np.isnan(eta_data), -9999.0, eta_data)
                
                date_str = date.strftime("%Y%m%d")
                output_file = os.path.join(
                    self.output_dir,
                    f"ETa_interpolated_{date_str}.tif"
                )
                
                self._write_geotiff(
                    output_file,
                    eta_data,
                    transform,
                    crs,
                    nodata=-9999.0
                )
        
        # Save extrapolated results
        if extrapolation_result.get('dates'):
            extrap_dates = extrapolation_result['dates']
            extrap_eta = extrapolation_result['ETa_daily']
            
            logger.info(f"Saving {len(extrap_dates)} extrapolated files")
            
            for i, date in enumerate(extrap_dates):
                if i >= len(extrap_eta):
                    break
                    
                eta_data = extrap_eta[i]
                
                # Handle NaN values
                eta_data = np.where(np.isnan(eta_data), -9999.0, eta_data)
                
                date_str = date.strftime("%Y%m%d")
                output_file = os.path.join(
                    self.output_dir,
                    f"ETa_extrapolated_{date_str}.tif"
                )
                
                self._write_geotiff(
                    output_file,
                    eta_data,
                    transform,
                    crs,
                    nodata=-9999.0
                )
        
        # Save combined NetCDF
        self._save_netcdf(interpolation_result, extrapolation_result)
        
        # Save summary CSV
        self._save_summary(interpolation_result, extrapolation_result)
        
        logger.info("All results saved successfully")
    
    def _write_geotiff(
        self,
        output_path: str,
        data: np.ndarray,
        transform: rasterio.transform.Affine,
        crs: rasterio.crs.CRS,
        nodata: float = -9999.0
    ) -> None:
        """Write a GeoTIFF file."""
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
            compress='lzw'
        ) as dst:
            dst.write(data, 1)
    
    def _save_netcdf(
        self,
        interpolation_result: Dict,
        extrapolation_result: Dict
    ) -> None:
        """Save combined results as NetCDF file."""
        logger.info("Creating combined NetCDF file")
        
        all_dates = []
        all_eta = []
        
        # Add original scenes
        for scene in self.scenes:
            all_dates.append(scene['date'])
            all_eta.append(scene['et_daily_data'])
        
        # Add interpolated dates
        if interpolation_result.get('dates'):
            for i, date in enumerate(interpolation_result['dates']):
                all_dates.append(date)
                all_eta.append(interpolation_result['ETa_daily'][i])
        
        # Add extrapolated dates
        if extrapolation_result.get('dates'):
            for i, date in enumerate(extrapolation_result['dates']):
                all_dates.append(date)
                all_eta.append(extrapolation_result['ETa_daily'][i])
        
        if not all_eta:
            logger.warning("No data to save in NetCDF")
            return
        
        # Create DataArray
        dates_sorted = sorted(all_dates)
        date_to_eta = {d: e for d, e in zip(all_dates, all_eta)}
        
        eta_array = np.array([date_to_eta[d] for d in dates_sorted])
        
        eta_ds = xr.DataArray(
            eta_array,
            dims=['time', 'y', 'x'],
            coords={
                'time': dates_sorted
            },
            attrs={
                'units': 'mm/day',
                'long_name': 'Daily Evapotranspiration',
                'source': 'METRIC with interpolation/extrapolation'
            }
        )
        
        # Save to NetCDF
        output_nc = os.path.join(self.output_dir, "ETa_daily_combined.nc")
        eta_ds.to_netcdf(output_nc)
        logger.info(f"NetCDF saved: {output_nc}")
    
    def _save_summary(
        self,
        interpolation_result: Dict,
        extrapolation_result: Dict
    ) -> None:
        """Save summary statistics as CSV."""
        logger.info("Creating summary statistics")
        
        records = []
        
        # Original scenes
        for scene in self.scenes:
            records.append({
                'date': scene['date_str'],
                'type': 'observed',
                'source': 'Landsat',
                'etrf_mean': scene['etrf_mean'],
                'et_daily_mean': float(np.nanmean(scene['et_daily_data']))
            })
        
        # Interpolated
        if interpolation_result.get('dates'):
            interp_eta = interpolation_result['ETa_daily']
            for i, date in enumerate(interpolation_result['dates']):
                records.append({
                    'date': date.strftime("%Y-%m-%d"),
                    'type': 'interpolated',
                    'source': 'METRIC interpolation',
                    'etrf_mean': float(np.nanmean(interpolation_result.get('etrf_interpolated', [np.nan])[i])) if i < len(interpolation_result.get('etrf_interpolated', [])) else np.nan,
                    'et_daily_mean': float(np.nanmean(interp_eta[i]))
                })
        
        # Extrapolated
        if extrapolation_result.get('dates'):
            extrap_eta = extrapolation_result['ETa_daily']
            for i, date in enumerate(extrapolation_result['dates']):
                records.append({
                    'date': date.strftime("%Y-%m-%d"),
                    'type': 'extrapolated',
                    'source': 'METRIC extrapolation',
                    'etrf_mean': float(np.nanmean(extrapolation_result['ETrF_used'])),
                    'et_daily_mean': float(np.nanmean(extrap_eta[i]))
                })
        
        # Save CSV
        df = pd.DataFrame(records)
        df = df.sort_values('date')
        csv_path = os.path.join(self.output_dir, "ETa_summary.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Summary saved: {csv_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("ET INTERPOLATION AND EXTRAPOLATION SUMMARY")
        print("="*60)
        print(f"Original scenes: {len(self.scenes)}")
        print(f"Interpolated days: {len(interpolation_result.get('dates', []))}")
        print(f"Extrapolated days: {len(extrapolation_result.get('dates', []))}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print("="*60)


def main():
    """Main function to run ET interpolation and extrapolation."""
    
    # Setup logging
    if Logger:
        try:
            Logger.setup("metric_et", level="DEBUG")
        except:
            pass
    
    # Define paths
    input_dir = r"E:\RSGIS\METRIC-main\METRIC-main\amirkabir2020output"
    output_dir = r"E:\RSGIS\METRIC-main\METRIC-main\amirkabir2020output\interpolated"
    
    # Create interpolator
    interpolator = ETInterpolator(
        input_dir=input_dir,
        output_dir=output_dir,
        extrapolation_days=14,
        interpolation_method="linear",
        gap_threshold=30
    )
    
    # Load scenes
    scenes = interpolator.load_scenes()
    
    if len(scenes) < 2:
        logger.error("Need at least 2 scenes for interpolation")
        return
    
    # Display scene dates
    logger.info("Scene dates:")
    for scene in scenes:
        logger.info(f"  {scene['date_str']}: ETrF = {scene['etrf_mean']:.3f}")
    
    # Fetch ET0 data
    et0_data = interpolator.fetch_et0_data()
    logger.info(f"ET0 data loaded: {len(et0_data)} days")
    
    # Perform interpolation
    logger.info("="*60)
    logger.info("Starting interpolation")
    logger.info("="*60)
    interp_result = interpolator.interpolate()
    
    # Perform extrapolation
    logger.info("="*60)
    logger.info("Starting extrapolation")
    logger.info("="*60)
    extrap_result = interpolator.extrapolate()
    
    # Save results
    interpolator.save_results(interp_result, extrap_result)
    
    logger.info("ET interpolation and extrapolation complete!")


if __name__ == "__main__":
    main()
