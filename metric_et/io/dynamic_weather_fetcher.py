"""
Dynamic weather data fetcher for METRIC ET processing.

Fetches spatially varying meteorological data from Open-Meteo API
for grid points within Landsat scene bounding box.
"""

import json
import os
import numpy as np
import pandas as pd
import requests
import xarray as xr
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import interpolate
import logging

from .weather_cache import WeatherCache

logger = logging.getLogger(__name__)

class DynamicWeatherFetcher:
    """Fetches weather data dynamically for Landsat scenes."""

    # Open-Meteo API base URL for historical data
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    # Variables to fetch
    HOURLY_VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "surface_pressure",
        "shortwave_radiation"
    ]

    DAILY_VARIABLES = [
        "et0_fao_evapotranspiration",
        "shortwave_radiation_sum"
    ]

    def __init__(self, grid_spacing_km: float = 9.0, cache_dir: str = "cache", enable_cache: bool = True):
        """
        Initialize the weather fetcher.

        Args:
            grid_spacing_km: Spacing between grid points in kilometers
            cache_dir: Directory for weather cache
            enable_cache: Whether to use caching
        """
        self.grid_spacing_km = grid_spacing_km
        self.enable_cache = enable_cache
        self.cache = WeatherCache(cache_dir) if enable_cache else None

    def _extract_scene_id(self, landsat_dir: str) -> str:
        """
        Extract scene ID from Landsat directory.

        Args:
            landsat_dir: Path to Landsat scene directory

        Returns:
            Scene identifier string
        """
        mtl_path = os.path.join(landsat_dir, "MTL.json")
        if os.path.exists(mtl_path):
            try:
                with open(mtl_path, 'r') as f:
                    mtl_data = json.load(f)
                scene_id = mtl_data.get('LANDSAT_PRODUCT_ID')
                if scene_id:
                    return scene_id
            except Exception as e:
                logger.warning(f"Failed to extract scene ID from MTL.json: {e}")

        # Fallback to directory name
        return os.path.basename(landsat_dir.strip('/\\'))

    def _extract_scene_date(self, landsat_dir: str) -> str:
        """
        Extract scene date from Landsat directory.

        Args:
            landsat_dir: Path to Landsat scene directory

        Returns:
            Scene date string (YYYY-MM-DD)
        """
        mtl_path = os.path.join(landsat_dir, "MTL.json")
        if os.path.exists(mtl_path):
            try:
                with open(mtl_path, 'r') as f:
                    mtl_data = json.load(f)
                scene_date = mtl_data.get('datetime') or mtl_data.get('DATE_ACQUIRED')
                if scene_date:
                    return scene_date
            except Exception as e:
                logger.warning(f"Failed to extract scene date from MTL.json: {e}")

        # Fallback to directory name parsing (assuming format contains date)
        dir_name = os.path.basename(landsat_dir.strip('/\\'))
        # Try to extract date from directory name (various formats)
        import re
        date_match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', dir_name)
        if date_match:
            return f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"

        logger.warning(f"Could not extract scene date from {landsat_dir}")
        return "unknown"

    def fetch_weather_for_scene(self, landsat_dir: str, target_coords: Dict, actual_extent: Tuple = None) -> Dict[str, xr.DataArray]:
        """
        Fetch weather data for a Landsat scene and interpolate to target grid.

        Uses caching to avoid redundant API calls for the same scene.

        Args:
            landsat_dir: Path to Landsat scene directory
            target_coords: Target coordinates dict with 'y', 'x' arrays
            actual_extent: Actual processed extent (min_lon, min_lat, max_lon, max_lat) or None to use MTL bbox

        Returns:
            Dictionary of weather variable DataArrays on target grid
        """
        # Extract scene information
        scene_id = self._extract_scene_id(landsat_dir)
        scene_date = self._extract_scene_date(landsat_dir)

        # Parse scene metadata for bbox
        mtl_path = os.path.join(landsat_dir, "MTL.json")
        if not os.path.exists(mtl_path):
            raise FileNotFoundError(f"MTL.json not found in {landsat_dir}")

        with open(mtl_path, 'r') as f:
            mtl_data = json.load(f)

        # Extract bbox
        if actual_extent:
            # Use the actual processed extent (clipped to ROI)
            min_lon, min_lat, max_lon, max_lat = actual_extent
            bbox = [min_lon, min_lat, max_lon, max_lat]
            logger.info(f"Using actual processed extent for weather grid: {bbox}")
        else:
            # Fallback to MTL bbox
            bbox = mtl_data.get('bbox')
            if not bbox:
                raise ValueError("No bounding box found in MTL data")
            logger.info(f"Using MTL bbox for weather grid: {bbox}")

        # Check cache first
        if self.enable_cache and self.cache:
            cached_weather = self.cache.load_scene_weather(scene_id)
            if cached_weather:
                logger.info(f"Cache hit for scene {scene_id}")
                
                # Validate cache compatibility with current target grid
                try:
                    self._validate_cache_compatibility(cached_weather, target_coords)
                    return self._interpolate_cached_data(cached_weather, target_coords)
                except ValueError as e:
                    logger.warning(f"Cache data incompatible with current target grid: {e}")
                    logger.info("Clearing incompatible cache entry and fetching fresh data")
                    # Clear the incompatible cache entry
                    self.cache.clear_cache()  # Clear all for now, could be more selective
                    # Continue to fetch fresh data below
        

        # Cache miss - fetch from API
        logger.info(f"Cache miss for scene {scene_id}, fetching from API")

        # Create grid points
        grid_points = self._create_grid_points(bbox)
        logger.info(f"Created {len(grid_points)} grid points for scene {scene_date}")

        # Fetch weather data for each point
        weather_data = self._fetch_all_points_weather(grid_points, scene_date)

        if not weather_data['valid_points']:
            raise ValueError("No valid weather data fetched for any grid point")

        logger.info(f"Successfully fetched data for {len(weather_data['valid_points'])} points")

        # Cache the weather data
        if self.enable_cache and self.cache:
            # Convert numpy arrays to lists for JSON serialization
            cache_data = {
                'grid_points': [list(point) for point in weather_data['valid_points']],
                'weather_variables': {var: weather_data[var].tolist() for var in self.HOURLY_VARIABLES + self.DAILY_VARIABLES}
            }
            self.cache.save_scene_weather(scene_id, scene_date, cache_data, bbox, self.grid_spacing_km)

        # Interpolate to target grid
        result = self._interpolate_to_target_grid(weather_data, weather_data['valid_points'], target_coords)

        return result

    def _fetch_all_points_weather(self, grid_points: List[Tuple[float, float]], scene_date: str) -> Dict:
        """
        Fetch weather data for all grid points.

        Args:
            grid_points: List of (lat, lon) tuples
            scene_date: Scene date string

        Returns:
            Dictionary with weather data and valid points
        """
        weather_data = {'valid_points': []}
        for var in self.HOURLY_VARIABLES + self.DAILY_VARIABLES:
            weather_data[var] = []

        for lat, lon in grid_points:
            point_data = self._fetch_weather_for_point(lat, lon, scene_date)
            if point_data and all(k in point_data for k in self.HOURLY_VARIABLES + self.DAILY_VARIABLES):
                weather_data['valid_points'].append((lat, lon))
                for var in self.HOURLY_VARIABLES + self.DAILY_VARIABLES:
                    weather_data[var].append(point_data[var])

        # Convert to numpy arrays
        for var in self.HOURLY_VARIABLES + self.DAILY_VARIABLES:
            weather_data[var] = np.array(weather_data[var])

        return weather_data

    def _validate_cache_compatibility(self, cached_weather: Dict, target_coords: Dict) -> None:
        """
        Validate that cached weather data is compatible with current target grid.

        Args:
            cached_weather: Cached weather data dictionary
            target_coords: Target coordinates dict with 'y', 'x' arrays
            
        Raises:
            ValueError: If cached data is incompatible with target grid
        """
        # Extract target grid information
        target_lats = target_coords['y']
        target_lons = target_coords['x']
        
        # Handle different coordinate types
        if hasattr(target_lats, 'values'):
            target_lats_array = target_lats.values
        else:
            target_lats_array = target_lats
            
        if hasattr(target_lons, 'values'):
            target_lons_array = target_lons.values
        else:
            target_lons_array = target_lons
            
        # Calculate expected target size
        if target_lats_array.ndim == 1 and target_lons_array.ndim == 1:
            expected_size = len(target_lats_array) * len(target_lons_array)
        elif target_lats_array.ndim == 2 and target_lons_array.ndim == 2:
            expected_size = target_lats_array.size
        else:
            raise ValueError(f"Unexpected coordinate dimensions: lat ndim={target_lats_array.ndim}, lon ndim={target_lons_array.ndim}")
            
        # Check cached data structure
        valid_points = cached_weather['grid_points']
        weather_data = cached_weather['weather_variables']
        
        if not valid_points:
            raise ValueError("Cached weather data has no valid grid points")
            
        # Check if weather variables have consistent size with grid points
        for var_name, var_data in weather_data.items():
            if isinstance(var_data, list):
                var_size = len(var_data)
            else:
                var_size = len(var_data) if hasattr(var_data, '__len__') else 1
                
            if var_size != len(valid_points):
                raise ValueError(f"Weather variable {var_name} has size {var_size} but {len(valid_points)} grid points")
                
        logger.debug(f"Cache validation passed: {len(valid_points)} grid points, expected target size {expected_size}")
        
        # Note: We don't validate exact spatial compatibility since bbox might differ between scenes
        # The interpolation will handle spatial differences
        
    def _interpolate_cached_data(self, cached_weather: Dict, target_coords: Dict) -> Dict[str, xr.DataArray]:
        """
        Interpolate cached weather data to target grid.

        Args:
            cached_weather: Cached weather data dictionary
            target_coords: Target coordinates dict with 'y', 'x' arrays

        Returns:
            Dictionary of interpolated DataArrays
        """
        valid_points = cached_weather['grid_points']
        weather_data = cached_weather['weather_variables']

        # Convert lists back to numpy arrays if needed
        for var in weather_data:
            if isinstance(weather_data[var], list):
                weather_data[var] = np.array(weather_data[var])

        return self._interpolate_to_target_grid(weather_data, valid_points, target_coords)

    def _interpolate_to_target_grid(self, weather_data: Dict, points: List[Tuple[float, float]],
                                   target_coords: Dict) -> Dict[str, xr.DataArray]:
        """
        Interpolate weather data from grid points to target spatial grid.

        Args:
            weather_data: Dict of variable arrays at grid points
            points: List of (lat, lon) tuples for grid points
            target_coords: Target coordinates dict with 'y', 'x' arrays

        Returns:
            Dictionary of interpolated DataArrays
        """
        # Extract target lat/lon grids
        target_lats = target_coords['y']  # Assuming 'y' is latitude
        target_lons = target_coords['x']  # Assuming 'x' is longitude

        # Handle different coordinate types (numpy arrays vs xarray DataArrays)
        if hasattr(target_lats, 'values'):
            target_lats_array = target_lats.values
        else:
            target_lats_array = target_lats
            
        if hasattr(target_lons, 'values'):
            target_lons_array = target_lons.values
        else:
            target_lons_array = target_lons

        logger.debug(f"Target coordinates - lat shape: {target_lats_array.shape}, lon shape: {target_lons_array.shape}")
        logger.debug(f"Target coordinates - lat ndim: {target_lats_array.ndim}, lon ndim: {target_lons_array.ndim}")

        # Create full 2D grids if they are 1D
        if target_lats_array.ndim == 1 and target_lons_array.ndim == 1:
            logger.debug("Creating meshgrid from 1D coordinates")
            lon_grid, lat_grid = np.meshgrid(target_lons_array, target_lats_array)
            target_shape = lat_grid.shape
            target_lat_flat = lat_grid.flatten()
            target_lon_flat = lon_grid.flatten()
        elif target_lats_array.ndim == 2 and target_lons_array.ndim == 2:
            logger.debug("Using 2D coordinates directly")
            target_shape = target_lats_array.shape
            target_lat_flat = target_lats_array.flatten()
            target_lon_flat = target_lons_array.flatten()
        else:
            # Handle mixed dimensionality
            logger.warning(f"Mixed coordinate dimensions: lat ndim={target_lats_array.ndim}, lon ndim={target_lons_array.ndim}")
            if target_lats_array.ndim == 1:
                # Assume lon is also 1D and create meshgrid
                lon_grid, lat_grid = np.meshgrid(target_lons_array, target_lats_array)
                target_shape = lat_grid.shape
                target_lat_flat = lat_grid.flatten()
                target_lon_flat = lon_grid.flatten()
            else:
                # Assume lat is already 2D, flatten both
                target_shape = target_lats_array.shape
                target_lat_flat = target_lats_array.flatten()
                target_lon_flat = target_lons_array.flatten()

        # Ensure we have the correct number of target points
        expected_size = target_shape[0] * target_shape[1]
        actual_size = len(target_lat_flat)
        
        logger.debug(f"Target grid analysis - shape: {target_shape}, expected size: {expected_size}, actual size: {actual_size}")
        
        if actual_size != expected_size:
            logger.error(f"Target grid size mismatch: expected {expected_size}, got {actual_size}")
            logger.error(f"This suggests coordinates may be duplicated or have inconsistent structure")
            logger.error(f"Lat size: {len(target_lat_flat)}, Lon size: {len(target_lon_flat)}")
            raise ValueError(f"Target coordinate arrays don't match target shape {target_shape}. "
                           f"Expected {expected_size} points but got {actual_size}. "
                           f"This may be due to coordinate structure inconsistency between scenes.")

        # Source points
        source_lats = np.array([p[0] for p in points])
        source_lons = np.array([p[1] for p in points])

        result = {}

        for var in weather_data:
            logger.debug(f"Interpolating {var}: {len(weather_data[var])} source points, {len(target_lat_flat)} target points, target shape {target_shape}")

            # Create target points as (n, 2) array
            target_points = np.column_stack([target_lat_flat, target_lon_flat])

            try:
                # Interpolate using scipy's griddata
                interpolated = interpolate.griddata(
                    (source_lats, source_lons),
                    weather_data[var],
                    target_points,
                    method='linear',
                    fill_value=np.nan
                )

                logger.debug(f"Interpolated array size: {len(interpolated)}, expected: {expected_size}")

                # Additional validation before reshape
                if len(interpolated) != expected_size:
                    logger.error(f"Interpolation result size mismatch: got {len(interpolated)}, expected {expected_size}")
                    raise ValueError(f"Interpolation returned array of size {len(interpolated)} but expected {expected_size}")

                # Reshape back to 2D
                interpolated_2d = interpolated.reshape(target_shape)

                # Handle NaN values by filling with nearest
                if np.any(np.isnan(interpolated_2d)):
                    logger.warning(f"NaN values in interpolated {var}, filling with nearest")
                    interpolated_2d = interpolate.griddata(
                        (source_lats, source_lons),
                        weather_data[var],
                        target_points,
                        method='nearest'
                    ).reshape(target_shape)

                # Create DataArray
                result[var] = xr.DataArray(
                    interpolated_2d,
                    dims=['y', 'x'],
                    coords={'y': target_lats, 'x': target_lons},
                    name=var
                )
                
            except ValueError as e:
                if "cannot reshape" in str(e) or "size" in str(e):
                    logger.error(f"Reshape error for variable {var}: {e}")
                    logger.error(f"This indicates a coordinate structure mismatch between cached data and current target grid")
                    logger.error(f"Target shape: {target_shape}, expected size: {expected_size}")
                    logger.error(f"Interpolated size: {len(interpolated) if 'interpolated' in locals() else 'unknown'}")
                    raise ValueError(f"Cannot interpolate {var} to target grid. "
                                   f"This may be due to scene shape mismatch. "
                                   f"Target: {target_shape} (size {expected_size}). "
                                   f"Try clearing the weather cache.") from e
                else:
                    raise

        return result

    def _create_grid_points(self, bbox: List[float]) -> List[Tuple[float, float]]:
        """
        Create grid points within bounding box at specified spacing.

        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]

        Returns:
            List of (lat, lon) tuples
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        # Approximate km per degree
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians((min_lat + max_lat) / 2))

        # Spacing in degrees
        delta_lat = self.grid_spacing_km / km_per_deg_lat
        delta_lon = self.grid_spacing_km / km_per_deg_lon

        # Create grid
        lat_points = np.arange(min_lat, max_lat + delta_lat, delta_lat)
        lon_points = np.arange(min_lon, max_lon + delta_lon, delta_lon)

        # Create mesh
        lon_grid, lat_grid = np.meshgrid(lon_points, lat_points)

        # Flatten to list of tuples
        points = list(zip(lat_grid.flatten(), lon_grid.flatten()))

        return points

    def _fetch_weather_for_point(self, lat: float, lon: float, date: str) -> Optional[Dict]:
        """
        Fetch weather data for a single point and date.

        Args:
            lat: Latitude
            lon: Longitude
            date: Date string YYYY-MM-DD

        Returns:
            Dictionary of weather variables or None if failed
        """
        try:
            # API parameters
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": date,
                "end_date": date,
                "hourly": self.HOURLY_VARIABLES,
                "daily": self.DAILY_VARIABLES,
                "timezone": "auto",  # Use local timezone
                "models": "best_match"
            }

            # Make request
            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            # Extract data at 10:30 local time
            result = self._extract_overpass_data(data, date)

            return result

        except Exception as e:
            logger.warning(f"Failed to fetch weather for ({lat}, {lon}) on {date}: {e}")
            return None

    def _extract_overpass_data(self, data: Dict, date: str) -> Dict:
        """
        Extract weather data at Landsat overpass time (10:30 local).

        Args:
            data: API response data
            date: Scene date

        Returns:
            Dictionary of extracted values
        """
        result = {}

        # For daily variables (ET0 and shortwave radiation sum)
        if 'daily' in data:
            daily_data = data['daily']
            if 'et0_fao_evapotranspiration' in daily_data:
                # Daily ET0 is for the whole day
                et0_values = daily_data['et0_fao_evapotranspiration']
                if et0_values:
                    result['et0_fao_evapotranspiration'] = et0_values[0]
            
            if 'shortwave_radiation_sum' in daily_data:
                # Daily shortwave radiation sum is for the whole day (MJ/m²/day)
                sw_rad_sum_values = daily_data['shortwave_radiation_sum']
                if sw_rad_sum_values:
                    result['shortwave_radiation_sum'] = sw_rad_sum_values[0]

        # For hourly variables, extract at 10:30 local
        if 'hourly' in data:
            hourly_data = data['hourly']
            times = pd.to_datetime(hourly_data['time'])

            # Find 10:30 local time
            target_time = pd.Timestamp(f"{date} 10:30:00")

            # Find closest time (should be exact with timezone=auto)
            time_diffs = np.abs((times - target_time).total_seconds())
            closest_idx = np.argmin(time_diffs)

            # Extract values
            for var in self.HOURLY_VARIABLES:
                if var in hourly_data:
                    values = hourly_data[var]
                    if closest_idx < len(values):
                        result[var] = values[closest_idx]

        return result