"""
Dynamic weather data fetcher for METRIC ET processing.

Fetches spatially varying meteorological data from Open-Meteo API
for grid points within Landsat scene bounding box.
"""

import json
import numpy as np
import pandas as pd
import requests
import xarray as xr
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import interpolate
import logging

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

    def __init__(self, grid_spacing_km: float = 9.0):
        """
        Initialize the weather fetcher.

        Args:
            grid_spacing_km: Spacing between grid points in kilometers
        """
        self.grid_spacing_km = grid_spacing_km

    def fetch_weather_for_scene(self, landsat_dir: str, target_coords: Dict, actual_extent: Tuple = None, target_crs: str = None) -> Dict[str, xr.DataArray]:
        """
        Fetch weather data for a Landsat scene and interpolate to target grid.

        Args:
            landsat_dir: Path to Landsat scene directory
            target_coords: Target coordinates dict with 'y', 'x' arrays
            actual_extent: Actual processed extent (min_lon, min_lat, max_lon, max_lat) or None to use MTL bbox
            target_crs: CRS of the target coordinates (e.g., 'EPSG:32639' for UTM Zone 39N)

        Returns:
            Dictionary of weather variable DataArrays on target grid
        """
        # Parse scene metadata for bbox
        mtl_path = landsat_dir + "/MTL.json"
        if not mtl_path:
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

        # Extract scene date
        scene_date = self._extract_scene_date(mtl_data)
        logger.info(f"Fetching weather data for date: {scene_date}")

        # Create grid points
        grid_points = self._create_grid_points(bbox)
        logger.info(f"Created {len(grid_points)} grid points")

        # Fetch weather data for each point
        weather_data = self._fetch_all_points_weather(grid_points, scene_date)

        if not weather_data['valid_points']:
            raise ValueError("No valid weather data fetched for any grid point")

        logger.info(f"Successfully fetched data for {len(weather_data['valid_points'])} points")

        # Interpolate to target grid
        result = self._interpolate_to_target_grid(weather_data, weather_data['valid_points'], target_coords, target_crs)

        return result

    def _extract_scene_date(self, mtl_data: Dict) -> str:
        """
        Extract scene date from MTL data.

        Args:
            mtl_data: MTL metadata dictionary

        Returns:
            Scene date string (YYYY-MM-DD)
        """
        scene_date = mtl_data.get('datetime') or mtl_data.get('DATE_ACQUIRED')
        if scene_date:
            # Handle different date formats
            if 'T' in scene_date:
                scene_date = scene_date.split('T')[0]
            return scene_date
        
        # Fallback to parsing from metadata
        logger.warning("Could not extract scene date from MTL, using placeholder")
        return "2023-04-27"  # Default fallback

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

    def _interpolate_to_target_grid(self, weather_data: Dict, points: List[Tuple[float, float]],
                                   target_coords: Dict, target_crs: str = None) -> Dict[str, xr.DataArray]:
        """
        Interpolate weather data from grid points to target spatial grid.

        Args:
            weather_data: Dict of variable arrays at grid points
            points: List of (lat, lon) tuples for grid points (in WGS84)
            target_coords: Target coordinates dict with 'y', 'x' arrays
            target_crs: CRS of the target coordinates (e.g., 'EPSG:32639' for UTM Zone 39N)

        Returns:
            Dictionary of interpolated DataArrays
        """
        # Store original target coordinates BEFORE transformation
        # These are the projected coordinates (e.g., UTM meters) that match Landsat data
        if target_crs and target_crs != "EPSG:4326":
            # Save original coordinates for use in DataArray creation
            original_y_coords = target_coords['y']
            original_x_coords = target_coords['x']
            
            # Handle coordinate types (numpy arrays vs xarray DataArrays)
            if hasattr(original_y_coords, 'values'):
                original_y_array = original_y_coords.values
            else:
                original_y_array = np.array(original_y_coords)
                
            if hasattr(original_x_coords, 'values'):
                original_x_array = original_x_coords.values
            else:
                original_x_array = np.array(original_x_coords)
            
            logger.debug(f"Original target coords - y shape: {original_y_array.shape}, x shape: {original_x_array.shape}")
        else:
            # No transformation - coordinates are already in lat/lon
            original_y_coords = target_coords['y']
            original_x_coords = target_coords['x']
        
        # Transform target coordinates from projected CRS to WGS84 lat/lon if needed
        # Target coordinates come from Landsat data which is typically in a projected CRS (e.g., UTM in meters)
        # Weather data from Open-Meteo is in WGS84 lat/lon degrees
        if target_crs and target_crs != "EPSG:4326":
            from pyproj import Transformer
            # Create transformer from target CRS to WGS84
            transformer = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
            
            # Extract target coordinate arrays
            target_lats_raw = target_coords['y']
            target_lons_raw = target_coords['x']
            
            # Handle different coordinate types (numpy arrays vs xarray DataArrays)
            if hasattr(target_lats_raw, 'values'):
                target_lats_array = target_lats_raw.values
                lats_is_dataarray = True
            else:
                target_lats_array = np.array(target_lats_raw)
                lats_is_dataarray = False
                
            if hasattr(target_lons_raw, 'values'):
                target_lons_array = target_lons_raw.values
                lons_is_dataarray = True
            else:
                target_lons_array = np.array(target_lons_raw)
                lons_is_dataarray = False
            
            logger.debug(f"Original target coords - lat range: [{target_lats_array.min()}, {target_lats_array.max()}], lon range: [{target_lons_array.min()}, {target_lons_array.max()}]")
            logger.debug(f"Target CRS: {target_crs}")
            
            # Transform coordinates from projected CRS to WGS84
            # Handle both 1D and 2D coordinate arrays
            if target_lats_array.ndim == 1 and target_lons_array.ndim == 1:
                # 1D coordinates - transform each point
                lons_wgs84 = np.zeros_like(target_lons_array)
                lats_wgs84 = np.zeros_like(target_lats_array)
                for i, (x, y) in enumerate(zip(target_lons_array, target_lats_array)):
                    lons_wgs84[i], lats_wgs84[i] = transformer.transform(x, y)
                target_lons_array = lons_wgs84
                target_lats_array = lats_wgs84
            elif target_lats_array.ndim == 2 and target_lons_array.ndim == 2:
                # 2D coordinates - transform each point
                target_lons_wgs84 = np.zeros_like(target_lons_array)
                target_lats_wgs84 = np.zeros_like(target_lats_array)
                for i in range(target_lats_array.shape[0]):
                    for j in range(target_lats_array.shape[1]):
                        target_lons_wgs84[i, j], target_lats_wgs84[i, j] = transformer.transform(
                            target_lons_array[i, j], target_lats_array[i, j]
                        )
                target_lons_array = target_lons_wgs84
                target_lats_array = target_lats_wgs84
            else:
                # Mixed dimensions - flatten, transform, reshape
                if target_lats_array.ndim == 1:
                    lon_grid, lat_grid = np.meshgrid(target_lons_array, target_lats_array)
                    flat_lons = lon_grid.flatten()
                    flat_lats = lat_grid.flatten()
                    flat_lons_wgs84 = np.zeros_like(flat_lons)
                    flat_lats_wgs84 = np.zeros_like(flat_lats)
                    for i, (x, y) in enumerate(zip(flat_lons, flat_lats)):
                        flat_lons_wgs84[i], flat_lats_wgs84[i] = transformer.transform(x, y)
                    target_lons_array = flat_lons_wgs84.reshape(lon_grid.shape)
                    target_lats_array = flat_lats_wgs84.reshape(lat_grid.shape)
                else:
                    # lat is already 2D
                    flat_lons = target_lons_array.flatten()
                    flat_lats = target_lats_array.flatten()
                    flat_lons_wgs84 = np.zeros_like(flat_lons)
                    flat_lats_wgs84 = np.zeros_like(flat_lats)
                    for i, (x, y) in enumerate(zip(flat_lons, flat_lats)):
                        flat_lons_wgs84[i], flat_lats_wgs84[i] = transformer.transform(x, y)
                    target_lons_array = flat_lons_wgs84.reshape(target_lons_array.shape)
                    target_lats_array = flat_lats_wgs84.reshape(target_lats_array.shape)
            
            logger.debug(f"Transformed target coords - lat range: [{target_lats_array.min():.4f}, {target_lats_array.max():.4f}], lon range: [{target_lons_array.min():.4f}, {target_lons_array.max():.4f}]")
            
            # Use transformed coordinates for interpolation
            target_lats = target_lats_array
            target_lons = target_lons_array
        else:
            # No transformation needed - coordinates are already in lat/lon
            # Extract target lat/lon grids
            target_lats = target_coords['y']  # Assuming 'y' is latitude
            target_lons = target_coords['x']  # Assuming 'x' is longitude
            
            # Handle different coordinate types (numpy arrays vs xarray DataArrays)
            if hasattr(target_lats, 'values'):
                target_lats_array = target_lats.values
            else:
                target_lats_array = np.array(target_lats)
                
            if hasattr(target_lons, 'values'):
                target_lons_array = target_lons.values
            else:
                target_lons_array = np.array(target_lons)

        logger.debug(f"Target coordinates - lat shape: {target_lats_array.shape}, lon shape: {target_lons_array.shape}")

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
            raise ValueError(f"Target coordinate arrays don't match target shape {target_shape}. "
                           f"Expected {expected_size} points but got {actual_size}")

        # Source points
        source_lats = np.array([p[0] for p in points])
        source_lons = np.array([p[1] for p in points])

        result = {}

        for var in self.HOURLY_VARIABLES + self.DAILY_VARIABLES:
            if var not in weather_data:
                logger.warning(f"Variable {var} not in weather data, skipping")
                continue
                
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

                # Create DataArray with ORIGINAL coordinates (projected, matching Landsat data)
                # NOT the transformed WGS84 coordinates used for interpolation
                result[var] = xr.DataArray(
                    interpolated_2d,
                    dims=['y', 'x'],
                    coords={'y': original_y_coords, 'x': original_x_coords},
                    name=var
                )
                
            except ValueError as e:
                if "cannot reshape" in str(e) or "size" in str(e):
                    logger.error(f"Reshape error for variable {var}: {e}")
                    raise ValueError(f"Cannot interpolate {var} to target grid. "
                                   f"Target: {target_shape} (size {expected_size}). "
                                   f"Check coordinate consistency.") from e
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
