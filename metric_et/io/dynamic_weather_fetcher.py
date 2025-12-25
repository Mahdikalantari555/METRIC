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
        "et0_fao_evapotranspiration"
    ]

    def __init__(self, grid_spacing_km: float = 9.0):
        """
        Initialize the weather fetcher.

        Args:
            grid_spacing_km: Spacing between grid points in kilometers
        """
        self.grid_spacing_km = grid_spacing_km

    def fetch_weather_for_scene(self, landsat_dir: str, target_coords: Dict, actual_extent: Tuple = None) -> Dict[str, xr.DataArray]:
        """
        Fetch weather data for a Landsat scene and interpolate to target grid.

        Args:
            landsat_dir: Path to Landsat scene directory
            target_coords: Target coordinates dict with 'y', 'x' arrays
            actual_extent: Actual processed extent (min_lon, min_lat, max_lon, max_lat) or None to use MTL bbox

        Returns:
            Dictionary of weather variable DataArrays on target grid
        """
        # Parse scene metadata
        mtl_path = os.path.join(landsat_dir, "MTL.json")
        if not os.path.exists(mtl_path):
            raise FileNotFoundError(f"MTL.json not found in {landsat_dir}")

        with open(mtl_path, 'r') as f:
            mtl_data = json.load(f)

        # Extract bbox and date
        if actual_extent:
            # Use the actual processed extent (clipped to ROI)
            min_lon, min_lat, max_lon, max_lat = actual_extent
            bbox = [min_lon, min_lat, max_lon, max_lat]
            logger.info(f"Using actual processed extent for weather grid: {bbox}")
        else:
            # Fallback to MTL bbox
            bbox = mtl_data['bbox']
            logger.info(f"Using MTL bbox for weather grid: {bbox}")

        scene_date = mtl_data['datetime']  # YYYY-MM-DD

        # Create grid points
        grid_points = self._create_grid_points(bbox)

        logger.info(f"Created {len(grid_points)} grid points for scene {scene_date}")

        # Fetch weather data for each point
        weather_data = {}
        valid_points = []
        for var in self.HOURLY_VARIABLES + self.DAILY_VARIABLES:
            weather_data[var] = []

        for lat, lon in grid_points:
            point_data = self._fetch_weather_for_point(lat, lon, scene_date)
            if point_data and all(k in point_data for k in self.HOURLY_VARIABLES + self.DAILY_VARIABLES):
                valid_points.append((lat, lon))
                for var in self.HOURLY_VARIABLES + self.DAILY_VARIABLES:
                    weather_data[var].append(point_data[var])

        if not valid_points:
            raise ValueError("No valid weather data fetched for any grid point")

        logger.info(f"Successfully fetched data for {len(valid_points)} points")

        # Convert to numpy arrays
        for var in weather_data:
            weather_data[var] = np.array(weather_data[var])

        # Interpolate to target grid
        result = self._interpolate_to_target_grid(weather_data, valid_points, target_coords)

        return result

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

        # Create full 2D grids if they are 1D
        if target_lats.ndim == 1 and target_lons.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(target_lons.values, target_lats.values)
            target_shape = lat_grid.shape
            target_lat_flat = lat_grid.flatten()
            target_lon_flat = lon_grid.flatten()
        else:
            # Assume already 2D
            target_shape = target_lats.shape
            target_lat_flat = target_lats.values.flatten()
            target_lon_flat = target_lons.values.flatten()

        # Source points
        source_lats = np.array([p[0] for p in points])
        source_lons = np.array([p[1] for p in points])

        result = {}

        for var in weather_data:
            # Interpolate using scipy's griddata
            interpolated = interpolate.griddata(
                (source_lats, source_lons),
                weather_data[var],
                (target_lat_flat, target_lon_flat),
                method='linear',
                fill_value=np.nan
            )

            # Reshape back to 2D
            interpolated_2d = interpolated.reshape(target_shape)

            # Handle NaN values by filling with nearest
            if np.any(np.isnan(interpolated_2d)):
                logger.warning(f"NaN values in interpolated {var}, filling with nearest")
                interpolated_2d = interpolate.griddata(
                    (source_lats, source_lons),
                    weather_data[var],
                    (target_lat_flat, target_lon_flat),
                    method='nearest'
                ).reshape(target_shape)

            # Create DataArray
            result[var] = xr.DataArray(
                interpolated_2d,
                dims=['y', 'x'],
                coords={'y': target_lats, 'x': target_lons},
                name=var
            )

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

        # For daily variables (ET0)
        if 'daily' in data:
            daily_data = data['daily']
            if 'et0_fao_evapotranspiration' in daily_data:
                # Daily ET0 is for the whole day
                et0_values = daily_data['et0_fao_evapotranspiration']
                if et0_values:
                    result['et0_fao_evapotranspiration'] = et0_values[0]

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