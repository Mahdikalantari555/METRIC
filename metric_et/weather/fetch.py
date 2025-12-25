"""Weather data fetching from Open-Meteo API.

This module provides functionality to download weather data for METRIC-ET
calculations, specifically targeting 7:00 UTC (10:30 Iran time) which
corresponds to the typical Landsat overpass time.

The module supports fetching weather data for multiple points within an ROI
to better represent spatial variability, with ~9km resolution from Open-Meteo.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests


class WeatherFetchError(Exception):
    """Base exception for weather fetching errors."""
    pass


class APIRequestError(WeatherFetchError):
    """Exception raised when API request fails."""
    pass


class DataNotFoundError(WeatherFetchError):
    """Exception raised when requested weather data is not available."""
    pass


# Weather variables for Open-Meteo API
HOURLY_WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
    "shortwave_radiation",
]


class WeatherFetcher:
    """
    Weather data fetcher from Open-Meteo API.
    
    This class handles downloading weather data for specific dates and
    locations, with a focus on 7:00 UTC (10:30 Iran time) which aligns
    with Landsat overpass times.
    
    The fetcher supports multiple points within an ROI to better represent
    spatial variability. Open-Meteo provides ~9km resolution weather data.
    
    Attributes:
        api_url: Base URL for Open-Meteo archive API
        timeout: Request timeout in seconds
        
    Example:
        >>> fetcher = WeatherFetcher()
        >>> weather_file = fetcher.download(
        ...     date="2025-12-23",
        ...     lat=35.6892,
        ...     lon=51.3890,
        ...     output_dir="data"
        ... )
    
    Example with multiple points:
        >>> points = [(35.68, 51.38), (35.70, 51.40), (35.72, 51.42)]
        >>> weather_file = fetcher.download(
        ...     date="2025-12-23",
        ...     points=points,
        ...     output_dir="data"
        ... )
    """
    
    def __init__(self, api_url: str = "https://archive-api.open-meteo.com/v1/archive",
                 timeout: int = 30):
        """
        Initialize the WeatherFetcher.
        
        Args:
            api_url: Open-Meteo API endpoint URL
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.timeout = timeout
    
    def _generate_grid_points(self, bbox: tuple, spacing_deg: float = 0.05) -> List[tuple]:
        """
        Generate a grid of points within a bounding box.
        
        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            spacing_deg: Spacing between points in degrees (default: 0.05° ~ 5.5km)
            
        Returns:
            List of (lat, lon) tuples
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        
        # Calculate number of points in each direction
        n_lon = max(1, int(np.ceil((max_lon - min_lon) / spacing_deg)))
        n_lat = max(1, int(np.ceil((max_lat - min_lat) / spacing_deg)))
        
        # Generate grid points
        points = []
        for i in range(n_lon):
            lon = min_lon + i * spacing_deg
            for j in range(n_lat):
                lat = min_lat + j * spacing_deg
                points.append((lat, lon))
        
        return points
    
    def download(
        self,
        date: Union[str, datetime],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        points: Optional[List[tuple]] = None,
        bbox: Optional[tuple] = None,
        output_dir: Union[str, Path] = "data",
        target_hour: int = 7,
        timezone: str = "UTC",
        grid_spacing: float = 0.05
    ) -> Path:
        """
        Download weather data for a specific date and location(s).
        
        This method supports three modes:
        1. Single point: Provide lat and lon
        2. Multiple points: Provide points list of (lat, lon) tuples
        3. Bounding box: Provide bbox and generate grid points
        
        Args:
            date: Date as YYYY-MM-DD string or datetime object
            lat: Latitude in decimal degrees (for single point mode)
            lon: Longitude in decimal degrees (for single point mode)
            points: List of (lat, lon) tuples for multiple points
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            output_dir: Directory to save weather data CSV file
            target_hour: Target hour for data extraction (default: 7 for UTC)
            timezone: Timezone for the request (default: "UTC")
            grid_spacing: Spacing between grid points in degrees (default: 0.05° ~ 5.5km)
            
        Returns:
            Path to the saved weather CSV file
            
        Raises:
            APIRequestError: If API request fails
            DataNotFoundError: If no data available for the requested time
            ValueError: If no location specified or invalid parameters
        """
        # Convert date to string if datetime object
        if isinstance(date, datetime):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = date
        
        # Validate date format
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Expected YYYY-MM-DD, got: {date_str}") from e
        
        # Determine points to query
        query_points = []
        
        if points is not None:
            # Multiple points mode
            if not isinstance(points, list):
                raise ValueError("points must be a list of (lat, lon) tuples")
            if len(points) == 0:
                raise ValueError("points list cannot be empty")
            query_points = points
        elif bbox is not None:
            # Bounding box mode - generate grid
            if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                raise ValueError("bbox must be a tuple of (min_lon, min_lat, max_lon, max_lat)")
            query_points = self._generate_grid_points(bbox, grid_spacing)
        elif lat is not None and lon is not None:
            # Single point mode
            query_points = [(lat, lon)]
        else:
            raise ValueError(
                "Must specify either (lat, lon), points list, or bbox. "
                "Got: lat={}, lon={}, points={}, bbox={}".format(
                    lat, lon, points, bbox
                )
            )
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        date_clean = date_str.replace("-", "")
        if len(query_points) == 1:
            weather_file = output_path / f"weather_{date_clean}_7am_UTC.csv"
        else:
            weather_file = output_path / f"weather_{date_clean}_7am_UTC_multi.csv"
        
        # Download weather for each point
        all_weather_data = []
        
        for idx, (lat, lon) in enumerate(query_points):
            try:
                # Prepare request parameters for single point
                weather_params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": date_str,
                    "end_date": date_str,
                    "hourly": HOURLY_WEATHER_VARS,
                    "daily": "et0_fao_evapotranspiration",
                    "models": "best_match",
                    "timezone": timezone
                }
                
                # Make API request
                response = requests.get(
                    self.api_url,
                    params=weather_params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                weather_data = response.json()
                hourly = weather_data.get("hourly", {})
                
                if not hourly or "time" not in hourly:
                    raise DataNotFoundError(f"No hourly data available for point {idx+1} ({lat}, {lon})")
                
                # Create DataFrame from hourly data
                df_dict = {"time": pd.to_datetime(hourly["time"])}
                for var in HOURLY_WEATHER_VARS:
                    if var in hourly:
                        df_dict[var] = hourly[var]
                
                weather_df = pd.DataFrame(df_dict)
                
                # Filter for target hour
                weather_target = weather_df[weather_df["time"].dt.hour == target_hour]
                
                if len(weather_target) == 0:
                    raise DataNotFoundError(
                        f"No {target_hour}:00 {timezone} data available for point {idx+1} ({lat}, {lon})"
                    )
                
                # Add location information
                weather_target = weather_target.copy()
                weather_target["latitude"] = lat
                weather_target["longitude"] = lon
                weather_target["point_id"] = idx + 1
                weather_target["local_time"] = weather_target["time"] + pd.Timedelta(hours=3, minutes=30)
                weather_target["time_zone"] = "UTC"
                weather_target["local_time_zone"] = "IRST (UTC+3:30)"
                
                all_weather_data.append(weather_target)
                
            except requests.exceptions.RequestException as e:
                raise APIRequestError(f"API request failed for point {idx+1} ({lat}, {lon}): {str(e)}") from e
            except Exception as e:
                raise WeatherFetchError(f"Error downloading weather for point {idx+1} ({lat}, {lon}): {str(e)}") from e
        
        # Combine all weather data
        if len(all_weather_data) > 1:
            combined_df = pd.concat(all_weather_data, ignore_index=True)
        else:
            combined_df = all_weather_data[0]
        
        # Save to CSV
        combined_df.to_csv(weather_file, index=False)
        
        return weather_file
    
    def download_for_multiple_dates(
        self,
        dates: list,
        lat: float,
        lon: float,
        output_dir: Union[str, Path]
    ) -> dict:
        """
        Download weather data for multiple dates.
        
        Args:
            dates: List of dates as YYYY-MM-DD strings or datetime objects
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            output_dir: Directory to save weather data CSV files
            
        Returns:
            Dictionary mapping dates to file paths or error messages
        """
        results = {}
        
        for date in dates:
            try:
                file_path = self.download(date, lat, lon, output_dir)
                results[date] = {
                    "file": str(file_path),
                    "status": "success",
                    "utc_time": "07:00",
                    "iran_time": "10:30"
                }
            except Exception as e:
                results[date] = {
                    "file": None,
                    "status": f"error: {str(e)}",
                    "utc_time": "07:00",
                    "iran_time": "10:30"
                }
        
        return results


def download_weather_for_7am_utc(
    date: Union[str, datetime],
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    points: Optional[List[tuple]] = None,
    bbox: Optional[tuple] = None,
    output_dir: Union[str, Path] = "data",
    api_url: str = "https://archive-api.open-meteo.com/v1/archive",
    grid_spacing: float = 0.05
) -> Tuple[Optional[Path], str]:
    """
    Download weather data specifically for 7:00 UTC (10:30 Iran time).
    
    This is a convenience function that wraps the WeatherFetcher class.
    
    Supports three modes:
    1. Single point: Provide lat and lon
    2. Multiple points: Provide points list of (lat, lon) tuples
    3. Bounding box: Provide bbox and generate grid points
    
    Args:
        date: Date as YYYY-MM-DD string or datetime object
        lat: Latitude in decimal degrees (for single point mode)
        lon: Longitude in decimal degrees (for single point mode)
        points: List of (lat, lon) tuples for multiple points
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
        output_dir: Directory to save weather data CSV file
        api_url: Open-Meteo API endpoint URL
        grid_spacing: Spacing between grid points in degrees (default: 0.05° ~ 5.5km)
        
    Returns:
        Tuple of (file_path, status_message)
        
    Example (single point):
        >>> file_path, status = download_weather_for_7am_utc(
        ...     "2025-12-23",
        ...     35.6892,
        ...     51.3890,
        ...     "data"
        ... )
        >>> if file_path:
        ...     print(f"Weather data saved to {file_path}")
    
    Example (multiple points):
        >>> points = [(35.68, 51.38), (35.70, 51.40), (35.72, 51.42)]
        >>> file_path, status = download_weather_for_7am_utc(
        ...     "2025-12-23",
        ...     points=points,
        ...     output_dir="data"
        ... )
    
    Example (bounding box):
        >>> bbox = (51.38, 35.68, 51.42, 35.72)  # min_lon, min_lat, max_lon, max_lat
        >>> file_path, status = download_weather_for_7am_utc(
        ...     "2025-12-23",
        ...     bbox=bbox,
        ...     output_dir="data"
        ... )
    """
    try:
        fetcher = WeatherFetcher(api_url=api_url)
        file_path = fetcher.download(
            date=date,
            lat=lat,
            lon=lon,
            points=points,
            bbox=bbox,
            output_dir=output_dir,
            grid_spacing=grid_spacing
        )
        return file_path, "Success"
    except Exception as e:
        return None, f"Error: {str(e)}"
