"""Meteo Reader for weather data.

This module provides functionality to read and interpolate weather data
for ETa calculations.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


class MeteoReaderError(Exception):
    """Base exception for MeteoReader errors."""
    pass


class DataLoadError(MeteoReaderError):
    """Exception raised when weather data loading fails."""
    pass


class TimeNotFoundError(MeteoReaderError):
    """Exception raised when requested time is outside data range."""
    pass


class MeteoReader:
    """
    Reader for meteorological data.
    
    This class handles reading weather station data from CSV files,
    interpolating values for specific times, and providing weather
    parameters for ETa calculations.
    
    Attributes:
        df: pandas DataFrame containing weather data
        columns: List of weather variable columns
        
    Example:
        >>> reader = MeteoReader()
        >>> reader.load("data/weather_data.csv")
        >>> weather = reader.get_at_time(datetime(2025, 12, 4, 7, 0, 0))
    """
    
    # Required columns for weather data
    REQUIRED_COLUMNS = [
        'time',
        'temperature_2m',
        'relative_humidity_2m',
        'wind_speed_10m',
        'surface_pressure',
        'shortwave_radiation',
        'et0_fao_evapotranspiration',
    ]
    
    def __init__(self):
        """Initialize the MeteoReader."""
        self.df: Optional[pd.DataFrame] = None
        self._time_column = 'time'
    
    def load(self, csv_path: Union[str, Path]) -> 'MeteoReader':
        """
        Load weather data from CSV file.
        
        Args:
            csv_path: Path to the weather data CSV file
            
        Returns:
            self for method chaining
            
        Raises:
            DataLoadError: If file doesn't exist or parsing fails
        """
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise DataLoadError(f"Weather data file not found: {csv_path}")
        
        try:
            self.df = pd.read_csv(csv_path, parse_dates=[self._time_column])
            self.df.set_index(self._time_column, inplace=True)
            self.df.sort_index(inplace=True)
            
            # Validate required columns (excluding 'time' as it's the index)
            required_data_cols = set(self.REQUIRED_COLUMNS) - {'time'}
            missing_cols = required_data_cols - set(self.df.columns)
            if missing_cols:
                raise DataLoadError(f"Missing required columns: {missing_cols}")
            
            return self
        
        except pd.errors.EmptyDataError:
            raise DataLoadError(f"Weather data file is empty: {csv_path}")
        except pd.errors.ParserError as e:
            raise DataLoadError(f"Failed to parse weather data: {e}")
    
    def get_at_time(self, dt: datetime) -> Dict[str, float]:
        """
        Get weather data at a specific time.
        
        If the exact time is not in the data, performs linear interpolation
        between the nearest hourly values.
        
        Args:
            dt: Datetime to query
            
        Returns:
            Dictionary containing weather variables at the requested time
            
        Raises:
            TimeNotFoundError: If time is outside data range
        """
        if self.df is None:
            raise DataLoadError("No data loaded. Call load() first.")
        
        return self.interpolate(dt)
    
    def interpolate(self, dt: datetime) -> Dict[str, float]:
        """
        Linearly interpolate weather data for a specific time.
        
        Args:
            dt: Datetime to interpolate for
            
        Returns:
            Dictionary with interpolated weather values
            
        Raises:
            TimeNotFoundError: If time is outside data range
        """
        if self.df is None:
            raise DataLoadError("No data loaded. Call load() first.")
        
        # Check if time is within data range
        if dt < self.df.index.min() or dt > self.df.index.max():
            available_range = f"{self.df.index.min()} to {self.df.index.max()}"
            raise TimeNotFoundError(
                f"Requested time {dt} is outside available data range: {available_range}"
            )
        
        # Check if exact time exists
        if dt in self.df.index:
            row = self.df.loc[dt]
            return {
                'temperature_2m': float(row['temperature_2m']),
                'relative_humidity_2m': float(row['relative_humidity_2m']),
                'wind_speed_10m': float(row['wind_speed_10m']),
                'surface_pressure': float(row['surface_pressure']),
                'shortwave_radiation': float(row['shortwave_radiation']),
                'et0_fao_evapotranspiration': float(row['et0_fao_evapotranspiration']),
                'datetime': dt,
            }
        
        # Interpolate between surrounding times
        # Find the two nearest timestamps
        before = self.df.index[self.df.index <= dt].max()
        after = self.df.index[self.df.index >= dt].min()
        
        if pd.isna(before) or pd.isna(after):
            # Edge case: dt is before first or after last timestamp
            row = self.df.loc[before if not pd.isna(before) else after]
            return {
                'temperature_2m': float(row['temperature_2m']),
                'relative_humidity_2m': float(row['relative_humidity_2m']),
                'wind_speed_10m': float(row['wind_speed_10m']),
                'surface_pressure': float(row['surface_pressure']),
                'shortwave_radiation': float(row['shortwave_radiation']),
                'et0_fao_evapotranspiration': float(row['et0_fao_evapotranspiration']),
                'datetime': dt,
            }
        
        # Calculate interpolation factor
        dt_before = pd.Timestamp(before)
        dt_after = pd.Timestamp(after)
        total_seconds = (dt_after - dt_before).total_seconds()
        elapsed_seconds = (pd.Timestamp(dt) - dt_before).total_seconds()
        
        if total_seconds == 0:
            alpha = 0.0
        else:
            alpha = elapsed_seconds / total_seconds
        
        # Interpolate each column
        result = {'datetime': dt}
        
        for col in self.REQUIRED_COLUMNS:
            if col == 'time':
                continue
            val_before = self.df.loc[before, col]
            val_after = self.df.loc[after, col]
            interpolated = val_before + alpha * (val_after - val_before)
            result[col] = float(interpolated)
        
        return result
    
    def get_hourly_range(self) -> tuple:
        """
        Get the start and end times of the weather data.
        
        Returns:
            Tuple of (start_time, end_time) as datetime objects
        """
        if self.df is None:
            raise DataLoadError("No data loaded. Call load() first.")
        
        return self.df.index.min().to_pydatetime(), self.df.index.max().to_pydatetime()
    
    def get_temperature_at_time(self, dt: datetime) -> float:
        """
        Get temperature at a specific time.
        
        Args:
            dt: Datetime to query
            
        Returns:
            Temperature in Celsius at 2m height
        """
        weather = self.get_at_time(dt)
        return weather['temperature_2m']
    
    def get_humidity_at_time(self, dt: datetime) -> float:
        """
        Get relative humidity at a specific time.
        
        Args:
            dt: Datetime to query
            
        Returns:
            Relative humidity in percentage (0-100)
        """
        weather = self.get_at_time(dt)
        return weather['relative_humidity_2m']
    
    def get_wind_speed_at_time(self, dt: datetime) -> float:
        """
        Get wind speed at a specific time.
        
        Args:
            dt: Datetime to query
            
        Returns:
            Wind speed in m/s at 10m height
        """
        weather = self.get_at_time(dt)
        return weather['wind_speed_10m']
    
    def get_pressure_at_time(self, dt: datetime) -> float:
        """
        Get surface pressure at a specific time.
        
        Args:
            dt: Datetime to query
            
        Returns:
            Surface pressure in hPa
        """
        weather = self.get_at_time(dt)
        return weather['surface_pressure']
    
    def get_radiation_at_time(self, dt: datetime) -> float:
        """
        Get shortwave radiation at a specific time.
        
        Args:
            dt: Datetime to query
            
        Returns:
            Shortwave radiation in W/m²
        """
        weather = self.get_at_time(dt)
        return weather['shortwave_radiation']
    
    def get_eto_at_time(self, dt: datetime) -> float:
        """
        Get reference ET (ET0) at a specific time.
        
        Args:
            dt: Datetime to query
            
        Returns:
            Reference ET in mm/day
        """
        weather = self.get_at_time(dt)
        return weather['et0_fao_evapotranspiration']
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Get the underlying DataFrame.
        
        Returns:
            Copy of the weather data DataFrame
        """
        if self.df is None:
            raise DataLoadError("No data loaded. Call load() first.")
        
        return self.df.copy()
    
    def __repr__(self) -> str:
        if self.df is None:
            return "MeteoReader(no data loaded)"
        
        count = len(self.df)
        time_range = f"{self.df.index.min()} to {self.df.index.max()}"
        return f"MeteoReader(data_points={count}, range={time_range})"


def read_weather_data(csv_path: str) -> MeteoReader:
    """
    Convenience function to load weather data.
    
    Args:
        csv_path: Path to the weather data CSV file
        
    Returns:
        MeteoReader instance with loaded data
    """
    reader = MeteoReader()
    reader.load(csv_path)
    return reader
