import numpy as np
import xarray as xr
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETExtrapolator:
    """
    Extrapolate and interpolate daily ETa for dates beyond Landsat observations.
    
    Formulas:
        1. Basic: ETa = ETrF × ET0
        2. Frozen ETrF: ETa(day_i) = ETrF_last × ET0(day_i)
        3. ETrF Interpolation: ETrF(t) = ETrF1 + (ETrF2 - ETrF1) × (t - t1) / (t2 - t1)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize ETExtrapolator with configuration options."""
        self.config = self._merge_config(config)
        logger.info("ETExtrapolator initialized with configuration: %s", self.config)
    
    def _merge_config(self, config: Optional[Dict]) -> Dict:
        """Merge user config with default values."""
        default_config = {
            "extrapolation_days": 3,
            "interpolation_method": "weighted",
            "min_et_daily": 0.0,
            "max_et_daily": 30.0,
            "etrf_min": 0.0,
            "etrf_max": 2.0
        }
        
        if config is None:
            return default_config
        
        return {**default_config, **config}
    
    def extrapolate(
        self,
        etrf_last: np.ndarray,
        et0_forecast: xr.DataArray
    ) -> Dict[str, np.ndarray]:
        """
        Extrapolate ETa for future dates using frozen ETrF.
        
        Formula: ETa(day_i) = ETrF_last × ET0(day_i)
        
        Args:
            etrf_last: ETrF raster from last Landsat scene (2D numpy array)
            et0_forecast: ET0 forecast data with time dimension
        
        Returns:
            Dictionary with:
                - 'ETa_daily': Daily ETa for each forecast day (3D: time × y × x)
                - 'dates': List of forecast dates
                - 'ETrF_used': The frozen ETrF value used
        """
        logger.info("Starting extrapolation with frozen ETrF method")
        
        # Validate inputs
        if etrf_last.ndim != 2:
            raise ValueError("etrf_last must be a 2D numpy array")
        
        if "time" not in et0_forecast.dims:
            raise ValueError("et0_forecast must have a 'time' dimension")
        
        # Calculate ETa for each forecast day
        et0_values = et0_forecast.values
        if et0_values.ndim == 1:
            # Spatially uniform ET0, reshape to match ETrF dimensions
            et0_values = et0_values[:, np.newaxis, np.newaxis]
            et0_values = np.repeat(np.repeat(et0_values, etrf_last.shape[0], axis=1), 
                                 etrf_last.shape[1], axis=2)
        elif et0_values.ndim == 3:
            # Spatially varying ET0, check dimensions match
            if et0_values.shape[1:] != etrf_last.shape:
                raise ValueError("ET0 forecast dimensions do not match ETrF dimensions")
        else:
            raise ValueError("ET0 forecast must be 1D (uniform) or 3D (spatial)")
        
        # Calculate ETa = ETrF_last × ET0
        eta_daily = etrf_last[np.newaxis, :, :] * et0_values
        
        # Apply ETa bounds
        eta_daily = np.clip(eta_daily, self.config["min_et_daily"], self.config["max_et_daily"])
        
        # Get dates from ET0 forecast
        dates = pd.to_datetime(et0_forecast.time.values).tolist()
        
        logger.info("Extrapolation complete: %d days forecasted", len(dates))
        
        return {
            'ETa_daily': eta_daily,
            'dates': dates,
            'ETrF_used': etrf_last
        }
    
    def interpolate(
        self,
        etrf_scenes: List[Tuple[datetime, np.ndarray]],  # [(date1, etrf1), (date2, etrf2), ..., (dateN, etrfN)]
        et0_data: xr.DataArray,
        method: str = "weighted",  # "linear" or "weighted"
        gap_threshold: int = 30  # Max days between scenes to interpolate (None = all gaps)
    ) -> Dict[str, np.ndarray]:
        """
        Interpolate ETa between Landsat scenes, handling multiple scene pairs and gaps.
        
        The method automatically detects gaps between consecutive Landsat scenes
        and interpolates only the dates within those gaps.
        
        Formula options:
        
        1. Linear interpolation of ETrF:
            ETrF(t) = ETrF1 + (ETrF2 - ETrF1) × (t - t1) / (t2 - t1)
            ETa(t) = ETrF(t) × ET0(t)
        
        2. Weighted interpolation (nearer scenes more influential):
            w1 = (t2 - t) / (t2 - t1)
            w2 = (t - t1) / (t2 - t1)
            ETa(t) = w1 × ETa1 + w2 × ETa2
            where ETa1 = ETrF1 × ET0(t1), ETa2 = ETrF2 × ET0(t2)
        
        Args:
            etrf_scenes: List of (date, etrf) tuples for each Landsat scene (must be sorted by date)
            et0_data: ET0 data with time dimension
            method: "linear" (ETrF interpolation) or "weighted" (ETa weighted by proximity)
            gap_threshold: Maximum number of days between scenes to interpolate (gaps larger than this are skipped)
        
        Returns:
            Dictionary with:
                - 'ETa_daily': Daily interpolated ETa (3D: time × y × x)
                - 'dates': List of interpolated dates
                - 'etrf_interpolated': ETrF values used for each day (for linear method)
                - 'weights': Weight values used (for weighted method)
                - 'scene_pairs': List of scene pairs used for interpolation
                - 'gaps': List of gap segments with start/end dates and number of days
        """
        logger.info("Starting interpolation with %s method", method)
        
        # Validate inputs
        if not etrf_scenes:
            raise ValueError("etrf_scenes must not be empty")
        
        # Check scenes are sorted by date
        dates = [date for date, _ in etrf_scenes]
        if dates != sorted(dates):
            raise ValueError("etrf_scenes must be sorted by date")
        
        if "time" not in et0_data.dims:
            raise ValueError("et0_data must have a 'time' dimension")
        
        # Detect gaps
        gaps = self._detect_gaps(etrf_scenes, gap_threshold)
        
        # Interpolate each gap
        all_eta = []
        all_dates = []
        all_etrf = []
        all_weights = []
        all_scene_pairs = []
        valid_gaps = []
        
        for gap in gaps:
            logger.info("Interpolating gap from %s to %s (%d days)", 
                      gap['start_date'].strftime('%Y-%m-%d'), 
                      gap['end_date'].strftime('%Y-%m-%d'), 
                      gap['days'])
            
            # Interpolate gap
            gap_result = self._interpolate_gap(
                gap['scene1'],
                gap['scene2'],
                et0_data,
                method
            )
            
            # Collect results
            all_eta.append(gap_result['ETa_daily'])
            all_dates.extend(gap_result['dates'])
            if 'etrf_interpolated' in gap_result:
                all_etrf.append(gap_result['etrf_interpolated'])
            if 'weights' in gap_result:
                all_weights.append(gap_result['weights'])
            all_scene_pairs.append((gap['scene1'][0], gap['scene2'][0]))
            valid_gaps.append(gap)
        
        # Combine results
        if all_eta:
            eta_daily = np.concatenate(all_eta, axis=0)
        else:
            eta_daily = np.array([])
        
        if all_etrf:
            etrf_interpolated = np.concatenate(all_etrf, axis=0)
        else:
            etrf_interpolated = np.array([])
        
        if all_weights:
            weights = np.concatenate(all_weights, axis=0)
        else:
            weights = np.array([])
        
        logger.info("Interpolation complete: %d days interpolated", len(all_dates))
        
        return {
            'ETa_daily': eta_daily,
            'dates': all_dates,
            'etrf_interpolated': etrf_interpolated,
            'weights': weights,
            'scene_pairs': all_scene_pairs,
            'gaps': valid_gaps
        }
    
    def _detect_gaps(
        self,
        etrf_scenes: List[Tuple[datetime, np.ndarray]],
        gap_threshold: Optional[int] = None
    ) -> List[Dict]:
        """
        Detect gaps between consecutive Landsat scenes.
        
        Args:
            etrf_scenes: List of (date, etrf) tuples sorted by date
            gap_threshold: Maximum days to consider interpolatable (None = all gaps)
        
        Returns:
            List of gap dictionaries:
                [{
                    'scene1': (date1, etrf1),
                    'scene2': (date2, etrf2),
                    'start_date': date1,
                    'end_date': date2,
                    'days': num_days_between,
                    'target_dates': [dates to interpolate]
                }, ...]
        """
        gaps = []
        
        for i in range(len(etrf_scenes) - 1):
            date1, etrf1 = etrf_scenes[i]
            date2, etrf2 = etrf_scenes[i + 1]
            
            days_between = (date2 - date1).days
            
            # Check if gap is within threshold
            if gap_threshold is not None and days_between > gap_threshold:
                logger.info("Skipping gap between %s and %s (too large: %d days)",
                          date1.strftime('%Y-%m-%d'),
                          date2.strftime('%Y-%m-%d'),
                          days_between)
                continue
            
            # Generate target dates for interpolation (all dates between date1 and date2)
            target_dates = []
            current_date = date1 + timedelta(days=1)
            while current_date < date2:
                target_dates.append(current_date)
                current_date += timedelta(days=1)
            
            gaps.append({
                'scene1': (date1, etrf1),
                'scene2': (date2, etrf2),
                'start_date': date1,
                'end_date': date2,
                'days': days_between,
                'target_dates': target_dates
            })
        
        logger.info("Detected %d valid gaps for interpolation", len(gaps))
        
        return gaps
    
    def _interpolate_gap(
        self,
        scene1: Tuple[datetime, np.ndarray],
        scene2: Tuple[datetime, np.ndarray],
        et0_data: xr.DataArray,
        method: str = "weighted"
    ) -> Dict[str, np.ndarray]:
        """
        Interpolate ETa for a single gap between two scenes.
        
        Args:
            scene1: (date, etrf) tuple for first scene
            scene2: (date, etrf) tuple for second scene
            et0_data: ET0 data with time dimension
            method: "linear" or "weighted"
        
        Returns:
            Dictionary with interpolated values for the gap
        """
        date1, etrf1 = scene1
        date2, etrf2 = scene2
        
        # Get target dates for interpolation
        target_dates = []
        current_date = date1 + timedelta(days=1)
        while current_date < date2:
            target_dates.append(current_date)
            current_date += timedelta(days=1)
        
        if not target_dates:
            return {
                'ETa_daily': np.array([]),
                'dates': [],
                'etrf_interpolated': np.array([]),
                'weights': np.array([])
            }
        
        # Extract ET0 data for target dates
        et0_gap = et0_data.sel(time=target_dates)
        
        if et0_gap.time.size != len(target_dates):
            raise ValueError("Missing ET0 data for some target dates")
        
        et0_values = et0_gap.values
        if et0_values.ndim == 1:
            et0_values = et0_values[:, np.newaxis, np.newaxis]
            et0_values = np.repeat(np.repeat(et0_values, etrf1.shape[0], axis=1),
                                 etrf1.shape[1], axis=2)
        
        # Interpolation
        if method == "linear":
            return self._interpolate_linear(scene1, scene2, et0_gap, et0_values, target_dates)
        elif method == "weighted":
            return self._interpolate_weighted(scene1, scene2, et0_gap, et0_values, target_dates)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
    
    def _interpolate_linear(
        self,
        scene1: Tuple[datetime, np.ndarray],
        scene2: Tuple[datetime, np.ndarray],
        et0_gap: xr.DataArray,
        et0_values: np.ndarray,
        target_dates: List[datetime]
    ) -> Dict[str, np.ndarray]:
        """Interpolate using linear ETrF method."""
        date1, etrf1 = scene1
        date2, etrf2 = scene2
        
        total_days = (date2 - date1).days
        
        eta_daily = []
        etrf_interpolated = []
        
        for i, target_date in enumerate(target_dates):
            days_from_scene1 = (target_date - date1).days
            fraction = days_from_scene1 / total_days
            
            # Linear interpolation of ETrF
            etrf_t = etrf1 + fraction * (etrf2 - etrf1)
            
            # Apply ETrF bounds
            etrf_t = np.clip(etrf_t, self.config["etrf_min"], self.config["etrf_max"])
            
            # Calculate ETa
            eta_t = etrf_t * et0_values[i]
            eta_t = np.clip(eta_t, self.config["min_et_daily"], self.config["max_et_daily"])
            
            eta_daily.append(eta_t)
            etrf_interpolated.append(etrf_t)
        
        return {
            'ETa_daily': np.array(eta_daily),
            'dates': target_dates,
            'etrf_interpolated': np.array(etrf_interpolated)
        }
    
    def _interpolate_weighted(
        self,
        scene1: Tuple[datetime, np.ndarray],
        scene2: Tuple[datetime, np.ndarray],
        et0_gap: xr.DataArray,
        et0_values: np.ndarray,
        target_dates: List[datetime]
    ) -> Dict[str, np.ndarray]:
        """Interpolate using weighted ETa method."""
        date1, etrf1 = scene1
        date2, etrf2 = scene2
        
        total_days = (date2 - date1).days
        
        eta_daily = []
        weights = []
        
        # Get ET0 for scene dates from the full dataset passed to _interpolate_gap
        # We'll use the gap ET0 values at boundaries as approximation
        # (In practice, scene1 and scene2 dates are at the boundaries of the gap)
        et0_scene1_val = et0_gap.isel(time=0).values if len(target_dates) > 0 else None
        et0_scene2_val = et0_gap.isel(time=-1).values if len(target_dates) > 0 else None
        
        # Handle scalar ET0 values
        if et0_scene1_val is not None and np.ndim(et0_scene1_val) == 0:
            et0_scene1_val = float(et0_scene1_val)
        if et0_scene2_val is not None and np.ndim(et0_scene2_val) == 0:
            et0_scene2_val = float(et0_scene2_val)
        
        eta_scene1 = etrf1 * et0_scene1_val if et0_scene1_val is not None else etrf1
        eta_scene2 = etrf2 * et0_scene2_val if et0_scene2_val is not None else etrf2
        
        for i, target_date in enumerate(target_dates):
            days_from_scene1 = (target_date - date1).days
            days_to_scene2 = (date2 - target_date).days
            
            # Calculate weights
            w1 = days_to_scene2 / total_days
            w2 = days_from_scene1 / total_days
            
            # Weighted ETa
            eta_t = w1 * eta_scene1 + w2 * eta_scene2
            eta_t = np.clip(eta_t, self.config["min_et_daily"], self.config["max_et_daily"])
            
            eta_daily.append(eta_t)
            weights.append(np.array([w1, w2]))
        
        return {
            'ETa_daily': np.array(eta_daily),
            'dates': target_dates,
            'weights': np.array(weights)
        }
    
    def fetch_et0_forecast(
        self,
        bbox: List[float],  # [min_lon, min_lat, max_lon, max_lat]
        start_date: str,    # YYYY-MM-DD
        days: int = None
    ) -> xr.DataArray:
        """
        Fetch ET0 forecast from Open-Meteo API.
        
        API: https://api.open-meteo.com/v1/forecast
        Variable: et0_fao_evapotranspiration
        
        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date for forecast (YYYY-MM-DD)
            days: Number of days to forecast (default: use config)
        
        Returns:
            xarray DataArray with time dimension
        """
        if days is None:
            days = self.config["extrapolation_days"]
        
        logger.info("Fetching ET0 forecast for %d days from %s", days, start_date)
        
        # Calculate center coordinates for bounding box
        lon = (bbox[0] + bbox[2]) / 2
        lat = (bbox[1] + bbox[3]) / 2
        
        # API URL and parameters
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=days - 1)).strftime("%Y-%m-%d"),
            "daily": "et0_fao_evapotranspiration",
            "timezone": "auto"
        }
        
        # Fetch data
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("Failed to fetch ET0 forecast: %s", str(e))
            raise
        
        # Convert to xarray DataArray
        dates = pd.date_range(start=start_date, periods=days)
        et0_data = np.array(data['daily']['et0_fao_evapotranspiration'])
        
        return xr.DataArray(
            et0_data,
            dims=['time'],
            coords={'time': dates},
            attrs={
                'units': 'mm/day',
                'long_name': 'ET0 FAO Penman-Monteith',
                'source': 'Open-Meteo Forecast API',
                'bbox': bbox,
                'latitude': lat,
                'longitude': lon
            }
        )
    
    def fetch_et0_historical(
        self,
        bbox: List[float],  # [min_lon, min_lat, max_lon, max_lat]
        start_date: str,    # YYYY-MM-DD
        end_date: str       # YYYY-MM-DD
    ) -> xr.DataArray:
        """
        Fetch historical ET0 from Open-Meteo Archive API.
        
        API: https://archive-api.open-meteo.com/v1/archive
        Variable: et0_fao_evapotranspiration
        
        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date for historical period (YYYY-MM-DD)
            end_date: End date for historical period (YYYY-MM-DD)
        
        Returns:
            xarray DataArray with time dimension covering the date range
        """
        logger.info("Fetching historical ET0 from %s to %s", start_date, end_date)
        
        # Calculate center coordinates for bounding box
        lon = (bbox[0] + bbox[2]) / 2
        lat = (bbox[1] + bbox[3]) / 2
        
        # API URL and parameters
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "et0_fao_evapotranspiration",
            "timezone": "auto"
        }
        
        # Fetch data
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("Failed to fetch historical ET0: %s", str(e))
            raise
        
        # Convert to xarray DataArray
        dates = pd.date_range(start=start_date, end=end_date)
        et0_data = np.array(data['daily']['et0_fao_evapotranspiration'])
        
        return xr.DataArray(
            et0_data,
            dims=['time'],
            coords={'time': dates},
            attrs={
                'units': 'mm/day',
                'long_name': 'ET0 FAO Penman-Monteith',
                'source': 'Open-Meteo Archive API',
                'bbox': bbox,
                'latitude': lat,
                'longitude': lon
            }
        )


def create_extrapolator(config: Optional[Dict] = None) -> ETExtrapolator:
    """
    Create an ETExtrapolator instance with optional configuration.
    
    Args:
        config: Configuration dictionary with keys like 'extrapolation_days', 'interpolation_method', etc.
    
    Returns:
        ETExtrapolator instance
    """
    return ETExtrapolator(config)
