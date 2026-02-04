"""
Validation utilities for METRIC ETa pipeline.

Provides validation functions for data checking, value range checks,
and data quality assurance.
"""

import re
from pathlib import Path
from typing import Union, Tuple
import numpy as np
import pandas as pd


# Required columns for weather data
WEATHER_REQUIRED_COLUMNS = [
    "temperature_2m",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "pressure",
    "solar_radiation"
]

# Required bands for Landsat data
LANDSAT_REQUIRED_BANDS = [
    "blue",
    "green",
    "red",
    "nir08",
    "swir16",
    "swir22",
    "lwir11",
    "qa_pixel"
]


def validate_scene_path(path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Validate Landsat scene directory structure.
    
    Args:
        path: Path to Landsat scene directory
        
    Returns:
        Tuple of (is_valid, message)
    """
    path = Path(path)
    
    if not path.exists():
        return False, f"Scene path does not exist: {path}"
    
    if not path.is_dir():
        return False, f"Scene path is not a directory: {path}"
    
    # Check for required files
    required_patterns = [
        r"blue\.tif$",
        r"green\.tif$",
        r"red\.tif$",
        r"nir08\.tif$",
        r"swir16\.tif$",
        r"swir22\.tif$",
        r"lwir11\.tif$",
        r"MTL\.json$"
    ]
    
    files = [f.name for f in path.iterdir() if f.is_file()]
    
    for pattern in required_patterns:
        if not any(re.search(pattern, f) for f in files):
            return False, f"Missing required file matching pattern: {pattern}"
    
    return True, f"Valid Landsat scene: {path.name}"


def validate_weather_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Check weather data has all required columns.
    
    Args:
        df: Weather data DataFrame
        
    Returns:
        Tuple of (is_valid, message)
    """
    if df is None or df.empty:
        return False, "Weather data is empty or None"
    
    missing_cols = set(WEATHER_REQUIRED_COLUMNS) - set(df.columns)
    
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check for null values
    null_counts = df[WEATHER_REQUIRED_COLUMNS].isnull().sum()
    if null_counts.any():
        return False, f"Null values found in columns: {null_counts[null_counts > 0].to_dict()}"
    
    return True, "Weather data is valid"


def validate_calibration_params(a: float, b: float) -> Tuple[bool, str]:
    """
    Validate calibration coefficients.
    
    METRIC calibration typically has:
    - a > 0 (positive slope)
    - b < Ts_hot (intercept less than hot pixel temperature)
    
    Args:
        a: Slope coefficient
        b: Intercept coefficient
        
    Returns:
        Tuple of (is_valid, message)
    """
    # First check if values are numeric and finite
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False, "Calibration parameters must be numeric"
    
    # Check for NaN and Inf
    if not np.isfinite(a) or not np.isfinite(b):
        return False, "Calibration parameters must be finite values"
    
    if a <= 0:
        return False, f"Slope 'a' must be positive, got: {a}"
    
    return True, "Calibration parameters are valid"


def check_ndvi_range(ndvi: np.ndarray, tol: float = 0.001) -> Tuple[bool, str]:
    """
    Check NDVI values are within valid range [-1, 1].
    
    Args:
        ndvi: NDVI array
        tol: Tolerance for boundary check
        
    Returns:
        Tuple of (is_valid, message)
    """
    if ndvi is None:
        return False, "NDVI array is None"
    
    if ndvi.size == 0:
        return False, "NDVI array is empty"
    
    valid_mask = (ndvi >= -1 - tol) & (ndvi <= 1 + tol)
    
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        invalid_percent = (invalid_count / ndvi.size) * 100
        return False, f"NDVI values out of range [-1, 1]: {invalid_count} pixels ({invalid_percent:.2f}%)"
    
    # Check for reasonable values (typical vegetation: 0.1 to 0.9)
    reasonable_mask = (ndvi >= -0.2) & (ndvi <= 1.0)
    if np.any(~reasonable_mask):
        Logger.warning(f"Found NDVI values outside typical range [-0.2, 1.0]")
    
    return True, "NDVI values are valid"


def check_albedo_range(albedo: np.ndarray, tol: float = 0.01) -> Tuple[bool, str]:
    """
    Check albedo values are within valid range [0, 1].
    
    Args:
        albedo: Albedo array
        tol: Tolerance for boundary check
        
    Returns:
        Tuple of (is_valid, message)
    """
    if albedo is None:
        return False, "Albedo array is None"
    
    if albedo.size == 0:
        return False, "Albedo array is empty"
    
    valid_mask = (albedo >= 0 - tol) & (albedo <= 1 + tol)
    
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        invalid_percent = (invalid_count / albedo.size) * 100
        return False, f"Albedo values out of range [0, 1]: {invalid_count} pixels ({invalid_percent:.2f}%)"
    
    return True, "Albedo values are valid"


def check_temperature_range(Ts: np.ndarray, min_temp: float = 250, max_temp: float = 400) -> Tuple[bool, str]:
    """
    Check surface temperature values are reasonable (250-400 K).
    
    Args:
        Ts: Surface temperature array in Kelvin
        min_temp: Minimum valid temperature (K)
        max_temp: Maximum valid temperature (K)
        
    Returns:
        Tuple of (is_valid, message)
    """
    if Ts is None:
        return False, "Temperature array is None"
    
    if Ts.size == 0:
        return False, "Temperature array is empty"
    
    valid_mask = (Ts >= min_temp) & (Ts <= max_temp)
    
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        invalid_percent = (invalid_count / Ts.size) * 100
        
        if invalid_count > 0:
            temp_stats = Ts[~valid_mask]
            Logger.warning(f"Temperature outliers: min={temp_stats.min():.1f}K, max={temp_stats.max():.1f}K")
        
        return False, f"Temperature values out of range [{min_temp}, {max_temp}] K: {invalid_count} pixels ({invalid_percent:.2f}%)"
    
    return True, f"Temperature values are valid (range: {Ts.min():.1f} - {Ts.max():.1f} K)"


def check_et_range(ET: np.ndarray, max_et: float = 15) -> Tuple[bool, str]:
    """
    Check ET values are within expected range [0, max_et] mm/day.
    
    Args:
        ET: ET array in mm/day
        max_et: Maximum expected ET value
        
    Returns:
        Tuple of (is_valid, message)
    """
    if ET is None:
        return False, "ET array is None"
    
    if ET.size == 0:
        return False, "ET array is empty"
    
    valid_mask = (ET >= 0) & (ET <= max_et)
    
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        invalid_percent = (invalid_count / ET.size) * 100
        
        if invalid_count > 0:
            et_stats = ET[~valid_mask]
            Logger.warning(f"ET outliers: min={et_stats.min():.2f} mm/day, max={et_stats.max():.2f} mm/day")
        
        return False, f"ET values out of range [0, {max_et}] mm/day: {invalid_count} pixels ({invalid_percent:.2f}%)"
    
    return True, f"ET values are valid (range: {ET.min():.2f} - {ET.max():.2f} mm/day)"


def check_for_nodata(data: np.ndarray, nodata_value: float = np.nan, max_nodata_ratio: float = 0.1) -> Tuple[bool, str]:
    """
    Check for excessive nodata values.
    
    Args:
        data: Input array
        nodata_value: Value indicating no data (e.g., NaN, -9999)
        max_nodata_ratio: Maximum allowed ratio of nodata pixels
        
    Returns:
        Tuple of (is_valid, message)
    """
    if data is None:
        return False, "Data array is None"
    
    if data.size == 0:
        return False, "Data array is empty"
    
    if np.isnan(nodata_value):
        nodata_count = np.sum(np.isnan(data))
    else:
        nodata_count = np.sum(data == nodata_value)
    
    nodata_ratio = nodata_count / data.size
    
    if nodata_ratio > max_nodata_ratio:
        return False, f"Excessive nodata values: {nodata_count} pixels ({nodata_ratio*100:.2f}%), max allowed: {max_nodata_ratio*100:.2f}%"
    
    if nodata_count > 0:
        return True, f"Nodata values found: {nodata_count} pixels ({nodata_ratio*100:.2f}%)"
    
    return True, "No nodata values found"


def check_cloud_coverage(cloud_mask: np.ndarray, threshold: float = 0.5) -> Tuple[bool, str]:
    """
    Check if cloud coverage exceeds threshold.
    
    Args:
        cloud_mask: Cloud mask array (True = cloud)
        threshold: Maximum cloud coverage ratio (default 50%)
        
    Returns:
        Tuple of (is_valid, message)
    """
    if cloud_mask is None:
        return False, "Cloud mask is None"
    
    if cloud_mask.size == 0:
        return False, "Cloud mask is empty"
    
    cloud_ratio = np.sum(cloud_mask) / cloud_mask.size
    
    if cloud_ratio > threshold:
        return False, f"Cloud coverage ({cloud_ratio*100:.2f}%) exceeds threshold ({threshold*100:.2f}%)"
    
    return True, f"Cloud coverage acceptable: {cloud_ratio*100:.2f}%"


def check_energy_balance(Rn: np.ndarray, G: np.ndarray, H: np.ndarray, LE: np.ndarray, 
                         tolerance: float = 0.1) -> dict:
    """
    Verify energy balance closure: Rn - G = H + LE
    
    Args:
        Rn: Net radiation array (W/m²)
        G: Soil heat flux array (W/m²)
        H: Sensible heat flux array (W/m²)
        LE: Latent heat flux array (W/m²)
        tolerance: Maximum residual ratio
        
    Returns:
        Dictionary with energy balance metrics
    """
    # Calculate available energy
    available_energy = Rn - G
    
    # Calculate turbulent fluxes
    turbulent_fluxes = H + LE
    
    # Calculate energy balance residual
    residual = available_energy - turbulent_fluxes
    
    # Calculate closure ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        closure_ratio = np.where(
            available_energy > 10,  # Only where available energy is significant
            turbulent_fluxes / available_energy,
            np.nan
        )
    
    # Filter valid pixels for statistics
    valid_mask = ~np.isnan(closure_ratio) & (available_energy > 10)
    
    if np.sum(valid_mask) == 0:
        return {
            "valid": False,
            "message": "No valid pixels for energy balance check",
            "mean_closure": np.nan,
            "std_closure": np.nan,
            "mean_residual": np.nan,
            "residual_ratio": np.nan
        }
    
    mean_closure = np.nanmean(closure_ratio[valid_mask])
    std_closure = np.nanstd(closure_ratio[valid_mask])
    mean_residual = np.nanmean(residual[valid_mask])
    
    # Check if closure is acceptable
    is_valid = (1 - tolerance) <= mean_closure <= (1 + tolerance)
    
    return {
        "valid": is_valid,
        "mean_closure": float(mean_closure),
        "std_closure": float(std_closure),
        "mean_residual": float(mean_residual),
        "residual_ratio": float(mean_residual / np.nanmean(np.abs(available_energy[valid_mask]))) if np.nanmean(np.abs(available_energy[valid_mask])) > 0 else 0,
        "valid_pixels": int(np.sum(valid_mask)),
        "total_pixels": int(np.sum(~np.isnan(Rn))),
        "message": f"Energy balance closure: {mean_closure*100:.1f}% (target: 100%, ±{tolerance*100:.0f}%)"
    }


def validate_input_data(**kwargs) -> Tuple[bool, str]:
    """
    Validate all input data for METRIC pipeline.
    
    Args:
        **kwargs: Input data arrays to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    errors = []
    warnings = []
    
    # Check required inputs
    required_inputs = ["ndvi", "albedo", "temperature", "net_radiation"]
    
    for input_name in required_inputs:
        if input_name not in kwargs:
            errors.append(f"Missing required input: {input_name}")
        elif kwargs[input_name] is None:
            errors.append(f"Input '{input_name}' is None")
        elif kwargs[input_name].size == 0:
            errors.append(f"Input '{input_name}' is empty")
    
    if errors:
        return False, "; ".join(errors)
    
    # Validate NDVI
    is_valid, msg = check_ndvi_range(kwargs["ndvi"])
    if not is_valid:
        errors.append(f"NDVI: {msg}")
    
    # Validate albedo
    is_valid, msg = check_albedo_range(kwargs["albedo"])
    if not is_valid:
        errors.append(f"Albedo: {msg}")
    
    # Validate temperature
    is_valid, msg = check_temperature_range(kwargs["temperature"])
    if not is_valid:
        errors.append(f"Temperature: {msg}")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, "All input data validated successfully"


# Import Logger for warning messages
from metric_et.utils.logger import Logger
