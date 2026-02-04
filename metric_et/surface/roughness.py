"""Roughness length calculations for METRIC ETa model."""

import numpy as np
import xarray as xr
from typing import Optional

from ..core.datacube import DataCube


class RoughnessCalculator:
    """
    Compute surface roughness parameters for momentum and heat transfer.
    
    Surface roughness length (z0m) and displacement height (d) are
    critical parameters for computing aerodynamic resistance in
    sensible heat flux calculations.
    
    Methods:
        - LAI-based roughness
        - NDVI-based roughness
        - Land cover based roughness
    
    Attributes:
        z0m_min: Minimum roughness length (m)
        z0m_max: Maximum roughness length (m)
    """
    
    # Default roughness values for different land cover types (meters)
    ROUGHNESS_LAND_COVER = {
        'water': 0.001,
        'bare_soil': 0.01,
        'sand': 0.005,
        'grass': 0.03,
        'shrub': 0.1,
        'forest': 1.0,
        'urban': 0.5,
        'crops': 0.1,
        'snow': 0.001,
        'default': 0.01
    }
    
    # Displacement height factors
    DISPLACEMENT_FACTOR = {
        'water': 0.0,
        'bare_soil': 0.0,
        'grass': 0.6,
        'shrub': 0.7,
        'forest': 0.75,
        'urban': 0.8,
        'crops': 0.65,
        'default': 0.0
    }
    
    def __init__(
        self,
        z0m_min: float = 0.01,
        z0m_max: float = 2.0,
        use_ndvi_method: bool = False
    ):
        """
        Initialize RoughnessCalculator.
        
        Args:
            z0m_min: Minimum valid roughness length (m)
            z0m_max: Maximum valid roughness length (m)
            use_ndvi_method: Use NDVI-based method instead of LAI
        """
        self.z0m_min = z0m_min
        self.z0m_max = z0m_max
        self.use_ndvi_method = use_ndvi_method
    
    def compute(self, cube: DataCube) -> DataCube:
        """
        Compute all roughness parameters and add to DataCube.
        
        Args:
            cube: Input DataCube containing LAI or NDVI
            
        Returns:
            DataCube with added roughness parameters
            
        Raises:
            ValueError: If required inputs are missing
        """
        z0m = self.compute_roughness_length(cube)
        cube.add("z0m", z0m)
        
        d = self.compute_displacement_height(cube)
        cube.add("d", d)
        
        # Also compute log of roughness for resistance calculations
        log_z0m = self.compute_log_roughness(cube)
        cube.add("log_z0m", log_z0m)
        
        return cube
    
    def _get_nodata_mask(self, cube: DataCube) -> xr.DataArray:
        """
        Get nodata mask from available QA bands.
        
        Args:
            cube: Input DataCube
            
        Returns:
            Boolean DataArray where True indicates valid data
        """
        if "qa_pixel" in cube.bands():
            qa = cube.get("qa_pixel")
            
            # Handle NaN values in QA band (from cloud masking)
            if np.any(np.isnan(qa.values)):
                # If QA band has NaN values, use the NaN mask directly
                # This means cloud masking has already been applied
                valid_mask = ~np.isnan(qa.values)
                return xr.DataArray(valid_mask, dims=['y', 'x'])
            
            # Original logic for non-NaN QA values
            # Ensure proper integer type for bit operations
            qa_values = qa.values
            if qa_values.dtype.kind not in ['u', 'i']:
                qa_values = qa_values.astype(np.uint32)
            
            # Cloud masks (bits 1-3 for Landsat 8/9)
            cloud_mask = ~(
                ((qa_values >> 1) & 1).astype(bool) |
                ((qa_values >> 2) & 1).astype(bool) |
                ((qa_values >> 3) & 1).astype(bool)
            )
            return xr.DataArray(cloud_mask, dims=['y', 'x'])
        
        first_band = next(iter(cube.data.values()))
        return xr.DataArray(
            np.ones(first_band.shape, dtype=bool),
            dims=['y', 'x']
        )
    
    def compute_roughness_length(
        self,
        cube: DataCube,
        lai: Optional[xr.DataArray] = None,
        ndvi: Optional[xr.DataArray] = None
    ) -> xr.DataArray:
        """
        Compute roughness length (z0m) for momentum transfer.
        
        LAI-based method:
            if LAI < 1: z0m = 0.01
            else: z0m = 0.1 * (1 - exp(-0.4 * LAI))
        
        NDVI-based method:
            z0m = 0.005 + 0.5 * NDVI  (clamped to 0.01-2.0 m)
        
        Args:
            cube: DataCube
            lai: Pre-computed LAI (uses cube.get("lai") if not provided)
            ndvi: Pre-computed NDVI (uses cube.get("ndvi") if not provided)
            
        Returns:
            Roughness length (z0m) in meters
        """
        # Get nodata mask
        valid_mask = self._get_nodata_mask(cube)
        
        if self.use_ndvi_method or "lai" not in cube.bands():
            # Use NDVI-based method
            if ndvi is None:
                if "ndvi" not in cube.bands():
                    raise ValueError("NDVI required for NDVI-based roughness. "
                                   "Compute NDVI first or provide as argument.")
                ndvi = cube.get("ndvi")
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Calculating NDVI-based roughness: ndvi range min={np.nanmin(ndvi):.3f}, max={np.nanmax(ndvi):.3f}")
            
            z0m = 0.005 + 0.5 * ndvi
            
            logger.info(f"Raw z0m range: min={np.nanmin(z0m):.3f}, max={np.nanmax(z0m):.3f}")
            
            # Check for negative values and handle them
            negative_mask = z0m < 0
            if np.any(negative_mask):
                logger.warning(f"Found {np.sum(negative_mask)} negative z0m values, setting to minimum")
                z0m = np.where(negative_mask, self.z0m_min, z0m)
            
            logger.info(f"Final z0m range: min={np.nanmin(z0m):.3f}, max={np.nanmax(z0m):.3f}")
            
            # Check for NaN values
            nan_count = np.sum(np.isnan(z0m))
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN z0m values, replacing with minimum")
                z0m = np.where(np.isnan(z0m), self.z0m_min, z0m)
            
            logger.info(f"Final z0m after NaN handling: min={np.nanmin(z0m):.3f}, max={np.nanmax(z0m):.3f}")
            
            # Check for zero values
            zero_count = np.sum(z0m == 0)
            if zero_count > 0:
                logger.warning(f"Found {zero_count} zero z0m values, replacing with minimum")
                z0m = np.where(z0m == 0, self.z0m_min, z0m)
            
            logger.info(f"Final z0m after zero handling: min={np.nanmin(z0m):.3f}, max={np.nanmax(z0m):.3f}")
            
            # Check for very small values
            small_mask = (z0m > 0) & (z0m < 0.001)
            if np.any(small_mask):
                logger.warning(f"Found {np.sum(small_mask)} very small z0m values, replacing with minimum")
                z0m = np.where(small_mask, self.z0m_min, z0m)
            
            logger.info(f"Final z0m after small value handling: min={np.nanmin(z0m):.3f}, max={np.nanmax(z0m):.3f}")
            
        else:
            # Use LAI-based method
            if lai is None:
                if "lai" not in cube.bands():
                    raise ValueError("LAI required for LAI-based roughness. "
                                   "Compute LAI first or provide as argument.")
                lai = cube.get("lai")
            
            z0m = xr.where(
                lai < 1,
                0.01,
                0.1 * (1 - np.exp(-0.4 * lai))
            )
        
        # Apply nodata mask
        z0m = z0m.where(valid_mask, np.nan)
        
        # Clamp to valid range
        z0m = z0m.clip(self.z0m_min, self.z0m_max)
        
        z0m.name = "z0m"
        z0m.attrs = {
            'long_name': 'Roughness Length for Momentum',
            'units': 'm',
            'range': f'[{self.z0m_min}, {self.z0m_max}]',
            'method': 'NDVI-based' if self.use_ndvi_method else 'LAI-based'
        }
        
        return z0m
    
    def compute_displacement_height(
        self,
        cube: DataCube,
        lai: Optional[xr.DataArray] = None,
        z0m: Optional[xr.DataArray] = None
    ) -> xr.DataArray:
        """
        Compute displacement height (d).
        
        Displacement height represents the level at which wind speed
        is zero due to vegetation drag. It is typically ~2/3 of
        vegetation height.
        
        Approximate relationship:
            d = 0.66 * LAI * z0m
        
        Args:
            cube: DataCube
            lai: Pre-computed LAI
            z0m: Pre-computed roughness length
            
        Returns:
            Displacement height in meters
        """
        # Get nodata mask
        valid_mask = self._get_nodata_mask(cube)
        
        if lai is None:
            if "lai" in cube.bands():
                lai = cube.get("lai")
            else:
                # Estimate from NDVI if LAI not available
                if "ndvi" in cube.bands():
                    ndvi = cube.get("ndvi")
                    lai = xr.where(ndvi < 0.2, 0.1, xr.where(ndvi > 0.8, 6.0, 0.57 * np.exp(2.33 * ndvi)))
                else:
                    raise ValueError("LAI or NDVI required for displacement height.")
        
        if z0m is None:
            if "z0m" in cube.bands():
                z0m = cube.get("z0m")
            else:
                z0m = self.compute_roughness_length(cube)
        
        # Compute displacement height
        d = 0.66 * lai * z0m
        
        # Apply nodata mask
        d = d.where(valid_mask, np.nan)
        
        # Clamp to reasonable range
        d = d.clip(0.0, 10.0)
        
        d.name = "d"
        d.attrs = {
            'long_name': 'Displacement Height',
            'units': 'm',
            'range': '[0, 10]',
            'method': 'LAI-based approximation'
        }
        
        return d
    
    def compute_log_roughness(
        self,
        cube: DataCube,
        z0m: Optional[xr.DataArray] = None,
        z0h: Optional[xr.DataArray] = None
    ) -> xr.DataArray:
        """
        Compute log of roughness length for resistance calculations.
        
        Used in aerodynamic resistance computation:
            ra = [ln((z - d) / z0m) * ln((z - d) / z0h)] / (k² * u)
        
        Args:
            cube: DataCube
            z0m: Roughness length for momentum
            z0h: Roughness length for heat (defaults to z0m/10)
            
        Returns:
            Log roughness parameter
        """
        # Get nodata mask
        valid_mask = self._get_nodata_mask(cube)
        
        if z0m is None:
            if "z0m" in cube.bands():
                z0m = cube.get("z0m")
            else:
                z0m = self.compute_roughness_length(cube)
        
        if z0h is None:
            # Heat roughness is typically ~1/10 of momentum roughness
            z0h = z0m / 10.0
        
        # Compute log of roughness ratio
        log_z0_ratio = np.log(z0m / z0h)
        
        # Apply nodata mask
        log_z0_ratio = log_z0_ratio.where(valid_mask, np.nan)
        
        log_z0_ratio.name = "log_z0m_z0h"
        log_z0_ratio.attrs = {
            'long_name': 'Log Ratio of Momentum to Heat Roughness',
            'units': 'dimensionless',
            'definition': 'ln(z0m / z0h)'
        }
        
        return log_z0_ratio
    
    def compute_roughness_from_land_cover(
        self,
        cube: DataCube,
        land_cover: Optional[xr.DataArray] = None
    ) -> xr.DataArray:
        """
        Compute roughness length based on land cover classification.
        
        Args:
            cube: DataCube
            land_cover: Land cover classification array
            
        Returns:
            Roughness length in meters
        """
        lc_roughness = self.ROUGHNESS_LAND_COVER
        
        if land_cover is None:
            # Fall back to NDVI-based method
            return self.compute_roughness_length(cube)
        
        # Create roughness array from land cover (placeholder)
        # In practice, this would use actual land cover classification
        z0m = land_cover * 0 + lc_roughness['default']
        
        z0m.name = "z0m_lc"
        z0m.attrs = {
            'long_name': 'Roughness Length (Land Cover Based)',
            'units': 'm',
            'method': 'Land Cover Classification'
        }
        
        return z0m
    
    def compute_aerodynamic_parameters(
        self,
        cube: DataCube,
        z_measurement: float = 2.0
    ) -> dict:
        """
        Compute all aerodynamic parameters for sensible heat flux.
        
        Args:
            cube: DataCube with computed roughness parameters
            z_measurement: Measurement height in meters (default: 2m)
            
        Returns:
            Dictionary with aerodynamic parameters
        """
        z0m = cube.get("z0m") or self.compute_roughness_length(cube)
        d = cube.get("d") or self.compute_displacement_height(cube)
        z0h = z0m / 10.0  # Heat roughness
        
        # Logarithmic profiles
        log_z0m_z = np.log((z_measurement - d) / z0m)
        log_z0h_z = np.log((z_measurement - d) / z0h)
        
        return {
            'z0m': z0m,
            'd': d,
            'z0h': z0h,
            'log_z0m_z': log_z0m_z,
            'log_z0h_z': log_z0h_z,
            'z_measurement': z_measurement
        }


# Alias for backward compatibility
Roughness = RoughnessCalculator
RoughnessLength = RoughnessCalculator


__all__ = ['RoughnessCalculator', 'Roughness', 'RoughnessLength']
