"""Albedo calculations for METRIC ETa model."""

import numpy as np
import xarray as xr
from typing import Optional

from ..core.datacube import DataCube


class AlbedoCalculator:
    """
    Compute surface broadband albedo from satellite imagery.
    
    This class provides methods for computing surface albedo (reflectance)
    in the shortwave range (0.3-3.0 μm), which is essential for
    computing the net radiation in energy balance calculations.
    
    Supported methods:
        - Landsat 8/9 TMAD (Terra Multi-sensor Algorithm for Data)
        - Landsat Collection 2 specific coefficients
    
    Attributes:
        use_collection2: Use Landsat Collection 2 coefficients (default: True)
        dark_pixel_correction: Apply dark pixel correction (default: True)
    """
    
    # Coefficients for Landsat Collection 1 (TMAD)
    COEFFS_COLLECTION1 = {
        'blue': 0.356,
        'red': 0.130,
        'nir': 0.373,
        'swir1': 0.085,
        'swir2': 0.072,
        'intercept': -0.0018
    }
    
    # Coefficients for Landsat Collection 2
    COEFFS_COLLECTION2 = {
        'blue': 0.254,
        'red': 0.149,
        'nir': 0.295,
        'swir1': 0.243,
        'swir2': 0.091,
        'intercept': 0.066
    }
    
    def __init__(
        self,
        use_collection2: bool = True,
        dark_pixel_correction: bool = True,
        min_albedo: float = 0.0,
        max_albedo: float = 0.5
    ):
        """
        Initialize AlbedoCalculator.
        
        Args:
            use_collection2: Use Landsat Collection 2 coefficients
            dark_pixel_correction: Apply dark pixel correction if albedo < 0
            min_albedo: Minimum valid albedo value
            max_albedo: Maximum valid albedo value
        """
        self.use_collection2 = use_collection2
        self.dark_pixel_correction = dark_pixel_correction
        self.min_albedo = min_albedo
        self.max_albedo = max_albedo
    
    def compute(self, cube: DataCube) -> DataCube:
        """
        Compute broadband albedo and add to DataCube.
        
        Args:
            cube: Input DataCube containing required bands
            
        Returns:
            DataCube with added albedo band
            
        Raises:
            ValueError: If required bands are missing
        """
        albedo = self.compute_albedo(cube)
        cube.add("albedo", albedo)
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
    
    def compute_albedo(self, cube: DataCube) -> xr.DataArray:
        """
        Compute broadband shortwave albedo.
        
        Uses TMAD coefficients for Landsat 8/9:
        α = Σ(coef_i * band_i) + intercept
        
        Args:
            cube: DataCube containing blue, red, nir08, swir16, swir22 bands
            
        Returns:
            Albedo as dimensionless DataArray [0, 1]
            
        Raises:
            ValueError: If required bands are missing
        """
        required_bands = ["blue", "red", "nir08", "swir16", "swir22"]
        missing = [b for b in required_bands if b not in cube.bands()]
        if missing:
            raise ValueError(
                f"Missing bands for albedo calculation: {missing}. "
                f"Available bands: {cube.bands()}"
            )
        
        # Get bands as float
        blue = cube.get("blue").astype(float)
        red = cube.get("red").astype(float)
        nir = cube.get("nir08").astype(float)
        swir1 = cube.get("swir16").astype(float)
        swir2 = cube.get("swir22").astype(float)
        
        # Select coefficients based on collection
        if self.use_collection2:
            coeffs = self.COEFFS_COLLECTION2
        else:
            coeffs = self.COEFFS_COLLECTION1
        
        # Compute albedo
        albedo = (
            coeffs['blue'] * blue +
            coeffs['red'] * red +
            coeffs['nir'] * nir +
            coeffs['swir1'] * swir1 +
            coeffs['swir2'] * swir2 +
            coeffs['intercept']
        )
        
        # Get nodata mask
        valid_mask = self._get_nodata_mask(cube)
        
        # Apply nodata mask
        albedo = albedo.where(valid_mask, np.nan)
        
        # Dark pixel correction: set negative values to minimum
        if self.dark_pixel_correction:
            albedo = xr.where(albedo < self.min_albedo, self.min_albedo, albedo)
        
        # Clamp to valid range
        albedo = albedo.clip(self.min_albedo, self.max_albedo)
        
        albedo.name = "albedo"
        albedo.attrs = {
            'long_name': 'Broadband Shortwave Albedo',
            'units': 'dimensionless',
            'range': f'[{self.min_albedo}, {self.max_albedo}]',
            'method': 'TMAD',
            'collection': 'Collection 2' if self.use_collection2 else 'Collection 1'
        }
        
        return albedo
    
    def compute_directional_hemispherical_albedo(
        self,
        cube: DataCube,
        solar_zenith_angle: float
    ) -> xr.DataArray:
        """
        Compute directional-hemispherical albedo (black-sky albedo).
        
        This accounts for the anisotropic reflection properties of surfaces.
        
        Args:
            cube: DataCube with broadband albedo
            solar_zenith_angle: Solar zenith angle in radians
            
        Returns:
            Black-sky albedo adjusted for solar angle
        """
        albedo = cube.get("albedo")
        if albedo is None:
            albedo = self.compute_albedo(cube)
        
        # Simple correction using hotspot parameterization
        # For near-nadir viewing, the correction is minimal
        mu0 = np.cos(solar_zenith_angle)
        
        # Avoid division by zero for high zenith angles
        mu0 = np.maximum(mu0, 0.1)
        
        # Apply directional correction
        # This is a simplified version; more complex BRDF models exist
        dhr_albedo = albedo * (1.0 + 0.1 * (1.0 - mu0))
        
        # Clamp to valid range
        dhr_albedo = dhr_albedo.clip(self.min_albedo, self.max_albedo)
        
        dhr_albedo.name = "albedo_black_sky"
        dhr_albedo.attrs = {
            'long_name': 'Directional-Hemispherical Reflectance (Black-sky)',
            'units': 'dimensionless',
            'solar_zenith_rad': solar_zenith_angle
        }
        
        return dhr_albedo
    
    def compute_bihemispherical_albedo(
        self,
        cube: DataCube,
        average_solar_zenith: float = 0.5
    ) -> xr.DataArray:
        """
        Compute bihemispherical albedo (white-sky albedo).
        
        This represents albedo under diffuse illumination conditions.
        
        Args:
            cube: DataCube with broadband albedo
            average_solar_zenith: Average solar zenith angle factor
            
        Returns:
            White-sky albedo
        """
        albedo = cube.get("albedo")
        if albedo is None:
            albedo = self.compute_albedo(cube)
        
        # White-sky albedo is typically higher than black-sky
        whr_albedo = albedo * (1.0 + 0.05 * (1.0 - average_solar_zenith))
        
        # Clamp to valid range
        whr_albedo = whr_albedo.clip(self.min_albedo, self.max_albedo)
        
        whr_albedo.name = "albedo_white_sky"
        whr_albedo.attrs = {
            'long_name': 'Bihemispherical Reflectance (White-sky)',
            'units': 'dimensionless'
        }
        
        return whr_albedo


# Alias for backward compatibility
Albedo = AlbedoCalculator


__all__ = ['AlbedoCalculator', 'Albedo']
