"""Emissivity calculations for METRIC ETa model."""

import numpy as np
import xarray as xr
from typing import Optional

from ..core.datacube import DataCube


class EmissivityCalculator:
    """
    Compute surface emissivity for thermal infrared calculations.
    
    Surface emissivity is essential for accurate estimation of longwave
    radiation and sensible heat flux in energy balance applications.
    
    Methods:
        - NDVI-based emissivity
        - Proportion of vegetation (Pv) based emissivity
        - Land cover type based emissivity
    
    Attributes:
        ndvi_min: Minimum NDVI for bare soil
        ndvi_max: Maximum NDVI for full vegetation
    """
    
    # Default emissivity values for different land cover types
    EMISSIVITY_LAND_COVER = {
        'water': 0.985,
        'bare_soil': 0.960,
        'sand': 0.955,
        'grass': 0.970,
        'forest': 0.985,
        'urban': 0.920,
        'crops': 0.970,
        'snow': 0.990,
        'default': 0.960
    }
    
    def __init__(
        self,
        ndvi_min: float = 0.0,
        ndvi_max: float = 0.8,
        use_pv_method: bool = False
    ):
        """
        Initialize EmissivityCalculator.
        
        Args:
            ndvi_min: Minimum NDVI for bare soil
            ndvi_max: Maximum NDVI for full vegetation
            use_pv_method: Use proportion of vegetation method instead of NDVI
        """
        self.ndvi_min = ndvi_min
        self.ndvi_max = ndvi_max
        self.use_pv_method = use_pv_method
    
    def compute(self, cube: DataCube) -> DataCube:
        """
        Compute emissivity and add to DataCube.
        
        Args:
            cube: Input DataCube containing NDVI or vegetation bands
            
        Returns:
            DataCube with added emissivity band
            
        Raises:
            ValueError: If required inputs are missing
        """
        emissivity = self.compute_emissivity(cube)
        cube.add("emissivity", emissivity)
        
        # Also add thermal emissivity correction factor
        thermal_corr = self.compute_thermal_emissivity_correction(cube)
        cube.add("emissivity_thermal_corr", thermal_corr)
        
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
    
    def compute_emissivity(
        self,
        cube: DataCube,
        ndvi: Optional[xr.DataArray] = None
    ) -> xr.DataArray:
        """
        Compute surface emissivity using NDVI-based method.
        
        Uses piecewise function based on NDVI value:
            NDVI < 0: ε = 0.985 (water/bare soil)
            NDVI < 0.2: ε = 0.96 + 0.025 * NDVI
            NDVI > 0.5: ε = 0.98 (dense vegetation)
            else: ε = 0.96 + 0.018 * NDVI
        
        Args:
            cube: DataCube
            ndvi: Pre-computed NDVI (uses cube.get("ndvi") if not provided)
            
        Returns:
            Emissivity as dimensionless DataArray
        """
        if ndvi is None:
            if "ndvi" not in cube.bands():
                raise ValueError("NDVI required for emissivity calculation. "
                               "Compute NDVI first or provide as argument.")
            ndvi = cube.get("ndvi")
        
        # Get nodata mask
        valid_mask = self._get_nodata_mask(cube)
        
        # Compute emissivity using piecewise function
        emissivity = xr.where(
            ndvi < 0,
            0.985,  # Water/bare soil
            xr.where(
                ndvi < 0.2,
                0.96 + 0.025 * ndvi,
                xr.where(
                    ndvi > 0.5,
                    0.98,  # Dense vegetation
                    0.96 + 0.018 * ndvi  # Moderate vegetation
                )
            )
        )
        
        # Apply nodata mask
        emissivity = emissivity.where(valid_mask, np.nan)
        
        # Clamp to valid range
        emissivity = emissivity.clip(0.90, 1.0)
        
        emissivity.name = "emissivity"
        emissivity.attrs = {
            'long_name': 'Surface Emissivity (Thermal)',
            'units': 'dimensionless',
            'range': '[0.90, 1.0]',
            'method': 'NDVI-based'
        }
        
        return emissivity
    
    def compute_emissivity_pv(
        self,
        cube: DataCube,
        ndvi: Optional[xr.DataArray] = None
    ) -> xr.DataArray:
        """
        Compute emissivity using proportion of vegetation (Pv).
        
        Method:
            Pv = ((NDVI - NDVI_min) / (NDVI_max - NDVI_min))²
            ε = 0.973 + 0.047 * ln(Pv)  for Pv > 0
        
        Args:
            cube: DataCube
            ndvi: Pre-computed NDVI
            
        Returns:
            Emissivity as dimensionless DataArray
        """
        if ndvi is None:
            if "ndvi" not in cube.bands():
                raise ValueError("NDVI required for Pv-based emissivity.")
            ndvi = cube.get("ndvi")
        
        # Get nodata mask
        valid_mask = self._get_nodata_mask(cube)
        
        # Compute proportion of vegetation
        pv = ((ndvi - self.ndvi_min) / (self.ndvi_max - self.ndvi_min)) ** 2
        pv = pv.clip(0.001, 1.0)  # Avoid log(0)
        
        # Compute emissivity using Pv relationship
        with np.errstate(invalid='ignore'):
            emissivity = xr.where(
                pv > 0,
                0.973 + 0.047 * np.log(pv),
                0.960  # Bare soil fallback
            )
        
        # Apply nodata mask
        emissivity = emissivity.where(valid_mask, np.nan)
        
        # Clamp to valid range
        emissivity = emissivity.clip(0.90, 1.0)
        
        emissivity.name = "emissivity_pv"
        emissivity.attrs = {
            'long_name': 'Surface Emissivity (Pv-based)',
            'units': 'dimensionless',
            'range': '[0.90, 1.0]',
            'method': 'Proportion of Vegetation'
        }
        
        return emissivity
    
    def compute_thermal_emissivity_correction(
        self,
        cube: DataCube,
        emissivity: Optional[xr.DataArray] = None
    ) -> xr.DataArray:
        """
        Compute thermal emissivity correction factor for Stefan-Boltzmann law.
        
        The correction factor accounts for non-blackbody behavior:
            ε_corr = 1 - ε
        
        This factor is used when computing emitted longwave radiation:
            L↑ = ε * σ * T⁴
        
        Args:
            cube: DataCube
            emissivity: Pre-computed emissivity
            
        Returns:
            Emissivity correction factor (1 - ε)
        """
        if emissivity is None:
            if "emissivity" not in cube.bands():
                emissivity = self.compute_emissivity(cube)
            else:
                emissivity = cube.get("emissivity")
        
        # Compute correction factor
        corr = 1.0 - emissivity
        
        corr.name = "emissivity_thermal_corr"
        corr.attrs = {
            'long_name': 'Thermal Emissivity Correction Factor',
            'units': 'dimensionless',
            'definition': '1 - emissivity'
        }
        
        return corr
    
    def compute_narrowband_to_broadband_emissivity(
        self,
        cube: DataCube,
        thermal_band: Optional[xr.DataArray] = None
    ) -> xr.DataArray:
        """
        Convert narrowband emissivity to broadband emissivity.
        
        Uses statistical relationship between narrowband (thermal band)
        and broadband emissivity.
        
        Args:
            cube: DataCube containing thermal band
            thermal_band: Pre-computed thermal band brightness temperature
            
        Returns:
            Broadband emissivity
        """
        if thermal_band is None:
            if "lwir11" not in cube.bands():
                raise ValueError("Thermal band (lwir11) required.")
            # Use raw thermal band values as proxy for emissivity
            thermal_band = cube.get("lwir11").astype(float)
        
        # Simple conversion: normalize to emissivity range
        # This is a placeholder; actual methods use spectral libraries
        emissivity = xr.where(
            thermal_band > 0,
            0.95 + 0.004 * (thermal_band / thermal_band.max()),
            0.96
        )
        
        emissivity = emissivity.clip(0.90, 1.0)
        
        emissivity.name = "emissivity_broadband"
        emissivity.attrs = {
            'long_name': 'Broadband Emissivity',
            'units': 'dimensionless',
            'range': '[0.90, 1.0]',
            'method': 'Narrowband to Broadband conversion'
        }
        
        return emissivity
    
    def compute_emissivity_from_land_cover(
        self,
        cube: DataCube,
        land_cover: Optional[xr.DataArray] = None
    ) -> xr.DataArray:
        """
        Compute emissivity based on land cover classification.
        
        Args:
            cube: DataCube
            land_cover: Land cover classification array
            
        Returns:
            Emissivity based on land cover types
        """
        # Default emissivity map for common land cover types
        lc_emissivity = self.EMISSIVITY_LAND_COVER
        
        # Create emissivity array from land cover (placeholder implementation)
        # In practice, this would use an actual land cover classification
        if land_cover is None:
            # Fall back to NDVI-based method
            return self.compute_emissivity(cube)
        
        emissivity = land_cover * 0 + lc_emissivity['default']
        
        emissivity.name = "emissivity_lc"
        emissivity.attrs = {
            'long_name': 'Surface Emissivity (Land Cover Based)',
            'units': 'dimensionless',
            'method': 'Land Cover Classification'
        }
        
        return emissivity


# Alias for backward compatibility
Emissivity = EmissivityCalculator


__all__ = ['EmissivityCalculator', 'Emissivity']
