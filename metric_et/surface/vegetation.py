"""Vegetation indices calculations for METRIC ETa model."""

import numpy as np
import xarray as xr
from typing import Optional, Union

from ..core.datacube import DataCube
from ..core.constants import NDVI_VEGETATION_THRESHOLD


class VegetationIndices:
    """
    Compute vegetation indices from satellite imagery.
    
    This class provides methods for computing common vegetation indices
    used in energy balance and evapotranspiration calculations.
    
    Supported indices:
        - NDVI (Normalized Difference Vegetation Index)
        - EVI (Enhanced Vegetation Index)
        - LAI (Leaf Area Index)
        - SAVI (Soil Adjusted Vegetation Index)
        - FVC (Fractional Vegetation Cover)
    
    Attributes:
        ndvi_min: Minimum NDVI value for vegetation (default: 0.0)
        ndvi_max: Maximum NDVI value for vegetation (default: 0.8)
        savi_l: Soil adjustment factor L (default: 0.5)
        use_asrar_lai: Use Asrar et al. LAI relationship (default: False)
    """
    
    def __init__(
        self,
        ndvi_min: float = 0.0,
        ndvi_max: float = 0.8,
        savi_l: float = 0.5,
        use_asrar_lai: bool = False
    ):
        """
        Initialize VegetationIndices calculator.
        
        Args:
            ndvi_min: Minimum NDVI value for vegetation fraction calculations
            ndvi_max: Maximum NDVI value for vegetation fraction calculations
            savi_l: Soil adjustment factor L for SAVI calculation
            use_asrar_lai: If True, use Asrar et al. LAI relationship;
                          otherwise use exponential relationship
        """
        self.ndvi_min = ndvi_min
        self.ndvi_max = ndvi_max
        self.savi_l = savi_l
        self.use_asrar_lai = use_asrar_lai
    
    def compute(self, cube: DataCube) -> DataCube:
        """
        Compute all vegetation indices and add to DataCube.
        
        Args:
            cube: Input DataCube containing required bands
            
        Returns:
            DataCube with added vegetation indices
            
        Raises:
            ValueError: If required bands are missing
        """
        self._validate_inputs(cube)
        
        # Compute NDVI (required for other indices)
        ndvi = self.compute_ndvi(cube)
        cube.add("ndvi", ndvi)
        
        # Compute EVI if blue band is available
        if "blue" in cube.bands():
            evi = self.compute_evi(cube)
            cube.add("evi", evi)
        
        # Compute LAI
        lai = self.compute_lai(cube)
        cube.add("lai", lai)
        
        # Compute SAVI
        savi = self.compute_savi(cube)
        cube.add("savi", savi)
        
        # Compute Fractional Vegetation Cover (FVC)
        fvc = self.compute_fvc(cube)
        cube.add("fvc", fvc)
        
        return cube
    
    def _validate_inputs(self, cube: DataCube) -> None:
        """
        Validate that required input bands exist in DataCube.
        
        Args:
            cube: Input DataCube to validate
            
        Raises:
            ValueError: If required bands are missing
        """
        required_bands = ["red", "nir08"]
        missing = [b for b in required_bands if b not in cube.bands()]
        if missing:
            raise ValueError(
                f"Missing required bands for vegetation indices: {missing}. "
                f"Available bands: {cube.bands()}"
            )
    
    def _get_nodata_mask(self, cube: DataCube) -> xr.DataArray:
        """
        Get nodata mask from available QA bands.

        Args:
            cube: Input DataCube

        Returns:
            Boolean DataArray where True indicates valid data
        """
        # Try to get nodata from QA pixel band first
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
                ((qa_values >> 1) & 1).astype(bool) |  # Cirrus
                ((qa_values >> 2) & 1).astype(bool) |  # Cloud
                ((qa_values >> 3) & 1).astype(bool)    # Shadow
            )
            return xr.DataArray(cloud_mask, dims=['y', 'x'])

        # Fallback: assume all data is valid
        first_band = next(iter(cube.data.values()))
        return xr.DataArray(
            np.ones(first_band.shape, dtype=bool),
            dims=['y', 'x']
        )
    
    def compute_ndvi(self, cube: DataCube) -> xr.DataArray:
        """
        Compute Normalized Difference Vegetation Index (NDVI).
        
        NDVI is a standardized index measuring vegetation health and density:
        NDVI = (NIR - Red) / (NIR + Red)
        
        Values range from -1 to 1:
            - Negative values: Water or clouds
            - Near zero: Bare soil or rocks
            - Positive values: Vegetation (higher = denser)
        
        Args:
            cube: DataCube containing red and nir08 bands
            
        Returns:
            NDVI as dimensionless DataArray (-1 to 1)
            
        Raises:
            ValueError: If red or nir08 bands are missing
        """
        if "red" not in cube.bands() or "nir08" not in cube.bands():
            raise ValueError("NDVI requires 'red' and 'nir08' bands")
        
        red = cube.get("red").astype(float)
        nir = cube.get("nir08").astype(float)
        
        # Get nodata mask
        valid_mask = self._get_nodata_mask(cube)
        
        # Compute NDVI
        denominator = nir + red
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / denominator
        
        # Set invalid pixels to nodata
        ndvi = ndvi.where(valid_mask & (denominator != 0), np.nan)
        
        # Clamp to valid range
        ndvi = ndvi.clip(-1.0, 1.0)
        
        # Add attributes
        ndvi.name = "ndvi"
        ndvi.attrs = {
            'long_name': 'Normalized Difference Vegetation Index',
            'units': 'dimensionless',
            'range': '[-1, 1]'
        }
        
        return ndvi
    
    def compute_evi(self, cube: DataCube) -> xr.DataArray:
        """
        Compute Enhanced Vegetation Index (EVI).
        
        EVI is optimized for high biomass regions and improves sensitivity:
        EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        
        Args:
            cube: DataCube containing blue, red, and nir08 bands
            
        Returns:
            EVI as dimensionless DataArray
            
        Raises:
            ValueError: If required bands are missing
        """
        required = ["blue", "red", "nir08"]
        missing = [b for b in required if b not in cube.bands()]
        if missing:
            raise ValueError(f"EVI requires bands: {missing}")
        
        blue = cube.get("blue").astype(float)
        red = cube.get("red").astype(float)
        nir = cube.get("nir08").astype(float)
        
        # Get nodata mask
        valid_mask = self._get_nodata_mask(cube)
        
        # Compute EVI with gain factor
        numerator = 2.5 * (nir - red)
        denominator = nir + 6.0 * red - 7.5 * blue + 1.0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            evi = numerator / denominator
        
        # Set invalid pixels to nodata
        evi = evi.where(valid_mask & (denominator != 0), np.nan)
        
        # Clamp to valid range [0, 1]
        evi = evi.clip(0.0, 1.0)
        
        evi.name = "evi"
        evi.attrs = {
            'long_name': 'Enhanced Vegetation Index',
            'units': 'dimensionless',
            'range': '[0, 1]'
        }
        
        return evi
    
    def compute_lai(self, cube: DataCube, ndvi: Optional[xr.DataArray] = None) -> xr.DataArray:
        """
        Compute Leaf Area Index (LAI) from NDVI.
        
        Uses either exponential or Asrar et al. relationship:
        - Exponential: LAI = 0.57 * exp(2.33 * NDVI) for 0.2 < NDVI < 0.8
        - Asrar: LAI = -ln((0.69 - NDVI)/0.59) / 0.91
        
        Args:
            cube: DataCube
            ndvi: Pre-computed NDVI (uses cube.get("ndvi") if not provided)
            
        Returns:
            LAI as dimensionless DataArray
        """
        if ndvi is None:
            if "ndvi" not in cube.bands():
                ndvi = self.compute_ndvi(cube)
            else:
                ndvi = cube.get("ndvi")
        
        # Get nodata mask
        valid_mask = self._get_nodata_mask(cube)
        
        # Compute LAI using xarray operations
        if self.use_asrar_lai:
            # Asrar et al. relationship
            # LAI = -ln((0.69 - NDVI)/0.59) / 0.91
            with np.errstate(invalid='ignore'):
                lai = -np.log((0.69 - ndvi) / 0.59) / 0.91
        else:
            # Exponential relationship with conditional logic
            lai = xr.where(
                ndvi < 0,
                0.0,
                xr.where(
                    ndvi < 0.2,
                    0.1,
                    xr.where(
                        ndvi > 0.8,
                        6.0,
                        0.57 * np.exp(2.33 * ndvi)
                    )
                )
            )
        
        # Apply nodata mask
        lai = lai.where(valid_mask, np.nan)
        
        # Clamp to reasonable range
        lai = lai.clip(0.0, 6.0)
        
        lai.name = "lai"
        lai.attrs = {
            'long_name': 'Leaf Area Index',
            'units': 'm²/m²',
            'range': '[0, 6]'
        }
        
        return lai
    
    def compute_savi(self, cube: DataCube, ndvi: Optional[xr.DataArray] = None) -> xr.DataArray:
        """
        Compute Soil Adjusted Vegetation Index (SAVI).
        
        SAVI accounts for soil brightness effects:
        SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        
        Args:
            cube: DataCube containing red and nir08 bands
            ndvi: Pre-computed NDVI (optional)
            
        Returns:
            SAVI as dimensionless DataArray
        """
        if "red" not in cube.bands() or "nir08" not in cube.bands():
            raise ValueError("SAVI requires 'red' and 'nir08' bands")
        
        red = cube.get("red").astype(float)
        nir = cube.get("nir08").astype(float)
        
        # Get nodata mask
        valid_mask = self._get_nodata_mask(cube)
        
        # Compute SAVI
        L = self.savi_l
        denominator = nir + red + L
        
        with np.errstate(divide='ignore', invalid='ignore'):
            savi = ((nir - red) / denominator) * (1 + L)
        
        # Set invalid pixels to nodata
        savi = savi.where(valid_mask & (denominator != 0), np.nan)
        
        # Clamp to reasonable range
        savi = savi.clip(-1.0, 1.0)
        
        savi.name = "savi"
        savi.attrs = {
            'long_name': 'Soil Adjusted Vegetation Index',
            'units': 'dimensionless',
            'L_factor': L
        }
        
        return savi
    
    def compute_fvc(
        self,
        cube: DataCube,
        ndvi: Optional[xr.DataArray] = None,
        ndvi_min: Optional[float] = None,
        ndvi_max: Optional[float] = None
    ) -> xr.DataArray:
        """
        Compute Fractional Vegetation Cover (FVC).
        
        FVC represents the proportion of ground covered by vegetation:
        FVC = ((NDVI - NDVI_min) / (NDVI_max - NDVI_min))²
        
        Args:
            cube: DataCube
            ndvi: Pre-computed NDVI
            ndvi_min: Minimum NDVI for bare soil (default from init)
            ndvi_max: Maximum NDVI for full vegetation (default from init)
            
        Returns:
            FVC as dimensionless DataArray [0, 1]
        """
        if ndvi is None:
            if "ndvi" not in cube.bands():
                ndvi = self.compute_ndvi(cube)
            else:
                ndvi = cube.get("ndvi")
        
        if ndvi_min is None:
            ndvi_min = self.ndvi_min
        if ndvi_max is None:
            ndvi_max = self.ndvi_max
        
        # Get nodata mask
        valid_mask = self._get_nodata_mask(cube)
        
        # Compute FVC
        with np.errstate(invalid='ignore'):
            fvc = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min)) ** 2
        
        # Apply nodata mask and clamp
        fvc = fvc.where(valid_mask, np.nan)
        fvc = fvc.clip(0.0, 1.0)
        
        fvc.name = "fvc"
        fvc.attrs = {
            'long_name': 'Fractional Vegetation Cover',
            'units': 'dimensionless',
            'range': '[0, 1]',
            'ndvi_min': ndvi_min,
            'ndvi_max': ndvi_max
        }
        
        return fvc


# Alias for backward compatibility
Vegetation = VegetationIndices


__all__ = ['VegetationIndices', 'Vegetation']
