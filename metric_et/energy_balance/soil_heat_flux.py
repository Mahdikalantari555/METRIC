"""
Soil Heat Flux (G) Calculation for METRIC ETa Model.

This module implements the soil heat flux component of the energy balance equation.
G is typically 5-10% of Rn for vegetated surfaces and higher for bare soil.

METRIC Empirical Equations:
- G/Rn = (T_s - 273.15) * (0.0038 + 0.0074 * α * (1 - 0.98 * NDVI^4)) (METRIC_Workflow.md)
- G/Rn = 0.05 + 0.25 * NDVI (Bastiaanssen 1995, vegetated areas)
- G/Rn = 0.31 * (1 - NDVI) (Bastiaanssen 1995, bare soil)
- G/Rn = 0.95 - 0.19 * NDVI (Allen et al., dense vegetation)
- G/Rn = 0.5 * (1 - NDVI) (Allen et al., bare soil)

Key relationship:
- High Ts + Low NDVI = High G
- Low Ts + High NDVI = Low G
"""

import logging
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from metric_et.core.constants import (
    VON_KARMAN, AIR_DENSITY, AIR_SPECIFIC_HEAT,
    FREEZING_POINT
)


@dataclass
class SoilHeatFluxConfig:
    """Configuration for soil heat flux calculation."""
    # Method selection: 'bastiaanssen', 'allen', 'moran', 'simple', 'document'
    method: str = 'document'
    
    # Vegetation threshold for switching between bare soil and vegetation formulas
    ndvi_threshold: float = 0.2
    
    # Minimum G/Rn ratio (safety floor)
    min_gn_ratio: float = 0.01
    
    # Maximum G/Rn ratio (safety ceiling)
    # Reduced from 0.35 to 0.25 for more realistic bounds:
    # - Typical vegetated surfaces: 2-10%
    # - Bare soil: 10-20%
    # - Extreme cases rarely exceed 25%
    max_gn_ratio: float = 0.25


class SoilHeatFlux:
    """
    Calculate soil heat flux (G) for METRIC energy balance.
    
    The soil heat flux represents the energy used to heat the soil.
    This is typically a small fraction of net radiation but can be
    significant for bare soil surfaces.
    
    Attributes:
        config: Configuration parameters for G calculation
    """
    
    def __init__(self, config: Optional[SoilHeatFluxConfig] = None):
        """
        Initialize SoilHeatFlux calculator.
        
        Args:
            config: Optional configuration parameters. Uses defaults if not provided.
        """
        self.config = config or SoilHeatFluxConfig()
    
    def calculate_gn_ratio(
        self,
        ndvi: np.ndarray,
        ts_kelvin: Optional[np.ndarray] = None,
        ta_kelvin: Optional[np.ndarray] = None,
        albedo: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate G/Rn ratio using selected method.

        Args:
            ndvi: Normalized Difference Vegetation Index (0-1)
            ts_kelvin: Surface temperature in Kelvin (optional, for Moran and document methods)
            ta_kelvin: Air temperature in Kelvin (optional, for Moran method)
            albedo: Surface albedo (optional, for document method)

        Returns:
            G/Rn ratio array
        """
        method = self.config.method.lower()

        if method == 'bastiaanssen':
            return self._bastiaanssen_method(ndvi)
        elif method == 'allen':
            return self._allen_method(ndvi)
        elif method == 'moran':
            return self._moran_method(ndvi, ts_kelvin, ta_kelvin)
        elif method == 'simple':
            return self._simple_method(ndvi)
        elif method == 'document':
            return self._document_method(ndvi, ts_kelvin, albedo)
        else:
            raise ValueError(f"Unknown soil heat flux method: {method}")
    
    def _bastiaanssen_method(self, ndvi: np.ndarray) -> np.ndarray:
        """
        Bastiaanssen (1995) formulation for G/Rn.
        
        G/Rn = 0.05 + 0.25 * NDVI (vegetated)
        G/Rn = 0.31 * (1 - NDVI) (bare soil)
        """
        gn_ratio = np.zeros_like(ndvi, dtype=np.float64)
        
        # Vegetated areas (NDVI > threshold)
        veg_mask = ndvi > self.config.ndvi_threshold
        gn_ratio[veg_mask] = 0.05 + 0.25 * ndvi[veg_mask]
        
        # Bare soil areas (NDVI <= threshold)
        bare_mask = ndvi <= self.config.ndvi_threshold
        gn_ratio[bare_mask] = 0.31 * (1.0 - ndvi[bare_mask])
        
        # Apply safety limits
        gn_ratio = np.clip(gn_ratio, self.config.min_gn_ratio, self.config.max_gn_ratio)
        
        return gn_ratio
    
    def _allen_method(self, ndvi: np.ndarray) -> np.ndarray:
        """
        Allen et al. METRIC-specific formulation for G/Rn.
        
        G/Rn = 0.95 - 0.19 * NDVI (dense vegetation)
        G/Rn = 0.5 * (1 - NDVI) (bare soil)
        """
        gn_ratio = np.zeros_like(ndvi, dtype=np.float64)
        
        # Vegetated areas
        veg_mask = ndvi > self.config.ndvi_threshold
        gn_ratio[veg_mask] = 0.95 - 0.19 * ndvi[veg_mask]
        
        # Bare soil areas
        bare_mask = ndvi <= self.config.ndvi_threshold
        gn_ratio[bare_mask] = 0.5 * (1.0 - ndvi[bare_mask])
        
        # Apply safety limits
        gn_ratio = np.clip(gn_ratio, self.config.min_gn_ratio, self.config.max_gn_ratio)
        
        return gn_ratio
    
    def _moran_method(
        self,
        ndvi: np.ndarray,
        ts_kelvin: Optional[np.ndarray] = None,
        ta_kelvin: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Moran et al. formulation for G/Rn.
        
        G/Rn = 0.4 * (Ts - Ta) / Rn - 0.5 * NDVI
        
        Note: This method requires Ts and Ta, and Rn for proper calculation.
        If Ts or Ta not provided, falls back to Bastiaanssen method.
        """
        if ts_kelvin is None or ta_kelvin is None:
            # Fallback to Bastiaanssen if temperature data not available
            return self._bastiaanssen_method(ndvi)
        
        # Calculate temperature difference
        delta_t = ts_kelvin - ta_kelvin
        
        # Moran formulation (simplified, assuming Rn normalization)
        gn_ratio = 0.4 * delta_t - 0.5 * ndvi
        
        # Apply safety limits
        gn_ratio = np.clip(gn_ratio, self.config.min_gn_ratio, self.config.max_gn_ratio)
        
        return gn_ratio
    
    def _simple_method(self, ndvi: np.ndarray) -> np.ndarray:
        """
        Simple METRIC approximation for G/Rn.

        G/Rn = (0.05 + 1 - NDVI * 0.9) * 0.1
        """
        gn_ratio = (0.05 + 1.0 - ndvi * 0.9) * 0.1

        # Apply safety limits
        gn_ratio = np.clip(gn_ratio, self.config.min_gn_ratio, self.config.max_gn_ratio)

        return gn_ratio

    def _document_method(
        self,
        ndvi: np.ndarray,
        ts_kelvin: Optional[np.ndarray] = None,
        albedo: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        METRIC Workflow document formulation for G/Rn.

        Implements the equation from METRIC_Workflow.md:
        G/Rn = (T_s - 273.15) * (0.0038 + 0.0074 * α * (1 - 0.98 * NDVI^4))

        This method requires surface temperature and albedo data.
        If albedo is not available, falls back to Bastiaanssen method.

        Args:
            ndvi: Normalized Difference Vegetation Index (0-1)
            ts_kelvin: Surface temperature in Kelvin (required)
            albedo: Surface broadband albedo (optional, falls back if None)

        Returns:
            G/Rn ratio array

        Raises:
            ValueError: If required parameters are missing
        """
        if ts_kelvin is None:
            raise ValueError("Surface temperature (ts_kelvin) is required for document method")

        if albedo is None:
            # Fallback to Bastiaanssen method if albedo not available
            return self._bastiaanssen_method(ndvi)

        # Convert surface temperature to Celsius
        ts_celsius = ts_kelvin - 273.15

        # Calculate G/Rn using the document equation
        # G/Rn = (T_s - 273.15) * (0.0038 + 0.0074 * α * (1 - 0.98 * NDVI^4))
        ndvi_term = 1.0 - 0.98 * (ndvi ** 4)
        albedo_term = 0.0074 * albedo * ndvi_term
        coefficient = 0.0038 + albedo_term

        gn_ratio = ts_celsius * coefficient

        # Apply safety limits
        gn_ratio = np.clip(gn_ratio, self.config.min_gn_ratio, self.config.max_gn_ratio)

        return gn_ratio
    
    def calculate(
        self,
        rn: np.ndarray,
        ndvi: np.ndarray,
        ts_kelvin: Optional[np.ndarray] = None,
        ta_kelvin: Optional[np.ndarray] = None,
        albedo: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate soil heat flux and related parameters.

        Args:
            rn: Net radiation array (W/m²)
            ndvi: Normalized Difference Vegetation Index (0-1)
            ts_kelvin: Surface temperature in Kelvin (optional)
            ta_kelvin: Air temperature in Kelvin (optional)
            albedo: Surface broadband albedo (optional, for document method)

        Returns:
            Dictionary containing:
                - 'G': Soil heat flux (W/m²)
                - 'G_Rn_ratio': G/Rn ratio
        """
        # Ensure arrays have same dtype
        def to_numpy(arr):
            if hasattr(arr, 'values'):
                return np.asarray(arr.values, dtype=np.float64)
            else:
                return np.asarray(arr, dtype=np.float64)

        rn = to_numpy(rn)
        ndvi = to_numpy(ndvi)

        if ts_kelvin is not None:
            ts_kelvin = to_numpy(ts_kelvin)
        if ta_kelvin is not None:
            ta_kelvin = to_numpy(ta_kelvin)
        if albedo is not None:
            albedo = to_numpy(albedo)

        # Calculate G/Rn ratio
        gn_ratio = self.calculate_gn_ratio(ndvi, ts_kelvin, ta_kelvin, albedo)

        # Calculate G (soil heat flux)
        # G = Rn * (G/Rn)
        G = rn * gn_ratio

        # Handle negative Rn (nighttime or errors)
        # G cannot exceed available energy when Rn is negative
        G = np.where(rn < 0, np.minimum(G, rn * 0.5), G)

        # Physical constraints
        G = np.maximum(G, 0.0)  # G cannot be negative

        return {
            'G': G,
            'G_Rn_ratio': gn_ratio
        }
    
    def compute(self, cube):
        """
        Compute soil heat flux and add to DataCube.

        Args:
            cube: DataCube with Rn and NDVI

        Returns:
            DataCube with added G and G_Rn_ratio
        """
        from ..core.datacube import DataCube

        # Get required inputs
        rn = cube.get("R_n")
        ndvi = cube.get("ndvi")
        ts_kelvin = cube.get("lwir11")  # Surface temperature in Kelvin
        ta_kelvin = cube.get("Ta")  # Air temperature in Kelvin, if available
        albedo = cube.get("albedo")  # Albedo for document method

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Soil heat flux calculation - albedo available: {albedo is not None}")
        if albedo is not None:
            logger.info(f"Albedo shape: {albedo.shape}, mean: {np.nanmean(albedo.values):.3f}")

        if rn is None:
            raise ValueError("Net radiation (R_n) not found in DataCube")
        if ndvi is None:
            raise ValueError("NDVI not found in DataCube")

        # Calculate soil heat flux
        result = self.calculate(
            rn.values, ndvi.values,
            ts_kelvin.values if ts_kelvin is not None else None,
            ta_kelvin if ta_kelvin is not None else None,
            albedo.values if albedo is not None else None
        )

        # Add to cube
        cube.add("G", result["G"])
        cube.add("G_Rn_ratio", result["G_Rn_ratio"])

        return cube

    def __call__(
        self,
        rn: np.ndarray,
        ndvi: np.ndarray,
        ts_kelvin: Optional[np.ndarray] = None,
        ta_kelvin: Optional[np.ndarray] = None,
        albedo: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Convenience method to calculate soil heat flux.

        Args:
            rn: Net radiation array (W/m²)
            ndvi: Normalized Difference Vegetation Index (0-1)
            ts_kelvin: Surface temperature in Kelvin (optional)
            ta_kelvin: Air temperature in Kelvin (optional)
            albedo: Surface broadband albedo (optional, for document method)

        Returns:
            Dictionary with 'G' and 'G_Rn_ratio' arrays
        """
        return self.calculate(rn, ndvi, ts_kelvin, ta_kelvin, albedo)


def create_soil_heat_flux(
    method: str = 'document',
    **kwargs
) -> SoilHeatFlux:
    """
    Factory function to create SoilHeatFlux instance.

    Args:
        method: Calculation method ('document', 'bastiaanssen', 'allen', 'moran', 'simple')
               'document' uses the equation from METRIC_Workflow.md
        **kwargs: Additional configuration parameters

    Returns:
        Configured SoilHeatFlux instance
    """
    config = SoilHeatFluxConfig(method=method, **kwargs)
    return SoilHeatFlux(config)
