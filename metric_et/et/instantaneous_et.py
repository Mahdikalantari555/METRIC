"""
Instantaneous Evapotranspiration (ET) Calculation for METRIC ETa Model.

This module converts latent heat flux (LE) to instantaneous ET rate
at the satellite overpass time.

Formulas:
    ET_inst = LE / λ
    
Where:
    - LE = latent heat flux (W/m²)
    - λ = latent heat of vaporization (J/kg) = 2.45e6 J/kg
    - Result: kg/m²/s = mm/s (since 1 kg/m² = 1 mm water)

Reference ET Fraction (ETrF):
    ETrF = ET_inst / ETr_inst
    
Where:
    - ETr_inst = instantaneous reference ET (alfalfa) at overpass time
    - ETr = ET0 × 1.15 (alfalfa reference is 15% higher than grass)

Physical Bounds:
    - 0.0 ≤ ETrF ≤ 2.0 (physical bounds)
    - ETrF = 1.0: ET equals reference ET (well-watered)
    - ETrF > 1.0: ET exceeds reference (overnight ET, advection)
    - ETrF < 1.0: ET below reference (water stress)

Spatial Patterns:
    - Wet/cold pixels: ETrF ≈ 1.0-1.2
    - Dry/hot pixels: ETrF ≈ 0.0-0.2
    - Water bodies: ETrF ≈ 1.0-1.1
    - Urban/built-up: ETrF ≈ 0.1-0.3

Regional Adaptations:
    - Midlatitude regions: max_et_rate = 1.5 mm/hr
    - Tropical regions: max_et_rate = 2.0 mm/hr (high insolation)
    - Arid regions: min_etrf = 0.0, max_etrf = 1.0
    - Irrigated regions: max_etrf = 2.0 (METRIC-compliant)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from metric_et.core.constants import (
    LATENT_HEAT_VAPORIZATION,
    SECONDS_PER_HOUR
)


@dataclass
class InstantaneousETConfig:
    """Configuration for instantaneous ET calculation."""
    
    # Latent heat of vaporization (J/kg)
    latent_heat_vaporization: float = LATENT_HEAT_VAPORIZATION
    
    # Minimum ETrF (reference ET fraction)
    min_etrf: float = 0.0

    # Maximum ETrF (reference ET fraction) - METRIC-compliant upper limit
    max_etrf: float = 2.0
    
    # Minimum ET rate (mm/hr)
    min_et_rate: float = 0.0
    
    # Maximum ET rate (mm/hr) - FIX: Increased to 2.5 for arid/Iran regions
    # Midlatitude: 2.0 mm/hr, Tropical: 2.5 mm/hr, Arid/Iran: 2.5 mm/hr
    max_et_rate: float = 2.5
    
    # Factor to convert ET0 to ETr (alfalfa reference)
    etr_factor: float = 1.15
    
    # Temperature-dependent lambda calculation
    use_temperature_lambda: bool = False
    
    # Region-specific ET rate limits
    region_max_et_rate: float = 1.5
    
    # Region-specific ETrF limits
    region_min_etrf: float = 0.0
    region_max_etrf: float = 2.0


class InstantaneousET:
    """
    Convert latent heat flux to instantaneous evapotranspiration rate.
    
    This class handles the conversion of latent heat flux (LE) measured in
    W/m² to instantaneous ET rate in mm/hr at the satellite overpass time.
    
    Attributes:
        config: Configuration parameters for ET calculation
        latent_heat: Latent heat of vaporization (J/kg)
    
    Example:
        >>> et_calc = InstantaneousET()
        >>> le = np.array([200.0, 300.0, 400.0])  # W/m²
        >>> et_inst = et_calc.calculate(le)
        >>> print(et_inst)  # mm/hr
    """
    
    def __init__(
        self,
        config: Optional[InstantaneousETConfig] = None
    ):
        """
        Initialize InstantaneousET calculator.

        Args:
            config: Optional configuration parameters. Uses defaults if not provided.
        """
        self.config = config or InstantaneousETConfig()
        self.latent_heat = self.config.latent_heat_vaporization
        
        # Set regional bounds if specified
        if self.config.region_max_et_rate != self.config.max_et_rate:
            self.config.max_et_rate = self.config.region_max_et_rate
        if self.config.region_min_etrf != self.config.min_etrf:
            self.config.min_etrf = self.config.region_min_etrf
        if self.config.region_max_etrf != self.config.max_etrf:
            self.config.max_etrf = self.config.region_max_etrf

    def _to_numpy(self, arr):
        """Convert array to numpy if it's xarray DataArray."""
        if hasattr(arr, 'values'):
            return np.asarray(arr.values, dtype=np.float64)
        else:
            return np.asarray(arr, dtype=np.float64)
    
    def calculate_et_rate(
        self,
        le: np.ndarray,
        temperature_k: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert latent heat flux to instantaneous ET rate.
        
        ET = LE / λ
        
        Where:
            - LE is in W/m² = J/(s·m²)
            - λ is in J/kg
            - Result is in kg/(m²·s) = mm/s
            - Convert to mm/hr by multiplying by 3600
        
        Args:
            le: Latent heat flux array (W/m²)
            temperature_k: Optional temperature array in Kelvin for temperature-
                          dependent latent heat calculation
            
        Returns:
            Instantaneous ET rate (mm/hr)
        """
        le = self._to_numpy(le)

        # Use temperature-dependent latent heat based on configuration
        if self.config.use_temperature_lambda or temperature_k is not None:
            temperature_k = self._to_numpy(temperature_k) if temperature_k is not None else None
            lambda_v = self._latent_heat_vaporization(temperature_k)
        else:
            lambda_v = self.latent_heat
        
        # ET (mm/hr) = LE (W/m²) × 3600 (s/hr) / λ (J/kg)
        # Since 1 kg/m² = 1 mm water
        et_mm_hr = le * SECONDS_PER_HOUR / lambda_v
        
        # Apply physical bounds
        et_mm_hr = np.clip(
            et_mm_hr,
            self.config.min_et_rate,
            self.config.max_et_rate
        )
        
        return et_mm_hr
    
    def _latent_heat_vaporization(
        self,
        temperature_k: np.ndarray
    ) -> np.ndarray:
        """
        Calculate temperature-dependent latent heat of vaporization.
        
        λ = 2.501e6 - 2361 × T(°C)
        
        Args:
            temperature_k: Temperature in Kelvin
            
        Returns:
            Latent heat of vaporization (J/kg)
        """
        t_celsius = temperature_k - 273.15
        return 2.501e6 - 2361.0 * t_celsius
    
    def _log_etrf_statistics(
        self,
        etrf_raw: np.ndarray,
        etrf_clipped: np.ndarray
    ) -> None:
        """
        Log comprehensive ETrF statistics for QA/QC.
        
        Args:
            etrf_raw: Raw ETrF values before clipping
            etrf_clipped: Clipped ETrF values
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Raw ETrF statistics
        valid_raw = etrf_raw[~np.isnan(etrf_raw)]
        if len(valid_raw) > 0:
            logger.info(f"ETrF raw (before clipping) - Min: {np.min(valid_raw):.6f}, Max: {np.max(valid_raw):.6f}, "
                       f"Mean: {np.mean(valid_raw):.6f}, Std: {np.std(valid_raw):.6f}")

        # Clipping statistics
        clipped_high = np.sum(etrf_raw > self.config.max_etrf)
        clipped_low = np.sum(etrf_raw < self.config.min_etrf)
        total_pixels = etrf_raw.size
        if clipped_high > 0:
            logger.info(f"ETrF clipping: {clipped_high}/{total_pixels} pixels clipped to max {self.config.max_etrf}")
        if clipped_low > 0:
            logger.info(f"ETrF clipping: {clipped_low}/{total_pixels} pixels clipped to min {self.config.min_etrf}")

        # Spatial variation analysis
        valid_etrf = etrf_clipped[~np.isnan(etrf_clipped)]
        if len(valid_etrf) > 0:
            unique_vals = len(np.unique(valid_etrf))
            max_min_diff = np.max(valid_etrf) - np.min(valid_etrf)
            logger.info(f"ETrF spatial variation: {unique_vals} unique values, range {max_min_diff:.6f}")
            if max_min_diff < 0.01:
                logger.warning("ETrF has minimal spatial variation - this will cause uniform ET_daily")
    
    def calculate_etrf(
        self,
        et_inst: np.ndarray,
        etr_inst: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Reference ET Fraction (ETrF).
        
        ETrF = ET_inst / ETr_inst
        
        Where:
            - ET_inst is instantaneous actual ET
            - ETr_inst is instantaneous reference ET (alfalfa)
        
        Args:
            et_inst: Instantaneous ET rate (mm/hr)
            etr_inst: Instantaneous reference ET for alfalfa (mm/hr)
            
        Returns:
            Reference ET fraction (dimensionless)
        """
        et_inst = self._to_numpy(et_inst)
        etr_inst = self._to_numpy(etr_inst)
        
        # Calculate ETrF with division by zero protection
        etrf = np.where(
            etr_inst > 0.01,  # Avoid division by near-zero values
            et_inst / etr_inst,
            0.0
        )
        
        # Apply physical bounds
        etrf_clipped = np.clip(
            etrf,
            self.config.min_etrf,
            self.config.max_etrf
        )

        # Enhanced logging for QA/QC
        self._log_etrf_statistics(etrf, etrf_clipped)

        return etrf_clipped
    
    def calculate_etof(
        self,
        et_inst: np.ndarray,
        eto_inst: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Reference ET Fraction for grass (EToF).
        
        EToF = ET_inst / ETo_inst
        
        Where:
            - ET_inst is instantaneous actual ET
            - ETo_inst is instantaneous reference ET for grass
            
        Args:
            et_inst: Instantaneous ET rate (mm/hr)
            eto_inst: Instantaneous reference ET for grass (mm/hr)
            
        Returns:
            Grass reference ET fraction (dimensionless)
        """
        et_inst = self._to_numpy(et_inst)
        eto_inst = self._to_numpy(eto_inst)
        
        # Calculate EToF with division by zero protection
        etof = np.where(
            eto_inst > 0.01,
            et_inst / eto_inst,
            0.0
        )
        
        # Apply physical bounds
        etof = np.clip(etof, 0.0, 1.5)
        
        return etof
    
    def calculate(
        self,
        le: np.ndarray,
        etr_inst: Optional[np.ndarray] = None,
        eto_inst: Optional[np.ndarray] = None,
        temperature_k: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate instantaneous ET and related metrics.
        
        Args:
            le: Latent heat flux (W/m²)
            etr_inst: Instantaneous reference ET for alfalfa (mm/hr, optional)
            eto_inst: Instantaneous reference ET for grass (mm/hr, optional)
            temperature_k: Temperature in Kelvin for temp-dependent lambda
            
        Returns:
            Dictionary containing:
                - 'ET_inst': Instantaneous ET rate (mm/hr)
                - 'ETrF': Reference ET fraction for alfalfa (dimensionless)
                - 'EToF': Reference ET fraction for grass (dimensionless)
                - 'LE': Input latent heat flux (W/m²)
        """
        le = self._to_numpy(le)
        
        # Calculate instantaneous ET rate
        et_inst = self.calculate_et_rate(le, temperature_k)
        
        result = {
            'ET_inst': et_inst,
            'LE': le
        }
        
        # Calculate ETrF if reference ET available
        if etr_inst is not None:
            result['ETrF'] = self.calculate_etrf(et_inst, etr_inst)
        
        # Calculate EToF if reference ET available
        if eto_inst is not None:
            result['EToF'] = self.calculate_etof(et_inst, eto_inst)
        
        return result
    
    def et0_to_etr(
        self,
        et0: np.ndarray
    ) -> np.ndarray:
        """
        Convert reference ET (grass) to alfalfa reference ET.
        
        ETr = ET0 × 1.15
        
        The alfalfa reference ET is approximately 15% higher than grass
        reference ET due to higher surface resistance and roughness.
        
        Args:
            et0: Reference ET for grass (mm/hr or mm/day)
            
        Returns:
            Reference ET for alfalfa (same units as input)
        """
        return et0 * self.config.etr_factor
    
    def spatial_pattern_analysis(
        self,
        etrf: np.ndarray
    ) -> Dict[str, str]:
        """
        Analyze spatial pattern of ETrF values.
        
        Args:
            etrf: Reference ET fraction array
            
        Returns:
            Dictionary with spatial pattern classification
        """
        etrf = self._to_numpy(etrf)
        
        # Calculate statistics
        mean_etrf = np.nanmean(etrf)
        std_etrf = np.nanstd(etrf)
        min_etrf = np.nanmin(etrf)
        max_etrf = np.nanmax(etrf)
        
        # Classify dominant land cover
        if mean_etrf >= 0.8:
            pattern = "Well-watered vegetation dominant"
        elif mean_etrf >= 0.5:
            pattern = "Mixed vegetation with moderate stress"
        elif mean_etrf >= 0.3:
            pattern = "Stressed vegetation or bare soil"
        else:
            pattern = "Dry/hot conditions or urban areas"
        
        return {
            'mean_etrf': str(mean_etrf),
            'std_etrf': str(std_etrf),
            'min_etrf': str(min_etrf),
            'max_etrf': str(max_etrf),
            'dominant_pattern': pattern
        }
    
    def compute(self, cube):
        """
        Compute instantaneous ET and add to DataCube.

        Args:
            cube: DataCube with LE and ETr_inst

        Returns:
            DataCube with added ET_inst, ETrF, EToF
        """
        from ..core.datacube import DataCube

        # Get required inputs
        le = cube.get("LE")
        etr_inst = cube.get("ETr_inst")

        if le is None:
            raise ValueError("Latent heat flux (LE) not found in DataCube")

        # Helper function to get values, handling scalars
        def get_values(data):
            if hasattr(data, 'values'):
                return data.values
            else:
                # Scalar: create array of same shape as le
                return np.full_like(le.values, data, dtype=np.float64)

        # Calculate instantaneous ET
        result = self.calculate(get_values(le), etr_inst=get_values(etr_inst) if etr_inst is not None else None)

        # Add to cube
        cube.add("ET_inst", result["ET_inst"])
        if "ETrF" in result:
            cube.add("ETrF", result["ETrF"])

        return cube

    def __call__(
        self,
        le: np.ndarray,
        etr_inst: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Convenience method to calculate instantaneous ET.
        """
        return self.calculate(le, etr_inst=etr_inst, **kwargs)


def create_instantaneous_et(
    min_etrf: float = 0.0,
    max_etrf: float = 2.0,
    max_et_rate: float = 2.5,  # FIX: Increased default
    use_temperature_lambda: bool = False,
    region: str = None,
    **kwargs
) -> InstantaneousET:
    """
    Factory function to create InstantaneousET instance with regional adaptations.
    
    Args:
        min_etrf: Minimum reference ET fraction (default: 0.0)
        max_etrf: Maximum reference ET fraction (default: 2.0, METRIC-compliant)
        max_et_rate: Maximum ET rate (mm/hr) - FIX: default 2.5 for arid regions
        use_temperature_lambda: Whether to always use temperature-dependent lambda
        region: Region identifier for preset configurations
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured InstantaneousET instance
        
    Examples:
        >>> # Standard METRIC configuration
        >>> et_calc = create_instantaneous_et()
        
        >>> # Tropical region with higher ET rates
        >>> et_calc = create_instantaneous_et(max_et_rate=2.5)
        
        >>> # Arid region with conservative bounds
        >>> et_calc = create_instantaneous_et(max_etrf=1.0, min_etrf=0.0)
        
        >>> # Always use temperature-dependent lambda
        >>> et_calc = create_instantaneous_et(use_temperature_lambda=True)
    """
    # Regional presets - FIX: Increased max_et_rate for all regions
    region_presets = {
        'tropical': {'max_et_rate': 2.5, 'max_etrf': 2.0},
        'arid': {'max_et_rate': 2.5, 'max_etrf': 1.5, 'min_etrf': 0.0},  # FIX: Increased
        'temperate': {'max_et_rate': 2.5, 'max_etrf': 2.0},  # FIX: Increased
        'irrigated': {'max_et_rate': 2.5, 'max_etrf': 2.0}  # FIX: Increased
    }
    
    # Apply regional presets
    if region and region in region_presets:
        kwargs.update(region_presets[region])
    
    config = InstantaneousETConfig(
        min_etrf=min_etrf,
        max_etrf=max_etrf,
        max_et_rate=max_et_rate,
        use_temperature_lambda=use_temperature_lambda,
        **kwargs
    )
    return InstantaneousET(config)
