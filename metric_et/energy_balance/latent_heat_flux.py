"""
Latent Heat Flux (LE) Calculation for METRIC ETa Model.

This module implements the latent heat flux component of the energy balance equation.
LE represents the energy used for evapotranspiration (water vaporization).

Energy Balance Residual:
    LE = Rn - G - H

Quality Checks:
    - LE must be ≥ 0 for evaporation to occur
    - LE ≤ Rn - G (energy conservation)
    - Negative LE indicates energy deficit (advection, measurement error)

Positive and Negative Energy Balance:
    if Rn - G > 0:
        LE = Rn - G - H
    else:
        LE = 0  # No latent heat flux possible

Fraction of Available Energy (EF):
    EF = LE / (Rn - G)
    - EF = 1.0 for wet/vegetated areas (all energy to ET)
    - EF = 0.0 for dry/bare areas (all energy to H)
    - EF > 1.0 possible for advection conditions (LE > Rn - G)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from metric_et.core.constants import (
    LATENT_HEAT_VAPORIZATION,
    FREEZING_POINT
)


@dataclass
class LatentHeatFluxConfig:
    """Configuration for latent heat flux calculation."""
    # Minimum EF (evaporative fraction) - safety floor
    min_ef: float = 0.0
    
    # Maximum EF (evaporative fraction) - safety ceiling
    # PHYSICAL LIMIT: EF can theoretically reach 2 for fully wet surfaces
    # Values > 2 indicate energy balance issues (negative H or measurement errors)
    max_ef: float = 2.0
    
    # Allow negative LE (advection conditions)
    allow_negative_le: bool = False
    
    # LE floor (W/m²)
    min_le: float = -100.0
    
    # LE ceiling (W/m²)
    max_le: float = 1000.0
    
    # Minimum available energy for LE calculation
    min_available_energy: float = 10.0
    
    # Quality flag threshold
    quality_threshold: float = 0.8


class LatentHeatFlux:
    """
    Calculate latent heat flux (LE) for METRIC energy balance.
    
    The latent heat flux represents the energy used for evapotranspiration.
    It is calculated as the residual of the energy balance equation:
    
        LE = Rn - G - H
    
    This is the most important component for ET estimation, as it directly
    relates to the water vaporization process.
    
    Attributes:
        config: Configuration parameters for LE calculation
    """
    
    def __init__(self, config: Optional[LatentHeatFluxConfig] = None):
        """
        Initialize LatentHeatFlux calculator.
        
        Args:
            config: Optional configuration parameters. Uses defaults if not provided.
        """
        self.config = config or LatentHeatFluxConfig()
    
    def calculate_available_energy(
        self,
        rn: np.ndarray,
        G: np.ndarray
    ) -> np.ndarray:
        """
        Calculate available energy (Rn - G).
        
        Args:
            rn: Net radiation (W/m²)
            G: Soil heat flux (W/m²)
            
        Returns:
            Available energy (W/m²)
        """
        return rn - G
    
    def calculate_le_residual(
        self,
        rn: np.ndarray,
        G: np.ndarray,
        H: np.ndarray
    ) -> np.ndarray:
        """
        Calculate LE as energy balance residual.
        
        LE = Rn - G - H
        
        Args:
            rn: Net radiation (W/m²)
            G: Soil heat flux (W/m²)
            H: Sensible heat flux (W/m²)
            
        Returns:
            Latent heat flux (W/m²)
        """
        return rn - G - H
    
    def calculate_ef(
        self,
        le: np.ndarray,
        available_energy: np.ndarray
    ) -> np.ndarray:
        """
        Calculate evaporative fraction (EF).
        
        EF = LE / (Rn - G)
        
        Args:
            le: Latent heat flux (W/m²)
            available_energy: Rn - G (W/m²)
            
        Returns:
            Evaporative fraction (dimensionless)
        """
        # Avoid division by zero
        ae_safe = np.where(
            np.abs(available_energy) < self.config.min_available_energy,
            np.sign(available_energy) * self.config.min_available_energy,
            available_energy
        )
        
        ef = le / ae_safe
        
        return ef
    
    def apply_energy_balance_constraints(
        self,
        le: np.ndarray,
        rn: np.ndarray,
        G: np.ndarray,
        H: np.ndarray
    ) -> np.ndarray:
        """
        Apply energy balance constraints to LE based on configuration.

        METRIC principle: LE is strictly residual, but constraints can be applied
        based on use case (daytime vs. nocturnal, advection conditions).
        By default, allows LE > available_energy to permit EF > 1.0.

        Args:
            le: Latent heat flux (W/m²)
            rn: Net radiation (W/m²)
            G: Soil heat flux (W/m²)
            H: Sensible heat flux (W/m²)

        Returns:
            Constrained latent heat flux (W/m²)
        """
        # Apply constraints based on configuration
        if self.config.allow_negative_le:
            # Allow negative LE for advection or nocturnal conditions
            # Apply physical bounds
            le = np.clip(le, self.config.min_le, self.config.max_le)
        else:
            # Standard METRIC daytime assumption: LE >= 0
            le = np.maximum(le, 0.0)

            # Removed constraint: LE should not exceed available energy
            # This allows EF > 1.0 for advection or special conditions
            # available_energy = rn - G
            # le = np.where(
            #     (available_energy > 0) & (le > available_energy),
            #     available_energy,
            #     le
            # )

        return le
    
    def calculate_et_instantaneous(
        self,
        le: np.ndarray,
        lambda_vaporization: float = LATENT_HEAT_VAPORIZATION
    ) -> np.ndarray:
        """
        Convert LE to instantaneous ET rate.
        
        ET = LE / λ (where λ is latent heat of vaporization)
        
        Result in mm/hour (assuming le is in W/m²)
        
        Args:
            le: Latent heat flux (W/m²)
            lambda_vaporization: Latent heat of vaporization (J/kg)
            
        Returns:
            Instantaneous ET rate (mm/hr)
        """
        # W/m² = J/(s·m²)
        # ET (mm/s) = LE (W/m²) / λ (J/kg)
        # Since 1 kg/m² = 1 mm water, density cancels out
        # ET (mm/hr) = LE * 3600 / λ
        
        et_mm_hr = le * 3600.0 / lambda_vaporization
        
        return et_mm_hr
    
    def calculate_et_fraction(
        self,
        le: np.ndarray,
        eto: np.ndarray,
        etr: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate ET fractions (ETrF and EToF).
        
        ETrF = ET / ETr (reference ET for alfalfa)
        EToF = ET / ETo (reference ET for grass)
        
        Args:
            le: Latent heat flux (W/m²)
            eto: Reference ET for grass (mm/hr)
            etr: Reference ET for alfalfa (mm/hr)
            
        Returns:
            Dictionary with ET fractions
        """
        et_inst = self.calculate_et_instantaneous(le)
        
        # Calculate fractions
        etrf = np.where(etr > 0, et_inst / etr, 0.0)
        etof = np.where(eto > 0, et_inst / eto, 0.0)
        
        # Clip to reasonable range based on configuration
        # Allow higher values for stressed or irrigated conditions
        max_et_fraction = 2.0  # Set to match ETrF max
        etrf = np.clip(etrf, 0.0, max_et_fraction)
        etof = np.clip(etof, 0.0, max_et_fraction)
        
        return {
            'ET_inst': et_inst,
            'ETrF': etrf,
            'EToF': etof
        }
    
    def quality_check(
        self,
        le: np.ndarray,
        rn: np.ndarray,
        G: np.ndarray,
        H: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Perform quality checks on LE calculation.
        
        Returns quality flags and metrics.
        
        Args:
            le: Latent heat flux (W/m²)
            rn: Net radiation (W/m²)
            G: Soil heat flux (W/m²)
            H: Sensible heat flux (W/m²)
            
        Returns:
            Dictionary with quality metrics
        """
        available_energy = rn - G
        
        # Energy balance residual
        residual = rn - G - H - le
        
        # Relative residual
        rel_residual = np.where(
            np.abs(available_energy) > 1.0,
            residual / np.abs(available_energy),
            0.0
        )
        
        # Quality flag (1.0 = good, 0.0 = bad)
        quality = np.ones_like(le, dtype=np.float64)
        
        # Check for energy balance violations
        # LE should be >= 0 (when AE > 0), but can exceed available_energy for advection
        quality = np.where(
            (available_energy > 0) & (le < 0),
            0.0,
            quality
        )
        
        # Check for large residuals
        # Improved scaling to prevent negative quality values
        quality_reduction = np.where(
            np.abs(rel_residual) > 0.1,
            np.minimum(np.abs(rel_residual) * 5.0, 1.0),  # Cap reduction at 1.0
            0.0
        )
        quality = np.maximum(quality - quality_reduction, 0.0)  # Ensure non-negative
        
        # Additional quality checks for energy balance violations
        # LE should be >= 0 (when AE > 0) for standard cases, but can exceed available_energy
        if not self.config.allow_negative_le:
            quality = np.where(
                (available_energy > 0) & (le < 0),
                0.0,
                quality
            )
        
        return {
            'quality_flag': quality,
            'residual': residual,
            'relative_residual': rel_residual,
            'available_energy': available_energy
        }
    
    def calculate(
        self,
        rn: np.ndarray,
        G: np.ndarray,
        H: np.ndarray,
        ts_kelvin: Optional[np.ndarray] = None,
        apply_constraints: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Calculate latent heat flux and related parameters with comprehensive validation.
        
        Args:
            rn: Net radiation array (W/m²)
            G: Soil heat flux array (W/m²)
            H: Sensible heat flux array (W/m²)
            ts_kelvin: Surface temperature in Kelvin (optional, for stress indicators)
            apply_constraints: Whether to apply energy balance constraints
            
        Returns:
            Dictionary containing:
                - 'LE': Latent heat flux (W/m²)
                - 'available_energy': Rn - G (W/m²)
                - 'EF': Evaporative fraction
                - 'ET_inst': Instantaneous ET rate (mm/hr)
                - 'stress_indicator': Optional stress indicator from surface temperature
                
        Raises:
            ValueError: If input arrays have incompatible shapes or invalid units
        """
        # Input validation
        def to_numpy(arr):
            if hasattr(arr, 'values'):
                return np.asarray(arr.values, dtype=np.float64)
            else:
                return np.asarray(arr, dtype=np.float64)

        rn = to_numpy(rn)
        G = to_numpy(G)
        H = to_numpy(H)
        
        # Validate input shapes
        if not (rn.shape == G.shape == H.shape):
            raise ValueError(f"Input arrays must have same shape. Got rn: {rn.shape}, G: {G.shape}, H: {H.shape}")
        
        # Validate units (basic checks)
        if np.any(np.abs(rn) > 2000) or np.any(np.abs(G) > 500) or np.any(np.abs(H) > 2000):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Potential unit mismatch: extreme values detected in input arrays")
        
        # Calculate LE as residual
        LE = self.calculate_le_residual(rn, G, H)
        
        # Calculate available energy
        available_energy = self.calculate_available_energy(rn, G)
        
        # Calculate evaporative fraction
        EF = self.calculate_ef(LE, available_energy)
        
        # Enhanced debugging for EF calculation
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"EF DEBUG: LE range [{np.nanmin(LE):.2f}, {np.nanmax(LE):.2f}], mean={np.nanmean(LE):.2f}")
        logger.debug(f"EF DEBUG: available_energy range [{np.nanmin(available_energy):.2f}, {np.nanmax(available_energy):.2f}], mean={np.nanmean(available_energy):.2f}")
        logger.debug(f"EF DEBUG: H range [{np.nanmin(H):.2f}, {np.nanmax(H):.2f}], mean={np.nanmean(H):.2f}")
        logger.debug(f"EF DEBUG: raw EF range [{np.nanmin(EF):.4f}, {np.nanmax(EF):.4f}], mean={np.nanmean(EF):.4f}")
        
        # Apply energy balance constraints
        if apply_constraints:
            LE = self.apply_energy_balance_constraints(LE, rn, G, H)
            # Recalculate EF after constraints
            EF = self.calculate_ef(LE, available_energy)
        
        # Calculate instantaneous ET
        ET_inst = self.calculate_et_instantaneous(LE)
        
        # Apply physical bounds to EF
        EF = np.clip(EF, self.config.min_ef, self.config.max_ef)
        
        # Optional surface temperature analysis for stress indicators
        stress_indicator = None
        if ts_kelvin is not None:
            ts_kelvin = to_numpy(ts_kelvin)
            if ts_kelvin.shape == rn.shape:
                # Calculate temperature-based stress indicator
                # Higher temperatures with low EF may indicate water stress
                stress_indicator = self._calculate_temperature_stress_indicator(LE, available_energy, ts_kelvin)
        
        result = {
            'LE': LE,
            'available_energy': available_energy,
            'EF': EF,
            'ET_inst': ET_inst
        }
        
        if stress_indicator is not None:
            result['stress_indicator'] = stress_indicator
            
        return result
    
    def calculate_full(
        self,
        rn: np.ndarray,
        G: np.ndarray,
        H: np.ndarray,
        ts_kelvin: Optional[np.ndarray] = None,
        eto: Optional[np.ndarray] = None,
        etr: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate latent heat flux with full quality checks and ET fractions.
        
        Args:
            rn: Net radiation array (W/m²)
            G: Soil heat flux array (W/m²)
            H: Sensible heat flux array (W/m²)
            ts_kelvin: Surface temperature in Kelvin (optional)
            eto: Reference ET for grass (mm/hr, optional)
            etr: Reference ET for alfalfa (mm/hr, optional)
            
        Returns:
            Dictionary containing all LE-related parameters and quality metrics
        """
        # Basic calculation
        result = self.calculate(rn, G, H, ts_kelvin)
        
        # Quality check
        quality = self.quality_check(result['LE'], rn, G, H)
        result.update(quality)
        
        # Calculate ET fractions if reference ET available
        if eto is not None or etr is not None:
            et_fractions = self.calculate_et_fraction(
                result['LE'],
                eto if eto is not None else np.zeros_like(result['LE']),
                etr if etr is not None else np.zeros_like(result['LE'])
            )
            result.update(et_fractions)
        
        return result
    
    def compute(self, cube):
        """
        Compute latent heat flux and add to DataCube.

        Args:
            cube: DataCube with Rn, G, H

        Returns:
            DataCube with added LE, EF, ET_inst
        """
        from ..core.datacube import DataCube

        # Get required inputs
        rn = cube.get("R_n")
        G = cube.get("G")
        H = cube.get("H")

        if rn is None:
            raise ValueError("Net radiation (R_n) not found in DataCube")
        if G is None:
            raise ValueError("Soil heat flux (G) not found in DataCube")
        if H is None:
            raise ValueError("Sensible heat flux (H) not found in DataCube")

        # Calculate latent heat flux
        result = self.calculate(rn.values, G.values, H.values)

        # Add to cube
        cube.add("LE", result["LE"])
        cube.add("EF", result["EF"])
        cube.add("ET_inst", result["ET_inst"])

        return cube

    def __call__(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Convenience method to calculate latent heat flux.
        """
        return self.calculate(*args, **kwargs)
    
    def _calculate_temperature_stress_indicator(
        self,
        le: np.ndarray,
        available_energy: np.ndarray,
        ts_kelvin: np.ndarray
    ) -> np.ndarray:
        """
        Calculate temperature-based stress indicator.
        
        High surface temperature combined with low evaporative fraction
        may indicate water stress conditions.
        
        Args:
            le: Latent heat flux (W/m²)
            available_energy: Available energy (Rn - G) (W/m²)
            ts_kelvin: Surface temperature (K)
            
        Returns:
            Stress indicator (0.0 to 1.0, higher values indicate more stress)
        """
        # Calculate evaporative fraction
        ef = self.calculate_ef(le, available_energy)
        
        # Normalize temperature (assume 280-320K range for typical conditions)
        # Higher temperatures get higher stress values
        temp_stress = np.clip((ts_kelvin - 290.0) / 30.0, 0.0, 1.0)
        
        # Combine temperature stress with low evaporative fraction
        # Low EF with high temperature indicates stress
        stress_indicator = temp_stress * (1.0 - np.clip(ef, 0.0, 1.0))
        
        return stress_indicator


def create_latent_heat_flux(
    allow_negative: bool = False,
    min_ef: float = 0.0,
    max_ef: float = 2.0,
    **kwargs
) -> LatentHeatFlux:
    """
    Factory function to create LatentHeatFlux instance with enhanced configuration.
    
    Args:
        allow_negative: Whether to allow negative LE (advection conditions)
        min_ef: Minimum evaporative fraction (default: 0.0)
        max_ef: Maximum evaporative fraction (default: 2.0, allows for advection conditions)
        **kwargs: Additional configuration parameters including:
            - min_le: Minimum LE value (W/m²) for negative LE cases
            - max_le: Maximum LE value (W/m²)
            - min_available_energy: Minimum available energy for EF calculation
            - quality_threshold: Quality flag threshold
            
    Returns:
        Configured LatentHeatFlux instance
        
    Examples:
        >>> # Standard METRIC daytime conditions
        >>> le_calculator = create_latent_heat_flux()
        
        >>> # Allow negative LE for advection studies
        >>> le_calculator = create_latent_heat_flux(allow_negative=True, min_le=-200.0)
        
        >>> # Flexible ET fraction scaling for stressed conditions
        >>> le_calculator = create_latent_heat_flux(max_ef=1.5)
    """
    config = LatentHeatFluxConfig(
        allow_negative_le=allow_negative,
        min_ef=min_ef,
        max_ef=max_ef,
        **kwargs
    )
    return LatentHeatFlux(config)
