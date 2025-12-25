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
    max_ef: float = 1.2
    
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
        Apply energy balance constraints to LE.
        
        Physical constraints:
        1. LE cannot exceed available energy (energy conservation)
        2. LE should be non-negative (no condensation in METRIC)
        3. LE cannot be more negative than available energy
        
        Args:
            le: Latent heat flux (W/m²)
            rn: Net radiation (W/m²)
            G: Soil heat flux (W/m²)
            H: Sensible heat flux (W/m²)
            
        Returns:
            Constrained latent heat flux (W/m²)
        """
        available_energy = rn - G
        
        if self.config.allow_negative_le:
            # Allow negative LE (advection conditions)
            # LE should be >= available energy (H <= 0)
            # and LE <= available energy (H >= 0)
            # This means LE can be negative only when H is positive
            # and magnitude cannot exceed available energy
            le = np.clip(le, available_energy, np.abs(available_energy) * 0.99)
        else:
            # Standard METRIC: LE >= 0
            # When available energy > 0, LE is bounded between 0 and available_energy
            # When available energy <= 0, LE = 0
            le = np.where(
                available_energy > 0,
                np.clip(le, 0.0, available_energy),
                0.0
            )
        
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
        # ET (mm/s) = LE (W/m²) / λ (J/kg) / 1000 (kg/m³ for mm conversion)
        # ET (mm/hr) = LE / λ / 1000 * 3600
        #            = LE * 3.6 / λ
        
        et_mm_hr = le * 3.6 / lambda_vaporization
        
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
        
        # Clip to reasonable range
        etrf = np.clip(etrf, 0.0, 2.0)
        etof = np.clip(etof, 0.0, 2.0)
        
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
        # LE should be between 0 and available_energy (when AE > 0)
        quality = np.where(
            (available_energy > 0) & ((le < 0) | (le > available_energy)),
            0.0,
            quality
        )
        
        # Check for large residuals
        quality = np.where(
            np.abs(rel_residual) > 0.1,
            1.0 - np.abs(rel_residual) * 5.0,
            quality
        )
        
        # Clip quality
        quality = np.clip(quality, 0.0, 1.0)
        
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
        Calculate latent heat flux and related parameters.
        
        Args:
            rn: Net radiation array (W/m²)
            G: Soil heat flux array (W/m²)
            H: Sensible heat flux array (W/m²)
            ts_kelvin: Surface temperature in Kelvin (optional, for quality check)
            apply_constraints: Whether to apply energy balance constraints
            
        Returns:
            Dictionary containing:
                - 'LE': Latent heat flux (W/m²)
                - 'available_energy': Rn - G (W/m²)
                - 'EF': Evaporative fraction
                - 'ET_inst': Instantaneous ET rate (mm/hr)
        """
        # Ensure arrays have same dtype
        def to_numpy(arr):
            if hasattr(arr, 'values'):
                return np.asarray(arr.values, dtype=np.float64)
            else:
                return np.asarray(arr, dtype=np.float64)

        rn = to_numpy(rn)
        G = to_numpy(G)
        H = to_numpy(H)
        
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
        
        return {
            'LE': LE,
            'available_energy': available_energy,
            'EF': EF,
            'ET_inst': ET_inst
        }
    
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


def create_latent_heat_flux(
    allow_negative: bool = False,
    min_ef: float = 0.0,
    max_ef: float = 1.2,
    **kwargs
) -> LatentHeatFlux:
    """
    Factory function to create LatentHeatFlux instance.
    
    Args:
        allow_negative: Whether to allow negative LE (advection)
        min_ef: Minimum evaporative fraction
        max_ef: Maximum evaporative fraction
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured LatentHeatFlux instance
    """
    config = LatentHeatFluxConfig(
        allow_negative_le=allow_negative,
        min_ef=min_ef,
        max_ef=max_ef,
        **kwargs
    )
    return LatentHeatFlux(config)
