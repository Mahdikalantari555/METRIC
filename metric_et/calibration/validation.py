"""
Energy Balance Validation for METRIC Calibration.

This module provides validation of energy balance closure and
anchor pixel quality checks for the METRIC model.
"""

from typing import Dict, Optional, List, Tuple, Any
import numpy as np
import xarray as xr
import logging
from dataclasses import dataclass


@dataclass
class EnergyBalanceResult:
    """Result container for energy balance validation."""
    residual: float  # Rn - G - H - LE (W/m²)
    fractional_residual: float  # residual / (Rn - G) (dimensionless)
    closure_ratio: float  # (H + LE) / (Rn - G) (dimensionless)
    cold_pixel_valid: bool
    hot_pixel_valid: bool
    cold_pixel_issues: List[str]
    hot_pixel_issues: List[str]
    et_inst_cold: float  # Instantaneous ET at cold pixel (mm/hr)
    et_inst_hot: float  # Instantaneous ET at hot pixel (mm/hr)


@dataclass
class AnchorPixelValidation:
    """Validation result for a single anchor pixel."""
    valid: bool
    issues: List[str]
    le_fraction: float  # LE / (Rn - G)
    h_fraction: float  # H / (Rn - G)
    residual: float  # Rn - G - H - LE
    et_inst: float  # Instantaneous ET (mm/hr)


class EnergyBalanceValidator:
    """
    Validate energy balance closure and anchor pixel quality.
    
    This class performs quality checks on the METRIC energy balance
    components and anchor pixel selection:
    
    1. Energy balance closure: Rn - G - H - LE ≈ 0
    2. Anchor pixel LE and H fractions
    3. Reference ET validation
    
    Attributes:
        closure_tolerance: Acceptable range for fractional residual
        le_cold_min: Minimum expected LE fraction at cold pixel
        le_cold_max: Maximum expected LE fraction at cold pixel
        h_hot_min: Minimum expected H fraction at hot pixel
        h_hot_max: Maximum expected H fraction at hot pixel
    """
    
    # Energy balance closure tolerance (fractional residual)
    CLOSURE_TOLERANCE = 0.2  # ±20% residual acceptable
    
    # Expected energy balance fractions at anchor pixels
    LE_COLD_MIN = 0.80  # Cold pixel: LE should be 80-100% of Rn-G
    LE_COLD_MAX = 1.00
    H_COLD_MAX = 0.20   # Cold pixel: H should be <20% of Rn-G
    
    LE_HOT_MIN = 0.05   # Hot pixel: LE should be 5-20% of Rn-G
    LE_HOT_MAX = 0.20
    H_HOT_MIN = 0.50    # Hot pixel: H should be 50-80% of Rn-G
    H_HOT_MAX = 0.80
    
    # Latent heat to ET conversion (W/m² to mm/hr)
    # ET (mm/hr) = LE (W/m²) / (2.45e6 J/kg * 1000 kg/m³) * 3600 s/hr * 1000 mm/m
    # Simplified: ET (mm/hr) = LE (W/m²) / 2452.7
    LE_TO_ET_CONVERSION = 1 / 2452.7
    
    def __init__(
        self,
        closure_tolerance: float = CLOSURE_TOLERANCE,
        le_cold_range: Tuple[float, float] = (LE_COLD_MIN, LE_COLD_MAX),
        h_cold_max: float = H_COLD_MAX,
        le_hot_range: Tuple[float, float] = (LE_HOT_MIN, LE_HOT_MAX),
        h_hot_range: Tuple[float, float] = (H_HOT_MIN, H_HOT_MAX)
    ):
        """
        Initialize the EnergyBalanceValidator.
        
        Args:
            closure_tolerance: Maximum acceptable fractional residual
            le_cold_range: Expected LE fraction range at cold pixel (min, max)
            h_cold_max: Maximum H fraction at cold pixel
            le_hot_range: Expected LE fraction range at hot pixel (min, max)
            h_hot_range: Expected H fraction range at hot pixel (min, max)
        """
        self.closure_tolerance = closure_tolerance
        self.le_cold_min, self.le_cold_max = le_cold_range
        self.h_cold_max = h_cold_max
        self.le_hot_min, self.le_hot_max = le_hot_range
        self.h_hot_min, self.h_hot_max = h_hot_range
    
    def compute_available_energy(
        self,
        rn: xr.DataArray,
        g: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute available energy (Rn - G).
        
        Args:
            rn: Net radiation (W/m²)
            g: Soil heat flux (W/m²)
            
        Returns:
            Available energy (W/m²)
        """
        return rn - g
    
    def compute_residual(
        self,
        rn: xr.DataArray,
        g: xr.DataArray,
        h: xr.DataArray,
        le: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute energy balance residual.
        
        Args:
            rn: Net radiation (W/m²)
            g: Soil heat flux (W/m²)
            h: Sensible heat flux (W/m²)
            le: Latent heat flux (W/m²)
            
        Returns:
            Residual (W/m²)
        """
        return rn - g - h - le
    
    def compute_fractional_residual(
        self,
        rn: xr.DataArray,
        g: xr.DataArray,
        h: xr.DataArray,
        le: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute fractional residual (residual / (Rn - G)).
        
        Args:
            rn: Net radiation (W/m²)
            g: Soil heat flux (W/m²)
            h: Sensible heat flux (W/m²)
            le: Latent heat flux (W/m²)
            
        Returns:
            Fractional residual (dimensionless)
        """
        available_energy = rn - g
        residual = rn - g - h - le
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_residual = np.where(
                available_energy > 1.0,
                residual / available_energy,
                np.nan
            )
        
        return xr.DataArray(
            frac_residual,
            dims=rn.dims,
            attrs={
                'units': 'dimensionless',
                'description': 'Fractional energy balance residual'
            }
        )
    
    def validate_anchor_pixel(
        self,
        pixel_type: str,
        le: float,
        h: float,
        rn: float,
        g: float,
        dT: float,
        etr_inst: float,
        etrf: float
    ) -> AnchorPixelValidation:
        """
        Validate an anchor pixel based on energy balance criteria.
        
        Args:
            pixel_type: 'cold' or 'hot'
            le: Latent heat flux (W/m²)
            h: Sensible heat flux (W/m²)
            rn: Net radiation (W/m²)
            g: Soil heat flux (W/m²)
            dT: Temperature difference Ts - Ta (K)
            etr_inst: Reference ET at overpass (mm/hr)
            etrf: Fraction of reference ET
            
        Returns:
            AnchorPixelValidation with validation results
        """
        issues = []
        available_energy = rn - g
        
        if available_energy <= 0:
            return AnchorPixelValidation(
                valid=False,
                issues=["Available energy (Rn - G) is not positive"],
                le_fraction=0.0,
                h_fraction=0.0,
                residual=0.0,
                et_inst=0.0
            )
        
        le_fraction = le / available_energy
        h_fraction = h / available_energy
        residual = rn - g - h - le
        
        # Compute instantaneous ET
        et_inst = le * self.LE_TO_ET_CONVERSION

        # METRIC Cold Pixel ET Validation
        if pixel_type == 'cold':
            expected_et_cold = etr_inst * etrf  # Should be ≈ 1.05 × ETr_inst
            et_cold_valid = abs(et_inst - expected_et_cold) / expected_et_cold < 0.1  # Within 10%
            logger = logging.getLogger(__name__)
            logger.info(f"Cold pixel ET validation: ET_inst={et_inst:.3f} mm/hr, Expected={expected_et_cold:.3f} mm/hr (ETr_inst={etr_inst:.3f} × ETrF={etrf:.3f}) - {'PASS' if et_cold_valid else 'FAIL'}")
        
        if pixel_type == 'cold':
            # Validate cold pixel
            if not (self.le_cold_min <= le_fraction <= self.le_cold_max):
                issues.append(
                    f"LE fraction {le_fraction:.2%} outside expected range "
                    f"[{self.le_cold_min:.0%}, {self.le_cold_max:.0%}]"
                )
            
            if h_fraction > self.h_cold_max:
                issues.append(
                    f"H fraction {h_fraction:.2%} exceeds maximum "
                    f"({self.h_cold_max:.0%})"
                )
            
            # dT should be small for cold pixel
            if abs(dT) > 5.0:
                issues.append(
                    f"|dT| = {abs(dT):.2f} K is larger than expected for cold pixel"
                )
        
        elif pixel_type == 'hot':
            # Validate hot pixel
            if not (self.le_hot_min <= le_fraction <= self.le_hot_max):
                issues.append(
                    f"LE fraction {le_fraction:.2%} outside expected range "
                    f"[{self.le_hot_min:.0%}, {self.le_hot_max:.0%}]"
                )
            
            if not (self.h_hot_min <= h_fraction <= self.h_hot_max):
                issues.append(
                    f"H fraction {h_fraction:.2%} outside expected range "
                    f"[{self.h_hot_min:.0%}, {self.h_hot_max:.0%}]"
                )
            
            # dT should be large for hot pixel
            if dT < 5.0:
                issues.append(
                    f"dT = {dT:.2f} K is smaller than expected for hot pixel"
                )
        
        valid = len(issues) == 0
        
        return AnchorPixelValidation(
            valid=valid,
            issues=issues,
            le_fraction=le_fraction,
            h_fraction=h_fraction,
            residual=residual,
            et_inst=et_inst
        )
    
    def validate_energy_balance(
        self,
        rn: xr.DataArray,
        g: xr.DataArray,
        h: xr.DataArray,
        le: xr.DataArray,
        location: Tuple[int, int],
        pixel_type: str
    ) -> Tuple[float, float]:
        """
        Validate energy balance at a specific pixel location.
        
        Args:
            rn: Net radiation (W/m²)
            g: Soil heat flux (W/m²)
            h: Sensible heat flux (W/m²)
            le: Latent heat flux (W/m²)
            location: (row, col) pixel location
            pixel_type: 'cold' or 'hot'
            
        Returns:
            Tuple of (residual, fractional_residual)
        """
        row, col = location
        
        rn_val = rn.values[row, col] if isinstance(rn, xr.DataArray) else rn[row, col]
        g_val = g.values[row, col] if isinstance(g, xr.DataArray) else g[row, col]
        h_val = h.values[row, col] if isinstance(h, xr.DataArray) else h[row, col]
        le_val = le.values[row, col] if isinstance(le, xr.DataArray) else le[row, col]
        
        available = rn_val - g_val
        residual = rn_val - g_val - h_val - le_val
        
        if available > 0:
            frac_residual = residual / available
        else:
            frac_residual = np.nan
        
        return residual, frac_residual
    
    def validate_full(
        self,
        cold_pixel_data: Dict[str, float],
        hot_pixel_data: Dict[str, float],
        etr_inst: float
    ) -> EnergyBalanceResult:
        """
        Perform full validation of anchor pixels and energy balance.
        
        Args:
            cold_pixel_data: Dictionary with cold pixel values
            hot_pixel_data: Dictionary with hot pixel values
            etr_inst: Reference ET at overpass (mm/hr)
            
        Returns:
            EnergyBalanceResult with all validation metrics
        """
        cold_issues = []
        hot_issues = []
        
        # Validate cold pixel
        cold_validation = self.validate_anchor_pixel(
            pixel_type='cold',
            le=cold_pixel_data['LE'],
            h=cold_pixel_data['H'],
            rn=cold_pixel_data['Rn'],
            g=cold_pixel_data['G'],
            dT=cold_pixel_data['dT'],
            etr_inst=etr_inst,
            etrf=cold_pixel_data['ETrF']
        )
        cold_issues.extend(cold_validation.issues)
        
        # Validate hot pixel
        hot_validation = self.validate_anchor_pixel(
            pixel_type='hot',
            le=hot_pixel_data['LE'],
            h=hot_pixel_data['H'],
            rn=hot_pixel_data['Rn'],
            g=hot_pixel_data['G'],
            dT=hot_pixel_data['dT'],
            etr_inst=etr_inst,
            etrf=hot_pixel_data['ETrF']
        )
        hot_issues.extend(hot_validation.issues)
        
        # Compute overall energy balance closure
        cold_residual = cold_validation.residual
        hot_residual = hot_validation.residual
        
        # Closure ratio = (H + LE) / (Rn - G)
        cold_ae = cold_pixel_data['Rn'] - cold_pixel_data['G']
        hot_ae = hot_pixel_data['Rn'] - hot_pixel_data['G']
        
        cold_closure = (
            (cold_pixel_data['H'] + cold_pixel_data['LE']) / cold_ae
            if cold_ae > 0 else np.nan
        )
        hot_closure = (
            (hot_pixel_data['H'] + hot_pixel_data['LE']) / hot_ae
            if hot_ae > 0 else np.nan
        )
        
        # Average closure for both pixels
        closure_ratio = np.nanmean([cold_closure, hot_closure])
        
        # Fractional residual (negative of closure - 1)
        fractional_residual = 1.0 - closure_ratio
        
        return EnergyBalanceResult(
            residual=np.nanmean([cold_residual, hot_residual]),
            fractional_residual=fractional_residual,
            closure_ratio=closure_ratio,
            cold_pixel_valid=cold_validation.valid,
            hot_pixel_valid=hot_validation.valid,
            cold_pixel_issues=cold_issues,
            hot_pixel_issues=hot_issues,
            et_inst_cold=cold_validation.et_inst,
            et_inst_hot=hot_validation.et_inst
        )
    
    def to_dict(self, result: EnergyBalanceResult) -> Dict[str, Any]:
        """
        Convert validation result to dictionary.
        
        Args:
            result: EnergyBalanceResult from validate_full()
            
        Returns:
            Dictionary representation of validation results
        """
        return {
            "residual": float(result.residual) if not np.isnan(result.residual) else None,
            "fractional_residual": (
                float(result.fractional_residual) if not np.isnan(result.fractional_residual) else None
            ),
            "closure_ratio": float(result.closure_ratio) if not np.isnan(result.closure_ratio) else None,
            "cold_pixel_valid": result.cold_pixel_valid,
            "hot_pixel_valid": result.hot_pixel_valid,
            "cold_pixel_issues": result.cold_pixel_issues,
            "hot_pixel_issues": result.hot_pixel_issues,
            "ET_inst_cold": float(result.et_inst_cold),
            "ET_inst_hot": float(result.et_inst_hot)
        }
    
    def __repr__(self) -> str:
        return (
            f"EnergyBalanceValidator(closure_tolerance={self.closure_tolerance:.0%}, "
            f"LE_cold_range=[{self.le_cold_min:.0%}, {self.le_cold_max:.0%}], "
            f"H_hot_range=[{self.h_hot_min:.0%}, {self.h_hot_max:.0%}])"
        )
