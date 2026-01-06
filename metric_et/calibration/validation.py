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
    # Correct physics: ET (mm/hr) = LE (W/m²) * 3600 (s/hr) / λ (J/kg)
    # Where λ = 2.45e6 J/kg (latent heat of vaporization)
    # Note: Water density cancels out since 1 kg/m² = 1 mm water depth
    # Simplified: ET (mm/hr) = LE (W/m²) * 3600 / 2.45e6
    LE_TO_ET_CONVERSION = 3600.0 / 2.45e6  # = 0.001469
    
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


@dataclass
class GlobalPostCalibrationValidation:
    """Result container for global post-calibration validation."""
    valid: bool  # Whether all constraints are satisfied
    ef_min: float  # Minimum EF value
    ef_p95: float  # 95th percentile of EF values
    ef_mean: float  # Mean EF value
    h_negative_fraction: float  # Fraction of H < 0 pixels
    h_extreme_negative_fraction: float  # Fraction of H < -20 W/m² pixels
    etrf_p95: float  # 95th percentile of ETrF values
    etrf_mean: float  # Mean ETrF value
    violations: List[str]  # List of constraint violations
    warnings: List[str]  # List of warnings


class GlobalPostCalibrationValidator:
    """
    Perform global post-calibration validation for METRIC model.
    
    This validator checks global constraints on calibrated scene data to ensure
    physically realistic energy balance and evapotranspiration estimates.
    
    Constraints validated:
    1. EF distribution sanity:
       - EF_min ≥ 0.15
       - EF_p95 ≤ 0.95
       - mean(EF) ≤ 0.85
    2. Sensible heat realism:
       - fraction(H < 0) ≤ 0.01
       - fraction(H < −20 W/m²) == 0
    3. Evapotranspiration ratio (ETrF):
       - P95(ETrF) ≤ 1.15
       - mean(ETrF) ≤ 1.05
    
    If any constraint is violated, the calibration should be rejected.
    """
    
    # Global post-calibration validation thresholds
    EF_MIN_THRESHOLD = 0.15
    EF_P95_THRESHOLD = 0.95
    EF_MEAN_THRESHOLD = 0.85
    
    H_NEGATIVE_FRACTION_THRESHOLD = 0.01
    H_EXTREME_NEGATIVE_THRESHOLD = 0.0  # Must be exactly 0
    
    ETRF_P95_THRESHOLD = 1.15
    ETRF_MEAN_THRESHOLD = 1.05
    
    def __init__(
        self,
        ef_min_threshold: float = EF_MIN_THRESHOLD,
        ef_p95_threshold: float = EF_P95_THRESHOLD,
        ef_mean_threshold: float = EF_MEAN_THRESHOLD,
        h_negative_fraction_threshold: float = H_NEGATIVE_FRACTION_THRESHOLD,
        h_extreme_negative_threshold: float = H_EXTREME_NEGATIVE_THRESHOLD,
        etrf_p95_threshold: float = ETRF_P95_THRESHOLD,
        etrf_mean_threshold: float = ETRF_MEAN_THRESHOLD
    ):
        """
        Initialize the GlobalPostCalibrationValidator.
        
        Args:
            ef_min_threshold: Minimum acceptable EF value
            ef_p95_threshold: Maximum acceptable 95th percentile EF value
            ef_mean_threshold: Maximum acceptable mean EF value
            h_negative_fraction_threshold: Maximum acceptable fraction of H < 0 pixels
            h_extreme_negative_threshold: Required fraction of H < -20 W/m² pixels (must be 0)
            etrf_p95_threshold: Maximum acceptable 95th percentile ETrF value
            etrf_mean_threshold: Maximum acceptable mean ETrF value
        """
        self.ef_min_threshold = ef_min_threshold
        self.ef_p95_threshold = ef_p95_threshold
        self.ef_mean_threshold = ef_mean_threshold
        self.h_negative_fraction_threshold = h_negative_fraction_threshold
        self.h_extreme_negative_threshold = h_extreme_negative_threshold
        self.etrf_p95_threshold = etrf_p95_threshold
        self.etrf_mean_threshold = etrf_mean_threshold
    
    def validate_global_calibration(
        self,
        ef: xr.DataArray,
        h: xr.DataArray,
        etrf: xr.DataArray,
        exclude_no_data: bool = True
    ) -> GlobalPostCalibrationValidation:
        """
        Perform global post-calibration validation on calibrated scene data.
        
        This method validates that the calibrated scene data meets physical
        realism constraints. If any constraint is violated, the validation fails.
        
        Args:
            ef: Evaporative fraction array (dimensionless)
            h: Sensible heat flux array (W/m²)
            etrf: Reference evapotranspiration fraction array (dimensionless)
            exclude_no_data: Whether to exclude NaN/invalid pixels from analysis
            
        Returns:
            GlobalPostCalibrationValidation with validation results and statistics
            
        Raises:
            ValueError: If required arrays are None or invalid
        """
        logger = logging.getLogger(__name__)
        
        # Validate inputs
        if ef is None:
            raise ValueError("EF array is required for global validation")
        if h is None:
            raise ValueError("H array is required for global validation")
        if etrf is None:
            raise ValueError("ETrF array is required for global validation")
        
        violations = []
        warnings = []
        
        # Create valid pixel mask if excluding no data
        if exclude_no_data:
            valid_mask = (
                ~np.isnan(ef.values) & 
                ~np.isnan(h.values) & 
                ~np.isnan(etrf.values) &
                (ef.values >= 0) & 
                (ef.values <= 1.0) &
                (etrf.values >= 0) &
                (etrf.values <= 2.0)  # ETrF should not exceed 2.0 in normal conditions
            )
        else:
            valid_mask = np.ones_like(ef.values, dtype=bool)
        
        # Extract valid values
        ef_valid = ef.values[valid_mask]
        h_valid = h.values[valid_mask]
        etrf_valid = etrf.values[valid_mask]
        
        if len(ef_valid) == 0:
            raise ValueError("No valid pixels found for global validation")
        
        logger.info(f"Global validation: {len(ef_valid)} valid pixels out of {ef.size}")
        
        # 1. EF distribution validation
        ef_min = float(np.nanmin(ef_valid))
        ef_p95 = float(np.nanpercentile(ef_valid, 95))
        ef_mean = float(np.nanmean(ef_valid))
        
        logger.debug(f"EF statistics: min={ef_min:.3f}, p95={ef_p95:.3f}, mean={ef_mean:.3f}")
        
        # Check EF minimum constraint
        if ef_min < self.ef_min_threshold:
            violations.append(
                f"EF minimum {ef_min:.3f} < {self.ef_min_threshold:.3f} threshold"
            )
        
        # Check EF 95th percentile constraint
        if ef_p95 > self.ef_p95_threshold:
            violations.append(
                f"EF 95th percentile {ef_p95:.3f} > {self.ef_p95_threshold:.3f} threshold"
            )
        
        # Check EF mean constraint
        if ef_mean > self.ef_mean_threshold:
            violations.append(
                f"EF mean {ef_mean:.3f} > {self.ef_mean_threshold:.3f} threshold"
            )
        
        # 2. Sensible heat flux validation
        h_negative_count = np.sum(h_valid < 0)
        h_negative_fraction = h_negative_count / len(h_valid)
        
        h_extreme_negative_count = np.sum(h_valid < -20.0)
        h_extreme_negative_fraction = h_extreme_negative_count / len(h_valid)
        
        logger.debug(
            f"H statistics: {h_negative_count} negative pixels ({h_negative_fraction:.1%}), "
            f"{h_extreme_negative_count} extreme negative pixels ({h_extreme_negative_fraction:.1%})"
        )
        
        # Check H negative fraction constraint
        if h_negative_fraction > self.h_negative_fraction_threshold:
            violations.append(
                f"H negative fraction {h_negative_fraction:.1%} > {self.h_negative_fraction_threshold:.1%} threshold"
            )
        
        # Check H extreme negative constraint (must be exactly 0)
        if h_extreme_negative_fraction > self.h_extreme_negative_threshold:
            violations.append(
                f"H extreme negative fraction {h_extreme_negative_fraction:.1%} > {self.h_extreme_negative_threshold:.1%} threshold (must be 0)"
            )
        
        # 3. ETrF validation
        etrf_p95 = float(np.nanpercentile(etrf_valid, 95))
        etrf_mean = float(np.nanmean(etrf_valid))
        
        logger.debug(f"ETrF statistics: p95={etrf_p95:.3f}, mean={etrf_mean:.3f}")
        
        # Check ETrF 95th percentile constraint
        if etrf_p95 > self.etrf_p95_threshold:
            violations.append(
                f"ETrF 95th percentile {etrf_p95:.3f} > {self.etrf_p95_threshold:.3f} threshold"
            )
        
        # Check ETrF mean constraint
        if etrf_mean > self.etrf_mean_threshold:
            violations.append(
                f"ETrF mean {etrf_mean:.3f} > {self.etrf_mean_threshold:.3f} threshold"
            )
        
        # Determine overall validity
        valid = len(violations) == 0
        
        # Log validation results
        if valid:
            logger.info("Global post-calibration validation: PASSED")
        else:
            logger.error("Global post-calibration validation: FAILED")
            for violation in violations:
                logger.error(f"  VIOLATION: {violation}")
        
        # Add warnings for concerning but not failing values
        if ef_p95 > 0.90 and ef_p95 <= self.ef_p95_threshold:
            warnings.append(f"EF 95th percentile {ef_p95:.3f} is high but within threshold")
        
        if etrf_p95 > 1.10 and etrf_p95 <= self.etrf_p95_threshold:
            warnings.append(f"ETrF 95th percentile {etrf_p95:.3f} is high but within threshold")
        
        if len(warnings) > 0:
            for warning in warnings:
                logger.warning(f"  WARNING: {warning}")
        
        return GlobalPostCalibrationValidation(
            valid=valid,
            ef_min=ef_min,
            ef_p95=ef_p95,
            ef_mean=ef_mean,
            h_negative_fraction=h_negative_fraction,
            h_extreme_negative_fraction=h_extreme_negative_fraction,
            etrf_p95=etrf_p95,
            etrf_mean=etrf_mean,
            violations=violations,
            warnings=warnings
        )
    
    def validate_or_raise(
        self,
        ef: xr.DataArray,
        h: xr.DataArray,
        etrf: xr.DataArray,
        exclude_no_data: bool = True
    ) -> GlobalPostCalibrationValidation:
        """
        Perform global validation and raise exception if constraints are violated.
        
        Args:
            ef: Evaporative fraction array (dimensionless)
            h: Sensible heat flux array (W/m²)
            etrf: Reference evapotranspiration fraction array (dimensionless)
            exclude_no_data: Whether to exclude NaN/invalid pixels from analysis
            
        Returns:
            GlobalPostCalibrationValidation if validation passes
            
        Raises:
            ValueError: If any validation constraint is violated
        """
        result = self.validate_global_calibration(ef, h, etrf, exclude_no_data)
        
        if not result.valid:
            violation_msg = "; ".join(result.violations)
            raise ValueError(
                f"Global post-calibration validation failed: {violation_msg}"
            )
        
        return result
    
    def to_dict(self, result: GlobalPostCalibrationValidation) -> Dict[str, Any]:
        """
        Convert validation result to dictionary.
        
        Args:
            result: GlobalPostCalibrationValidation from validate_global_calibration()
            
        Returns:
            Dictionary representation of validation results
        """
        return {
            "valid": result.valid,
            "ef_min": float(result.ef_min),
            "ef_p95": float(result.ef_p95),
            "ef_mean": float(result.ef_mean),
            "h_negative_fraction": float(result.h_negative_fraction),
            "h_extreme_negative_fraction": float(result.h_extreme_negative_fraction),
            "etrf_p95": float(result.etrf_p95),
            "etrf_mean": float(result.etrf_mean),
            "violations": result.violations,
            "warnings": result.warnings
        }
    
    def __repr__(self) -> str:
        return (
            f"GlobalPostCalibrationValidator("
            f"EF_thresholds=[min≥{self.ef_min_threshold:.2f}, p95≤{self.ef_p95_threshold:.2f}, mean≤{self.ef_mean_threshold:.2f}], "
            f"H_constraints=[neg_frac≤{self.h_negative_fraction_threshold:.0%}, extreme_neg={self.h_extreme_negative_threshold:.0%}], "
            f"ETrF_thresholds=[p95≤{self.etrf_p95_threshold:.2f}, mean≤{self.etrf_mean_threshold:.2f}])"
        )
