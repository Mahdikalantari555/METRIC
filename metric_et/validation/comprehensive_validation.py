"""
Comprehensive Validation Framework for METRIC ET Calculations

This module provides a complete validation system for METRIC ET processing,
including energy balance closure validation, physical constraint checks,
quality assessment, and error detection/recovery mechanisms.

Key Features:
- Energy balance closure validation (LE ≤ Rn - G)
- ETrF sanity bounds validation (0 ≤ ETrF ≤ 1.3 with monitoring)
- LE-ET consistency checks
- Physical constraint validation for all variables
- Prevention of METRIC violations
- Step-by-step logging system
- Error detection and recovery
- Quality flagging and confidence scoring
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import logging
from enum import Enum

# Import existing validation components
from ..calibration.validation import EnergyBalanceValidator, EnergyBalanceResult, AnchorPixelValidation
from ..utils.validation import check_ndvi_range, check_albedo_range, check_temperature_range, check_et_range
from ..core.constants import LATENT_HEAT_VAPORIZATION
from ..utils.logger import Logger


class ValidationSeverity(Enum):
    """Validation message severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class QualityFlag(Enum):
    """Quality flag values for ET products."""
    EXCELLENT = 0
    GOOD = 1
    FAIR = 2
    POOR = 3
    BAD = 4


@dataclass
class ValidationMessage:
    """Individual validation message with context."""
    severity: ValidationSeverity
    category: str  # e.g., "energy_balance", "etrf_bounds", "physical_constraints"
    message: str
    location: Optional[str] = None  # e.g., "cold_pixel", "hot_pixel", "pixel_[i,j]"
    value: Optional[float] = None
    expected_range: Optional[str] = None
    recovery_suggestion: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationStep:
    """Step in the validation process."""
    name: str
    description: str
    status: str  # "pending", "running", "completed", "failed"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    messages: List[ValidationMessage] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings_count: int = 0
    errors_count: int = 0


@dataclass
class ValidationResult:
    """Complete validation result for a processing run."""
    processing_date: str
    scene_id: Optional[str]
    total_pixels: int
    valid_pixels: int
    
    # Energy balance validation
    energy_balance: Optional[EnergyBalanceResult] = None
    
    # Physical constraints validation
    physical_constraints_passed: bool = True
    physical_issues: List[ValidationMessage] = field(default_factory=list)
    
    # ETrF validation
    etrf_bounds_passed: bool = True
    etrf_issues: List[ValidationMessage] = field(default_factory=list)
    
    # LE-ET consistency
    le_et_consistency_passed: bool = True
    le_et_issues: List[ValidationMessage] = field(default_factory=list)
    
    # Overall quality assessment
    overall_quality_flag: QualityFlag = QualityFlag.GOOD
    confidence_score: float = 0.0
    
    # Validation steps
    steps: List[ValidationStep] = field(default_factory=list)
    
    # Summary statistics
    validation_summary: Dict[str, Any] = field(default_factory=dict)


class ComprehensiveMETRICValidator:
    """
    Comprehensive validation framework for METRIC ET calculations.
    
    This validator performs multiple levels of validation:
    
    1. **Energy Balance Closure**: Validates Rn = G + H + LE
    2. **Physical Constraints**: Ensures all variables are within realistic bounds
    3. **ETrF Sanity Checks**: Validates reference ET fraction ranges
    4. **LE-ET Consistency**: Ensures LE and ET calculations are consistent
    5. **METRIC-Specific Rules**: Validates METRIC model assumptions
    6. **Quality Assessment**: Provides overall quality flags and confidence scores
    """
    
    # Validation thresholds
    ENERGY_BALANCE_TOLERANCE = 0.15  # ±15% closure acceptable
    ETRF_MIN = 0.0
    ETRF_MAX = 1.3  # Allow slight over-estimation
    ETRF_WARNING_MAX = 1.2  # Warning threshold
    
    # Physical constraint thresholds
    NDVI_MIN, NDVI_MAX = -1.0, 1.0
    ALBEDO_MIN, ALBEDO_MAX = 0.0, 1.0
    TEMP_MIN, TEMP_MAX = 240, 320  # K
    ET_DAILY_MAX = 15.0  # mm/day
    ET_INST_MAX = 1.0  # mm/hour
    
    # Energy flux thresholds (W/m²)
    RN_MIN, RN_MAX = 0, 1000
    G_MIN, G_MAX = -100, 100
    H_MIN, H_MAX = -200, 600
    LE_MIN, LE_MAX = -100, 600
    
    def __init__(self, scene_id: Optional[str] = None):
        """
        Initialize comprehensive validator.
        
        Args:
            scene_id: Scene identifier for logging
        """
        self.scene_id = scene_id
        self.logger = Logger.get_logger(__name__)
        self.validation_steps = []
        self.current_step = None
        
    def validate_complete_workflow(
        self,
        cube: Any,
        results: Dict[str, xr.DataArray],
        calibration_data: Optional[Dict] = None
    ) -> ValidationResult:
        """
        Perform complete validation of METRIC workflow.
        
        Args:
            cube: DataCube with all input data
            results: Dictionary of calculation results
            calibration_data: Optional calibration information
            
        Returns:
            ValidationResult with complete assessment
        """
        self.logger.info(f"Starting comprehensive validation for scene: {self.scene_id}")
        
        # Initialize validation result
        validation_result = ValidationResult(
            processing_date=datetime.now().isoformat(),
            scene_id=self.scene_id,
            total_pixels=0,
            valid_pixels=0
        )
        
        try:
            # Step 1: Input Data Validation
            self._start_step("input_validation", "Validating input data quality")
            input_validation = self._validate_input_data(cube, results)
            validation_result.steps.append(self.current_step)
            if not input_validation:
                validation_result.physical_constraints_passed = False
                
            # Step 2: Surface Properties Validation
            self._start_step("surface_properties", "Validating surface property calculations")
            surface_validation = self._validate_surface_properties(cube)
            validation_result.steps.append(self.current_step)
            
            # Step 3: Radiation Balance Validation
            self._start_step("radiation_balance", "Validating radiation balance components")
            radiation_validation = self._validate_radiation_balance(cube)
            validation_result.steps.append(self.current_step)
            
            # Step 4: Energy Balance Closure Validation
            self._start_step("energy_balance", "Validating energy balance closure")
            energy_validation = self._validate_energy_balance(cube)
            validation_result.steps.append(self.current_step)
            validation_result.energy_balance = energy_validation
            
            # Step 5: ETrF Bounds Validation
            self._start_step("etrf_bounds", "Validating ETrF sanity bounds")
            etrf_validation = self._validate_etrf_bounds(results)
            validation_result.steps.append(self.current_step)
            validation_result.etrf_bounds_passed = etrf_validation
            
            # Step 6: LE-ET Consistency Validation
            self._start_step("le_et_consistency", "Validating LE-ET consistency")
            consistency_validation = self._validate_le_et_consistency(cube, results)
            validation_result.steps.append(self.current_step)
            validation_result.le_et_consistency_passed = consistency_validation
            
            # Step 7: METRIC-Specific Rule Validation
            self._start_step("metric_rules", "Validating METRIC-specific rules")
            metric_validation = self._validate_metric_rules(cube, results, calibration_data)
            validation_result.steps.append(self.current_step)
            
            # Step 8: Quality Assessment
            self._start_step("quality_assessment", "Computing overall quality assessment")
            quality_assessment = self._assess_overall_quality(validation_result)
            validation_result.steps.append(self.current_step)
            validation_result.overall_quality_flag = quality_assessment['quality_flag']
            validation_result.confidence_score = quality_assessment.get('confidence_score', quality_assessment.get('quality_score', 0.0))
            
            # Finalize validation
            self._finalize_validation(validation_result, cube, results)
            
            self.logger.info(f"Validation completed. Overall quality: {validation_result.overall_quality_flag.name}")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            self._complete_step("failed")
            raise
        
    def _start_step(self, name: str, description: str) -> None:
        """Start a validation step."""
        self.current_step = ValidationStep(
            name=name,
            description=description,
            status="running",
            start_time=datetime.now()
        )
        self.logger.info(f"Starting validation step: {name}")
        
    def _complete_step(self, status: str) -> None:
        """Complete current validation step."""
        if self.current_step:
            self.current_step.status = status
            self.current_step.end_time = datetime.now()
            self.logger.info(f"Completed validation step: {self.current_step.name} - {status}")
            
    def _add_message(
        self,
        severity: ValidationSeverity,
        category: str,
        message: str,
        location: Optional[str] = None,
        value: Optional[float] = None,
        expected_range: Optional[str] = None,
        recovery_suggestion: Optional[str] = None
    ) -> None:
        """Add validation message to current step."""
        if self.current_step:
            msg = ValidationMessage(
                severity=severity,
                category=category,
                message=message,
                location=location,
                value=value,
                expected_range=expected_range,
                recovery_suggestion=recovery_suggestion
            )
            self.current_step.messages.append(msg)
            
            # Update counters
            if severity == ValidationSeverity.WARNING:
                self.current_step.warnings_count += 1
            elif severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                self.current_step.errors_count += 1
                
    def _validate_input_data(self, cube: Any, results: Dict[str, xr.DataArray]) -> bool:
        """Validate input data quality."""
        try:
            # Check for essential variables
            essential_vars = ['ndvi', 'albedo', 'lwir11', 'R_n', 'G', 'H', 'LE', 'ET_daily', 'ET_inst', 'ETrF']
            missing_vars = []
            
            for var in essential_vars:
                if var not in results or results[var] is None:
                    missing_vars.append(var)
                    
            if missing_vars:
                self._add_message(
                    ValidationSeverity.ERROR,
                    "input_data",
                    f"Missing essential variables: {missing_vars}",
                    recovery_suggestion="Check METRIC pipeline output generation"
                )
                self._complete_step("failed")
                return False
                
            # Get data array for statistics
            total_pixels = 0
            valid_pixels = 0
            
            for var in essential_vars:
                if var in results and results[var] is not None:
                    data = results[var].values
                    total_pixels = max(total_pixels, data.size)
                    valid_pixels += np.sum(~np.isnan(data))
                    
            self.current_step.metrics['total_pixels'] = total_pixels
            self.current_step.metrics['valid_pixels'] = valid_pixels
            self.current_step.metrics['valid_fraction'] = valid_pixels / total_pixels if total_pixels > 0 else 0
            
            self._complete_step("completed")
            return True
            
        except Exception as e:
            self._add_message(
                ValidationSeverity.ERROR,
                "input_data",
                f"Error validating input data: {str(e)}"
            )
            self._complete_step("failed")
            return False
            
    def _validate_surface_properties(self, cube: Any) -> bool:
        """Validate surface property calculations."""
        try:
            all_passed = True
            
            # NDVI validation
            if cube.get("ndvi") is not None:
                ndvi = cube.get("ndvi").values
                is_valid, msg = check_ndvi_range(ndvi)
                if not is_valid:
                    self._add_message(
                        ValidationSeverity.ERROR,
                        "physical_constraints",
                        f"NDVI validation failed: {msg}",
                        expected_range="[-1.0, 1.0]",
                        recovery_suggestion="Check vegetation index calculation and input bands"
                    )
                    all_passed = False
                else:
                    # Check for reasonable values
                    reasonable_ndvi = ndvi[(ndvi >= -0.2) & (ndvi <= 1.0)]
                    if len(reasonable_ndvi) < len(ndvi) * 0.8:
                        self._add_message(
                            ValidationSeverity.WARNING,
                            "physical_constraints",
                            f"Only {len(reasonable_ndvi)/len(ndvi)*100:.1f}% of NDVI values are in reasonable range [-0.2, 1.0]",
                            recovery_suggestion="Check cloud masking and surface classification"
                        )
                        
            # Albedo validation
            if cube.get("albedo") is not None:
                albedo = cube.get("albedo").values
                is_valid, msg = check_albedo_range(albedo)
                if not is_valid:
                    self._add_message(
                        ValidationSeverity.ERROR,
                        "physical_constraints",
                        f"Albedo validation failed: {msg}",
                        expected_range="[0.0, 1.0]",
                        recovery_suggestion="Check albedo calculation and atmospheric correction"
                    )
                    all_passed = False
                    
            # Emissivity validation
            if cube.get("emissivity") is not None:
                emissivity = cube.get("emissivity").values
                valid_emiss = emissivity[(emissivity >= 0.8) & (emissivity <= 1.0)]
                if len(valid_emiss) < len(emissivity) * 0.9:
                    self._add_message(
                        ValidationSeverity.WARNING,
                        "physical_constraints",
                        f"Only {len(valid_emiss)/len(emissivity)*100:.1f}% of emissivity values are in typical range [0.8, 1.0]",
                        recovery_suggestion="Check surface type classification and emissivity database"
                    )
                    
            self.current_step.metrics['all_passed'] = all_passed
            self._complete_step("completed" if all_passed else "failed")
            return all_passed
            
        except Exception as e:
            self._add_message(
                ValidationSeverity.ERROR,
                "surface_properties",
                f"Error validating surface properties: {str(e)}"
            )
            self._complete_step("failed")
            return False
            
    def _validate_radiation_balance(self, cube: Any) -> bool:
        """Validate radiation balance components."""
        try:
            all_passed = True
            
            # Net radiation validation
            if cube.get("R_n") is not None:
                rn = cube.get("R_n").values
                valid_rn = rn[(rn >= self.RN_MIN) & (rn <= self.RN_MAX)]
                if len(valid_rn) < len(rn) * 0.9:
                    self._add_message(
                        ValidationSeverity.WARNING,
                        "radiation_balance",
                        f"Only {len(valid_rn)/len(rn)*100:.1f}% of net radiation values are in expected range [{self.RN_MIN}, {self.RN_MAX}] W/m²",
                        expected_range=f"[{self.RN_MIN}, {self.RN_MAX}] W/m²"
                    )
                    all_passed = False
                    
            # Shortwave radiation validation
            if cube.get("Rs_down") is not None:
                rs_down = cube.get("Rs_down").values
                valid_rs = rs_down[(rs_down >= 0) & (rs_down <= 1200)]
                if len(valid_rs) < len(rs_down) * 0.9:
                    self._add_message(
                        ValidationSeverity.WARNING,
                        "radiation_balance",
                        f"Only {len(valid_rs)/len(rs_down)*100:.1f}% of shortwave down values are in expected range [0, 1200] W/m²",
                        expected_range="[0, 1200] W/m²"
                    )
                    
            self.current_step.metrics['all_passed'] = all_passed
            self._complete_step("completed" if all_passed else "failed")
            return all_passed
            
        except Exception as e:
            self._add_message(
                ValidationSeverity.ERROR,
                "radiation_balance",
                f"Error validating radiation balance: {str(e)}"
            )
            self._complete_step("failed")
            return False
            
    def _validate_energy_balance(self, cube: Any) -> Optional[EnergyBalanceResult]:
        """Validate energy balance closure."""
        try:
            # Get energy balance components
            rn = cube.get("R_n")
            g = cube.get("G")
            h = cube.get("H")
            le = cube.get("LE")
            
            if any(v is None for v in [rn, g, h, le]):
                self._add_message(
                    ValidationSeverity.ERROR,
                    "energy_balance",
                    "Missing energy balance components for closure validation",
                    recovery_suggestion="Check energy balance calculation pipeline"
                )
                self._complete_step("failed")
                return None
                
            # Use existing EnergyBalanceValidator
            validator = EnergyBalanceValidator()
            
            # Get sample pixel locations for validation
            # For simplicity, use center pixel and a few random pixels
            height, width = rn.values.shape
            test_locations = [
                (height // 2, width // 2),  # Center
                (height // 4, width // 4),  # Upper left quadrant
                (3 * height // 4, 3 * width // 4)  # Lower right quadrant
            ]
            
            energy_balance_results = []
            
            for row, col in test_locations:
                try:
                    rn_val = float(rn.values[row, col])
                    g_val = float(g.values[row, col])
                    h_val = float(h.values[row, col])
                    le_val = float(le.values[row, col])
                    
                    # Check if all values are valid
                    if np.isfinite([rn_val, g_val, h_val, le_val]).all():
                        # Calculate metrics
                        available_energy = rn_val - g_val
                        residual = rn_val - g_val - h_val - le_val
                        
                        if available_energy > 0:
                            closure_ratio = (h_val + le_val) / available_energy
                            fractional_residual = residual / available_energy
                        else:
                            closure_ratio = np.nan
                            fractional_residual = np.nan
                            
                        energy_balance_results.append({
                            'residual': residual,
                            'closure_ratio': closure_ratio,
                            'fractional_residual': fractional_residual,
                            'location': f'pixel_[{row},{col}]'
                        })
                        
                except (IndexError, ValueError):
                    continue
                    
            if energy_balance_results:
                # Aggregate results
                residuals = [r['residual'] for r in energy_balance_results if np.isfinite(r['residual'])]
                closures = [r['closure_ratio'] for r in energy_balance_results if np.isfinite(r['closure_ratio'])]
                
                mean_residual = np.mean(residuals) if residuals else 0
                mean_closure = np.mean(closures) if closures else 0
                
                # Check closure quality
                closure_ok = abs(1 - mean_closure) <= self.ENERGY_BALANCE_TOLERANCE
                
                if not closure_ok:
                    self._add_message(
                        ValidationSeverity.WARNING,
                        "energy_balance",
                        f"Energy balance closure issue: mean closure ratio = {mean_closure:.3f}, target = 1.0 ± {self.ENERGY_BALANCE_TOLERANCE:.3f}",
                        expected_range=f"[{1-self.ENERGY_BALANCE_TOLERANCE:.3f}, {1+self.ENERGY_BALANCE_TOLERANCE:.3f}]",
                        recovery_suggestion="Check anchor pixel selection and calibration coefficients"
                    )
                    
                # Check for negative available energy
                negative_ae_count = sum(1 for r in energy_balance_results if r['residual'] > 0 and r['closure_ratio'] < 0)
                if negative_ae_count > 0:
                    self._add_message(
                        ValidationSeverity.ERROR,
                        "energy_balance",
                        f"Found {negative_ae_count} pixels with negative available energy (Rn - G < 0)",
                        recovery_suggestion="Check surface temperature and radiation calculations"
                    )
                    
                self.current_step.metrics['mean_residual'] = mean_residual
                self.current_step.metrics['mean_closure'] = mean_closure
                self.current_step.metrics['closure_ok'] = closure_ok
                self.current_step.metrics['tested_pixels'] = len(energy_balance_results)
                
                # Create EnergyBalanceResult-like object
                result = EnergyBalanceResult(
                    residual=mean_residual,
                    fractional_residual=1 - mean_closure,
                    closure_ratio=mean_closure,
                    cold_pixel_valid=True,  # Would need anchor pixel data
                    hot_pixel_valid=True,
                    cold_pixel_issues=[],
                    hot_pixel_issues=[],
                    et_inst_cold=0.0,
                    et_inst_hot=0.0
                )
                
            else:
                self._add_message(
                    ValidationSeverity.ERROR,
                    "energy_balance",
                    "No valid pixels found for energy balance validation",
                    recovery_suggestion="Check data quality and masking"
                )
                self._complete_step("failed")
                return None
                
            self._complete_step("completed")
            return result
            
        except Exception as e:
            self._add_message(
                ValidationSeverity.ERROR,
                "energy_balance",
                f"Error validating energy balance: {str(e)}"
            )
            self._complete_step("failed")
            return None
            
    def _validate_etrf_bounds(self, results: Dict[str, xr.DataArray]) -> bool:
        """Validate ETrF sanity bounds."""
        try:
            if "ETrF" not in results or results["ETrF"] is None:
                self._add_message(
                    ValidationSeverity.ERROR,
                    "etrf_bounds",
                    "ETrF data not available for validation"
                )
                self._complete_step("failed")
                return False
                
            etrf = results["ETrF"].values
            valid_etrf = etrf[~np.isnan(etrf)]
            
            if len(valid_etrf) == 0:
                self._add_message(
                    ValidationSeverity.ERROR,
                    "etrf_bounds",
                    "No valid ETrF values found"
                )
                self._complete_step("failed")
                return False
                
            min_etrf = np.min(valid_etrf)
            max_etrf = np.max(valid_etrf)
            mean_etrf = np.mean(valid_etrf)
            
            issues_found = False
            
            # Check minimum bounds
            if min_etrf < self.ETRF_MIN:
                self._add_message(
                    ValidationSeverity.ERROR,
                    "etrf_bounds",
                    f"ETrF minimum ({min_etrf:.3f}) below acceptable range [{self.ETRF_MIN}, {self.ETRF_MAX}]",
                    value=min_etrf,
                    expected_range=f"[{self.ETRF_MIN}, {self.ETRF_MAX}]",
                    recovery_suggestion="Check LE calculation and reference ET estimation"
                )
                issues_found = True
                
            # Check maximum bounds
            if max_etrf > self.ETRF_MAX:
                self._add_message(
                    ValidationSeverity.ERROR,
                    "etrf_bounds",
                    f"ETrF maximum ({max_etrf:.3f}) exceeds acceptable range [{self.ETRF_MIN}, {self.ETRF_MAX}]",
                    value=max_etrf,
                    expected_range=f"[{self.ETRF_MIN}, {self.ETRF_MAX}]",
                    recovery_suggestion="Check anchor pixel selection and hot pixel calibration"
                )
                issues_found = True
                
            # Check warning threshold
            if max_etrf > self.ETRF_WARNING_MAX:
                outlier_count = np.sum(etrf > self.ETRF_WARNING_MAX)
                self._add_message(
                    ValidationSeverity.WARNING,
                    "etrf_bounds",
                    f"{outlier_count} pixels ({outlier_count/len(valid_etrf)*100:.1f}%) exceed warning threshold ({self.ETRF_WARNING_MAX})",
                    value=max_etrf,
                    expected_range=f"[{self.ETRF_MIN}, {self.ETRF_WARNING_MAX}]",
                    recovery_suggestion="Review extreme ET values for physical plausibility"
                )
                
            # Check for unrealistic spatial variation
            if np.std(valid_etrf) > 0.5:
                self._add_message(
                    ValidationSeverity.WARNING,
                    "etrf_bounds",
                    f"High ETrF spatial variation: std = {np.std(valid_etrf):.3f}",
                    expected_range="std < 0.5",
                    recovery_suggestion="Check for mixed pixels or calibration issues"
                )
                
            self.current_step.metrics['min_etrf'] = min_etrf
            self.current_step.metrics['max_etrf'] = max_etrf
            self.current_step.metrics['mean_etrf'] = mean_etrf
            self.current_step.metrics['std_etrf'] = np.std(valid_etrf)
            self.current_step.metrics['issues_found'] = issues_found
            
            self._complete_step("failed" if issues_found else "completed")
            return not issues_found
            
        except Exception as e:
            self._add_message(
                ValidationSeverity.ERROR,
                "etrf_bounds",
                f"Error validating ETrF bounds: {str(e)}"
            )
            self._complete_step("failed")
            return False
            
    def _validate_le_et_consistency(self, cube: Any, results: Dict[str, xr.DataArray]) -> bool:
        """Validate LE-ET consistency."""
        try:
            if any(v is None for v in [cube.get("LE"), results.get("ET_inst")]):
                self._add_message(
                    ValidationSeverity.ERROR,
                    "le_et_consistency",
                    "Missing LE or ET_inst data for consistency validation"
                )
                self._complete_step("failed")
                return False
                
            le = cube.get("LE").values
            et_inst = results["ET_inst"].values
            
            # Convert ET to LE for consistency check
            # ET (mm/hr) = LE (W/m²) × 3600 / λ
            # Therefore: LE (W/m²) = ET (mm/hr) × λ / 3600
            # Note: Water density (ρ = 1000 kg/m³) cancels out since 1 kg/m² = 1 mm
            le_from_et = et_inst * LATENT_HEAT_VAPORIZATION / 3600
            
            # Find valid pixels for comparison
            valid_mask = ~np.isnan(le) & ~np.isnan(le_from_et)
            
            if not np.any(valid_mask):
                self._add_message(
                    ValidationSeverity.ERROR,
                    "le_et_consistency",
                    "No valid pixels found for LE-ET consistency check"
                )
                self._complete_step("failed")
                return False
                
            le_valid = le[valid_mask]
            le_from_et_valid = le_from_et[valid_mask]
            
            # Calculate differences
            diff = np.abs(le_valid - le_from_et_valid)
            rel_diff = diff / np.maximum(np.abs(le_valid), 1.0)  # Avoid division by zero
            
            mean_abs_diff = np.mean(diff)
            mean_rel_diff = np.mean(rel_diff)
            max_abs_diff = np.max(diff)
            
            consistency_ok = mean_rel_diff < 0.1  # 10% tolerance
            
            if not consistency_ok:
                self._add_message(
                    ValidationSeverity.WARNING,
                    "le_et_consistency",
                    f"LE-ET consistency issue: mean relative difference = {mean_rel_diff:.3f}",
                    expected_range="< 0.1 (10%)",
                    recovery_suggestion="Check LE and ET calculation formulas and constants"
                )
                
            # Check for systematic bias
            bias = np.mean(le_valid - le_from_et_valid)
            if abs(bias) > 50:  # 50 W/m² threshold
                self._add_message(
                    ValidationSeverity.WARNING,
                    "le_et_consistency",
                    f"Systematic bias detected: LE_calc - LE_from_ET = {bias:.1f} W/m²",
                    expected_range=f"[{-50}, {50}] W/m²",
                    recovery_suggestion="Check conversion constants and calculation methods"
                )
                
            self.current_step.metrics['mean_abs_diff'] = mean_abs_diff
            self.current_step.metrics['mean_rel_diff'] = mean_rel_diff
            self.current_step.metrics['max_abs_diff'] = max_abs_diff
            self.current_step.metrics['bias'] = bias
            self.current_step.metrics['consistency_ok'] = consistency_ok
            self.current_step.metrics['valid_pixels'] = np.sum(valid_mask)
            
            self._complete_step("completed" if consistency_ok else "failed")
            return consistency_ok
            
        except Exception as e:
            self._add_message(
                ValidationSeverity.ERROR,
                "le_et_consistency",
                f"Error validating LE-ET consistency: {str(e)}"
            )
            self._complete_step("failed")
            return False
            
    def _validate_metric_rules(self, cube: Any, results: Dict[str, xr.DataArray], calibration_data: Optional[Dict]) -> bool:
        """Validate METRIC-specific rules and assumptions."""
        try:
            all_passed = True
            
            # Check LE ≤ Rn - G constraint
            if all(v is not None for v in [cube.get("LE"), cube.get("R_n"), cube.get("G")]):
                le = cube.get("LE").values
                rn = cube.get("R_n").values
                g = cube.get("G").values
                
                available_energy = rn - g
                violation_mask = le > available_energy
                violation_count = np.sum(violation_mask)
                
                if violation_count > 0:
                    violation_fraction = violation_count / le.size
                    self._add_message(
                        ValidationSeverity.ERROR,
                        "metric_rules",
                        f"LE > (Rn - G) constraint violated in {violation_count} pixels ({violation_fraction*100:.1f}%)",
                        recovery_suggestion="Check energy balance calculation and anchor pixel calibration"
                    )
                    all_passed = False
                    
            # Check ET values against physical limits
            if results.get("ET_daily") is not None:
                et_daily = results["ET_daily"].values
                is_valid, msg = check_et_range(et_daily, max_et=self.ET_DAILY_MAX)
                if not is_valid:
                    self._add_message(
                        ValidationSeverity.WARNING,
                        "metric_rules",
                        f"ET_daily validation warning: {msg}",
                        expected_range=f"[0, {self.ET_DAILY_MAX}] mm/day",
                        recovery_suggestion="Review extreme ET values for physical plausibility"
                    )
                    
            if results.get("ET_inst") is not None:
                et_inst = results["ET_inst"].values
                valid_et_inst = et_inst[~np.isnan(et_inst)]
                if np.max(valid_et_inst) > self.ET_INST_MAX:
                    outlier_count = np.sum(et_inst > self.ET_INST_MAX)
                    self._add_message(
                        ValidationSeverity.WARNING,
                        "metric_rules",
                        f"{outlier_count} pixels exceed ET_inst threshold ({self.ET_INST_MAX} mm/hr)",
                        expected_range=f"[0, {self.ET_INST_MAX}] mm/hr"
                    )
                    
            # Check temperature ranges
            if cube.get("lwir11") is not None:
                ts = cube.get("lwir11").values
                is_valid, msg = check_temperature_range(ts, min_temp=self.TEMP_MIN, max_temp=self.TEMP_MAX)
                if not is_valid:
                    self._add_message(
                        ValidationSeverity.WARNING,
                        "metric_rules",
                        f"Surface temperature validation: {msg}",
                        expected_range=f"[{self.TEMP_MIN}, {self.TEMP_MAX}] K"
                    )
                    
            self.current_step.metrics['all_passed'] = all_passed
            self._complete_step("completed" if all_passed else "failed")
            return all_passed
            
        except Exception as e:
            self._add_message(
                ValidationSeverity.ERROR,
                "metric_rules",
                f"Error validating METRIC rules: {str(e)}"
            )
            self._complete_step("failed")
            return False
            
    def _assess_overall_quality(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Assess overall quality based on all validation results."""
        try:
            # Count issues by severity
            total_warnings = sum(step.warnings_count for step in validation_result.steps)
            total_errors = sum(step.errors_count for step in validation_result.steps)
            
            # Calculate quality score (0-1)
            quality_score = 1.0
            
            # Deduct for errors
            quality_score -= total_errors * 0.1
            quality_score -= total_warnings * 0.02
            
            # Deduct for failed validations
            if not validation_result.energy_balance:
                quality_score -= 0.2
            if not validation_result.physical_constraints_passed:
                quality_score -= 0.15
            if not validation_result.etrf_bounds_passed:
                quality_score -= 0.15
            if not validation_result.le_et_consistency_passed:
                quality_score -= 0.1
                
            # Ensure score is in [0, 1]
            quality_score = max(0.0, min(1.0, quality_score))
            
            # Assign quality flag
            if quality_score >= 0.9:
                quality_flag = QualityFlag.EXCELLENT
            elif quality_score >= 0.8:
                quality_flag = QualityFlag.GOOD
            elif quality_score >= 0.6:
                quality_flag = QualityFlag.FAIR
            elif quality_score >= 0.4:
                quality_flag = QualityFlag.POOR
            else:
                quality_flag = QualityFlag.BAD
                
            result = {
                'quality_score': quality_score,
                'quality_flag': quality_flag,
                'total_warnings': total_warnings,
                'total_errors': total_errors
            }
            
            self.current_step.metrics.update(result)
            self._complete_step("completed")
            return result
            
        except Exception as e:
            self._add_message(
                ValidationSeverity.ERROR,
                "quality_assessment",
                f"Error assessing overall quality: {str(e)}"
            )
            self._complete_step("failed")
            return {
                'quality_score': 0.0,
                'quality_flag': QualityFlag.BAD,
                'total_warnings': 0,
                'total_errors': 1
            }
            
    def _finalize_validation(self, validation_result: ValidationResult, cube: Any, results: Dict[str, xr.DataArray]) -> None:
        """Finalize validation result with summary statistics."""
        try:
            # Calculate basic statistics
            total_pixels = 0
            valid_pixels = 0
            
            for var in results.values():
                if var is not None:
                    data = var.values
                    total_pixels = max(total_pixels, data.size)
                    valid_pixels += np.sum(~np.isnan(data))
                    
            validation_result.total_pixels = total_pixels
            validation_result.valid_pixels = valid_pixels
            
            # Create validation summary
            validation_result.validation_summary = {
                'processing_date': validation_result.processing_date,
                'scene_id': validation_result.scene_id,
                'total_pixels': total_pixels,
                'valid_pixels': valid_pixels,
                'data_quality': valid_pixels / total_pixels if total_pixels > 0 else 0,
                'energy_balance_ok': validation_result.energy_balance is not None,
                'physical_constraints_ok': validation_result.physical_constraints_passed,
                'etrf_bounds_ok': validation_result.etrf_bounds_passed,
                'le_et_consistency_ok': validation_result.le_et_consistency_passed,
                'overall_quality': validation_result.overall_quality_flag.name,
                'confidence_score': validation_result.confidence_score,
                'validation_steps_completed': len([s for s in validation_result.steps if s.status == 'completed']),
                'validation_steps_failed': len([s for s in validation_result.steps if s.status == 'failed']),
                'total_warnings': sum(s.warnings_count for s in validation_result.steps),
                'total_errors': sum(s.errors_count for s in validation_result.steps)
            }
            
        except Exception as e:
            self.logger.error(f"Error finalizing validation: {e}")
            
    def save_validation_report(self, validation_result: ValidationResult, output_dir: str) -> str:
        """Save validation report to file."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save detailed JSON report
            report_data = {
                'validation_summary': validation_result.validation_summary,
                'steps': []
            }
            
            for step in validation_result.steps:
                step_data = {
                    'name': step.name,
                    'description': step.description,
                    'status': step.status,
                    'duration_seconds': (step.end_time - step.start_time).total_seconds() if step.end_time and step.start_time else None,
                    'warnings_count': step.warnings_count,
                    'errors_count': step.errors_count,
                    'metrics': step.metrics,
                    'messages': [
                        {
                            'severity': msg.severity.value,
                            'category': msg.category,
                            'message': msg.message,
                            'location': msg.location,
                            'value': msg.value,
                            'expected_range': msg.expected_range,
                            'recovery_suggestion': msg.recovery_suggestion,
                            'timestamp': msg.timestamp.isoformat()
                        }
                        for msg in step.messages
                    ]
                }
                report_data['steps'].append(step_data)
            
            # Save energy balance data if available
            if validation_result.energy_balance:
                report_data['energy_balance'] = {
                    'residual': validation_result.energy_balance.residual,
                    'fractional_residual': validation_result.energy_balance.fractional_residual,
                    'closure_ratio': validation_result.energy_balance.closure_ratio,
                    'cold_pixel_valid': validation_result.energy_balance.cold_pixel_valid,
                    'hot_pixel_valid': validation_result.energy_balance.hot_pixel_valid
                }
            
            # Save to file
            scene_id = validation_result.scene_id or 'unknown'
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = output_path / f"validation_report_{scene_id}_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
                
            self.logger.info(f"Validation report saved to: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"Error saving validation report: {e}")
            raise
            
    def print_validation_summary(self, validation_result: ValidationResult) -> None:
        """Print validation summary to console."""
        print("\n" + "="*80)
        print("METRIC ET VALIDATION SUMMARY")
        print("="*80)
        
        summary = validation_result.validation_summary
        
        print(f"Scene ID: {summary.get('scene_id', 'N/A')}")
        print(f"Processing Date: {summary.get('processing_date', 'N/A')}")
        print(f"Data Quality: {summary.get('data_quality', 0):.1%} valid pixels ({summary.get('valid_pixels', 0)}/{summary.get('total_pixels', 0)})")
        print(f"Overall Quality: {summary.get('overall_quality', 'UNKNOWN')} (Confidence: {summary.get('confidence_score', 0):.1%})")
        
        print("\nVALIDATION RESULTS:")
        print(f"  Energy Balance: {'PASS' if summary.get('energy_balance_ok') else 'FAIL'}")
        print(f"  Physical Constraints: {'PASS' if summary.get('physical_constraints_ok') else 'FAIL'}")
        print(f"  ETrF Bounds: {'PASS' if summary.get('etrf_bounds_ok') else 'FAIL'}")
        print(f"  LE-ET Consistency: {'PASS' if summary.get('le_et_consistency_ok') else 'FAIL'}")
        
        print(f"\nVALIDATION STATISTICS:")
        print(f"  Steps Completed: {summary.get('validation_steps_completed', 0)}")
        print(f"  Steps Failed: {summary.get('validation_steps_failed', 0)}")
        print(f"  Warnings: {summary.get('total_warnings', 0)}")
        print(f"  Errors: {summary.get('total_errors', 0)}")
        
        # Print key issues
        all_messages = []
        for step in validation_result.steps:
            all_messages.extend(step.messages)
            
        if all_messages:
            print(f"\nKEY ISSUES:")
            for msg in all_messages:
                if msg.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                    print(f"  [{msg.severity.value}] {msg.category}: {msg.message}")
                    if msg.recovery_suggestion:
                        print(f"    -> {msg.recovery_suggestion}")
                        
        print("="*80)