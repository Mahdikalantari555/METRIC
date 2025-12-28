"""
METRIC ET Validation Framework Integration

This module provides functions to integrate the comprehensive validation framework
into the existing download_and_process_debal.py workflow.

The integration functions are designed to be easily added to the existing pipeline
without requiring major code changes.
"""

import os
import sys
import numpy as np
import xarray as xr
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import logging

# Add metric_et to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .comprehensive_validation import ComprehensiveMETRICValidator, ValidationResult, QualityFlag
from ..utils.logger import Logger


class METRICValidationIntegrator:
    """
    Integration class for METRIC validation framework.
    
    This class provides methods to integrate validation into the existing
    download_and_process_debal.py workflow with minimal code changes.
    """
    
    def __init__(self, scene_id: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize validation integrator.
        
        Args:
            scene_id: Scene identifier for logging and reporting
            log_level: Logging level for validation messages
        """
        self.scene_id = scene_id
        self.logger = Logger.get_logger(__name__)
        
        # Configure logging level
        Logger.setup(name="metric_validation", level=log_level)
        
        # Initialize validator
        self.validator = ComprehensiveMETRICValidator(scene_id=scene_id)
        
        # Track validation results
        self.validation_results: List[ValidationResult] = []
        
    def validate_after_pipeline_execution(
        self,
        results: Dict[str, xr.DataArray],
        cube: Any,
        calibration_data: Optional[Dict] = None,
        output_dir: str = "validation_output"
    ) -> ValidationResult:
        """
        Run complete validation after METRIC pipeline execution.
        
        This function is designed to be called after the METRIC pipeline
        has completed and results are available.
        
        Args:
            results: Dictionary of METRIC calculation results
            cube: DataCube with all processed data
            calibration_data: Optional calibration information
            output_dir: Directory to save validation reports
            
        Returns:
            ValidationResult with complete assessment
        """
        self.logger.info("Starting post-pipeline validation")
        
        try:
            # Run comprehensive validation
            validation_result = self.validator.validate_complete_workflow(
                cube=cube,
                results=results,
                calibration_data=calibration_data
            )
            
            # Store results
            self.validation_results.append(validation_result)
            
            # Save validation report
            report_file = self.validator.save_validation_report(
                validation_result, output_dir
            )
            
            # Print summary to console
            self.validator.print_validation_summary(validation_result)
            
            self.logger.info(f"Validation completed. Report saved to: {report_file}")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise
            
    def validate_step_by_step(
        self,
        step_name: str,
        cube: Any,
        results: Dict[str, xr.DataArray],
        step_data: Dict[str, Any]
    ) -> bool:
        """
        Validate a specific step in the pipeline.
        
        This function can be called after each major step in the pipeline
        to catch issues early.
        
        Args:
            step_name: Name of the pipeline step being validated
            cube: Current DataCube state
            results: Current results dictionary
            step_data: Additional data specific to this step
            
        Returns:
            True if validation passed, False otherwise
        """
        self.logger.info(f"Validating step: {step_name}")
        
        try:
            if step_name == "surface_properties":
                return self._validate_surface_properties_step(cube, results, step_data)
            elif step_name == "radiation_balance":
                return self._validate_radiation_step(cube, results, step_data)
            elif step_name == "energy_balance":
                return self._validate_energy_balance_step(cube, results, step_data)
            elif step_name == "et_calculation":
                return self._validate_et_step(cube, results, step_data)
            else:
                self.logger.warning(f"No specific validation for step: {step_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Step validation failed for {step_name}: {e}")
            return False
            
    def _validate_surface_properties_step(self, cube: Any, results: Dict[str, xr.DataArray], step_data: Dict[str, Any]) -> bool:
        """Validate surface properties calculation step."""
        issues_found = False
        
        # NDVI validation
        if cube.get("ndvi") is not None:
            ndvi = cube.get("ndvi").values
            valid_ndvi = ndvi[(ndvi >= -1.0) & (ndvi <= 1.0)]
            if len(valid_ndvi) < len(ndvi) * 0.8:
                self.logger.warning(f"Only {len(valid_ndvi)/len(ndvi)*100:.1f}% of NDVI values are valid")
                issues_found = True
            else:
                self.logger.info(f"NDVI validation passed: {len(valid_ndvi)} valid pixels")
                
        # Albedo validation
        if cube.get("albedo") is not None:
            albedo = cube.get("albedo").values
            valid_albedo = albedo[(albedo >= 0.0) & (albedo <= 1.0)]
            if len(valid_albedo) < len(albedo) * 0.9:
                self.logger.warning(f"Only {len(valid_albedo)/len(albedo)*100:.1f}% of albedo values are valid")
                issues_found = True
            else:
                self.logger.info(f"Albedo validation passed: {len(valid_albedo)} valid pixels")
                
        return not issues_found
        
    def _validate_radiation_step(self, cube: Any, results: Dict[str, xr.DataArray], step_data: Dict[str, Any]) -> bool:
        """Validate radiation balance step."""
        issues_found = False
        
        # Net radiation validation
        if cube.get("R_n") is not None:
            rn = cube.get("R_n").values
            valid_rn = rn[(rn >= 0) & (rn <= 1000)]
            if len(valid_rn) < len(rn) * 0.8:
                self.logger.warning(f"Only {len(valid_rn)/len(rn)*100:.1f}% of net radiation values are in expected range")
                issues_found = True
            else:
                self.logger.info(f"Net radiation validation passed: {len(valid_rn)} valid pixels")
                
        return not issues_found
        
    def _validate_energy_balance_step(self, cube: Any, results: Dict[str, xr.DataArray], step_data: Dict[str, Any]) -> bool:
        """Validate energy balance step."""
        issues_found = False
        
        # Check energy balance components
        if all(v is not None for v in [cube.get("R_n"), cube.get("G"), cube.get("H"), cube.get("LE")]):
            rn = cube.get("R_n").values
            g = cube.get("G").values
            h = cube.get("H").values
            le = cube.get("LE").values
            
            # Check for negative available energy
            available_energy = rn - g
            negative_ae_count = np.sum(available_energy < 0)
            
            if negative_ae_count > 0:
                self.logger.error(f"Found {negative_ae_count} pixels with negative available energy (Rn - G < 0)")
                issues_found = True
            else:
                self.logger.info("Energy balance step validation passed")
                
        return not issues_found
        
    def _validate_et_step(self, cube: Any, results: Dict[str, xr.DataArray], step_data: Dict[str, Any]) -> bool:
        """Validate ET calculation step."""
        issues_found = False
        
        # ETrF validation
        if "ETrF" in results and results["ETrF"] is not None:
            etrf = results["ETrF"].values
            valid_etrf = etrf[(etrf >= 0.0) & (etrf <= 1.3)]
            if len(valid_etrf) < len(etrf) * 0.9:
                outlier_count = len(etrf) - len(valid_etrf)
                self.logger.warning(f"ETrF validation: {outlier_count} pixels outside expected bounds [0, 1.3]")
                issues_found = True
            else:
                self.logger.info(f"ETrF validation passed: {len(valid_etrf)} valid pixels")
                
        # ET daily validation
        if "ET_daily" in results and results["ET_daily"] is not None:
            et_daily = results["ET_daily"].values
            valid_et_daily = et_daily[(et_daily >= 0) & (et_daily <= 15)]
            if len(valid_et_daily) < len(et_daily) * 0.9:
                outlier_count = len(et_daily) - len(valid_et_daily)
                self.logger.warning(f"ET_daily validation: {outlier_count} pixels exceed 15 mm/day threshold")
                issues_found = True
            else:
                self.logger.info(f"ET_daily validation passed: {len(valid_et_daily)} valid pixels")
                
        return not issues_found
        
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.validation_results:
            return {"message": "No validation results available"}
            
        # Aggregate results
        total_validations = len(self.validation_results)
        passed_validations = sum(1 for r in self.validation_results if r.overall_quality_flag in [QualityFlag.EXCELLENT, QualityFlag.GOOD])
        
        return {
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "success_rate": passed_validations / total_validations if total_validations > 0 else 0,
            "average_quality": np.mean([r.confidence_score for r in self.validation_results]),
            "latest_result": self.validation_results[-1].validation_summary if self.validation_results else None
        }


# Integration functions for download_and_process_debal.py
def add_validation_to_download_process():
    """
    Generate code snippets to add validation to download_and_process_debal.py.
    
    This function provides the code modifications needed to integrate
    the validation framework into the existing workflow.
    """
    
    integration_code = '''
# ============================================================================
# VALIDATION FRAMEWORK INTEGRATION
# ============================================================================

# Add these imports at the top of download_and_process_debal.py
from metric_et.validation import METRICValidationIntegrator
from metric_et.validation.comprehensive_validation import QualityFlag

# Add validation integrator initialization in main()
validation_integrator = METRICValidationIntegrator(scene_id=scene_date)

# Modify the main() function to include validation
def main():
    """Main function with integrated validation."""
    
    print("Starting Landsat download and METRIC-ET processing for Debal ROI")
    
    try:
        # Step 1: Download Landsat data
        print("Step 1: Downloading Landsat data from MPC...")
        landsat_dir, scene_date = download_landsat_for_debal()
        
        # Step 2: Run METRIC pipeline
        print("Step 2: Running METRIC-ET pipeline...")
        results, cube = run_metric_pipeline(landsat_dir)
        
        # Step 2.5: Run validation after pipeline completion
        print("Step 2.5: Running comprehensive validation...")
        validation_result = validation_integrator.validate_after_pipeline_execution(
            results=results,
            cube=cube,
            output_dir="validation_output"
        )
        
        # Check validation quality and provide recommendations
        if validation_result.overall_quality_flag in [QualityFlag.POOR, QualityFlag.BAD]:
            print(f"⚠️  WARNING: Validation quality is {validation_result.overall_quality_flag.name}")
            print("Consider reviewing the following:")
            for step in validation_result.steps:
                if step.errors_count > 0:
                    print(f"  - {step.name}: {step.errors_count} errors")
        else:
            print(f"✓ Validation passed with {validation_result.overall_quality_flag.name} quality")
        
        # Step 3: Save desired outputs
        print("Step 3: Saving desired outputs...")
        saved_files = save_desired_outputs(results, cube, "debal_outputs", scene_date.replace('-', ''))
        
        # Step 4: Save validation summary
        print("Step 4: Saving validation summary...")
        validation_summary = validation_integrator.get_validation_summary()
        
        # Add validation info to summary
        saved_files["validation_report"] = "validation_output"
        saved_files["validation_quality"] = validation_result.overall_quality_flag.name
        saved_files["confidence_score"] = validation_result.confidence_score
        
        print("\\nProcessing completed successfully!")
        print(f"Landsat data: {landsat_dir}")
        print(f"Outputs saved to: debal_outputs/")
        print(f"Validation report saved to: validation_output/")
        print(f"Available outputs: {list(saved_files.keys())}")
        print(f"Validation Quality: {validation_result.overall_quality_flag.name} ({validation_result.confidence_score:.1%})")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0
'''
    
    return integration_code


def create_validation_config():
    """
    Create a validation configuration file.
    
    Returns:
        Dictionary with validation configuration settings
    """
    
    config = {
        "validation": {
            "enabled": True,
            "log_level": "INFO",
            "output_directory": "validation_output",
            "thresholds": {
                "energy_balance_tolerance": 0.15,
                "etrf_min": 0.0,
                "etrf_max": 1.3,
                "etrf_warning_max": 1.2,
                "et_daily_max": 15.0,
                "et_inst_max": 1.0,
                "temperature_min": 240,
                "temperature_max": 320
            },
            "quality_thresholds": {
                "excellent": 0.9,
                "good": 0.8,
                "fair": 0.6,
                "poor": 0.4
            },
            "validation_steps": {
                "input_validation": True,
                "surface_properties": True,
                "radiation_balance": True,
                "energy_balance": True,
                "etrf_bounds": True,
                "le_et_consistency": True,
                "metric_rules": True,
                "quality_assessment": True
            },
            "reporting": {
                "save_json_report": True,
                "print_console_summary": True,
                "include_detailed_messages": True,
                "include_recovery_suggestions": True
            }
        }
    }
    
    return config


def save_validation_config(config: Dict[str, Any], output_path: str) -> str:
    """
    Save validation configuration to file.
    
    Args:
        config: Validation configuration dictionary
        output_path: Path to save configuration file
        
    Returns:
        Path to saved configuration file
    """
    import json
    
    config_path = Path(output_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    return str(config_path)


def print_integration_instructions():
    """Print instructions for integrating validation framework."""
    
    instructions = """
METRIC ET VALIDATION FRAMEWORK INTEGRATION INSTRUCTIONS
======================================================

1. REQUIRED IMPORTS:
   Add these imports to download_and_process_debal.py:
   
   from metric_et.validation import METRICValidationIntegrator
   from metric_et.validation.comprehensive_validation import QualityFlag

2. INITIALIZE VALIDATION:
   Add this initialization code in your main() function:
   
   # Initialize validation integrator
   validation_integrator = METRICValidationIntegrator(scene_id=scene_date)

3. ADD VALIDATION STEP:
   After running the METRIC pipeline, add validation:
   
   # Run comprehensive validation
   validation_result = validation_integrator.validate_after_pipeline_execution(
       results=results,
       cube=cube,
       output_dir="validation_output"
   )

4. CHECK VALIDATION QUALITY:
   Add quality check after validation:
   
   if validation_result.overall_quality_flag in [QualityFlag.POOR, QualityFlag.BAD]:
       print(f"⚠️  WARNING: Validation quality is {validation_result.overall_quality_flag.name}")
       print("Consider reviewing the validation report for details.")
   else:
       print(f"✓ Validation passed with {validation_result.overall_quality_flag.name} quality")

5. SAVE VALIDATION INFO:
   Add validation information to your output summary:
   
   saved_files["validation_report"] = "validation_output"
   saved_files["validation_quality"] = validation_result.overall_quality_flag.name
   saved_files["confidence_score"] = validation_result.confidence_score

OUTPUTS:
- Validation report: validation_output/validation_report_[scene_id]_[timestamp].json
- Console summary: Printed during execution
- Quality flags: EXCELLENT, GOOD, FAIR, POOR, BAD
- Confidence score: 0.0 to 1.0

VALIDATION CHECKS:
✓ Input data completeness and quality
✓ Surface properties (NDVI, albedo, emissivity)
✓ Radiation balance components
✓ Energy balance closure (Rn = G + H + LE)
✓ ETrF sanity bounds (0 to 1.3)
✓ LE-ET consistency
✓ METRIC-specific physical constraints
✓ Overall quality assessment

For detailed implementation, see the integration code provided by add_validation_to_download_process()
"""
    
    print(instructions)


# Example usage function
def example_integration():
    """Example of how to integrate validation into existing workflow."""
    
    # Create example validation integrator
    integrator = METRICValidationIntegrator(scene_id="20251221")
    
    print("Example validation integration:")
    print("1. Initialize: integrator = METRICValidationIntegrator(scene_id='20251221')")
    print("2. After pipeline: validation_result = integrator.validate_after_pipeline_execution(...)")
    print("3. Check quality: if validation_result.overall_quality_flag == QualityFlag.GOOD:")
    print("4. Save report: report_file = integrator.validator.save_validation_report(...)")
    print("\nConfiguration options:")
    print("- Log level: DEBUG, INFO, WARNING, ERROR")
    print("- Output directory: Custom validation report location")
    print("- Quality thresholds: Configurable validation tolerances")
    
    return integrator


if __name__ == "__main__":
    # Print integration instructions
    print_integration_instructions()
    
    # Show example integration
    example_integration()
    
    # Create and save default configuration
    config = create_validation_config()
    config_path = save_validation_config(config, "validation_config.json")
    print(f"\nDefault configuration saved to: {config_path}")