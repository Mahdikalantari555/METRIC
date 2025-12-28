"""
Validation Module for METRIC ET

This module provides comprehensive validation capabilities for METRIC ET calculations,
including energy balance validation, quality assessment, and error detection.

Main Components:
- ComprehensiveMETRICValidator: Main validation framework
- ValidationResult: Result container for validation outputs
- QualityFlag: Quality classification system
"""

from .comprehensive_validation import (
    ComprehensiveMETRICValidator,
    ValidationResult,
    ValidationMessage,
    ValidationStep,
    QualityFlag,
    ValidationSeverity
)

__all__ = [
    'ComprehensiveMETRICValidator',
    'ValidationResult', 
    'ValidationMessage',
    'ValidationStep',
    'QualityFlag',
    'ValidationSeverity'
]

__version__ = '1.0.0'