"""
METRIC Calibration Module.

This module provides anchor pixel selection, dT calibration, and
energy balance validation for the METRIC evapotranspiration model.

Submodules:
- anchor_pixels: Selection of hot and cold anchor pixels
- dt_calibration: Calibration of the dT-Ts relationship
- validation: Energy balance closure and quality checks

Example:
    >>> from metric_et.calibration import (
    ...     AnchorPixelSelector,
    ...     DTCalibration,
    ...     EnergyBalanceValidator
    ... )
    >>>
    >>> # Select anchor pixels
    >>> selector = AnchorPixelSelector()
    >>> anchor_result = selector.select_anchor_pixels(
    ...     ts=ts_data, ndvi=ndvi_data, albedo=albedo_data,
    ...     le=le_data, h=h_data, rn=rn_data, g=g_data,
    ...     cloud_mask=cloud_mask, air_temperature=air_temp
    ... )
    >>>
    >>> # Calibrate dT relationship
    >>> calibrator = DTCalibration(et0_inst=0.65)
    >>> calibration = calibrator.calibrate(
    ...     ts_cold=cold_pixel.ts,
    ...     ts_hot=hot_pixel.ts,
    ...     air_temperature=air_temp
    ... )
    >>>
    >>> # Validate energy balance
    >>> validator = EnergyBalanceValidator()
    >>> validation = validator.validate_full(
    ...     cold_pixel_data={...}, hot_pixel_data={...}, etr_inst=etr_inst
    ... )
"""

from .anchor_pixels import (
    AnchorPixel,
    AnchorPixelSelector,
    AnchorPixelsResult
)

from .dt_calibration import (
    CalibrationResult,
    DTCalibration
)

from .validation import (
    EnergyBalanceResult,
    EnergyBalanceValidator,
    AnchorPixelValidation
)

__all__ = [
    # Anchor pixel classes
    'AnchorPixel',
    'AnchorPixelSelector',
    'AnchorPixelsResult',
    
    # Calibration classes
    'CalibrationResult',
    'DTCalibration',
    
    # Validation classes
    'EnergyBalanceResult',
    'EnergyBalanceValidator',
    'AnchorPixelValidation'
]
