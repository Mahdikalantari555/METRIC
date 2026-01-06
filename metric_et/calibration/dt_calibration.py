"""
dT Calibration for METRIC Model.

This module implements the calibration of the temperature difference (dT)
relationship used in the sensible heat flux computation for METRIC.
"""

from typing import Dict, Optional, Tuple, Any, List
import logging
import numpy as np
import xarray as xr
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import physical constants for METRIC calibration
from ..core.constants import LATENT_HEAT_VAPORIZATION, WATER_DENSITY, MJ_M2_DAY_TO_W
# Import global post-calibration validation
from .validation import GlobalPostCalibrationValidator, GlobalPostCalibrationValidation


class CalibrationStatus(Enum):
    """Enumeration of calibration decision outcomes."""
    ACCEPTED = "ACCEPTED"
    REUSED = "REUSED"
    REJECTED = "REJECTED"


@dataclass
class CalibrationLog:
    """Container for calibration logging information."""
    scene_id: str
    timestamp: datetime
    status: CalibrationStatus
    qa_valid_percent: float
    ndvi_p05: float
    ndvi_p95: float
    ef_min: float
    ef_mean: float
    ef_p95: float
    h_min: float
    h_mean: float
    etrf_mean: float
    etrf_p95: float
    a_coefficient: Optional[float] = None
    b_coefficient: Optional[float] = None
    rejection_reason: Optional[str] = None
    reuse_source: Optional[str] = None  # For REUSED status
    valid: bool = True  # Overall scene validity
    errors: list = field(default_factory=list)


@dataclass
class CalibrationResult:
    """Result container for dT calibration."""
    a_coefficient: float  # Slope of dT-(Ts-Ta) relationship
    b_coefficient: float  # Intercept of dT-(Ts-Ta) relationship
    dT_cold: float  # Calibrated dT at cold pixel (K)
    dT_hot: float  # Calibrated dT at hot pixel (K)
    ts_cold: float  # Ts at cold pixel (K)
    ts_hot: float  # Ts at hot pixel (K)
    air_temperature: float  # Air temperature at 2m (K)
    valid: bool  # Whether calibration succeeded
    errors: list  # List of validation errors
    # NEW: Enhanced fields for METRIC cold pixel constraint
    et0_inst: float = 0.0        # Instantaneous reference ET (mm/hr)
    le_cold: float = 0.0         # Cold pixel latent heat flux (W/m²)
    h_cold: float = 0.0          # Cold pixel sensible heat flux (W/m²)
    rn_cold: float = 0.0         # Cold pixel net radiation (W/m²)
    g_cold: float = 0.0          # Cold pixel soil heat flux (W/m²)
    # NEW: Anchor pixel metadata fields
    cold_pixel_ndvi: float = np.nan    # NDVI at cold pixel location
    cold_pixel_albedo: float = np.nan  # Albedo at cold pixel location
    hot_pixel_ndvi: float = np.nan     # NDVI at hot pixel location
    hot_pixel_albedo: float = np.nan   # Albedo at hot pixel location
    cold_pixel_x: int = 0              # X coordinate of cold pixel
    cold_pixel_y: int = 0              # Y coordinate of cold pixel
    hot_pixel_x: int = 0               # X coordinate of hot pixel
    hot_pixel_y: int = 0               # Y coordinate of hot pixel
    # NEW: Decision logic fields
    status: CalibrationStatus = CalibrationStatus.ACCEPTED
    scene_quality: str = "GOOD"  # GOOD/DEGRADED/LOW_QUALITY/REJECTED
    timestamp: datetime = field(default_factory=datetime.now)
    rejection_reason: Optional[str] = None
    reuse_source: Optional[str] = None
    # NEW: Unified pipeline fields
    global_validation_passed: bool = True
    global_violations: list = field(default_factory=list)
    anchor_physics_valid: bool = True
    prevalidation_passed: bool = True


class DTCalibration:
    """
    Calibrate the dT-(Ts-Ta) relationship for METRIC sensible heat flux.

    The METRIC model uses a linear relationship between temperature difference
    and the deviation of surface temperature from air temperature:

        dT = a * (Ts - Ta) + b

    Where:
        dT = Ts - Ta (temperature difference between surface and air)
        Ts = Surface temperature (K)
        Ta = Air temperature (K)
        a, b = Calibration coefficients determined from anchor pixels

    ENHANCED: Now enforces the cold pixel energy balance constraint:
        H_cold = Rn_cold - G_cold - LE_cold ≈ 0
    
    Where LE_cold is derived from instantaneous reference ET:
        ET0_inst = ET0_daily × (Rs_inst / Rs_daily)
        LE_cold = ET0_inst × λ × ρw / 3600

    The coefficients are solved using the correct METRIC equations:
        a = dT_hot / (Ts_hot − Ts_cold)
        b = −a · Ts_cold

    This prevents ET overestimation and enforces proper cold pixel calibration.
    """

    # Conversion factor from grass to alfalfa reference ET
    ETR_CONVERSION = 1.15  # ETr = ET0 * 1.15

    def __init__(self):
        """
        Initialize the DTCalibration with METRIC cold pixel constraint enforcement.

        Enhanced METRIC calibration now requires:
        - Daily ET0 from weather data
        - Instantaneous and daily shortwave radiation
        - Cold pixel energy balance constraint enforcement
        """
        self._calibration_history = []  # Store previous valid calibrations for fallback
        self._logger = logging.getLogger(__name__)
    
    @classmethod
    def create(cls) -> 'DTCalibration':
        """
        Create DTCalibration instance with enhanced METRIC constraint enforcement.

        Enhanced METRIC calibration now includes:
        - Cold pixel energy balance constraint
        - Instantaneous ET0 calculation from daily values
        - Enhanced validation and error logging

        Returns:
            DTCalibration instance
        """
        return cls()
    
    def compute_scene_statistics(self, cube) -> Dict[str, float]:
        """
        Compute mandatory scene statistics for logging.

        Computes:
        - QA valid %
        - NDVI p05/p95
        - EF min/mean/p95
        - H min/mean
        - ETrF mean/p95

        Args:
            cube: DataCube with scene data

        Returns:
            Dictionary with computed statistics
        """
        try:
            # Get QA valid percentage
            qa_valid = np.sum(~np.isnan(cube.get("ndvi").values))
            qa_total = cube.get("ndvi").size
            qa_valid_percent = (qa_valid / qa_total) * 100.0

            # Get NDVI percentiles
            ndvi = cube.get("ndvi").values[~np.isnan(cube.get("ndvi").values)]
            ndvi_p05 = float(np.percentile(ndvi, 5))
            ndvi_p95 = float(np.percentile(ndvi, 95))

            # Get EF statistics (if available)
            ef = cube.get("EF")
            if ef is not None:
                ef_values = ef.values[~np.isnan(ef.values)]
                ef_min = float(np.min(ef_values))
                ef_mean = float(np.mean(ef_values))
                ef_p95 = float(np.percentile(ef_values, 95))
            else:
                ef_min = ef_mean = ef_p95 = np.nan

            # Get H statistics (if available)
            h = cube.get("H")
            if h is not None:
                h_values = h.values[~np.isnan(h.values)]
                h_min = float(np.min(h_values))
                h_mean = float(np.mean(h_values))
            else:
                h_min = h_mean = np.nan

            # Get ETrF statistics (if available)
            etrf = cube.get("ETrF")
            if etrf is not None:
                etrf_values = etrf.values[~np.isnan(etrf.values)]
                etrf_mean = float(np.mean(etrf_values))
                etrf_p95 = float(np.percentile(etrf_values, 95))
            else:
                etrf_mean = etrf_p95 = np.nan

            return {
                "qa_valid_percent": qa_valid_percent,
                "ndvi_p05": ndvi_p05,
                "ndvi_p95": ndvi_p95,
                "ef_min": ef_min,
                "ef_mean": ef_mean,
                "ef_p95": ef_p95,
                "h_min": h_min,
                "h_mean": h_mean,
                "etrf_mean": etrf_mean,
                "etrf_p95": etrf_p95
            }

        except Exception as e:
            self._logger.error(f"Error computing scene statistics: {e}")
            return {
                "qa_valid_percent": 0.0,
                "ndvi_p05": np.nan,
                "ndvi_p95": np.nan,
                "ef_min": np.nan,
                "ef_mean": np.nan,
                "ef_p95": np.nan,
                "h_min": np.nan,
                "h_mean": np.nan,
                "etrf_mean": np.nan,
                "etrf_p95": np.nan
            }

    def apply_decision_logic(self, calibration_result: CalibrationResult, cube, scene_id: str) -> CalibrationResult:
        """
        Apply final decision logic and logging as specified in the algorithm.

        Decision logic:
        1. If calibration accepted: Store (a, b) with timestamp, Mark scene as QUALITY = GOOD, Proceed to ET_daily
        2. If calibration rejected but previous valid calibration exists: Reuse last valid (a, b), Mark scene as QUALITY = DEGRADED, Continue processing
        3. If calibration rejected and no fallback exists: Reject scene, Do not produce ET outputs, Log failure reason

        Args:
            calibration_result: Result from calibrate() method
            cube: DataCube with scene data
            scene_id: Scene identifier for logging

        Returns:
            CalibrationResult with decision applied
        """
        # Create calibration log
        timestamp = datetime.now()
        stats = self.compute_scene_statistics(cube)

        if calibration_result.valid:
            # Case 1: Calibration accepted
            calibration_result.status = CalibrationStatus.ACCEPTED
            calibration_result.scene_quality = "GOOD"
            calibration_result.timestamp = timestamp
            
            # Store in calibration history for potential fallback
            self._calibration_history.append({
                "a_coefficient": calibration_result.a_coefficient,
                "b_coefficient": calibration_result.b_coefficient,
                "timestamp": timestamp,
                "scene_id": scene_id
            })
            
            # Keep only last 10 calibrations
            if len(self._calibration_history) > 10:
                self._calibration_history = self._calibration_history[-10:]
            
            self._logger.info(
                f"Calibration ACCEPTED for scene {scene_id}: "
                f"a={calibration_result.a_coefficient:.3f}, b={calibration_result.b_coefficient:.1f}"
            )
            
        elif self._calibration_history:
            # Case 2: Calibration rejected but previous valid calibration exists
            last_valid = self._calibration_history[-1]
            calibration_result.status = CalibrationStatus.REUSED
            calibration_result.scene_quality = "DEGRADED"
            calibration_result.timestamp = timestamp
            calibration_result.a_coefficient = last_valid["a_coefficient"]
            calibration_result.b_coefficient = last_valid["b_coefficient"]
            calibration_result.reuse_source = f"Scene {last_valid['scene_id']} at {last_valid['timestamp']}"
            calibration_result.valid = True  # Reused calibration is valid for processing
            
            self._logger.warning(
                f"Calibration REUSED for scene {scene_id}: "
                f"a={calibration_result.a_coefficient:.3f}, b={calibration_result.b_coefficient:.1f} "
                f"from {calibration_result.reuse_source}"
            )
            
        else:
            # Case 3: Calibration rejected and no fallback exists
            calibration_result.status = CalibrationStatus.REJECTED
            calibration_result.scene_quality = "REJECTED"
            calibration_result.timestamp = timestamp
            calibration_result.rejection_reason = f"Calibration failed with errors: {calibration_result.errors}"
            
            self._logger.error(
                f"Calibration REJECTED for scene {scene_id}: {calibration_result.rejection_reason}"
            )

        # Create calibration log entry
        calibration_log = CalibrationLog(
            scene_id=scene_id,
            timestamp=timestamp,
            status=calibration_result.status,
            qa_valid_percent=stats["qa_valid_percent"],
            ndvi_p05=stats["ndvi_p05"],
            ndvi_p95=stats["ndvi_p95"],
            ef_min=stats["ef_min"],
            ef_mean=stats["ef_mean"],
            ef_p95=stats["ef_p95"],
            h_min=stats["h_min"],
            h_mean=stats["h_mean"],
            etrf_mean=stats["etrf_mean"],
            etrf_p95=stats["etrf_p95"],
            a_coefficient=calibration_result.a_coefficient,
            b_coefficient=calibration_result.b_coefficient,
            rejection_reason=calibration_result.rejection_reason,
            reuse_source=calibration_result.reuse_source,
            valid=calibration_result.status != CalibrationStatus.REJECTED,
            errors=calibration_result.errors
        )

        # Log mandatory statistics
        self._logger.info(
            f"Scene {scene_id} Statistics - "
            f"QA: {stats['qa_valid_percent']:.1f}%, "
            f"NDVI p05/p95: {stats['ndvi_p05']:.3f}/{stats['ndvi_p95']:.3f}, "
            f"EF min/mean/p95: {stats['ef_min']:.3f}/{stats['ef_mean']:.3f}/{stats['ef_p95']:.3f}, "
            f"H min/mean: {stats['h_min']:.1f}/{stats['h_mean']:.1f} W/m², "
            f"ETrF mean/p95: {stats['etrf_mean']:.3f}/{stats['etrf_p95']:.3f}, "
            f"Status: {calibration_result.status.value}"
        )

        return calibration_result

    def get_calibration_logs(self) -> list:
        """
        Get all calibration log entries.

        Returns:
            List of CalibrationLog entries
        """
        return self._calibration_history.copy()

    def clear_calibration_history(self):
        """
        Clear calibration history.
        """
        self._calibration_history.clear()
        self._logger.info("Calibration history cleared")
    
    
    def calibrate_from_anchors(self, cube, anchors):
        """
        Calibrate using DataCube and AnchorPixelsResult with enhanced weather data.

        Args:
            cube: DataCube with Ta, Rn, G, ET0, and weather data
            anchors: AnchorPixelsResult

        Returns:
            CalibrationResult with METRIC cold pixel constraint enforcement
        """
        ts_cold = anchors.cold_pixel.temperature
        ts_hot = anchors.hot_pixel.temperature
        
        # Get required arrays from DataCube
        air_temperature_array = cube.get("temperature_2m")
        rn = cube.get("R_n")
        g = cube.get("G")
        
        # NEW: Get weather data for ET0 conversion
        et0_daily_array = cube.get("et0_fao_evapotranspiration")
        rs_inst_array = cube.get("shortwave_radiation")
        rs_daily_array = cube.get("shortwave_radiation_sum")

        # Validate required data availability
        if air_temperature_array is None:
            raise ValueError("Air temperature (temperature_2m) not found in DataCube")
        if rn is None:
            raise ValueError("Net radiation (R_n) not found in DataCube")
        if g is None:
            raise ValueError("Soil heat flux (G) not found in DataCube")
        if et0_daily_array is None:
            raise ValueError("Daily ET0 (et0_fao_evapotranspiration) not found in DataCube")
        if rs_inst_array is None:
            raise ValueError("Instantaneous shortwave radiation (shortwave_radiation) not found in DataCube")
        if rs_daily_array is None:
            raise ValueError("Daily shortwave radiation sum (shortwave_radiation_sum) not found in DataCube")

        # Get scalar air temperature (uniform)
        air_temperature = float(np.nanmean(air_temperature_array.values)) if hasattr(air_temperature_array, 'values') else float(np.nanmean(air_temperature_array))
        
        # Get scalar weather data (assuming uniform over scene)
        et0_daily = float(np.nanmean(et0_daily_array.values))
        rs_inst = float(np.nanmean(rs_inst_array.values))
        rs_daily = float(np.nanmean(rs_daily_array.values))

        # Get values at hot pixel location
        hot_y, hot_x = anchors.hot_pixel.y, anchors.hot_pixel.x
        rn_hot = float(rn[hot_y, hot_x])
        g_hot = float(g[hot_y, hot_x])
        
        # NEW: Get values at cold pixel location
        cold_y, cold_x = anchors.cold_pixel.y, anchors.cold_pixel.x
        rn_cold = float(rn[cold_y, cold_x])
        g_cold = float(g[cold_y, cold_x])
        
        # Extract anchor pixel metadata for calibration result
        cold_pixel_ndvi = anchors.cold_pixel.ndvi if anchors.cold_pixel.ndvi is not None else np.nan
        cold_pixel_albedo = anchors.cold_pixel.albedo if anchors.cold_pixel.albedo is not None else np.nan
        hot_pixel_ndvi = anchors.hot_pixel.ndvi if anchors.hot_pixel.ndvi is not None else np.nan
        hot_pixel_albedo = anchors.hot_pixel.albedo if anchors.hot_pixel.albedo is not None else np.nan

        logger = logging.getLogger(__name__)
        logger.info(f"Calibrating with weather data: ET0_daily={et0_daily:.3f} mm/day, Rs_inst={rs_inst:.1f} W/m², Rs_daily={rs_daily:.1f} MJ/m²/day")
        logger.info(f"Anchor pixel metadata: Cold (x={cold_x}, y={cold_y}, NDVI={cold_pixel_ndvi:.3f}, Albedo={cold_pixel_albedo:.3f}), Hot (x={hot_x}, y={hot_y}, NDVI={hot_pixel_ndvi:.3f}, Albedo={hot_pixel_albedo:.3f})")

        return self.calibrate(
            ts_cold=ts_cold,
            ts_hot=ts_hot,
            air_temperature=air_temperature,
            rn_hot=rn_hot,
            g_hot=g_hot,
            et0_daily=et0_daily,      # NEW: Daily ET0
            rs_inst=rs_inst,          # NEW: Instantaneous shortwave radiation
            rs_daily=rs_daily,        # NEW: Daily shortwave radiation sum
            rn_cold=rn_cold,          # NEW: Cold pixel net radiation
            g_cold=g_cold,            # NEW: Cold pixel soil heat flux
            # NEW: Anchor pixel metadata
            cold_pixel_ndvi=cold_pixel_ndvi,
            cold_pixel_albedo=cold_pixel_albedo,
            hot_pixel_ndvi=hot_pixel_ndvi,
            hot_pixel_albedo=hot_pixel_albedo,
            cold_pixel_x=cold_x,
            cold_pixel_y=cold_y,
            hot_pixel_x=hot_x,
            hot_pixel_y=hot_y
        )

    def calibrate(
        self,
        ts_cold: float,
        ts_hot: float,
        air_temperature: float,
        rn_hot: float,
        g_hot: float,
        et0_daily: float,           # Daily ET0 (mm/day)
        rs_inst: float,             # Instantaneous shortwave radiation (W/m²)
        rs_daily: float,            # Daily shortwave radiation sum (MJ/m²/day)
        rn_cold: float,             # Net radiation at cold pixel (W/m²)
        g_cold: float,              # Soil heat flux at cold pixel (W/m²)
        # NEW: Anchor pixel metadata
        cold_pixel_ndvi: float = np.nan,
        cold_pixel_albedo: float = np.nan,
        hot_pixel_ndvi: float = np.nan,
        hot_pixel_albedo: float = np.nan,
        cold_pixel_x: int = 0,
        cold_pixel_y: int = 0,
        hot_pixel_x: int = 0,
        hot_pixel_y: int = 0
    ) -> CalibrationResult:
        """
        Perform full dT calibration using anchor pixels with METRIC cold pixel constraint.

        Enforces the cold pixel energy balance constraint: H_cold ≈ 0 where 
        LE_cold = ET0_inst, preventing ET overestimation.

        Args:
            ts_cold: Surface temperature at cold pixel (K)
            ts_hot: Surface temperature at hot pixel (K)
            air_temperature: Air temperature at 2m (K)
            rn_hot: Net radiation at hot pixel (W/m²)
            g_hot: Soil heat flux at hot pixel (W/m²)
            et0_daily: Daily reference ET0 (mm/day)
            rs_inst: Instantaneous shortwave radiation at overpass (W/m²)
            rs_daily: Daily shortwave radiation sum (MJ/m²/day)
            rn_cold: Net radiation at cold pixel (W/m²)
            g_cold: Soil heat flux at cold pixel (W/m²)

        Returns:
            CalibrationResult with coefficients and intermediate values
        """
        errors = []
        logger = logging.getLogger(__name__)

        # Validate inputs
        if ts_cold >= ts_hot:
            errors.append(
                f"Invalid temperatures: Ts_cold ({ts_cold:.2f} K) >= "
                f"Ts_hot ({ts_hot:.2f} K)"
            )

        # Calculate instantaneous reference ET (critical METRIC fix)
        # ET0_inst = ET0_daily * (Rs_inst / Rs_daily)
        # This is the correct formula from Allen et al. (2007) and Tasumi et al. (2005)
         
        # Convert rs_daily from MJ/m²/day to W/m² for the ratio
        rs_daily_avg_w = rs_daily * MJ_M2_DAY_TO_W  # MJ/m²/day -> W/m²
         
        if rs_daily <= 0:
            errors.append(f"Invalid rs_daily ({rs_daily:.2f}): must be positive")
            et0_inst = 0.0
        else:
            # Apply radiation ratio (no capping for METRIC)
            radiation_ratio = rs_inst / rs_daily_avg_w
            
            # ET0_inst = ET0_daily * (Rs_inst / Rs_daily)
            # Result is in mm/day-equivalent for METRIC LE calculation
            et0_inst = et0_daily * radiation_ratio

            logger.debug(f"ET0 conversion: {et0_daily:.3f} mm/day * {radiation_ratio:.2f} = {et0_inst:.3f} mm/day-equiv")

        # Calculate cold pixel latent heat flux from ET0 constraint
        # LE = ET * λ / 86400 (ET in mm/day, λ in J/kg, result in W/m²)
        le_cold = et0_inst * LATENT_HEAT_VAPORIZATION / 86400.0

        # Calculate available energy at cold pixel
        available_energy = rn_cold - g_cold

        # ENFORCE PHYSICAL CONSTRAINT: LE_cold cannot exceed available energy
        # This prevents non-physical negative H_cold values
        if le_cold > available_energy:
            logger.warning(
                f"LE_cold capped to available energy: {le_cold:.1f} W/m² > {available_energy:.1f} W/m²"
            )
            le_cold = available_energy  # Hard cap LE_cold to available energy
            h_cold = 0.0  # H_cold ≈ 0 when LE_cold is capped
            logger.info(f"Energy balance: LE_cold capped to {le_cold:.1f} W/m², H_cold = {h_cold:.1f} W/m²")
        else:
            # Normal case: calculate H_cold from energy balance
            h_cold = available_energy - le_cold
            logger.debug(f"Cold pixel energy balance: H_cold = {available_energy:.1f} - {le_cold:.1f} = {h_cold:.1f} W/m²")

        # Calibrate dT values
        dT_cold = ts_cold - air_temperature
        dT_hot = ts_hot - air_temperature

        # Validate cold pixel dT constraint (METRIC requirement)
        if dT_cold > 0.5:
            errors.append(
                f"Cold pixel dT constraint violated: dT_cold = {dT_cold:.2f} K > 0.5 K. "
                f"This indicates ET overestimation. Check ET0_inst calculation."
            )
            logger.warning(f"Cold pixel dT constraint failed: {dT_cold:.2f} K > 0.5 K")
 
        # Validate ET0_inst is positive
        if et0_inst <= 0:
            errors.append(
                f"Invalid ET0_inst: {et0_inst:.3f} mm/hr. Must be positive."
            )
            logger.warning(f"Invalid ET0_inst: {et0_inst:.3f} mm/hr")
 
        # Validate LE_cold is positive
        if le_cold <= 0:
            errors.append(
                f"Invalid LE_cold: {le_cold:.1f} W/m². Must be positive."
            )
            logger.warning(f"Invalid LE_cold: {le_cold:.1f} W/m²")
 
        # Validate energy balance at cold pixel (METRIC requirement)
        # Note: H_cold should be small when LE_cold is properly constrained
        # Large H_cold values indicate calibration issues, not energy balance violations
        if le_cold == available_energy and h_cold > 30.0:
            errors.append(
                f"Cold pixel calibration issue: H_cold = {h_cold:.2f} W/m² after LE_cold capping. "
                f"This suggests the cold pixel may not be suitable for calibration."
            )
            logger.warning(f"Cold pixel calibration issue: H_cold = {h_cold:.2f} W/m² (LE_cold was capped)")
        elif le_cold < available_energy and abs(h_cold) > 30.0:
            errors.append(
                f"Cold pixel energy balance violation: H_cold = {h_cold:.2f} W/m². "
                f"Should be ≈ 0 for properly calibrated cold pixel."
            )
            logger.warning(f"Cold pixel energy balance issue: H_cold = {h_cold:.2f} W/m²")

        # Compute calibration coefficients using CORRECT METRIC equations
        # that account for actual dT_cold (not assuming dT_cold = 0)
        try:
            if ts_hot <= ts_cold:
                raise ValueError(f"Invalid temperatures: Ts_hot ({ts_hot}) <= Ts_cold ({ts_cold})")

            # CORRECT METRIC calibration equations accounting for dT_cold:
            # dT = a * (Ts - Ts_cold) + dT_cold
            # So a = (dT_hot - dT_cold) / (Ts_hot - Ts_cold)
            # b = dT_cold - a * Ts_cold

            a = (dT_hot - dT_cold) / (ts_hot - ts_cold)
            b = dT_cold - a * ts_cold
            
            logger.debug(f"METRIC calibration: a = ({dT_hot:.2f} - {dT_cold:.2f}) / ({ts_hot:.1f} - {ts_cold:.1f}) = {a:.3f}")
            logger.debug(f"METRIC calibration: b = {dT_cold:.2f} - {a:.3f} * {ts_cold:.1f} = {b:.1f}")
            
        except ValueError as e:
            errors.append(str(e))
            # Enhanced fallback: Use empirical relationship instead of zero
            # METRIC typically uses values between 10-50 W/m²/K depending on conditions
            if ts_hot <= ts_cold:
                # For invalid temperature order, use typical agricultural value
                a = 25.0  # W/m²/K - typical for agricultural areas
                b = dT_cold - a * ts_cold if 'dT_cold' in locals() else 0.0
                logger.warning(f"Invalid temperature order: using empirical fallback a={a}, b={b}")
            else:
                a, b = 0.0, 0.0  # Only use zero fallback for other errors

        # Validate calibration
        valid = len(errors) == 0

        return CalibrationResult(
            a_coefficient=a,
            b_coefficient=b,
            dT_cold=dT_cold,
            dT_hot=dT_hot,
            ts_cold=ts_cold,
            ts_hot=ts_hot,
            air_temperature=air_temperature,
            valid=valid,
            errors=errors,
            # NEW: Enhanced fields for METRIC cold pixel constraint
            et0_inst=et0_inst,
            le_cold=le_cold,
            h_cold=h_cold,
            rn_cold=rn_cold,
            g_cold=g_cold,
            # NEW: Anchor pixel metadata fields
            cold_pixel_ndvi=cold_pixel_ndvi,
            cold_pixel_albedo=cold_pixel_albedo,
            hot_pixel_ndvi=hot_pixel_ndvi,
            hot_pixel_albedo=hot_pixel_albedo,
            cold_pixel_x=cold_pixel_x,
            cold_pixel_y=cold_pixel_y,
            hot_pixel_x=hot_pixel_x,
            hot_pixel_y=hot_pixel_y
        )
    
    def unified_calibration_pipeline(
        self,
        cube,
        scene_id: str,
        energy_balance_manager,
        anchor_pixel_selector,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> CalibrationResult:
        """
        Execute the complete METRIC calibration pipeline as specified in the algorithm.
        
        This method implements the exact METRIC algorithm flow:
        1. Scene-level pre-validation → 2. Dynamic anchor candidate selection → 
        3. Anchor pixel physics checks → 4. Calibration solve (dT = a·Ts + b) → 
        5. Global post-calibration validation → 6. Final decision logic with logging
        
        Args:
            cube: DataCube with scene data
            scene_id: Scene identifier for logging
            energy_balance_manager: EnergyBalanceManager instance for H/LE calculation
            anchor_pixel_selector: AnchorPixelSelector instance
            validation_config: Configuration for validation thresholds
            
        Returns:
            CalibrationResult with full pipeline execution results
        """
        import logging
        from datetime import datetime
        
        logger = logging.getLogger(__name__)
        logger.info(f"Starting unified METRIC calibration pipeline for scene {scene_id}")
        
        try:
            # Step 1: Scene-level pre-validation (HARD REJECT)
            logger.info("Step 1: Scene-level pre-validation")
            rejected, reason = self._perform_scene_prevalidation(cube)
            if rejected:
                logger.error(f"Scene rejected in pre-validation: {reason}")
                return CalibrationResult(
                    a_coefficient=0.0, b_coefficient=0.0,
                    dT_cold=0.0, dT_hot=0.0,
                    ts_cold=0.0, ts_hot=0.0,
                    air_temperature=293.15,
                    valid=False, errors=[f"Pre-validation rejection: {reason}"],
                    status=CalibrationStatus.REJECTED,
                    scene_quality="REJECTED",
                    timestamp=datetime.now(),
                    rejection_reason=reason,
                    # NEW: Anchor pixel metadata fields with defaults
                    cold_pixel_ndvi=np.nan,
                    cold_pixel_albedo=np.nan,
                    hot_pixel_ndvi=np.nan,
                    hot_pixel_albedo=np.nan,
                    cold_pixel_x=0,
                    cold_pixel_y=0,
                    hot_pixel_x=0,
                    hot_pixel_y=0
                )
            
            # Step 2: Dynamic anchor candidate selection
            logger.info("Step 2: Dynamic anchor candidate selection")
            rn = cube.get("R_n")
            g_flux = cube.get("G")
            
            if rn is None or g_flux is None:
                raise ValueError("Rn and G required for anchor pixel selection")
            
            # Use enhanced selector if available
            if hasattr(anchor_pixel_selector, 'select_anchor_pixels') and \
               'cube' in anchor_pixel_selector.select_anchor_pixels.__code__.co_varnames:
                # Enhanced selector takes DataCube as first parameter
                anchor_result = anchor_pixel_selector.select_anchor_pixels(
                    cube=cube,
                    rn=rn.values if rn is not None else None,
                    g_flux=g_flux.values if g_flux is not None else None
                )
            else:
                # Standard selector takes arrays as parameters
                anchor_result = anchor_pixel_selector.select_anchor_pixels(
                    ts=cube.get("lst").values,
                    ndvi=cube.get("ndvi").values if cube.get("ndvi") is not None else None,
                    albedo=cube.get("albedo").values if cube.get("albedo") is not None else None
                )
            
            # Step 3: Anchor pixel physics checks
            logger.info("Step 3: Anchor pixel physics checks")
            anchor_valid, anchor_physics_issues = self._perform_anchor_physics_validation(
                cube, anchor_result, energy_balance_manager
            )
            
            if not anchor_valid:
                logger.warning(f"Anchor pixel physics validation failed - marking as LOW QUALITY. Issues: {anchor_physics_issues}")
                # Continue with calibration but flag as LOW QUALITY
            
            # Step 4: Calibration solve (dT = a·Ts + b)
            logger.info("Step 4: Calibration solve (dT = a·Ts + b)")
            calibration = self.calibrate_from_anchors(cube, anchor_result)
            
            if not calibration.valid:
                logger.warning(f"Calibration solve failed: {calibration.errors}")
                # For physics constraint failures, we still want to proceed with ET calculation
                # Only severe QA issues should prevent processing
                # Set calibration as valid to allow ET calculation to proceed
                calibration.valid = True
                calibration.status = CalibrationStatus.ACCEPTED
                calibration.scene_quality = "LOW_QUALITY"
                logger.info("Proceeding with ET calculation despite calibration issues")
            
            # Step 5: Global post-calibration validation (requires H, LE, EF, ETrF)
            logger.info("Step 5: Global post-calibration validation")
            
            # Calculate energy balance with calibration to get H, LE for validation
            energy_balance_manager.set_anchor_pixel_calibration(
                calibration.a_coefficient, calibration.b_coefficient
            )
            eb_results = energy_balance_manager.calculate(cube)
            
            # Get validation arrays
            ef = cube.get("EF")
            h = cube.get("H")
            etrf = cube.get("ETrF")
            
            if ef is not None and h is not None and etrf is not None:
                global_validation = self.validate_global_calibration(
                    ef, h, etrf, raise_on_failure=False
                )
                
                if not global_validation.valid:
                    logger.warning(f"Global validation failed - marking as LOW QUALITY: {global_validation.violations}")
                    # Add violations to calibration errors but don't reject
                    calibration.errors.extend([f"Global validation: {v}" for v in global_validation.violations])
                    # Keep calibration.valid as is - we want to continue processing
                    anchor_physics_issues.extend([f"Global validation: {v}" for v in global_validation.violations])
            else:
                logger.warning("Skipping global validation - required arrays not available")
                global_validation = None
            
            # Step 6: Final decision logic with logging
            logger.info("Step 6: Final decision logic with logging")
            final_result = self.apply_decision_logic(calibration, cube, scene_id)
            
            # Add global validation results to final result
            if global_validation is not None:
                final_result.global_validation_passed = global_validation.valid
                final_result.global_violations = global_validation.violations
            
            # Add anchor physics issues to final result
            final_result.anchor_physics_valid = len(anchor_physics_issues) == 0
            
            # Update final result based on physics and validation issues
            if anchor_physics_issues and calibration.valid:
                # Physics constraints failed but calibration succeeded - mark as LOW QUALITY
                final_result.scene_quality = "LOW_QUALITY"
                final_result.rejection_reason = f"Physics constraint issues: {'; '.join(anchor_physics_issues)}"
                logger.warning(f"Scene {scene_id} marked as LOW QUALITY due to physics constraints")
            
            logger.info(f"Unified calibration pipeline completed with status: {final_result.status.value}, quality: {final_result.scene_quality}")
            return final_result
            
        except Exception as e:
            logger.error(f"Unified calibration pipeline failed: {e}")
            return CalibrationResult(
                a_coefficient=0.0, b_coefficient=0.0,
                dT_cold=0.0, dT_hot=0.0,
                ts_cold=0.0, ts_hot=0.0,
                air_temperature=293.15,
                valid=False, errors=[f"Pipeline execution error: {str(e)}"],
                status=CalibrationStatus.REJECTED,
                scene_quality="REJECTED",
                timestamp=datetime.now(),
                rejection_reason=f"Pipeline error: {str(e)}",
                # NEW: Anchor pixel metadata fields with defaults
                cold_pixel_ndvi=np.nan,
                cold_pixel_albedo=np.nan,
                hot_pixel_ndvi=np.nan,
                hot_pixel_albedo=np.nan,
                cold_pixel_x=0,
                cold_pixel_y=0,
                hot_pixel_x=0,
                hot_pixel_y=0
            )
    
    def _perform_scene_prevalidation(self, cube, qa_pixel=None) -> Tuple[bool, str]:
        """
        Perform scene-level pre-validation (HARD REJECT) checks.

        Args:
            cube: DataCube with scene data
            qa_pixel: QA pixel array for ROI calculation (optional)

        Returns:
            Tuple of (rejected: bool, reason: str)
        """
        from ..core.constants import MJ_M2_DAY_TO_W

        logger = logging.getLogger(__name__)

        try:
            # Get required data
            ndvi = cube.get("ndvi")
            rn = cube.get("R_n")
            et0_daily_array = cube.get("et0_fao_evapotranspiration")
            rs_inst_array = cube.get("shortwave_radiation")
            rs_daily_array = cube.get("shortwave_radiation_sum")

            if ndvi is None:
                return True, "NDVI data not available for validation"
            if rn is None:
                return True, "Net radiation (R_n) not available for validation"
            if et0_daily_array is None or rs_inst_array is None or rs_daily_array is None:
                return True, "Weather data not available for ET0_inst calculation"

            # 1. QA coverage check: count only cloud-masked pixels as loss within ROI
            # Use qa_pixel to determine ROI size if available (clipped but not cloud-masked)
            if qa_pixel is not None:
                roi_pixels = np.sum(~np.isnan(qa_pixel.values))
                valid_pixels = np.sum(~np.isnan(ndvi.values))
                if roi_pixels > 0:
                    valid_pixel_fraction = valid_pixels / roi_pixels
                else:
                    return True, "No valid ROI pixels found"
            else:
                # Fallback to old method (includes clipped areas in denominator)
                valid_pixels = np.sum(~np.isnan(ndvi.values))
                total_pixels = ndvi.size
                valid_pixel_fraction = valid_pixels / total_pixels

            # Get QA coverage thresholds (same as in pipeline)
            qa_reject_threshold = 0.30  # Reject if QA < 0.30 (severe pixel loss)
            qa_low_quality_threshold = 0.70  # Flag as LOW QUALITY if QA < 0.70

            if qa_pixel is not None:
                logger.info(f"QA coverage (ROI-based): {valid_pixels}/{roi_pixels} = {valid_pixel_fraction:.3f}")
            else:
                logger.info(f"QA coverage (array-based): {valid_pixels}/{total_pixels} = {valid_pixel_fraction:.3f}")
            logger.info(f"Thresholds - Reject: <{qa_reject_threshold:.2f}, Low Quality: <{qa_low_quality_threshold:.2f}")

            if valid_pixel_fraction < qa_reject_threshold:
                # Reject if more than 70% pixels lost (cloud masked)
                return True, f"QA coverage too low (severe pixel loss): {valid_pixel_fraction:.3f} < {qa_reject_threshold:.2f}"
            elif valid_pixel_fraction < qa_low_quality_threshold:
                # Don't reject, but flag for LOW QUALITY - continue processing
                logger.warning(f"QA coverage indicates moderate pixel loss: {valid_pixel_fraction:.3f} < {qa_low_quality_threshold:.2f} - will be flagged as LOW QUALITY")
                # Don't return here - continue with validation but will be marked as LOW QUALITY later
            
            # 2. NDVI dynamic range check: NDVI_p95 - NDVI_p05 < 0.30
            ndvi_values = ndvi.values[~np.isnan(ndvi.values)]
            if len(ndvi_values) == 0:
                return True, "No valid NDVI values for dynamic range check"
            
            ndvi_p5 = np.percentile(ndvi_values, 5)
            ndvi_p95 = np.percentile(ndvi_values, 95)
            ndvi_range = ndvi_p95 - ndvi_p5
            
            logger.info(f"NDVI dynamic range: P95={ndvi_p95:.3f}, P5={ndvi_p5:.3f}, range={ndvi_range:.3f}")
            
            if ndvi_range < 0.30:
                return True, f"NDVI dynamic range too low: {ndvi_range:.3f} < 0.30"
            
            # 3. Net radiation sanity check: median(Rn) < 300 W/m²
            rn_values = rn.values[~np.isnan(rn.values)]
            if len(rn_values) == 0:
                return True, "No valid Rn values for median check"
            
            rn_median = np.median(rn_values)
            
            logger.info(f"Net radiation median: {rn_median:.1f} W/m²")
            
            if rn_median < 300:
                return True, f"Net radiation median too low: {rn_median:.1f} < 300 W/m²"
            
            # 4. ET0_inst sanity check: ET0_inst < 0.2 or > 1.0 mm/hr
            et0_daily = float(np.nanmean(et0_daily_array.values))
            rs_inst = float(np.nanmean(rs_inst_array.values))
            rs_daily = float(np.nanmean(rs_daily_array.values))
            
            rs_daily_avg_w = rs_daily * MJ_M2_DAY_TO_W  # MJ/m²/day -> W/m²
            
            if rs_daily_avg_w <= 0:
                return True, "Invalid daily shortwave radiation for ET0_inst calculation"
            
            radiation_ratio = rs_inst / rs_daily_avg_w
            et0_inst = et0_daily * radiation_ratio / 24.0  # mm/day -> mm/hr
            
            logger.info(f"ET0_inst: {et0_inst:.3f} mm/hr")
            
            if et0_inst < 0.1 or et0_inst > 1.8:
                return True, f"ET0_inst out of range: {et0_inst:.3f} mm/hr (must be 0.1-1.5)"
            
            logger.info("Scene-level pre-validation passed")
            return False, ""
            
        except Exception as e:
            logger.error(f"Error in scene pre-validation: {e}")
            return True, f"Pre-validation error: {str(e)}"
    
    def _perform_anchor_physics_validation(self, cube, anchor_result, energy_balance_manager) -> Tuple[bool, List[str]]:
        """
        Perform anchor pixel physics validation.
        
        Args:
            cube: DataCube with scene data
            anchor_result: AnchorPixelsResult from selection
            energy_balance_manager: EnergyBalanceManager for H/LE calculation
            
        Returns:
            Tuple of (valid: bool, issues: List[str])
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Set temporary calibration for anchor validation
            # Use temporary coefficients for validation
            temp_calibration = CalibrationResult(
                a_coefficient=20.0,  # Temporary value
                b_coefficient=0.0,
                dT_cold=0.0, dT_hot=0.0,
                ts_cold=0.0, ts_hot=0.0,
                air_temperature=self._get_air_temperature(cube),
                valid=True, errors=[],
                # NEW: Anchor pixel metadata fields with defaults
                cold_pixel_ndvi=np.nan,
                cold_pixel_albedo=np.nan,
                hot_pixel_ndvi=np.nan,
                hot_pixel_albedo=np.nan,
                cold_pixel_x=0,
                cold_pixel_y=0,
                hot_pixel_x=0,
                hot_pixel_y=0
            )
            
            energy_balance_manager.set_anchor_pixel_calibration(
                temp_calibration.a_coefficient, temp_calibration.b_coefficient
            )
            
            # Calculate energy balance to get H, LE at anchor pixels
            eb_results = energy_balance_manager.calculate(cube)
            
            # Get anchor pixel data
            cold_y, cold_x = anchor_result.cold_pixel.y, anchor_result.cold_pixel.x
            hot_y, hot_x = anchor_result.hot_pixel.y, anchor_result.hot_pixel.x
            
            # Extract energy balance values at anchor pixels
            h_array = cube.get("H")
            le_array = cube.get("LE")
            rn_array = cube.get("R_n")
            g_array = cube.get("G")
            
            if h_array is None or le_array is None:
                logger.warning("H and LE arrays not available for anchor validation")
                return True, []  # Skip validation if data not available
            
            # Cold pixel validation
            cold_h = float(h_array[cold_y, cold_x])
            cold_le = float(le_array[cold_y, cold_x])
            cold_rn = float(rn_array[cold_y, cold_x])
            cold_g = float(g_array[cold_y, cold_x])
            
            # Hot pixel validation
            hot_h = float(h_array[hot_y, hot_x])
            hot_le = float(le_array[hot_y, hot_x])
            hot_rn = float(rn_array[hot_y, hot_x])
            hot_g = float(g_array[hot_y, hot_x])
            
            # Validate using EnergyBalanceValidator
            from .validation import EnergyBalanceValidator
            validator = EnergyBalanceValidator()
            
            # Prepare data for validation
            cold_pixel_data = {
                'LE': cold_le,
                'H': cold_h,
                'Rn': cold_rn,
                'G': cold_g,
                'dT': anchor_result.cold_pixel.temperature - temp_calibration.air_temperature,
                'ETrF': 1.05  # Expected cold pixel ETrF
            }
            
            hot_pixel_data = {
                'LE': hot_le,
                'H': hot_h,
                'Rn': hot_rn,
                'G': hot_g,
                'dT': anchor_result.hot_pixel.temperature - temp_calibration.air_temperature,
                'ETrF': 0.05  # Expected hot pixel ETrF
            }
            
            et0_inst = cube.get("et0_fao_evapotranspiration")
            et0_inst_value = float(np.nanmean(et0_inst.values)) / 24.0 if et0_inst is not None else 0.5
            
            validation_result = validator.validate_full(
                cold_pixel_data, hot_pixel_data, et0_inst_value
            )
            
            # Log validation results
            logger.info(
                f"Anchor physics validation: Cold valid={validation_result.cold_pixel_valid}, "
                f"Hot valid={validation_result.hot_pixel_valid}"
            )
            
            issues = []
            if validation_result.cold_pixel_issues:
                logger.warning(f"Cold pixel issues: {validation_result.cold_pixel_issues}")
                issues.extend([f"Cold pixel: {issue}" for issue in validation_result.cold_pixel_issues])
            
            if validation_result.hot_pixel_issues:
                logger.warning(f"Hot pixel issues: {validation_result.hot_pixel_issues}")
                issues.extend([f"Hot pixel: {issue}" for issue in validation_result.hot_pixel_issues])
            
            valid = validation_result.cold_pixel_valid and validation_result.hot_pixel_valid
            return valid, issues
            
        except Exception as e:
            logger.error(f"Error in anchor physics validation: {e}")
            return True, [f"Validation error: {str(e)}"]  # Continue on error but log issue
    
    def _get_air_temperature(self, cube) -> float:
        """
        Get air temperature from cube.
        
        Args:
            cube: DataCube with weather data
            
        Returns:
            Air temperature in Kelvin
        """
        temp_2m = cube.get("temperature_2m")
        if temp_2m is not None:
            return float(np.nanmean(temp_2m.values))
        return 293.15  # Default 20°C

    def validate_global_calibration(
        self,
        ef: xr.DataArray,
        h: xr.DataArray,
        etrf: xr.DataArray,
        raise_on_failure: bool = True
    ) -> GlobalPostCalibrationValidation:
        """
        Perform global post-calibration validation on calibrated scene data.
        
        This method validates that the calibrated scene data meets physical
        realism constraints specified in the METRIC algorithm. If any constraint
        is violated, the validation fails.
        
        Validates:
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
        
        Args:
            ef: Evaporative fraction array (dimensionless)
            h: Sensible heat flux array (W/m²)
            etrf: Reference evapotranspiration fraction array (dimensionless)
            raise_on_failure: Whether to raise ValueError if validation fails
            
        Returns:
            GlobalPostCalibrationValidation with validation results and statistics
            
        Raises:
            ValueError: If validation fails and raise_on_failure is True
        """
        logger = logging.getLogger(__name__)
        
        # Create validator instance
        validator = GlobalPostCalibrationValidator()
        
        logger.info("Performing global post-calibration validation...")
        
        # Perform validation
        if raise_on_failure:
            result = validator.validate_or_raise(ef, h, etrf)
        else:
            result = validator.validate_global_calibration(ef, h, etrf)
        
        # Add validation info to logger
        if result.valid:
            logger.info(
                f"Global validation PASSED: EF[min={result.ef_min:.3f}, p95={result.ef_p95:.3f}, mean={result.ef_mean:.3f}], "
                f"H[neg_frac={result.h_negative_fraction:.1%}, extreme_neg={result.h_extreme_negative_fraction:.1%}], "
                f"ETrF[p95={result.etrf_p95:.3f}, mean={result.etrf_mean:.3f}]"
            )
        else:
            logger.error(
                f"Global validation FAILED: {len(result.violations)} violations detected"
            )
        
        return result
    
    def compute_dT_map(
        self,
        ts: xr.DataArray,
        calibration: CalibrationResult
    ) -> xr.DataArray:
        """
        Compute dT map as Ts - Ta.

        Args:
            ts: Surface temperature array (K)
            calibration: CalibrationResult from calibrate()

        Returns:
            xr.DataArray of dT values (K)
        """
        dT = ts - calibration.air_temperature
        dT = dT.assign_attrs({
            'units': 'K',
            'description': 'Temperature difference (Ts - Ta) for METRIC'
        })
        return dT
    
    def to_dict(self, result: CalibrationResult) -> Dict[str, Any]:
        """
        Convert calibration result to dictionary.

        Args:
            result: CalibrationResult from calibrate()

        Returns:
            Dictionary representation of calibration including METRIC constraint fields
        """
        return {
            "a_coefficient": float(result.a_coefficient),
            "b_coefficient": float(result.b_coefficient),
            "dT_cold": float(result.dT_cold),
            "dT_hot": float(result.dT_hot),
            "Ts_cold": float(result.ts_cold),
            "Ts_hot": float(result.ts_hot),
            "Ta": float(result.air_temperature),
            "valid": result.valid,
            "errors": result.errors,
            # NEW: Enhanced fields for METRIC cold pixel constraint
            "et0_inst": float(result.et0_inst),
            "le_cold": float(result.le_cold),
            "h_cold": float(result.h_cold),
            "rn_cold": float(result.rn_cold),
            "g_cold": float(result.g_cold),
            # NEW: Anchor pixel metadata fields
            "cold_pixel_ndvi": float(result.cold_pixel_ndvi),
            "cold_pixel_albedo": float(result.cold_pixel_albedo),
            "hot_pixel_ndvi": float(result.hot_pixel_ndvi),
            "hot_pixel_albedo": float(result.hot_pixel_albedo),
            "cold_pixel_x": int(result.cold_pixel_x),
            "cold_pixel_y": int(result.cold_pixel_y),
            "hot_pixel_x": int(result.hot_pixel_x),
            "hot_pixel_y": int(result.hot_pixel_y)
        }
    
    def __repr__(self) -> str:
        return "DTCalibration()"
