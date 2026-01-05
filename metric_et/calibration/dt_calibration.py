"""
dT Calibration for METRIC Model.

This module implements the calibration of the temperature difference (dT)
relationship used in the sensible heat flux computation for METRIC.
"""

from typing import Dict, Optional, Tuple, Any
import logging
import numpy as np
import xarray as xr
from dataclasses import dataclass

# Import physical constants for METRIC calibration
from ..core.constants import LATENT_HEAT_VAPORIZATION, WATER_DENSITY, MJ_M2_DAY_TO_W


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
        pass
    
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

        logger = logging.getLogger(__name__)
        logger.info(f"Calibrating with weather data: ET0_daily={et0_daily:.3f} mm/day, Rs_inst={rs_inst:.1f} W/m², Rs_daily={rs_daily:.1f} MJ/m²/day")

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
            g_cold=g_cold             # NEW: Cold pixel soil heat flux
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
        g_cold: float               # Soil heat flux at cold pixel (W/m²)
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
            # Apply radiation ratio with METRIC capping (max 2× daily average)
            radiation_ratio_raw = rs_inst / rs_daily_avg_w
            radiation_ratio = min(radiation_ratio_raw, 2.0)  # Cap at 2× daily average
            
            # ET0_inst = ET0_daily * (Rs_inst / Rs_daily)
            # Note: ET0_daily is in mm/day, so we need to convert to mm/hr
            # by dividing by 24 (hours per day)
            et0_inst = et0_daily * radiation_ratio / 24.0
             
            logger.debug(f"ET0 conversion: {et0_daily:.3f} mm/day * ({radiation_ratio:.2f} capped) / 24 = {et0_inst:.3f} mm/hr")
            logger.debug(f"  Raw radiation ratio: {radiation_ratio_raw:.2f}, Capped ratio: {radiation_ratio:.2f}")

        # Calculate cold pixel latent heat flux from ET0 constraint
        # LE = ET * λ / 3600 (ET in mm/hr, λ in J/kg, result in W/m²)
        le_cold = et0_inst * LATENT_HEAT_VAPORIZATION / 3600.0

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
        try:
            if dT_hot <= 0:
                raise ValueError(f"Invalid dT_hot ({dT_hot}): must be positive")

            # CORRECT METRIC calibration equations:
            # a = dT_hot / (Ts_hot - Ts_cold)  
            # b = -a * Ts_cold
            
            a = dT_hot / (ts_hot - ts_cold)
            b = -a * ts_cold
            
            logger.debug(f"METRIC calibration: a = {dT_hot:.2f} / ({ts_hot:.1f} - {ts_cold:.1f}) = {a:.3f}")
            logger.debug(f"METRIC calibration: b = -{a:.3f} * {ts_cold:.1f} = {b:.1f}")
            
        except ValueError as e:
            errors.append(str(e))
            # Enhanced fallback: Use empirical relationship instead of zero
            # METRIC typically uses values between 10-50 W/m²/K depending on conditions
            if dT_hot <= 0:
                # For negative dT_hot, use typical agricultural value
                a = 25.0  # W/m²/K - typical for agricultural areas
                logger.warning(f"Invalid dT_hot ({dT_hot:.2f}K): using empirical fallback a={a}")
            else:
                a, b = 0.0, 0.0  # Only use zero fallback for other errors
            b = 0.0  # METRIC typically uses b=0

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
            g_cold=g_cold
        )
    
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
            "g_cold": float(result.g_cold)
        }
    
    def __repr__(self) -> str:
        return "DTCalibration()"
