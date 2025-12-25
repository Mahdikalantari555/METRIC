"""
dT Calibration for METRIC Model.

This module implements the calibration of the temperature difference (dT)
relationship used in the sensible heat flux computation for METRIC.
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import xarray as xr
from dataclasses import dataclass


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
    etr_inst: float  # Reference ET at satellite overpass (mm/hr)
    valid: bool  # Whether calibration succeeded
    errors: list  # List of validation errors


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

    The coefficients are solved using the anchor pixels with H_cold ≈ 0 and
    H_hot derived from energy balance assumptions.

    Attributes:
        et0_inst: Instantaneous reference ET (grass) at overpass
        etr_inst: Instantaneous reference ET (alfalfa) at overpass
    """

    # Conversion factor from grass to alfalfa reference ET
    ETR_CONVERSION = 1.15  # ETr = ET0 * 1.15

    def __init__(
        self,
        et0_inst: float
    ):
        """
        Initialize the DTCalibration with weather data.

        Args:
            et0_inst: Instantaneous reference ET (grass) at satellite overpass (mm/hr)
        """
        self.et0_inst = et0_inst
        self.etr_inst = et0_inst * self.ETR_CONVERSION
    
    @classmethod
    def from_weather_data(
        cls,
        et0_inst: float
    ) -> 'DTCalibration':
        """
        Create DTCalibration from weather data.

        Args:
            et0_inst: Instantaneous reference ET at overpass (mm/hr)

        Returns:
            DTCalibration instance
        """
        return cls(
            et0_inst=et0_inst
        )
    
    def calibrate_dT(
        self,
        ts_cold: float,
        ts_hot: float,
        air_temperature: float,
        rn_hot: float,
        g_hot: float
    ) -> Tuple[float, float]:
        """
        Calibrate dT values at anchor pixels.

        dT = Ts - Ta for both pixels.
        H_hot = Rn_hot - G_hot (assuming LE_hot ≈ 0)

        Args:
            ts_cold: Surface temperature at cold pixel (K)
            ts_hot: Surface temperature at hot pixel (K)
            air_temperature: Air temperature at 2m (K)
            rn_hot: Net radiation at hot pixel (W/m²)
            g_hot: Soil heat flux at hot pixel (W/m²)

        Returns:
            Tuple of (dT_cold, dT_hot) in Kelvin
        """
        # dT is always Ts - Ta
        dT_cold = ts_cold - air_temperature
        dT_hot = ts_hot - air_temperature

        return dT_cold, dT_hot
    
    def compute_coefficients(
        self,
        dT_hot: float,
        rn_hot: float,
        g_hot: float
    ) -> Tuple[float, float]:
        """
        Compute the calibration coefficient for H = a * dT.

        a = H_hot / dT_hot, where H_hot = Rn_hot - G_hot

        Args:
            dT_hot: dT at hot pixel (K)
            rn_hot: Net radiation at hot pixel (W/m²)
            g_hot: Soil heat flux at hot pixel (W/m²)

        Returns:
            Tuple of (a_coefficient, b_coefficient) where b=0

        Raises:
            ValueError: If dT_hot is zero or negative
        """
        if dT_hot <= 0:
            raise ValueError(f"Invalid dT_hot ({dT_hot}): must be positive")

        h_hot = rn_hot - g_hot
        a = h_hot / dT_hot
        b = 0.0

        return a, b
    
    def calibrate_from_anchors(self, cube, anchors):
        """
        Calibrate using DataCube and AnchorPixelsResult.

        Args:
            cube: DataCube with Ta, Rn, G
            anchors: AnchorPixelsResult

        Returns:
            CalibrationResult
        """
        ts_cold = anchors.cold_pixel.temperature
        ts_hot = anchors.hot_pixel.temperature
        air_temperature_array = cube.get("temperature_2m")
        rn = cube.get("R_n")
        g = cube.get("G")

        if air_temperature_array is None:
            raise ValueError("Air temperature (temperature_2m) not found in DataCube")
        if rn is None:
            raise ValueError("Net radiation (R_n) not found in DataCube")
        if g is None:
            raise ValueError("Soil heat flux (G) not found in DataCube")

        # Get scalar air temperature (uniform)
        air_temperature = float(np.nanmean(air_temperature_array.values)) if hasattr(air_temperature_array, 'values') else float(np.nanmean(air_temperature_array))

        # Get values at hot pixel location
        hot_y, hot_x = anchors.hot_pixel.y, anchors.hot_pixel.x
        rn_hot = float(rn[hot_y, hot_x])
        g_hot = float(g[hot_y, hot_x])

        return self.calibrate(
            ts_cold=ts_cold,
            ts_hot=ts_hot,
            air_temperature=air_temperature,
            rn_hot=rn_hot,
            g_hot=g_hot
        )

    def calibrate(
        self,
        ts_cold: float,
        ts_hot: float,
        air_temperature: float,
        rn_hot: float,
        g_hot: float
    ) -> CalibrationResult:
        """
        Perform full dT calibration using anchor pixels.

        Args:
            ts_cold: Surface temperature at cold pixel (K)
            ts_hot: Surface temperature at hot pixel (K)
            air_temperature: Air temperature at 2m (K)
            rn_hot: Net radiation at hot pixel (W/m²)
            g_hot: Soil heat flux at hot pixel (W/m²)

        Returns:
            CalibrationResult with coefficients and intermediate values
        """
        errors = []

        # Validate inputs
        if ts_cold >= ts_hot:
            errors.append(
                f"Invalid temperatures: Ts_cold ({ts_cold:.2f} K) >= "
                f"Ts_hot ({ts_hot:.2f} K)"
            )

        # Calibrate dT values
        dT_cold, dT_hot = self.calibrate_dT(
            ts_cold=ts_cold,
            ts_hot=ts_hot,
            air_temperature=air_temperature,
            rn_hot=rn_hot,
            g_hot=g_hot
        )

        # Compute calibration coefficients
        try:
            a, b = self.compute_coefficients(
                dT_hot=dT_hot,
                rn_hot=rn_hot,
                g_hot=g_hot
            )
        except ValueError as e:
            errors.append(str(e))
            # Enhanced fallback: Use empirical relationship instead of zero
            # METRIC typically uses values between 10-50 W/m²/K depending on conditions
            if dT_hot <= 0:
                # For negative dT_hot, use typical agricultural value
                a = 25.0  # W/m²/K - typical for agricultural areas
                logger = logging.getLogger(__name__)
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
            etr_inst=self.etr_inst,
            valid=valid,
            errors=errors
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
            Dictionary representation of calibration
        """
        return {
            "a_coefficient": float(result.a_coefficient),
            "b_coefficient": float(result.b_coefficient),
            "dT_cold": float(result.dT_cold),
            "dT_hot": float(result.dT_hot),
            "Ts_cold": float(result.ts_cold),
            "Ts_hot": float(result.ts_hot),
            "Ta": float(result.air_temperature),
            "ETr_inst": float(result.etr_inst),
            "valid": result.valid,
            "errors": result.errors
        }
    
    def __repr__(self) -> str:
        return (
            f"DTCalibration(ET0_inst={self.et0_inst:.4f} mm/hr, "
            f"ETr_inst={self.etr_inst:.4f} mm/hr)"
        )
