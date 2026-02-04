"""
Verification script for calibration physical value ranges.

Tests that dt_a (a_coefficient) is in the physically reasonable range of 10-50 W/m²/K.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from metric_et.calibration.dt_calibration import DTCalibration
from metric_et.core.datacube import DataCube


class TestCalibrationPhysicalValues:
    """Test calibration produces physically reasonable values."""
    
    def test_dt_a_in_physical_range(self):
        """
        Test that dt_a (a_coefficient) is in the physically reasonable range.
        
        Physical range: 10-50 W/m²/K corresponds to rah = 20-100 s/m
        which is realistic for atmospheric conditions.
        """
        calibration = DTCalibration.create()
        
        # Test case 1: Typical agricultural conditions
        result = calibration.calibrate(
            ts_cold=300.0,  # K (cold pixel - well-watered vegetation)
            ts_hot=325.0,   # K (hot pixel - dry bare soil)
            air_temperature=298.0,  # K (2m air temperature)
            rn_hot=450.0,   # W/m² (net radiation at hot pixel)
            g_hot=22.5,     # W/m² (soil heat flux ~5% of Rn)
            et0_daily=5.0,  # mm/day (daily ET0)
            rs_inst=600.0,  # W/m² (instantaneous shortwave)
            rs_daily=20.0,  # MJ/m²/day (daily shortwave)
            rn_cold=500.0,  # W/m² (net radiation at cold pixel)
            g_cold=25.0     # W/m² (soil heat flux at cold pixel)
        )
        
        # dt_a should be in 10-50 W/m²/K range
        assert 10.0 <= result.a_coefficient <= 50.0, \
            f"dt_a = {result.a_coefficient:.2f} W/m²/K outside physical range [10, 50]"
        
        # dT_hot should be ~27 K (325 - 298)
        expected_dT_hot = 325.0 - 298.0
        assert abs(result.dT_hot - expected_dT_hot) < 0.1, \
            f"dT_hot = {result.dT_hot:.2f} K, expected ~{expected_dT_hot:.2f} K"
        
        print(f"Test case 1 PASSED: dt_a = {result.a_coefficient:.2f} W/m²/K")
    
    def test_dt_a_high_radiation(self):
        """Test dt_a with high radiation conditions (clear sky, midday)."""
        calibration = DTCalibration.create()
        
        result = calibration.calibrate(
            ts_cold=295.0,  # Cool vegetation
            ts_hot=330.0,   # Hot dry soil
            air_temperature=293.0,  # Warm air
            rn_hot=600.0,   # High net radiation
            g_hot=30.0,     # Soil heat flux
            et0_daily=7.0,  # High ET0
            rs_inst=900.0,  # High insolation
            rs_daily=25.0,  # High daily radiation
            rn_cold=650.0,  # High Rn at cold pixel
            g_cold=32.5     # Higher G
        )
        
        assert 10.0 <= result.a_coefficient <= 50.0, \
            f"dt_a = {result.a_coefficient:.2f} W/m²/K outside physical range"
        
        print(f"Test case 2 PASSED: dt_a = {result.a_coefficient:.2f} W/m²/K")
    
    def test_dt_a_low_radiation(self):
        """Test dt_a with low radiation conditions (cloudy, morning)."""
        calibration = DTCalibration.create()
        
        result = calibration.calibrate(
            ts_cold=290.0,  # Cool vegetation
            ts_hot=305.0,   # Mild hot pixel
            air_temperature=288.0,  # Cool air
            rn_hot=200.0,   # Low net radiation
            g_hot=10.0,     # Low soil heat flux
            et0_daily=2.0,  # Low ET0
            rs_inst=200.0,  # Low insolation
            rs_daily=10.0,  # Low daily radiation
            rn_cold=250.0,  # Low Rn at cold pixel
            g_cold=12.5     # Low G
        )
        
        assert 10.0 <= result.a_coefficient <= 50.0, \
            f"dt_a = {result.a_coefficient:.2f} W/m²/K outside physical range"
        
        print(f"Test case 3 PASSED: dt_a = {result.a_coefficient:.2f} W/m²/K")
    
    def test_dt_a_small_temperature_difference(self):
        """Test dt_a when Ts_hot - Ts_cold is small (edge case)."""
        calibration = DTCalibration.create()
        
        result = calibration.calibrate(
            ts_cold=310.0,  # Warm vegetation
            ts_hot=320.0,   # Only 10K difference
            air_temperature=308.0,
            rn_hot=400.0,
            g_hot=20.0,
            et0_daily=4.0,
            rs_inst=500.0,
            rs_daily=18.0,
            rn_cold=420.0,
            g_cold=21.0
        )
        
        # Should still be in range, may have warning about small temperature difference
        assert 10.0 <= result.a_coefficient <= 50.0, \
            f"dt_a = {result.a_coefficient:.2f} W/m²/K outside physical range"
        
        print(f"Test case 4 PASSED: dt_a = {result.a_coefficient:.2f} W/m²/K")
    
    def test_dt_a_extreme_temperature_difference(self):
        """Test dt_a with large temperature difference."""
        calibration = DTCalibration.create()
        
        result = calibration.calibrate(
            ts_cold=285.0,  # Very cool vegetation
            ts_hot=350.0,   # Very hot dry soil
            air_temperature=283.0,
            rn_hot=550.0,
            g_hot=27.5,
            et0_daily=6.0,
            rs_inst=800.0,
            rs_daily=22.0,
            rn_cold=580.0,
            g_cold=29.0
        )
        
        assert 10.0 <= result.a_coefficient <= 50.0, \
            f"dt_a = {result.a_coefficient:.2f} W/m²/K outside physical range"
        
        print(f"Test case 5 PASSED: dt_a = {result.a_coefficient:.2f} W/m²/K")
    
    def test_b_coefficient_is_negative_ta(self):
        """Test that b coefficient equals -air_temperature (METRIC standard)."""
        calibration = DTCalibration.create()
        
        air_temp = 298.0
        result = calibration.calibrate(
            ts_cold=300.0,
            ts_hot=325.0,
            air_temperature=air_temp,
            rn_hot=450.0,
            g_hot=22.5,
            et0_daily=5.0,
            rs_inst=600.0,
            rs_daily=20.0,
            rn_cold=500.0,
            g_cold=25.0
        )
        
        # b should be approximately -air_temperature
        expected_b = -air_temp
        assert abs(result.b_coefficient - expected_b) < 1.0, \
            f"b = {result.b_coefficient:.2f}, expected ~{expected_b:.2f}"
        
        print(f"Test case 6 PASSED: b = {result.b_coefficient:.2f} K")
    
    def test_cold_pixel_dt_near_zero(self):
        """Test that dT_cold is near zero for well-watered cold pixel."""
        calibration = DTCalibration.create()
        
        result = calibration.calibrate(
            ts_cold=300.0,  # Cold pixel close to air temperature
            ts_hot=325.0,
            air_temperature=298.0,  # Air temp close to cold pixel
            rn_hot=450.0,
            g_hot=22.5,
            et0_daily=5.0,
            rs_inst=600.0,
            rs_daily=20.0,
            rn_cold=500.0,
            g_cold=25.0
        )
        
        # dT_cold should be small (2K in this case)
        assert result.dT_cold < 5.0, \
            f"dT_cold = {result.dT_cold:.2f} K, expected < 5 K for cold pixel"
        
        print(f"Test case 7 PASSED: dT_cold = {result.dT_cold:.2f} K")
    
    def test_calibration_clips_extreme_values(self):
        """Test that calibration clips extreme dt_a values to physical range."""
        calibration = DTCalibration.create()
        
        # Create conditions that might produce extreme dt_a
        # Very small dT_hot (close to zero)
        result = calibration.calibrate(
            ts_cold=298.0,
            ts_hot=298.5,  # Only 0.5K difference (below threshold)
            air_temperature=297.0,
            rn_hot=500.0,
            g_hot=25.0,
            et0_daily=5.0,
            rs_inst=600.0,
            rs_daily=20.0,
            rn_cold=520.0,
            g_cold=26.0
        )
        
        # Should be clipped to default (20.0) or within range
        assert 10.0 <= result.a_coefficient <= 50.0, \
            f"dt_a clipped to {result.a_coefficient:.2f} W/m²/K"
        
        print(f"Test case 8 PASSED: dt_a = {result.a_coefficient:.2f} W/m²/K (clipped)")


class TestCalibrationMultipleScenarios:
    """Run multiple calibration scenarios and verify physical ranges."""
    
    def test_calibration_series(self):
        """Test calibration with multiple random scenarios."""
        calibration = DTCalibration.create()
        
        np.random.seed(42)
        results = []
        
        for i in range(20):
            # Random but realistic conditions
            ts_cold = np.random.uniform(285, 310)
            ts_hot = ts_cold + np.random.uniform(15, 40)
            air_temperature = ts_cold - np.random.uniform(-3, 3)
            rn_hot = np.random.uniform(300, 600)
            g_hot = rn_hot * 0.05
            et0_daily = np.random.uniform(3, 8)
            rs_inst = np.random.uniform(400, 900)
            rs_daily = np.random.uniform(15, 25)
            rn_cold = np.random.uniform(400, 650)
            g_cold = rn_cold * 0.05
            
            result = calibration.calibrate(
                ts_cold=ts_cold,
                ts_hot=ts_hot,
                air_temperature=air_temperature,
                rn_hot=rn_hot,
                g_hot=g_hot,
                et0_daily=et0_daily,
                rs_inst=rs_inst,
                rs_daily=rs_daily,
                rn_cold=rn_cold,
                g_cold=g_cold
            )
            
            results.append({
                'dt_a': result.a_coefficient,
                'b': result.b_coefficient,
                'dT_hot': result.dT_hot,
                'dT_cold': result.dT_cold,
                'valid': result.valid
            })
            
            # Check physical range for each result
            assert 10.0 <= result.a_coefficient <= 50.0, \
                f"Scenario {i}: dt_a = {result.a_coefficient:.2f} outside range"
        
        # Analyze results
        dt_a_values = [r['dt_a'] for r in results]
        print(f"\nCalibration Series Results:")
        print(f"  dt_a mean: {np.mean(dt_a_values):.2f} W/m²/K")
        print(f"  dt_a std:  {np.std(dt_a_values):.2f} W/m²/K")
        print(f"  dt_a min:  {np.min(dt_a_values):.2f} W/m²/K")
        print(f"  dt_a max:  {np.max(dt_a_values):.2f} W/m²/K")
        print(f"  All 20 scenarios: dt_a in [10, 50] W/m²/K range")
        
        assert all(10.0 <= r['dt_a'] <= 50.0 for r in results), \
            "Some scenarios produced dt_a outside physical range"
        
        print("Test case 9 PASSED: All 20 random scenarios in physical range")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
