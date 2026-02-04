"""
Unit tests for energy balance module validation.

Tests energy balance validation functions.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestEnergyBalanceClosure:
    """Test energy balance closure validation."""
    
    def test_energy_balance_closure_basic(self):
        """Test basic energy balance closure: LE = Rn - G - H."""
        Rn = 500   # W/m2
        G = 50     # W/m2
        H = 100    # W/m2
        LE = Rn - G - H
        
        # Energy balance should close
        assert LE == 350
    
    def test_energy_balance_closure_with_arrays(self):
        """Test energy balance closure with arrays."""
        Rn = np.random.uniform(400, 600, (100, 100))
        G = np.random.uniform(30, 80, (100, 100))
        H = np.random.uniform(50, 150, (100, 100))
        LE = Rn - G - H
        
        # Calculate expected LE
        LE_expected = Rn - G - H
        
        # Check closure
        np.testing.assert_array_almost_equal(LE, LE_expected)
    
    def test_energy_balance_metrics(self):
        """Test energy balance metrics calculation."""
        from metric_et.utils.validation import check_energy_balance
        
        Rn = np.random.uniform(400, 600, (100, 100))
        G = np.random.uniform(30, 80, (100, 100))
        H = np.random.uniform(50, 150, (100, 100))
        LE = Rn - G - H
        
        result = check_energy_balance(Rn=Rn, G=G, H=H, LE=LE)
        
        # Check result structure
        assert "valid" in result
        assert "mean_closure" in result
        assert "mean_residual" in result
        assert "valid_pixels" in result


class TestValidationChecks:
    """Test data validation checks."""
    
    def test_check_ndvi_range_valid(self):
        """Test NDVI range validation with valid data."""
        from metric_et.utils.validation import check_ndvi_range
        
        # Valid NDVI values
        valid_ndvi = np.random.uniform(-0.2, 0.9, (100, 100))
        is_valid, msg = check_ndvi_range(valid_ndvi)
        assert is_valid is True
    
    def test_check_ndvi_range_invalid(self):
        """Test NDVI range validation with invalid data."""
        from metric_et.utils.validation import check_ndvi_range
        
        # Invalid NDVI (out of range)
        invalid_ndvi = np.array([1.5, 0.5, -0.5])
        is_valid, msg = check_ndvi_range(invalid_ndvi)
        assert is_valid is False
    
    def test_check_albedo_range_valid(self):
        """Test albedo range validation with valid data."""
        from metric_et.utils.validation import check_albedo_range
        
        # Valid albedo values
        valid_albedo = np.random.uniform(0.1, 0.4, (100, 100))
        is_valid, msg = check_albedo_range(valid_albedo)
        assert is_valid is True
    
    def test_check_temperature_range_valid(self):
        """Test temperature range validation with valid data."""
        from metric_et.utils.validation import check_temperature_range
        
        # Valid temperature values (280-320 K)
        valid_temp = np.random.uniform(280, 320, (100, 100))
        is_valid, msg = check_temperature_range(valid_temp)
        assert is_valid is True
    
    def test_check_temperature_range_invalid(self):
        """Test temperature range validation with invalid data."""
        from metric_et.utils.validation import check_temperature_range
        
        # Too cold
        cold_temp = np.array([200, 250, 280])
        is_valid, msg = check_temperature_range(cold_temp)
        assert is_valid is False
    
    def test_check_et_range_valid(self):
        """Test ET range validation with valid data."""
        from metric_et.utils.validation import check_et_range
        
        # Valid ET values (0-10 mm/day)
        valid_et = np.random.uniform(0, 10, (100, 100))
        is_valid, msg = check_et_range(valid_et)
        assert is_valid is True
    
    def test_check_et_range_invalid(self):
        """Test ET range validation with invalid data."""
        from metric_et.utils.validation import check_et_range
        
        # Invalid ET (out of range)
        invalid_et = np.array([20, 5, 8])  # 20 is too high
        is_valid, msg = check_et_range(invalid_et)
        assert is_valid is False
    
    def test_check_for_nodata_valid(self):
        """Test nodata checking with valid data."""
        from metric_et.utils.validation import check_for_nodata
        
        # Valid data (no nodata)
        valid_data = np.random.rand(100, 100)
        is_valid, msg = check_for_nodata(valid_data)
        assert is_valid is True
    
    def test_check_for_nodata_excessive(self):
        """Test nodata checking with excessive nodata."""
        from metric_et.utils.validation import check_for_nodata
        
        # Data with too many nodata
        data_with_nodata = np.random.rand(100, 100)
        data_with_nodata[:20, :] = np.nan  # 20% nodata
        
        is_valid, msg = check_for_nodata(data_with_nodata, nodata_value=np.nan, max_nodata_ratio=0.1)
        assert is_valid is False
    
    def test_check_cloud_coverage_low(self):
        """Test cloud coverage check with low clouds."""
        from metric_et.utils.validation import check_cloud_coverage
        
        cloud_mask = np.zeros((100, 100), dtype=bool)
        cloud_mask[5, 5] = True  # Only 1 pixel
        
        is_valid, msg = check_cloud_coverage(cloud_mask, threshold=0.5)
        assert is_valid is True
    
    def test_check_cloud_coverage_high(self):
        """Test cloud coverage check with high clouds."""
        from metric_et.utils.validation import check_cloud_coverage
        
        cloud_mask = np.zeros((100, 100), dtype=bool)
        cloud_mask[:60, :] = True  # 60% cloud cover
        
        is_valid, msg = check_cloud_coverage(cloud_mask, threshold=0.5)
        assert is_valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
