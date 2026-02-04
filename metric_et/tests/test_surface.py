"""
Unit tests for surface validation functions.

Tests NDVI, albedo, and temperature validation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestNDVICalculation:
    """Test NDVI calculation validation."""
    
    def test_ndvi_calculation_basic(self):
        """Test NDVI from red and NIR."""
        red = 0.1
        nir = 0.5
        
        ndvi = (nir - red) / (nir + red)
        
        # NDVI = (NIR - Red) / (NIR + Red)
        expected = (0.5 - 0.1) / (0.5 + 0.1)
        assert abs(ndvi - expected) < 1e-10
    
    def test_ndvi_positive_values(self):
        """Test NDVI for vegetated pixels."""
        red = 0.1
        nir = 0.5
        
        ndvi = (nir - red) / (nir + red)
        assert 0 < ndvi < 1
    
    def test_ndvi_negative_values(self):
        """Test NDVI for water/non-vegetation."""
        red = 0.3
        nir = 0.1
        
        ndvi = (nir - red) / (nir + red)
        assert -1 < ndvi < 0
    
    def test_ndvi_zero_values(self):
        """Test NDVI when NIR equals Red."""
        red = 0.2
        nir = 0.2
        
        ndvi = (nir - red) / (nir + red)
        assert ndvi == 0
    
    def test_ndvi_range_check(self):
        """Test NDVI range validation."""
        from metric_et.utils.validation import check_ndvi_range
        
        # Valid NDVI
        valid_ndvi = np.random.uniform(-0.2, 0.9, (100, 100))
        is_valid, msg = check_ndvi_range(valid_ndvi)
        assert is_valid is True
        
        # Invalid NDVI (out of range)
        invalid_ndvi = np.array([1.5, 0.5, -0.5])
        is_valid, msg = check_ndvi_range(invalid_ndvi)
        assert is_valid is False


class TestAlbedoCalculation:
    """Test albedo validation."""
    
    def test_albedo_calculation_basic(self):
        """Test albedo calculation formula."""
        blue = 0.1
        red = 0.15
        nir = 0.4
        swir1 = 0.1
        swir2 = 0.05
        
        # Using TMAD coefficients (Collection 2)
        albedo = (0.254 * blue + 0.149 * red + 0.295 * nir + 
                  0.243 * swir1 + 0.091 * swir2 + 0.066)
        
        # Albedo should be between 0 and 1
        assert 0 <= albedo <= 1
    
    def test_albedo_range_check(self):
        """Test albedo range validation."""
        from metric_et.utils.validation import check_albedo_range
        
        # Valid albedo
        valid_albedo = np.random.uniform(0.1, 0.4, (100, 100))
        is_valid, msg = check_albedo_range(valid_albedo)
        assert is_valid is True
        
        # Invalid albedo (out of range)
        invalid_albedo = np.array([1.5, 0.5, 0.2])
        is_valid, msg = check_albedo_range(invalid_albedo)
        assert is_valid is False


class TestTemperatureValidation:
    """Test temperature validation."""
    
    def test_temperature_range_check_valid(self):
        """Test temperature range validation with valid data."""
        from metric_et.utils.validation import check_temperature_range
        
        # Valid temperature values
        valid_temp = np.random.uniform(280, 320, (100, 100))
        is_valid, msg = check_temperature_range(valid_temp)
        assert is_valid is True
    
    def test_temperature_range_check_low(self):
        """Test temperature range validation with too low values."""
        from metric_et.utils.validation import check_temperature_range
        
        # Too cold
        cold_temp = np.array([200, 250, 280])
        is_valid, msg = check_temperature_range(cold_temp)
        assert is_valid is False
    
    def test_temperature_range_check_high(self):
        """Test temperature range validation with too high values."""
        from metric_et.utils.validation import check_temperature_range
        
        # Too hot
        hot_temp = np.array([350, 400, 420])
        is_valid, msg = check_temperature_range(hot_temp)
        assert is_valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
