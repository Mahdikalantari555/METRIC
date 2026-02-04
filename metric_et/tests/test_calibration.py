"""
Unit tests for calibration validation.

Tests calibration parameter validation functions.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestCalibrationValidation:
    """Test calibration validation functions."""
    
    def test_valid_calibration_params(self):
        """Test validation of valid calibration parameters."""
        from metric_et.utils.validation import validate_calibration_params
        
        is_valid, msg = validate_calibration_params(1.5, 280.0)
        assert is_valid is True
    
    def test_invalid_negative_slope(self):
        """Test validation rejects negative slope."""
        from metric_et.utils.validation import validate_calibration_params
        
        is_valid, msg = validate_calibration_params(-1.0, 280.0)
        assert is_valid is False
        assert "positive" in msg.lower()
    
    def test_invalid_zero_slope(self):
        """Test validation rejects zero slope."""
        from metric_et.utils.validation import validate_calibration_params
        
        is_valid, msg = validate_calibration_params(0, 280.0)
        assert is_valid is False
    
    def test_invalid_nan_values(self):
        """Test validation rejects NaN values."""
        from metric_et.utils.validation import validate_calibration_params
        
        is_valid, msg = validate_calibration_params(float('nan'), 280.0)
        assert is_valid is False
        assert "finite" in msg.lower() or "numeric" in msg.lower()
    
    def test_invalid_inf_values(self):
        """Test validation rejects infinite values."""
        from metric_et.utils.validation import validate_calibration_params
        
        is_valid, msg = validate_calibration_params(float('inf'), 280.0)
        assert is_valid is False


class TestInputValidation:
    """Test overall input validation."""
    
    def test_validate_input_data_complete(self):
        """Test validation with complete input data."""
        from metric_et.utils.validation import validate_input_data
        
        ndvi = np.random.uniform(-0.2, 0.9, (100, 100))
        albedo = np.random.uniform(0.1, 0.4, (100, 100))
        temperature = np.random.uniform(280, 320, (100, 100))
        net_radiation = np.random.uniform(400, 600, (100, 100))
        
        is_valid, msg = validate_input_data(
            ndvi=ndvi,
            albedo=albedo,
            temperature=temperature,
            net_radiation=net_radiation
        )
        
        assert is_valid is True
    
    def test_validate_input_data_missing_ndvi(self):
        """Test validation with missing NDVI."""
        from metric_et.utils.validation import validate_input_data
        
        albedo = np.random.uniform(0.1, 0.4, (100, 100))
        temperature = np.random.uniform(280, 320, (100, 100))
        
        is_valid, msg = validate_input_data(
            albedo=albedo,
            temperature=temperature
        )
        
        assert is_valid is False
        assert "Missing" in msg
    
    def test_validate_input_data_invalid_ndvi(self):
        """Test validation with invalid NDVI."""
        from metric_et.utils.validation import validate_input_data
        
        ndvi = np.array([1.5, 0.5, -0.5])  # Out of range
        albedo = np.random.uniform(0.1, 0.4, (100, 100))
        temperature = np.random.uniform(280, 320, (100, 100))
        net_radiation = np.random.uniform(400, 600, (100, 100))
        
        is_valid, msg = validate_input_data(
            ndvi=ndvi,
            albedo=albedo,
            temperature=temperature,
            net_radiation=net_radiation
        )
        
        assert is_valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
