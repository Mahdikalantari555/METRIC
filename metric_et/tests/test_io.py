"""
Unit tests for IO module of METRIC ETa pipeline.

Tests IO validation and scene path validation.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestLandsatReader:
    """Test Landsat data reader validation."""
    
    def test_validate_scene_path_valid(self, sample_scene_path):
        """Test scene path validation with valid path."""
        from metric_et.utils.validation import validate_scene_path
        
        is_valid, msg = validate_scene_path(sample_scene_path)
        assert is_valid is True
    
    def test_validate_scene_path_invalid(self, tmp_path):
        """Test scene path validation with invalid path."""
        from metric_et.utils.validation import validate_scene_path
        
        invalid_path = tmp_path / "nonexistent_scene"
        is_valid, msg = validate_scene_path(invalid_path)
        assert is_valid is False
    
    def test_validate_scene_path_missing_files(self, tmp_path):
        """Test scene path validation with missing required files."""
        from metric_et.utils.validation import validate_scene_path
        
        # Create directory without required files
        scene_path = tmp_path / "incomplete_scene"
        scene_path.mkdir()
        (scene_path / "blue.tif").touch()
        
        is_valid, msg = validate_scene_path(scene_path)
        assert is_valid is False
        assert "Missing required file" in msg


class TestMeteoReader:
    """Test meteorological data reader validation."""
    
    def test_validate_weather_data_valid(self, sample_weather_dataframe):
        """Test weather data validation with valid data."""
        from metric_et.utils.validation import validate_weather_data
        
        is_valid, msg = validate_weather_data(sample_weather_dataframe)
        assert is_valid is True
    
    def test_validate_weather_data_empty(self):
        """Test weather data validation with empty data."""
        from metric_et.utils.validation import validate_weather_data
        
        empty_df = pd.DataFrame()
        
        is_valid, msg = validate_weather_data(empty_df)
        assert is_valid is False
    
    def test_validate_weather_data_missing_columns(self):
        """Test weather data validation with missing columns."""
        from metric_et.utils.validation import validate_weather_data
        
        df = pd.DataFrame({"temperature": [280, 285, 290]})
        
        is_valid, msg = validate_weather_data(df)
        assert is_valid is False


class TestValidationUtilities:
    """Test additional validation utilities."""
    
    def test_check_for_nodata(self):
        """Test nodata checking."""
        from metric_et.utils.validation import check_for_nodata
        
        # Valid data
        valid_data = np.random.rand(100, 100)
        is_valid, msg = check_for_nodata(valid_data)
        assert is_valid is True
        
        # Data with NaN
        data_with_nan = np.random.rand(100, 100)
        data_with_nan[0, 0] = np.nan
        is_valid, msg = check_for_nodata(data_with_nan, nodata_value=np.nan)
        assert is_valid is True  # Only 1 nodata is OK
    
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
