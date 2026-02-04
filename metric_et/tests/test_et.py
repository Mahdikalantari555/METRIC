"""
Unit tests for ET validation functions.

Tests ET range validation and calculations.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestInstantaneousET:
    """Test instantaneous ET calculations."""
    
    def test_le_to_et_conversion(self):
        """Test LE to instantaneous ET conversion."""
        LE = 200  # W/m2
        
        # ET_inst = LE / LATENT_HEAT_VAPORIZATION * 3600 (mm/hr)
        from metric_et.core.constants import LATENT_HEAT_VAPORIZATION, SECONDS_PER_HOUR
        
        ET_inst = LE / LATENT_HEAT_VAPORIZATION * SECONDS_PER_HOUR
        
        # ET should be positive
        assert ET_inst > 0
    
    def test_zero_le_zero_et(self):
        """Test zero LE gives zero ET."""
        from metric_et.core.constants import LATENT_HEAT_VAPORIZATION, SECONDS_PER_HOUR
        
        ET_inst = 0 / LATENT_HEAT_VAPORIZATION * SECONDS_PER_HOUR
        
        assert ET_inst == 0
    
    def test_daily_et_from_instantaneous(self):
        """Test daily ET from instantaneous ET."""
        # Instantaneous ET at satellite overpass (~10:30 local time)
        ET_inst = 0.8  # mm/hr
        
        # Daily ET is approximately instantaneous ET * daylight hours
        # For Landsat (10:30), approximately 0.5 * daily
        daylight_hours = 12
        ET_daily = ET_inst * daylight_hours
        
        assert ET_daily > 0
        # Reasonable daily ET range
        assert ET_daily < 15


class TestDailyET:
    """Test daily ET calculations."""
    
    def test_etrf_calculation(self):
        """Test reference ET fraction (ETrF) calculation."""
        ET_inst = 0.6  # mm/hr
        ETr_inst = 0.75  # mm/hr reference ET
        
        ETrF = ET_inst / ETr_inst
        
        # ETrF should be between 0 and ~1.5
        assert 0 < ETrF < 2.0
    
    def test_daily_et_from_etrf(self):
        """Test daily ET from ETrF."""
        ETrF = 0.8
        ETr_daily = 5.0  # mm/day
        
        ET_daily = ETrF * ETr_daily
        
        # ET should be less than reference ET (with some tolerance)
        assert ET_daily < ETr_daily * 1.5
        assert ET_daily > 0
    
    def test_daily_et_range_check(self):
        """Test ET range validation."""
        from metric_et.utils.validation import check_et_range
        
        # Valid ET
        valid_et = np.random.uniform(0, 10, (100, 100))
        is_valid, msg = check_et_range(valid_et)
        assert is_valid is True
        
        # Invalid ET (out of range)
        invalid_et = np.array([20, 5, 8])  # 20 is too high
        is_valid, msg = check_et_range(invalid_et)
        assert is_valid is False
    
    def test_daily_et_typical_values(self):
        """Test daily ET for typical conditions."""
        ETrF = 0.9
        ETr_daily = 6.0
        
        ET_daily = ETrF * ETr_daily
        
        # Should be reasonable
        assert 0 < ET_daily < 10
        assert ET_daily < ETr_daily * 1.5
    
    def test_daily_et_low_values(self):
        """Test daily ET for water-stressed conditions."""
        ETrF = 0.3
        ETr_daily = 6.0
        
        ET_daily = ETrF * ETr_daily
        
        # Should be lower than reference
        assert ET_daily < ETr_daily
    
    def test_daily_et_high_values(self):
        """Test daily ET for well-watered conditions."""
        ETrF = 1.1
        ETr_daily = 5.0
        
        ET_daily = ETrF * ETr_daily
        
        # Can exceed reference under ideal conditions
        assert ET_daily > ETr_daily


class TestETValidation:
    """Test ET quality assessment."""
    
    def test_et_quality_flags(self):
        """Test ET quality flag generation."""
        # Create sample ET array
        et = np.random.uniform(0, 8, (100, 100))
        
        # Quality assessment based on range
        valid_mask = (et >= 0) & (et <= 15)
        valid_pixels = np.sum(valid_mask)
        outlier_pixels = np.sum(~valid_mask)
        mean_et = np.mean(et[valid_mask])
        
        # Check quality assessment
        assert valid_pixels > 0
        assert mean_et > 0
    
    def test_et_range_distribution(self):
        """Test ET range distribution."""
        et = np.random.uniform(0, 8, (100, 100))
        
        # Check distribution
        min_et = np.min(et)
        max_et = np.max(et)
        mean_et = np.mean(et)
        
        assert min_et >= 0
        assert max_et <= 15
        assert mean_et > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
