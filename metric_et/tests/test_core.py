"""
Unit tests for core module of METRIC ETa pipeline.

Tests DataCube, constants, and core functionality.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestConstants:
    """Test physical constants are correct."""
    
    def test_stefan_boltzmann_constant(self):
        """Test Stefan-Boltzmann constant value."""
        from metric_et.core.constants import STEFAN_BOLTZMANN
        expected = 5.670374419e-8
        assert abs(STEFAN_BOLTZMANN - expected) < 1e-15
    
    def test_von_karman_constant(self):
        """Test von Karman constant value."""
        from metric_et.core.constants import VON_KARMAN
        expected = 0.41
        assert abs(VON_KARMAN - expected) < 0.01
    
    def test_solar_constant(self):
        """Test solar constant is reasonable."""
        from metric_et.core.constants import SOLAR_CONSTANT
        assert 1300 < SOLAR_CONSTANT < 1450
    
    def test_latent_heat_vaporization(self):
        """Test latent heat of vaporization is reasonable."""
        from metric_et.core.constants import LATENT_HEAT_VAPORIZATION
        # At 25C, L ~ 2.44e6 J/kg
        assert 2.4e6 < LATENT_HEAT_VAPORIZATION < 2.5e6
    
    def test_specific_heat_air(self):
        """Test specific heat of air is reasonable."""
        from metric_et.core.constants import AIR_SPECIFIC_HEAT
        # ~1013 J/(kg.K)
        assert 1000 < AIR_SPECIFIC_HEAT < 1030
    
    def test_air_density(self):
        """Test air density is reasonable."""
        from metric_et.core.constants import AIR_DENSITY
        # ~1.225 kg/m3 at sea level
        assert 1.1 < AIR_DENSITY < 1.3
    
    def test_kelvin_offset(self):
        """Test Kelvin to Celsius offset."""
        from metric_et.core.constants import CELSIUS_TO_KELVIN
        assert CELSIUS_TO_KELVIN == 273.15
    
    def test_degrees_to_radians(self):
        """Test degree to radian conversion constant."""
        from metric_et.core.constants import DEG_TO_RAD
        expected = np.pi / 180.0
        assert abs(DEG_TO_RAD - expected) < 1e-15
    
    def test_radians_to_degrees(self):
        """Test radian to degree conversion constant."""
        from metric_et.core.constants import RAD_TO_DEG
        expected = 180.0 / np.pi
        assert abs(RAD_TO_DEG - expected) < 1e-15


class TestDataCube:
    """Test DataCube initialization and methods."""
    
    def test_datacube_creation(self):
        """Test DataCube initialization."""
        from metric_et.core.datacube import DataCube
        
        cube = DataCube()
        assert cube is not None
    
    def test_datacube_add_bands(self):
        """Test adding bands to DataCube."""
        from metric_et.core.datacube import DataCube
        
        cube = DataCube()
        data = np.random.rand(10, 10)
        cube.add("blue", data)
        
        # Check that the band was added
        assert "blue" in cube.data
        assert cube.get("blue") is not None
    
    def test_datacube_metadata(self):
        """Test DataCube metadata operations."""
        from metric_et.core.datacube import DataCube
        
        cube = DataCube()
        cube.metadata["scene_id"] = "test_scene_001"
        
        assert cube.metadata.get("scene_id") == "test_scene_001"
    
    def test_datacube_shape(self):
        """Test DataCube array shapes are consistent."""
        from metric_et.core.datacube import DataCube
        
        cube = DataCube()
        shape = (10, 10)
        
        cube.add("blue", np.random.rand(*shape))
        cube.add("red", np.random.rand(*shape))
        
        assert cube.get("blue").shape == shape
        assert cube.get("red").shape == shape


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_degrees_to_radians(self):
        """Test degree to radian conversion."""
        from metric_et.core.constants import DEG_TO_RAD
        import numpy as np
        
        assert abs(DEG_TO_RAD * 0 - 0) < 1e-10
        assert abs(DEG_TO_RAD * 90 - np.pi/2) < 1e-10
        assert abs(DEG_TO_RAD * 180 - np.pi) < 1e-10
    
    def test_radians_to_degrees(self):
        """Test radian to degree conversion."""
        from metric_et.core.constants import RAD_TO_DEG
        import numpy as np
        
        assert abs(RAD_TO_DEG * 0 - 0) < 1e-10
        assert abs(RAD_TO_DEG * np.pi/2 - 90) < 1e-10
        assert abs(RAD_TO_DEG * np.pi - 180) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
