"""
Pytest configuration and fixtures for METRIC ETa pipeline tests.

Provides common fixtures for testing.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_datacube():
    """Create sample DataCube for testing."""
    from metric_et.core.datacube import DataCube
    
    cube = DataCube()
    
    # Create sample bands (10x10 arrays)
    shape = (10, 10)
    
    cube.add("blue", np.random.rand(*shape) * 0.2)
    cube.add("green", np.random.rand(*shape) * 0.3)
    cube.add("red", np.random.rand(*shape) * 0.2)
    cube.add("nir08", np.random.rand(*shape) * 0.6)
    cube.add("swir16", np.random.rand(*shape) * 0.2)
    cube.add("swir22", np.random.rand(*shape) * 0.1)
    cube.add("lwir11", np.random.rand(*shape) * 20 + 280)  # 280-300 K
    cube.add("qa_pixel", np.zeros(shape, dtype=np.uint8))
    
    # Add metadata
    cube.metadata["scene_id"] = "test_scene_001"
    cube.metadata["acquisition_date"] = "2024-01-15"
    cube.metadata["wrs_path"] = 166
    cube.metadata["wrs_row"] = 38
    
    return cube


@pytest.fixture
def sample_weather():
    """Create sample weather data for testing."""
    return {
        "temperature_2m": np.random.uniform(280, 300),  # K
        "relative_humidity": np.random.uniform(30, 80),  # %
        "wind_speed": np.random.uniform(2, 8),  # m/s
        "wind_direction": np.random.uniform(0, 360),  # degrees
        "pressure": np.random.uniform(990, 1015),  # hPa
        "solar_radiation": np.random.uniform(400, 800),  # W/m2
        "datetime": "2024-01-15T10:00:00"
    }


@pytest.fixture
def sample_weather_dataframe():
    """Create sample weather DataFrame for testing."""
    data = {
        "temperature_2m": [285, 287, 289, 291, 293],
        "relative_humidity": [60, 55, 50, 45, 40],
        "wind_speed": [3, 4, 5, 6, 7],
        "wind_direction": [90, 120, 150, 180, 210],
        "pressure": [1010, 1008, 1005, 1003, 1000],
        "solar_radiation": [500, 550, 600, 650, 700]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_ndvi():
    """Create sample NDVI array for testing."""
    # Create a realistic NDVI pattern
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    xx, yy = np.meshgrid(x, y)
    
    # Circular pattern with some variation
    ndvi = 0.5 * np.exp(-(xx**2 + yy**2)) + np.random.uniform(-0.1, 0.1, (100, 100))
    
    # Clip to valid range
    ndvi = np.clip(ndvi, -0.2, 0.9)
    
    return ndvi


@pytest.fixture
def sample_albedo():
    """Create sample albedo array for testing."""
    albedo = np.random.uniform(0.1, 0.4, (100, 100))
    return albedo


@pytest.fixture
def sample_temperature():
    """Create sample surface temperature array for testing."""
    # Create a temperature pattern (280-320 K)
    temp = 300 + np.random.uniform(-15, 15, (100, 100))
    return temp


@pytest.fixture
def sample_energy_fluxes():
    """Create sample energy flux arrays for testing."""
    Rn = np.random.uniform(400, 600, (100, 100))  # Net radiation
    G = np.random.uniform(30, 80, (100, 100))     # Soil heat flux
    H = np.random.uniform(50, 150, (100, 100))    # Sensible heat flux
    LE = Rn - G - H                               # Latent heat flux (energy balance)
    
    return {
        "Rn": Rn,
        "G": G,
        "H": H,
        "LE": LE
    }


@pytest.fixture
def sample_calibration_params():
    """Create sample calibration parameters for testing."""
    return {
        "a": 1.5,
        "b": 280.0,
        "Ts_cold": 295.0,
        "Ts_hot": 315.0,
        "dT_cold": 2.0,
        "dT_hot": 15.0
    }


@pytest.fixture
def sample_scene_path(tmp_path):
    """Create a mock Landsat scene directory for testing."""
    scene_path = tmp_path / "landsat_20240115_166_038"
    scene_path.mkdir()
    
    # Create mock band files
    bands = ["blue", "green", "red", "nir08", "swir16", "swir22", "lwir11", "qa_pixel"]
    for band in bands:
        (scene_path / f"{band}.tif").touch()
    
    # Create MTL file
    mtl_content = """{
    "LANDSAT_METADATA_FILE": {
        "IMAGE_ATTRIBUTES": {
            "CLOUD_COVER": 15.5,
            "DATE_ACQUIRED": "2024-01-15"
        },
        "RADIOMETRIC_RESCALING": {
            "RADIANCE_ADD_BAND_1": -0.1,
            "RADIANCE_MULT_BAND_1": 0.001
        }
    }
}"""
    (scene_path / "MTL.json").write_text(mtl_content)
    
    return str(scene_path)


@pytest.fixture
def sample_weather_csv(tmp_path):
    """Create a mock weather CSV file for testing."""
    data = """temperature_2m,relative_humidity,wind_speed,wind_direction,pressure,solar_radiation
285.5,60.0,3.2,180.0,1010.0,500.0
286.0,58.0,3.5,185.0,1009.0,550.0
287.0,55.0,4.0,190.0,1008.0,600.0
288.0,52.0,4.2,195.0,1007.0,650.0
289.0,50.0,4.5,200.0,1005.0,700.0"""
    
    csv_path = tmp_path / "weather_data.csv"
    csv_path.write_text(data)
    return str(csv_path)


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Configure logging for tests."""
    import logging
    logging.getLogger("metric_et").setLevel(logging.DEBUG)
    yield
    logging.getLogger("metric_et").setLevel(logging.WARNING)
