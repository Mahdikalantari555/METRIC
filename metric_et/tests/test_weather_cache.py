"""
Unit tests for weather cache functionality.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from metric_et.io.weather_cache import WeatherCache


class TestWeatherCache:
    """Test cases for WeatherCache class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = WeatherCache(cache_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        # Close any open connections
        if hasattr(self.cache, '_conn'):
            self.cache._conn.close()

        # Remove temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test cache initialization."""
        assert os.path.exists(self.cache.db_path)
        assert self.cache._ensure_cache_dir() is None  # Should not raise

    def test_save_and_load_scene_weather(self):
        """Test saving and loading scene weather data."""
        scene_id = "LC09_L2SP_165039_20241015_02_T1"
        scene_date = "2024-10-15"
        bbox = [-122.0, 37.0, -121.0, 38.0]
        grid_spacing = 9.0

        # Sample weather data
        weather_data = {
            'grid_points': [(37.5, -121.5), (37.5, -121.0)],
            'weather_variables': {
                'temperature_2m': [25.3, 24.8],
                'relative_humidity_2m': [45.2, 46.1],
                'wind_speed_10m': [2.1, 2.3],
                'surface_pressure': [1013.2, 1012.8],
                'shortwave_radiation': [850.5, 842.1],
                'et0_fao_evapotranspiration': [5.2, 5.1],
                'shortwave_radiation_sum': [25.3, 25.1]
            }
        }

        # Save data
        success = self.cache.save_scene_weather(scene_id, scene_date, weather_data, bbox, grid_spacing)
        assert success

        # Load data
        loaded_data = self.cache.load_scene_weather(scene_id)
        assert loaded_data is not None
        # JSON serialization converts tuples to lists
        assert loaded_data['grid_points'] == [list(point) for point in weather_data['grid_points']]
        assert loaded_data['weather_variables']['temperature_2m'] == weather_data['weather_variables']['temperature_2m']

    def test_cache_miss(self):
        """Test cache miss for non-existent scene."""
        loaded_data = self.cache.load_scene_weather("non_existent_scene")
        assert loaded_data is None

    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.cache.get_cache_stats()
        assert isinstance(stats, dict)
        assert 'total_scenes' in stats
        assert stats['total_scenes'] == 0  # Empty cache

    def test_list_cached_scenes(self):
        """Test listing cached scenes."""
        scenes = self.cache.list_cached_scenes()
        assert isinstance(scenes, list)
        assert len(scenes) == 0  # Empty cache

    def test_clear_cache(self):
        """Test cache clearing."""
        # Add some data first
        scene_id = "test_scene"
        weather_data = {
            'grid_points': [(0.0, 0.0)],
            'weather_variables': {'temperature_2m': [20.0]}
        }
        self.cache.save_scene_weather(scene_id, "2024-01-01", weather_data, [0, 0, 1, 1], 9.0)

        # Verify data exists
        assert self.cache.load_scene_weather(scene_id) is not None

        # Clear cache
        cleared = self.cache.clear_cache()
        assert cleared == 1

        # Verify data is gone
        assert self.cache.load_scene_weather(scene_id) is None

    def test_compression(self):
        """Test data compression and decompression."""
        test_data = {"large_data": "x" * 10000, "numbers": list(range(1000))}

        compressed = self.cache._compress_data(test_data)
        assert isinstance(compressed, bytes)
        assert len(compressed) < len(json.dumps(test_data).encode())  # Should be compressed

        decompressed = self.cache._decompress_data(compressed)
        assert decompressed == test_data

    def test_fallback_scene_id_generation(self):
        """Test fallback scene ID generation."""
        bbox = [-122.0, 37.0, -121.0, 38.0]
        scene_date = "2024-10-15"
        grid_spacing = 9.0

        fallback_id = self.cache._generate_fallback_scene_id(bbox, scene_date, grid_spacing)
        assert scene_date in fallback_id
        assert "9.0" in fallback_id
        # Should be deterministic
        fallback_id2 = self.cache._generate_fallback_scene_id(bbox, scene_date, grid_spacing)
        assert fallback_id == fallback_id2

    @patch('sqlite3.connect')
    def test_database_error_handling(self, mock_connect):
        """Test error handling for database operations."""
        mock_connect.side_effect = Exception("Database error")

        # Should handle errors gracefully
        success = self.cache.save_scene_weather("test", "2024-01-01", {}, [0, 0, 1, 1], 9.0)
        assert not success

        loaded = self.cache.load_scene_weather("test")
        assert loaded is None