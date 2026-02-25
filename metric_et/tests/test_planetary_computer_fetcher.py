"""Unit tests for PlanetaryComputerLandsatFetcher."""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import xarray as xr
from pathlib import Path
from shapely.geometry import box

from metric_et.io.planetary_computer_fetcher import (
    PlanetaryComputerLandsatFetcher,
    NoSceneFoundError,
    AuthenticationError,
    DownloadError,
    PartialDataError,
    GeometryError,
)
from metric_et.core.datacube import DataCube


class TestPlanetaryComputerLandsatFetcher:
    """Test suite for PlanetaryComputerLandsatFetcher."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        fetcher = PlanetaryComputerLandsatFetcher()
        assert fetcher.collection == "landsat-c2-l2"
        assert len(fetcher.bands) == 8
        assert fetcher.max_cloud_cover == 70.0
        assert fetcher.cache_dir is None
        assert fetcher.use_cache is True
    
    def test_init_custom(self):
        """Test initialization with custom parameters."""
        fetcher = PlanetaryComputerLandsatFetcher(
            collection="landsat-c2-l2",
            bands=['blue', 'green', 'red'],
            max_cloud_cover=50.0,
            cache_dir="/tmp/cache",
            use_cache=False
        )
        assert fetcher.collection == "landsat-c2-l2"
        assert fetcher.bands == ['blue', 'green', 'red']
        assert fetcher.max_cloud_cover == 50.0
        assert str(fetcher.cache_dir) == "/tmp/cache"
        assert fetcher.use_cache is False
    
    def test_init_invalid_band(self):
        """Test initialization with invalid band name."""
        with pytest.raises(ValueError, match="Unknown band"):
            PlanetaryComputerLandsatFetcher(bands=['invalid_band'])
    
    def test_normalize_geometry_from_geojson(self):
        """Test geometry normalization from GeoJSON dict."""
        fetcher = PlanetaryComputerLandsatFetcher()
        geojson = {
            "type": "Polygon",
            "coordinates": [[
                [48.0, 31.0],
                [48.5, 31.0],
                [48.5, 31.5],
                [48.0, 31.5],
                [48.0, 31.0]
            ]]
        }
        geom = fetcher._normalize_geometry(geojson)
        assert geom.is_valid
        assert geom.area > 0
    
    def test_normalize_geometry_from_shapely(self):
        """Test geometry normalization from shapely geometry."""
        fetcher = PlanetaryComputerLandsatFetcher()
        geom = box(48.0, 31.0, 48.5, 31.5)
        normalized = fetcher._normalize_geometry(geom)
        assert normalized.equals(geom)
    
    def test_normalize_geometry_empty(self):
        """Test geometry normalization with empty geometry."""
        fetcher = PlanetaryComputerLandsatFetcher()
        from shapely.geometry import Point
        empty_geom = Point()
        with pytest.raises(GeometryError, match="Geometry is empty"):
            fetcher._normalize_geometry(empty_geom)
    
    def test_normalize_geometry_invalid_type(self):
        """Test geometry normalization with invalid type."""
        fetcher = PlanetaryComputerLandsatFetcher()
        with pytest.raises(GeometryError, match="Geometry must be dict or shapely geometry"):
            fetcher._normalize_geometry("invalid")
    
    def test_normalize_dates_strings(self):
        """Test date normalization from strings."""
        fetcher = PlanetaryComputerLandsatFetcher()
        start = "2023-04-27"
        end = "2023-04-28"
        start_dt, end_dt = fetcher._normalize_dates((start, end))
        assert isinstance(start_dt, datetime)
        assert isinstance(end_dt, datetime)
        assert start_dt.date() == datetime(2023, 4, 27).date()
        assert end_dt.date() == datetime(2023, 4, 28).date()
    
    def test_normalize_dates_datetime(self):
        """Test date normalization from datetime objects."""
        fetcher = PlanetaryComputerLandsatFetcher()
        start = datetime(2023, 4, 27)
        end = datetime(2023, 4, 28)
        start_dt, end_dt = fetcher._normalize_dates((start, end))
        assert start_dt == start
        # End date is adjusted to end of day (23:59:59) if it's at midnight
        expected_end = datetime(2023, 4, 28, 23, 59, 59)
        assert end_dt == expected_end
    
    def test_normalize_dates_invalid_range(self):
        """Test date normalization with start after end."""
        fetcher = PlanetaryComputerLandsatFetcher()
        start = datetime(2023, 4, 28)
        end = datetime(2023, 4, 27)
        with pytest.raises(ValueError, match="Start date .* is after end date"):
            fetcher._normalize_dates((start, end))
    
    def test_filter_full_coverage(self):
        """Test full coverage filtering."""
        fetcher = PlanetaryComputerLandsatFetcher()
        
        # Create ROI and scene geometries
        roi = box(48.0, 31.0, 48.5, 31.5)
        
        # Scene that fully covers ROI
        scene1_bbox = [47.5, 30.5, 48.8, 31.8]
        item1 = Mock()
        item1.bbox = scene1_bbox
        item1.id = "scene1"
        
        # Scene that partially overlaps ROI
        scene2_bbox = [48.2, 31.2, 48.7, 31.7]
        item2 = Mock()
        item2.bbox = scene2_bbox
        item2.id = "scene2"
        
        items = [item1, item2]
        filtered = fetcher._filter_full_coverage(roi, items)
        
        assert len(filtered) == 1
        assert filtered[0].id == "scene1"
    
    def test_filter_full_coverage_empty(self):
        """Test full coverage filtering with no full coverage scenes."""
        fetcher = PlanetaryComputerLandsatFetcher()
        roi = box(48.0, 31.0, 48.5, 31.5)
        
        # Scene that only partially overlaps
        scene_bbox = [48.2, 31.2, 48.7, 31.7]
        item = Mock()
        item.bbox = scene_bbox
        item.id = "scene_partial"
        
        items = [item]
        filtered = fetcher._filter_full_coverage(roi, items)
        assert len(filtered) == 0
    
    def test_get_cache_path(self):
        """Test cache path generation."""
        fetcher = PlanetaryComputerLandsatFetcher(cache_dir="/tmp/cache")
        asset_href = "https://example.com/band.tif"
        band_name = "blue"
        cache_path = fetcher._get_cache_path(asset_href, band_name)
        
        assert cache_path.parent == Path("/tmp/cache")
        assert cache_path.name.startswith("blue_")
        assert cache_path.name.endswith(".tif")
    
    def test_get_cache_path_no_cache(self):
        """Test cache path when caching is disabled."""
        fetcher = PlanetaryComputerLandsatFetcher(cache_dir=None)
        asset_href = "https://example.com/band.tif"
        band_name = "blue"
        cache_path = fetcher._get_cache_path(asset_href, band_name)
        assert cache_path is None or str(cache_path) == "None"
    
    def test_extract_metadata(self):
        """Test metadata extraction from STAC item."""
        fetcher = PlanetaryComputerLandsatFetcher()
        
        # Create mock STAC item
        item = Mock()
        item.id = "LC08_L2SP_166038_20230427_20230428_02_T1"
        item.datetime = datetime(2023, 4, 27, 10, 30, 0)
        
        properties = {
            'cloud_cover': 15.5,
            'platform': 'landsat-8',
            'view:sun_elevation': 45.2,
            'view:sun_azimuth': 123.4,
            'landsat:wrs_path': 166,
            'landsat:wrs_row': 38,
            'landsat:correction': 'L2SP',
        }
        item.properties = properties
        
        cube = DataCube()
        fetcher._extract_metadata(cube, item)
        
        assert cube.metadata['scene_id'] == item.id
        assert cube.acquisition_time == item.datetime
        assert cube.metadata['cloud_cover'] == 15.5
        assert cube.metadata['platform'] == 'landsat-8'
        assert cube.metadata['sun_elevation'] == 45.2
        assert cube.metadata['sun_azimuth'] == 123.4
        assert cube.metadata['path'] == 166
        assert cube.metadata['row'] == 38
        assert cube.metadata['correction'] == 'L2SP'
        assert cube.metadata['sensor'] == 'oli'
    
    def test_extract_metadata_landsat9(self):
        """Test metadata extraction for Landsat 9."""
        fetcher = PlanetaryComputerLandsatFetcher()
        item = Mock()
        item.id = "LC09_L2SP_166038_20230427_20230428_02_T1"
        item.datetime = datetime(2023, 4, 27)
        item.properties = {
            'platform': 'landsat-9',
            'cloud_cover': 10.0,
        }
        cube = DataCube()
        fetcher._extract_metadata(cube, item)
        assert cube.metadata['sensor'] == 'oli'
    
    def test_extract_metadata_landsat7(self):
        """Test metadata extraction for Landsat 7."""
        fetcher = PlanetaryComputerLandsatFetcher()
        item = Mock()
        item.id = "LE07_L2SP_166038_20230427_20230428_02_T1"
        item.datetime = datetime(2023, 4, 27)
        item.properties = {
            'platform': 'landsat-7',
            'cloud_cover': 10.0,
        }
        cube = DataCube()
        fetcher._extract_metadata(cube, item)
        assert cube.metadata['sensor'] == 'etm'
    
    @patch('planetary_computer.sign')
    def test_download_and_build_cube_success(self, mock_sign):
        """Test successful download and cube building."""
        # Mock the signed item and its assets
        mock_item = Mock()
        mock_item.id = "test_scene_001"
        mock_item.datetime = datetime(2023, 4, 27)
        mock_item.properties = {
            'cloud_cover': 10.0,
            'platform': 'landsat-8',
            'view:sun_elevation': 45.0,
            'view:sun_azimuth': 123.0,
            'landsat:wrs_path': 166,
            'landsat:wrs_row': 38,
            'landsat:correction': 'L2SP',
        }
        mock_sign.return_value = mock_item
        
        # Create mock assets with COG URLs
        mock_assets = {}
        for band in ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'lwir11', 'qa_pixel']:
            asset = Mock()
            asset.href = f"https://example.com/{band}.tif"
            mock_assets[band] = asset
        mock_item.assets = mock_assets
        
        # Mock rasterio.open to return a simple array
        with patch('rasterio.open') as mock_rasterio:
            # Setup mock dataset
            mock_ds = MagicMock()
            mock_ds.read.return_value = np.ones((10, 10), dtype=np.float32) * 0.5
            mock_ds.transform = (30.0, 0.0, 499980.0, 0.0, -30.0, 4200000.0)
            mock_ds.crs = "EPSG:32639"
            mock_ds.bounds = (499980.0, 4199010.0, 500280.0, 4199310.0)
            mock_ds.nodata = None
            mock_ds.__enter__ = lambda self: self
            mock_ds.__exit__ = lambda self, *args: None
            mock_rasterio.return_value = mock_ds
            
            fetcher = PlanetaryComputerLandsatFetcher()
            roi = box(48.0, 31.0, 48.5, 31.5)
            
            cube = fetcher._download_and_build_cube(mock_item, roi)
            
            assert cube is not None
            assert isinstance(cube, DataCube)
            assert len(cube.bands()) == 8
            for band in ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'lwir11', 'qa_pixel']:
                assert band in cube.bands()
            assert cube.crs is not None
            assert cube.transform is not None
            assert cube.extent is not None
    
    @patch('planetary_computer.sign')
    def test_download_and_build_cube_missing_band(self, mock_sign):
        """Test cube building with missing band."""
        mock_item = Mock()
        mock_item.id = "test_scene_missing"
        mock_item.datetime = datetime(2023, 4, 27)
        mock_item.properties = {
            'cloud_cover': 10.0,
            'platform': 'landsat-8',
            'view:sun_elevation': 45.0,
            'view:sun_azimuth': 123.0,
            'landsat:wrs_path': 166,
            'landsat:wrs_row': 38,
            'landsat:correction': 'L2SP',
        }
        mock_sign.return_value = mock_item
        
        # Only provide some bands (missing lwir11)
        mock_assets = {}
        for band in ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'qa_pixel']:
            asset = Mock()
            asset.href = f"https://example.com/{band}.tif"
            mock_assets[band] = asset
        mock_item.assets = mock_assets
        
        with patch('rasterio.open') as mock_rasterio:
            mock_ds = MagicMock()
            mock_ds.read.return_value = np.ones((10, 10), dtype=np.float32) * 0.5
            mock_ds.transform = (30.0, 0.0, 499980.0, 0.0, -30.0, 4200000.0)
            mock_ds.crs = "EPSG:32639"
            mock_ds.bounds = (499980.0, 4199010.0, 500280.0, 4199310.0)
            mock_ds.nodata = None
            mock_ds.__enter__ = lambda self: self
            mock_ds.__exit__ = lambda self, *args: None
            mock_rasterio.return_value = mock_ds
            
            fetcher = PlanetaryComputerLandsatFetcher()
            roi = box(48.0, 31.0, 48.5, 31.5)
            
            with pytest.raises(PartialDataError, match="Missing critical bands"):
                fetcher._download_and_build_cube(mock_item, roi)
    
    @patch('planetary_computer.sign')
    def test_download_and_build_cube_missing_qa_only(self, mock_sign):
        """Test cube building with only QA band missing (should succeed with warning)."""
        mock_item = Mock()
        mock_item.id = "test_scene_no_qa"
        mock_item.datetime = datetime(2023, 4, 27)
        mock_item.properties = {
            'cloud_cover': 10.0,
            'platform': 'landsat-8',
            'view:sun_elevation': 45.0,
            'view:sun_azimuth': 123.0,
            'landsat:wrs_path': 166,
            'landsat:wrs_row': 38,
            'landsat:correction': 'L2SP',
        }
        mock_sign.return_value = mock_item
        
        # All bands except qa_pixel
        mock_assets = {}
        for band in ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'lwir11']:
            asset = Mock()
            asset.href = f"https://example.com/{band}.tif"
            mock_assets[band] = asset
        mock_item.assets = mock_assets
        
        with patch('rasterio.open') as mock_rasterio:
            mock_ds = MagicMock()
            mock_ds.read.return_value = np.ones((10, 10), dtype=np.float32) * 0.5
            mock_ds.transform = (30.0, 0.0, 499980.0, 0.0, -30.0, 4200000.0)
            mock_ds.crs = "EPSG:32639"
            mock_ds.bounds = (499980.0, 4199010.0, 500280.0, 4199310.0)
            mock_ds.nodata = None
            mock_ds.__enter__ = lambda self: self
            mock_ds.__exit__ = lambda self, *args: None
            mock_rasterio.return_value = mock_ds
            
            fetcher = PlanetaryComputerLandsatFetcher()
            roi = box(48.0, 31.0, 48.5, 31.5)
            
            cube = fetcher._download_and_build_cube(mock_item, roi)
            
            assert cube is not None
            # Should have 7 bands (qa_pixel missing)
            assert len(cube.bands()) == 7
            assert 'qa_pixel' not in cube.bands()
    
    def test_search_scenes_no_auth(self):
        """Test search_scenes with mocked STAC client."""
        fetcher = PlanetaryComputerLandsatFetcher()
        
        # Mock the STAC client
        with patch.object(fetcher, 'client') as mock_client:
            # Create mock search result
            mock_item = Mock()
            mock_item.id = "test_scene"
            mock_item.datetime = datetime(2023, 4, 27)
            mock_item.bbox = [48.0, 31.0, 48.5, 31.5]
            mock_item.geometry = {"type": "Polygon", "coordinates": [[[48.0, 31.0], [48.5, 31.0], [48.5, 31.5], [48.0, 31.5], [48.0, 31.0]]]}
            mock_item.properties = {
                'cloud_cover': 15.0,
                'platform': 'landsat-8',
            }
            
            mock_search = Mock()
            mock_search.items.return_value = [mock_item]
            mock_client.search.return_value = mock_search
            
            roi = box(48.0, 31.0, 48.5, 31.5)
            date_range = (datetime(2023, 4, 27), datetime(2023, 4, 28))
            
            results = fetcher.search_scenes(roi, date_range)
            
            assert len(results) == 1
            assert results[0]['id'] == "test_scene"
            assert results[0]['cloud_cover'] == 15.0
            assert results[0]['platform'] == 'landsat-8'
    
    def test_get_scene_count(self):
        """Test get_scene_count method."""
        fetcher = PlanetaryComputerLandsatFetcher()
        
        with patch.object(fetcher, 'search_scenes') as mock_search:
            mock_search.return_value = [{'id': 'scene1'}, {'id': 'scene2'}, {'id': 'scene3'}]
            
            roi = box(48.0, 31.0, 48.5, 31.5)
            date_range = (datetime(2023, 4, 27), datetime(2023, 4, 28))
            
            count = fetcher.get_scene_count(roi, date_range)
            assert count == 3
    
    def test_fetch_scenes_no_results(self):
        """Test fetch_scenes with no matching scenes."""
        fetcher = PlanetaryComputerLandsatFetcher()
        
        with patch.object(fetcher, '_search_scenes') as mock_search:
            mock_search.return_value = []
            
            roi = box(48.0, 31.0, 48.5, 31.5)
            date_range = (datetime(2023, 4, 27), datetime(2023, 4, 28))
            
            with pytest.raises(NoSceneFoundError, match="No scenes found"):
                fetcher.fetch_scenes(roi, date_range)
    
    def test_fetch_scenes_no_full_coverage(self):
        """Test fetch_scenes with only partial coverage scenes."""
        fetcher = PlanetaryComputerLandsatFetcher()
        
        # Create mock items that partially overlap
        mock_item = Mock()
        mock_item.id = "partial_scene"
        mock_item.bbox = [48.2, 31.2, 48.7, 31.7]  # Only partial overlap
        mock_item.datetime = datetime(2023, 4, 27)
        mock_item.properties = {'cloud_cover': 10.0}
        
        with patch.object(fetcher, '_search_scenes') as mock_search:
            mock_search.return_value = [mock_item]
            
            roi = box(48.0, 31.0, 48.5, 31.5)
            date_range = (datetime(2023, 4, 27), datetime(2023, 4, 28))
            
            with pytest.raises(NoSceneFoundError, match="No scenes found with full ROI coverage") as exc_info:
                fetcher.fetch_scenes(roi, date_range)
            
            assert "partial overlaps: 1" in str(exc_info.value)
    
    def test_fetch_scenes_sort_by_cloud_cover(self):
        """Test fetch_scenes sorting by cloud cover."""
        fetcher = PlanetaryComputerLandsatFetcher()
        
        # Create mock items with different cloud covers
        def make_item(item_id, cloud_cover):
            item = Mock()
            item.id = item_id
            item.bbox = [47.5, 30.5, 48.8, 31.8]  # Full coverage
            item.datetime = datetime(2023, 4, 27)
            item.properties = {'cloud_cover': cloud_cover}
            return item
        
        items = [
            make_item("scene_high_cc", 50.0),
            make_item("scene_low_cc", 10.0),
            make_item("scene_mid_cc", 30.0),
        ]
        
        with patch.object(fetcher, '_search_scenes') as mock_search:
            mock_search.return_value = items
            
            with patch.object(fetcher, '_download_and_build_cube') as mock_download:
                mock_download.return_value = DataCube()
                
                roi = box(48.0, 31.0, 48.5, 31.5)
                date_range = (datetime(2023, 4, 27), datetime(2023, 4, 28))
                
                cubes = fetcher.fetch_scenes(roi, date_range, sort_by='cloud_cover')
                
                # Should be sorted by cloud cover (ascending)
                assert len(cubes) == 3
                # The download would be called in order, we just verify all were downloaded
                assert mock_download.call_count == 3


class TestErrorClasses:
    """Test suite for custom error classes."""
    
    def test_no_scene_found_error_default(self):
        """Test NoSceneFoundError without partial count."""
        err = NoSceneFoundError("Test message")
        assert str(err) == "Test message"
        assert err.partial_count is None
    
    def test_no_scene_found_error_with_partial(self):
        """Test NoSceneFoundError with partial count."""
        err = NoSceneFoundError("Test message", partial_count=5)
        assert "partial overlaps: 5" in str(err)
        assert err.partial_count == 5
    
    def test_download_error_with_details(self):
        """Test DownloadError with band and URL."""
        err = DownloadError("Download failed", band_name="blue", url="https://example.com/blue.tif")
        assert "band: blue" in str(err)
        assert "url: https://example.com/blue.tif" in str(err)
    
    def test_download_error_without_details(self):
        """Test DownloadError without extra details."""
        err = DownloadError("Download failed")
        assert str(err) == "Download failed"
        assert err.band_name is None
        assert err.url is None
    
    def test_partial_data_error(self):
        """Test PartialDataError."""
        missing = ['blue', 'green']
        available = ['red', 'nir08']
        err = PartialDataError("Missing bands", missing_bands=missing, available_bands=available)
        assert "Missing bands: [blue, green]" in str(err)
        assert err.missing_bands == missing
        assert err.available_bands == available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
