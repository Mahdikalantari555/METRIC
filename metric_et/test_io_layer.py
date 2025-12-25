#!/usr/bin/env python
"""Test script for IO layer modules."""

from datetime import datetime
import sys

def test_landsat_reader():
    """Test LandsatReader functionality."""
    from metric_et.io import LandsatReader
    
    print('Testing LandsatReader...')
    reader = LandsatReader()
    cube = reader.load('data/landsat_20251204_166_038')
    
    print(f'  Loaded cube: {cube}')
    print(f'  Bands: {cube.bands()}')
    print(f'  Scalars: {cube.scalars()}')
    print(f'  Sun elevation: {cube.metadata.get("sun_elevation")}')
    print(f'  Sun azimuth: {cube.metadata.get("sun_azimuth")}')
    print(f'  CRS: {cube.crs}')
    print(f'  Shape: ({cube.y_dim}, {cube.x_dim})')
    
    # Validate bands
    expected_bands = ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'lwir11', 'qa', 'qa_pixel']
    for band in expected_bands:
        if band not in cube.data:
            raise AssertionError(f'Missing band: {band}')
    
    print('  All expected bands present.')
    print('  [PASS] LandsatReader test passed!')
    return cube

def test_meteo_reader():
    """Test MeteoReader functionality."""
    from metric_et.io import MeteoReader
    
    print()
    print('Testing MeteoReader...')
    meteo = MeteoReader()
    meteo.load('data/weather_data.csv')
    
    print(f'  Loaded meteo: {meteo}')
    
    # Test getting weather at exact time
    weather = meteo.get_at_time(datetime(2025, 12, 4, 7, 0, 0))
    print(f'  Weather at 2025-12-04 07:00:')
    print(f'    Temperature: {weather["temperature_2m"]}°C')
    print(f'    Humidity: {weather["relative_humidity_2m"]}%')
    print(f'    Wind speed: {weather["wind_speed_10m"]} m/s')
    print(f'    Pressure: {weather["surface_pressure"]} hPa')
    
    # Test interpolation (use a time within the data range)
    weather_interp = meteo.interpolate(datetime(2025, 12, 4, 6, 30, 0))
    print(f'  Weather at 2025-12-04 06:30 (interpolated):')
    print(f'    Temperature: {weather_interp["temperature_2m"]}°C')
    
    print('  [PASS] MeteoReader test passed!')
    return meteo

def test_cloud_masker(cube):
    """Test CloudMasker functionality."""
    from metric_et.preprocess import CloudMasker
    
    print()
    print('Testing CloudMasker...')
    masker = CloudMasker()
    
    qa_pixel = cube.get('qa_pixel')
    mask = masker.create_mask(qa_pixel)
    
    print(f'  Cloud mask shape: {mask.shape}')
    print(f'  Cloud mask dtype: {mask.dtype}')
    
    # Test applying mask
    masked_cube = masker.apply_mask(cube, mask, fill_value=-9999.0)
    print(f'  Masked cube: {masked_cube}')
    
    # Test cloud coverage
    coverage = masker.compute_cloud_coverage(cube)
    print(f'  Cloud coverage: {coverage:.2f}%')
    
    print('  [PASS] CloudMasker test passed!')
    return mask

def test_resampler():
    """Test Resampler functionality."""
    from metric_et.preprocess import Resampler
    
    print()
    print('Testing Resampler...')
    resampler = Resampler(target_resolution=30, target_crs='EPSG:32639')
    
    # Test with DataCube
    from metric_et.io import LandsatReader
    reader = LandsatReader()
    cube = reader.load('data/landsat_20251204_166_038')
    
    print(f'  Original cube shape: ({cube.y_dim}, {cube.x_dim})')
    
    # Test resample to same resolution (should work without error)
    resampled = resampler.to_resolution(cube, target_resolution=30, interpolation='bilinear')
    print(f'  Resampled cube: {resampled}')
    
    print('  [PASS] Resampler test passed!')

def main():
    """Run all tests."""
    print('=' * 60)
    print('METRIC ETa - IO Layer Test Suite')
    print('=' * 60)
    
    try:
        cube = test_landsat_reader()
        test_meteo_reader()
        test_cloud_masker(cube)
        test_resampler()
        
        print()
        print('=' * 60)
        print('All tests passed successfully!')
        print('=' * 60)
        return 0
        
    except Exception as e:
        print(f'\n[FAIL] Test failed: {e}')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
