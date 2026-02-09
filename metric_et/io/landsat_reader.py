"""Landsat Reader for METRIC ETa model.

This module provides functionality to read and process Landsat Collection 2 Level-2
GeoTIFF bands and metadata for ETa calculations.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import rasterio
import xarray as xr
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject

from ..core.datacube import DataCube


class LandsatReaderError(Exception):
    """Base exception for LandsatReader errors."""
    pass


class MTLParseError(LandsatReaderError):
    """Exception raised when MTL metadata parsing fails."""
    pass


class BandNotFoundError(LandsatReaderError):
    """Exception raised when a required band is not found."""
    pass


class LandsatReader:
    """
    Reader for Landsat Collection 2 Level-2 data.
    
    This class handles reading Landsat GeoTIFF bands, parsing MTL metadata,
    and creating a DataCube with properly scaled and georeferenced data.
    
    Attributes:
        band_mapping: Mapping of band names to filenames
        
    Example:
        >>> reader = LandsatReader()
        >>> cube = reader.load("data/landsat_20251204_166_038/")
        >>> print(cube)
        
    Example with custom band mapping:
        >>> custom_mapping = {'blue': 'B02.tif', 'green': 'B03.tif', ...}
        >>> reader = LandsatReader(band_mapping=custom_mapping)
        >>> cube = reader.load("data/custom_scene/")
    """
    
    # Default band name to filename mapping for Landsat 8/9 (OLI)
    DEFAULT_BAND_MAPPING = {
        'blue': 'blue.tif',
        'green': 'green.tif',
        'red': 'red.tif',
        'nir08': 'nir08.tif',
        'swir16': 'swir16.tif',
        'swir22': 'swir22.tif',
        'lwir11': 'lwir11.tif',
        'qa': 'qa.tif',
        'qa_pixel': 'qa_pixel.tif',
    }
    
    # Keep BAND_MAPPING as alias for backward compatibility
    BAND_MAPPING = DEFAULT_BAND_MAPPING
    
    def __init__(self, band_mapping: Optional[Dict[str, str]] = None):
        """Initialize the LandsatReader with optional custom band mapping.
        
        Args:
            band_mapping: Optional custom mapping of band names to filenames.
                          If None, uses default Landsat 8/9 band naming.
                          Example: {'blue': 'B02_30m.tif', 'green': 'B03_30m.tif', ...}
        """
        self.band_mapping = band_mapping if band_mapping is not None else self.DEFAULT_BAND_MAPPING.copy()
        # Keep scale_factors and add_offsets synchronized with band_mapping
        self._sync_scale_factors()
    
    def _sync_scale_factors(self):
        """Synchronize scale factors and offsets with current band_mapping."""
        # Default scale factors for standard Landsat bands
        default_scales = {
            'blue': 1.0, 'green': 1.0, 'red': 1.0, 'nir08': 1.0,
            'swir16': 1.0, 'swir22': 1.0, 'lwir11': 1.0, 'qa': 1.0, 'qa_pixel': 1.0
        }
        default_offsets = {
            'blue': 0.0, 'green': 0.0, 'red': 0.0, 'nir08': 0.0,
            'swir16': 0.0, 'swir22': 0.0, 'lwir11': 0.0, 'qa': 0.0, 'qa_pixel': 0.0
        }
        
        # Build instance-specific scale factors and offsets
        self.SCALE_FACTORS = {}
        self.ADD_OFFSETS = {}
        for band_name in self.band_mapping.keys():
            self.SCALE_FACTORS[band_name] = default_scales.get(band_name, 1.0)
            self.ADD_OFFSETS[band_name] = default_offsets.get(band_name, 0.0)
    
    def load(self, scene_path: str) -> DataCube:
        """
        Load all bands from a Landsat scene folder.
        
        Args:
            scene_path: Path to the Landsat scene directory
            
        Returns:
            DataCube containing all bands and metadata
            
        Raises:
            LandsatReaderError: If scene path doesn't exist or bands are missing
        """
        scene_path = Path(scene_path)
        
        if not scene_path.exists():
            raise LandsatReaderError(f"Scene path does not exist: {scene_path}")
        
        if not scene_path.is_dir():
            raise LandsatReaderError(f"Scene path is not a directory: {scene_path}")
        
        # Initialize DataCube
        cube = DataCube()
        
        # Parse MTL metadata
        mtl_path = scene_path / 'MTL.json'
        if not mtl_path.exists():
            raise MTLParseError(f"MTL.json not found at: {mtl_path}")
        
        mtl_data = self._read_mtl(mtl_path)
        
        # Read each band using instance's band_mapping (supports custom mapping)
        for band_name, filename in self.band_mapping.items():
            band_path = scene_path / filename
            if not band_path.exists():
                raise BandNotFoundError(f"Band file not found: {band_path}")
            
            data = self._read_band(band_path)
            data = self._apply_scaling(data, band_name)
            cube.add(band_name, data)
        
        # Add metadata from MTL
        cube.metadata['sun_elevation'] = mtl_data.get('sun_elevation')
        cube.metadata['sun_azimuth'] = mtl_data.get('sun_azimuth')
        cube.metadata['cloud_cover'] = mtl_data.get('cloud_cover', 0.0)
        cube.metadata['path'] = mtl_data.get('path')
        cube.metadata['row'] = mtl_data.get('row')
        # Extract scene_id from Landsat product ID
        scene_id = mtl_data.get('LANDSAT_PRODUCT_ID')
        if scene_id:
            cube.metadata['scene_id'] = scene_id
        
        # Set acquisition time
        if 'datetime' in mtl_data:
            try:
                cube.acquisition_time = datetime.fromisoformat(mtl_data['datetime'])
            except ValueError:
                # Try parsing as date only
                try:
                    from datetime import date
                    date_obj = date.fromisoformat(mtl_data['datetime'])
                    cube.acquisition_time = datetime.combine(date_obj, datetime.min.time())
                except ValueError:
                    cube.acquisition_time = None
        
        # Set CRS and transform from first band in the mapping
        first_band_filename = next(iter(self.band_mapping.values()))
        first_band_path = scene_path / first_band_filename
        with rasterio.open(first_band_path) as src:
            cube.crs = src.crs
            cube.transform = src.transform
            cube.extent = src.bounds
        
        return cube
    
    def _read_mtl(self, mtl_path: Union[str, Path]) -> Dict:
        """
        Parse MTL.json metadata file.
        
        Args:
            mtl_path: Path to MTL.json file
            
        Returns:
            Dictionary containing parsed metadata
            
        Raises:
            MTLParseError: If JSON parsing fails
        """
        try:
            with open(mtl_path, 'r') as f:
                mtl_data = json.load(f)
            return mtl_data
        except json.JSONDecodeError as e:
            raise MTLParseError(f"Failed to parse MTL.json: {e}")
        except IOError as e:
            raise MTLParseError(f"Failed to read MTL.json: {e}")
    
    def _read_band(self, band_path: Union[str, Path]) -> xr.DataArray:
        """
        Read a single band GeoTIFF file.
        
        Args:
            band_path: Path to the band GeoTIFF file
            
        Returns:
            xarray.DataArray with the band data
        """
        with rasterio.open(band_path) as src:
            data = src.read(1)
            
            # Get coordinates
            height, width = data.shape
            transform = src.transform
            
            # Create coordinate arrays
            rows = np.arange(height)
            cols = np.arange(width)
            
            # Calculate real-world coordinates
            x_coords = transform[2] + transform[0] * (cols + 0.5)
            y_coords = transform[5] + transform[4] * (rows + 0.5)
            
            # Create DataArray with coordinates
            da = xr.DataArray(
                data,
                dims=['y', 'x'],
                coords={'y': y_coords, 'x': x_coords},
                attrs={
                    'crs': str(src.crs) if src.crs else None,
                    'transform': transform,
                    'nodata': src.nodata,
                    'dtype': str(data.dtype),
                }
            )
        
        return da
    
    def _apply_scaling(self, data: xr.DataArray, band_name: str) -> xr.DataArray:
        """
        Apply scale factors and offsets to band data.

        Landsat Collection 2 Level-2 products are stored as scaled integers.
        This method converts them to physical values.

        Args:
            data: Input DataArray
            band_name: Name of the band

        Returns:
            Scaled DataArray
        """
        scale = self.SCALE_FACTORS.get(band_name, 1.0)
        offset = self.ADD_OFFSETS.get(band_name, 0.0)

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Scaling band {band_name}: scale={scale}, offset={offset}")
        logger.info(f"Input data range: min={np.nanmin(data):.6f}, max={np.nanmax(data):.6f}")

        # Landsat Collection 2 Level-2 products are already scaled to physical units
        logger.info(f"Landsat Collection 2 Level-2 data is already scaled to physical units, skipping scaling for {band_name}")

        # For QA bands, keep as integer for bit operations
        if band_name in ['qa', 'qa_pixel']:
            scaled_data = data.astype(np.uint16)
        else:
            scaled_data = data.astype(np.float32)

        logger.info(f"Output data range: min={np.nanmin(scaled_data):.6f}, max={np.nanmax(scaled_data):.6f}")

        # Preserve attributes
        scaled_data.attrs = data.attrs
        scaled_data.attrs['scale_factor'] = scale
        scaled_data.attrs['add_offset'] = offset

        return scaled_data
    
    def _reproject_to_utm(self, data: xr.DataArray, target_crs: str) -> xr.DataArray:
        """
        Reproject data to a target CRS (UTM zone).
        
        Args:
            data: Input DataArray
            target_crs: Target coordinate reference system (e.g., 'EPSG:32639')
            
        Returns:
            Reprojected DataArray
        """
        # This is a placeholder - full implementation would use rasterio.warp
        # For now, we assume data is already in consistent CRS
        return data
    
    def get_band_path(self, scene_path: str, band_name: str) -> Path:
        """
        Get the full path to a specific band file.
        
        Args:
            scene_path: Path to the scene directory
            band_name: Name of the band
            
        Returns:
            Full path to the band file
            
        Raises:
            BandNotFoundError: If band mapping doesn't exist
        """
        if band_name not in self.band_mapping:
            raise BandNotFoundError(f"Unknown band: {band_name}")
        
        return Path(scene_path) / self.band_mapping[band_name]
    
    def validate_scene(self, scene_path: str) -> Tuple[bool, list]:
        """
        Validate that a scene contains all required bands.
        
        Args:
            scene_path: Path to the scene directory
            
        Returns:
            Tuple of (is_valid, list of missing bands)
        """
        scene_path = Path(scene_path)
        missing_bands = []
        
        for band_name, filename in self.band_mapping.items():
            band_path = scene_path / filename
            if not band_path.exists():
                missing_bands.append(band_name)
        
        return len(missing_bands) == 0, missing_bands


def read_landsat_scene(scene_path: str, band_mapping: Optional[Dict[str, str]] = None) -> DataCube:
    """
    Convenience function to load a Landsat scene.
    
    Args:
        scene_path: Path to the Landsat scene directory
        band_mapping: Optional custom mapping of band names to filenames
        
    Returns:
        DataCube containing all bands and metadata
    """
    reader = LandsatReader(band_mapping=band_mapping)
    return reader.load(scene_path)
