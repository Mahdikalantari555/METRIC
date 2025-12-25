"""Resampling module for raster data.

This module provides functionality for resampling raster data to target
resolutions and reprojecting to different coordinate reference systems.
"""

from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import calculate_default_transform, reproject

from ..core.datacube import DataCube


class ResamplerError(Exception):
    """Base exception for Resampler errors."""
    pass


class Resampler:
    """
    Resampler for raster data.
    
    This class handles resampling raster data to target resolutions
    and reprojecting to different coordinate reference systems.
    
    Attributes:
        target_resolution: Default target resolution in meters
        target_crs: Default target CRS
        
    Example:
        >>> resampler = Resampler(target_resolution=30, target_crs='EPSG:32639')
        >>> resampled = resampler.to_resolution(data, target_resolution=30)
    """
    
    def __init__(
        self,
        target_resolution: int = 30,
        target_crs: str = 'EPSG:32639',
    ):
        """
        Initialize the Resampler.
        
        Args:
            target_resolution: Default target resolution in meters
            target_crs: Default target CRS (e.g., 'EPSG:32639' for UTM zone 39N)
        """
        self.target_resolution = target_resolution
        self.target_crs = target_crs
    
    def to_resolution(
        self,
        data: Union[xr.DataArray, DataCube],
        target_resolution: int,
        interpolation: str = 'bilinear',
    ) -> Union[xr.DataArray, DataCube]:
        """
        Resample data to a target resolution.
        
        Args:
            data: Input DataArray or DataCube
            target_resolution: Target resolution in meters
            interpolation: Interpolation method ('bilinear', 'nearest', 'cubic')
            
        Returns:
            Resampled data with same type as input
            
        Raises:
            ResamplerError: If resampling fails
        """
        if isinstance(data, DataCube):
            resampled_cube = DataCube()
            resampled_cube.crs = data.crs
            resampled_cube.transform = data.transform
            resampled_cube.acquisition_time = data.acquisition_time
            resampled_cube.metadata = data.metadata.copy()
            
            for band_name, band_data in data.data.items():
                if isinstance(band_data, xr.DataArray):
                    resampled_band = self._resample_dataarray(
                        band_data, target_resolution, interpolation
                    )
                    resampled_cube.add(band_name, resampled_band)
                else:
                    resampled_cube.add(band_name, band_data)
            
            return resampled_cube
        else:
            return self._resample_dataarray(data, target_resolution, interpolation)
    
    def _resample_dataarray(
        self,
        data: xr.DataArray,
        target_resolution: int,
        interpolation: str,
    ) -> xr.DataArray:
        """
        Resample a single DataArray to target resolution.
        
        Args:
            data: Input DataArray
            target_resolution: Target resolution in meters
            interpolation: Interpolation method
            
        Returns:
            Resampled DataArray
        """
        if data.attrs.get('transform') is None:
            raise ResamplerError("DataArray missing transform information")
        
        transform = data.attrs['transform']
        current_resolution = abs(transform[0])  # Pixel width
        
        if abs(current_resolution - target_resolution) < 0.1:
            # Resolutions are essentially the same
            return data
        
        # Calculate scaling factor
        scale_factor = current_resolution / target_resolution
        
        # Calculate new dimensions
        height, width = data.shape
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Use scipy for resampling
        try:
            from scipy.ndimage import zoom
            
            # Select interpolation filter
            if interpolation == 'nearest':
                order = 0
            elif interpolation == 'bilinear':
                order = 1
            elif interpolation == 'cubic':
                order = 3
            else:
                order = 1
            
            # Resample data
            resampled_data = zoom(data.values, scale_factor, order=order)
            
            # Create new transform
            new_transform = (
                transform[0] / scale_factor,  # Width of pixel
                transform[1],
                transform[2],
                transform[4],
                transform[4] / scale_factor,  # Height of pixel (negative)
                transform[5],
            )
            
            # Create new coordinates
            rows = np.arange(resampled_data.shape[0])
            cols = np.arange(resampled_data.shape[1])
            
            new_x_coords = new_transform[2] + new_transform[0] * (cols + 0.5)
            new_y_coords = new_transform[5] + new_transform[4] * (rows + 0.5)
            
            result = xr.DataArray(
                resampled_data,
                dims=['y', 'x'],
                coords={'y': new_y_coords, 'x': new_x_coords},
                attrs={
                    **data.attrs,
                    'transform': new_transform,
                    'resampled': True,
                    'original_resolution': current_resolution,
                    'target_resolution': target_resolution,
                }
            )
            
            return result
        
        except ImportError:
            raise ResamplerError("scipy is required for resampling")
    
    def to_crs(
        self,
        data: Union[xr.DataArray, DataCube],
        target_crs: str,
        interpolation: str = 'bilinear',
    ) -> Union[xr.DataArray, DataCube]:
        """
        Reproject data to a target coordinate reference system.
        
        Args:
            data: Input DataArray or DataCube
            target_crs: Target CRS (e.g., 'EPSG:32639')
            interpolation: Interpolation method
            
        Returns:
            Reprojected data with same type as input
            
        Raises:
            ResamplerError: If reprojection fails
        """
        target_crs_obj = CRS.from_user_input(target_crs)
        
        if isinstance(data, DataCube):
            if data.crs is None:
                raise ResamplerError("DataCube missing CRS information")
            
            reprojected_cube = DataCube()
            reprojected_cube.acquisition_time = data.acquisition_time
            reprojected_cube.metadata = data.metadata.copy()
            
            for band_name, band_data in data.data.items():
                if isinstance(band_data, xr.DataArray):
                    reprojected_band = self._reproject_dataarray(
                        band_data, target_crs_obj, interpolation
                    )
                    reprojected_cube.add(band_name, reprojected_band)
                else:
                    reprojected_cube.add(band_name, band_data)
            
            return reprojected_cube
        else:
            return self._reproject_dataarray(data, target_crs_obj, interpolation)
    
    def _reproject_dataarray(
        self,
        data: xr.DataArray,
        target_crs: CRS,
        interpolation: str,
    ) -> xr.DataArray:
        """
        Reproject a single DataArray to target CRS.
        
        Args:
            data: Input DataArray
            target_crs: Target CRS object
            interpolation: Interpolation method
            
        Returns:
            Reprojected DataArray
        """
        if data.attrs.get('crs') is None:
            raise ResamplerError("DataArray missing CRS information")
        
        source_crs = CRS.from_user_input(data.attrs['crs'])
        
        if source_crs == target_crs:
            return data
        
        # Get transform from data
        transform = data.attrs.get('transform')
        if transform is None:
            raise ResamplerError("DataArray missing transform information")
        
        height, width = data.shape
        
        # Calculate new transform and dimensions
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=data.dtype,
                crs=source_crs,
                transform=transform,
            ) as src:
                # Calculate default transform
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    source_crs, target_crs, width, height, *src.bounds
                )
            
            # Reproject using rasterio
            dst_data = np.zeros((dst_height, dst_width), dtype=data.dtype)
            
            reproject(
                source=data.values,
                destination=dst_data,
                src_transform=transform,
                src_crs=source_crs,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                resampling=self._get_resampling_method(interpolation),
            )
            
            # Create new coordinates
            rows = np.arange(dst_height)
            cols = np.arange(dst_width)
            
            new_x_coords = dst_transform[2] + dst_transform[0] * (cols + 0.5)
            new_y_coords = dst_transform[5] + dst_transform[4] * (rows + 0.5)
            
            result = xr.DataArray(
                dst_data,
                dims=['y', 'x'],
                coords={'y': new_y_coords, 'x': new_x_coords},
                attrs={
                    **data.attrs,
                    'crs': str(target_crs),
                    'transform': dst_transform,
                    'reprojected': True,
                    'original_crs': str(source_crs),
                }
            )
        
        return result
    
    def match_grids(
        self,
        data1: Union[xr.DataArray, DataCube],
        data2: Union[xr.DataArray, DataCube],
    ) -> Tuple[Union[xr.DataArray, DataCube], Union[xr.DataArray, DataCube]]:
        """
        Align two datasets to the same grid.
        
        The second dataset is resampled to match the grid of the first.
        
        Args:
            data1: Reference dataset (grid to match)
            data2: Dataset to resample
            
        Returns:
            Tuple of (data1, resampled_data2) with matching grids
        """
        if isinstance(data1, DataCube) and isinstance(data2, DataCube):
            if data1.crs is None or data2.crs is None:
                raise ResamplerError("Both DataCubes must have CRS information")
            
            # Use data1's grid as reference
            ref_transform = data1.transform
            ref_crs = data1.crs
            ref_height = data1.y_dim
            ref_width = data1.x_dim
            
            # Reproject data2 to match data1's grid
            aligned_cube = DataCube()
            aligned_cube.crs = ref_crs
            aligned_cube.transform = ref_transform
            aligned_cube.acquisition_time = data2.acquisition_time
            aligned_cube.metadata = data2.metadata.copy()
            
            for band_name, band_data in data2.data.items():
                if isinstance(band_data, xr.DataArray):
                    aligned_band = self._align_to_grid(
                        band_data, ref_transform, ref_crs, ref_height, ref_width
                    )
                    aligned_cube.add(band_name, aligned_band)
                else:
                    aligned_cube.add(band_name, band_data)
            
            return data1, aligned_cube
        
        elif isinstance(data1, xr.DataArray) and isinstance(data2, xr.DataArray):
            ref_transform = data1.attrs.get('transform')
            ref_crs = data1.attrs.get('crs')
            
            if ref_transform is None or ref_crs is None:
                raise ResamplerError("data1 missing transform or CRS")
            
            return data1, self._align_to_grid(
                data2, ref_transform, ref_crs, data1.shape[0], data1.shape[1]
            )
        
        else:
            raise ResamplerError("Both inputs must be DataArray or DataCube")
    
    def _align_to_grid(
        self,
        data: xr.DataArray,
        ref_transform,
        ref_crs: str,
        ref_height: int,
        ref_width: int,
    ) -> xr.DataArray:
        """
        Align data to a reference grid.
        
        Args:
            data: Input DataArray
            ref_transform: Reference transform
            ref_crs: Reference CRS
            ref_height: Reference height
            ref_width: Reference width
            
        Returns:
            Aligned DataArray
        """
        source_crs = CRS.from_user_input(data.attrs.get('crs', ref_crs))
        target_crs = CRS.from_user_input(ref_crs)
        
        if source_crs == target_crs:
            # Same CRS, just need to match grid
            return self._resample_to_match(data, ref_transform, ref_height, ref_width)
        
        # Reproject and match grid
        with rasterio.io.MemoryFile() as memfile:
            reprojected_data = np.zeros((ref_height, ref_width), dtype=data.dtype)
            
            reproject(
                source=data.values,
                destination=reprojected_data,
                src_transform=data.attrs.get('transform'),
                src_crs=source_crs,
                dst_transform=ref_transform,
                dst_crs=target_crs,
                dst_height=ref_height,
                dst_width=ref_width,
                resampling=self._get_resampling_method('bilinear'),
            )
            
            # Create coordinates
            rows = np.arange(ref_height)
            cols = np.arange(ref_width)
            
            new_x_coords = ref_transform[2] + ref_transform[0] * (cols + 0.5)
            new_y_coords = ref_transform[5] + ref_transform[4] * (rows + 0.5)
            
            return xr.DataArray(
                reprojected_data,
                dims=['y', 'x'],
                coords={'y': new_y_coords, 'x': new_x_coords},
                attrs={
                    **data.attrs,
                    'crs': ref_crs,
                    'transform': ref_transform,
                    'aligned': True,
                }
            )
    
    def _resample_to_match(
        self,
        data: xr.DataArray,
        ref_transform,
        ref_height: int,
        ref_width: int,
    ) -> xr.DataArray:
        """
        Resample data to match reference grid dimensions.
        
        Args:
            data: Input DataArray
            ref_transform: Reference transform
            ref_height: Reference height
            ref_width: Reference width
            
        Returns:
            Resampled DataArray
        """
        height, width = data.shape
        scale_y = ref_height / height
        scale_x = ref_width / width
        
        # Use zoom for resampling
        try:
            from scipy.ndimage import zoom
            
            scale_factor = (scale_y, scale_x)
            resampled_data = zoom(data.values, scale_factor, order=1)
            
            # Crop or pad if needed
            if resampled_data.shape[0] != ref_height or resampled_data.shape[1] != ref_width:
                resampled_data = self._crop_or_pad(resampled_data, ref_height, ref_width)
            
            # Create coordinates
            rows = np.arange(ref_height)
            cols = np.arange(ref_width)
            
            new_x_coords = ref_transform[2] + ref_transform[0] * (cols + 0.5)
            new_y_coords = ref_transform[5] + ref_transform[4] * (rows + 0.5)
            
            return xr.DataArray(
                resampled_data,
                dims=['y', 'x'],
                coords={'y': new_y_coords, 'x': new_x_coords},
                attrs={
                    **data.attrs,
                    'transform': ref_transform,
                    'resampled': True,
                }
            )
        
        except ImportError:
            raise ResamplerError("scipy is required for resampling")
    
    def _crop_or_pad(
        self,
        data: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        """
        Crop or pad data to target dimensions.
        
        Args:
            data: Input array
            target_height: Target height
            target_width: Target width
            
        Returns:
            Cropped or padded array
        """
        height, width = data.shape
        
        result = np.zeros((target_height, target_width), dtype=data.dtype)
        
        # Calculate crop/pad boundaries
        y_start = max(0, (height - target_height) // 2)
        y_end = min(height, y_start + target_height)
        x_start = max(0, (width - target_width) // 2)
        x_end = min(width, x_start + target_width)
        
        # Copy available data
        result_y_start = max(0, (target_height - height) // 2)
        result_x_start = max(0, (target_width - width) // 2)
        
        result_slice = result[
            result_y_start:result_y_start + (y_end - y_start),
            result_x_start:result_x_start + (x_end - x_start)
        ]
        
        data_slice = data[y_start:y_end, x_start:x_end]
        
        result[
            result_y_start:result_y_start + data_slice.shape[0],
            result_x_start:result_x_start + data_slice.shape[1]
        ] = data_slice
        
        return result
    
    def _get_resampling_method(self, interpolation: str):
        """
        Get rasterio resampling method from interpolation string.
        
        Args:
            interpolation: Interpolation method name
            
        Returns:
            rasterio resampling enum value
        """
        from rasterio.warp import Resampling
        
        methods = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic,
            'cubic_spline': Resampling.cubic_spline,
            'lanczos': Resampling.lanczos,
            'average': Resampling.average,
            'mode': Resampling.mode,
        }
        
        return methods.get(interpolation.lower(), Resampling.bilinear)


def resample_to_resolution(
    data: Union[xr.DataArray, DataCube],
    target_resolution: int,
    interpolation: str = 'bilinear',
) -> Union[xr.DataArray, DataCube]:
    """
    Convenience function to resample data to target resolution.
    
    Args:
        data: Input DataArray or DataCube
        target_resolution: Target resolution in meters
        interpolation: Interpolation method
        
    Returns:
        Resampled data
    """
    resampler = Resampler(target_resolution=target_resolution)
    return resampler.to_resolution(data, target_resolution, interpolation)


# Alias for Resampler
Resampling = Resampler
