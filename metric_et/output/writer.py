"""Output Writer for METRIC ETa pipeline.

This module provides functions for writing georeferenced output files
including GeoTIFF, NetCDF, CSV, and JSON formats.
"""

from typing import Dict, Optional, Any, Union
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
import json
import csv
from datetime import datetime
from pathlib import Path

from metric_et.core.datacube import DataCube
from metric_et.calibration.dt_calibration import CalibrationResult


def write_geotiff(
    path: str,
    data: Union[np.ndarray, xr.DataArray],
    cube: DataCube,
    compression: str = "LZW",
    nodata: float = np.nan,
    dtype: Optional[str] = None
) -> None:
    """Write single-band GeoTIFF with CRS and transform from DataCube.
    
    Args:
        path: Output file path
        data: 2D numpy array or xarray.DataArray
        cube: DataCube containing CRS and transform information
        compression: Compression algorithm (default: LZW)
        nodata: No-data value (default: NaN)
        dtype: Output data type (inferred from data if not specified)
    """
    # Convert to numpy array if needed
    if isinstance(data, xr.DataArray):
        data = data.values
    
    # Handle NaN values
    # Use -9999.0 as nodata for better compatibility with GIS software
    nodata_value = -9999.0
    if np.issubdtype(dtype, np.integer):
        # For integer types, replace NaN with 0
        data = np.where(np.isnan(data), 0, data)
        nodata_value = 0
    else:
        # For float types, replace NaN with -9999.0
        data = np.where(np.isnan(data), nodata_value, data)

    # Determine output dtype
    if dtype is None:
        if data.dtype == np.float32 or data.dtype == np.float64:
            dtype = 'float32'
        else:
            dtype = data.dtype
    
    # Get CRS and transform from DataCube
    crs = cube.crs
    transform = cube.transform
    
    if crs is None or transform is None:
        raise ValueError("DataCube must have CRS and transform set")
    
    height, width = data.shape
    
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress=compression,
        nodata=nodata_value
    ) as dst:
        dst.write(data.astype(dtype), 1)
        
        # Add metadata
        dst.update_tags(
            DATE=datetime.now().isoformat(),
            SENSOR=cube.metadata.get('sensor', 'Unknown'),
            PRODUCT='METRIC_ET'
        )


def write_multiband_geotiff(
    path: str,
    bands_dict: Dict[str, np.ndarray],
    cube: DataCube,
    compression: str = "LZW"
) -> None:
    """Write multi-band GeoTIFF with all ET products.
    
    Args:
        path: Output file path
        bands_dict: Dictionary mapping band names to 2D arrays
        cube: DataCube containing CRS and transform information
        compression: Compression algorithm (default: LZW)
    """
    # Determine common dtype (prefer float32 for ET products)
    dtypes = [arr.dtype for arr in bands_dict.values()]
    if any(dt == np.float64 for dt in dtypes):
        dtype = 'float32'
    else:
        dtype = dtypes[0] if dtypes else 'float32'
    
    # Get CRS and transform
    crs = cube.crs
    transform = cube.transform
    
    if crs is None or transform is None:
        raise ValueError("DataCube must have CRS and transform set")
    
    # Check all bands have same shape
    shapes = [arr.shape for arr in bands_dict.values()]
    if len(set(shapes)) > 1:
        raise ValueError("All bands must have the same shape")
    
    height, width = shapes[0]
    count = len(bands_dict)
    
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress=compression
    ) as dst:
        for idx, (band_name, data) in enumerate(bands_dict.items(), start=1):
            # Handle NaN values
            data_clean = np.where(np.isnan(data), -9999.0, data)
            dst.write(data_clean.astype(dtype), idx)
            
        # Write band descriptions
        for idx, band_name in enumerate(bands_dict.keys(), start=1):
            dst.set_band_description(idx, band_name)
        
        # Add metadata
        dst.update_tags(
            DATE=datetime.now().isoformat(),
            SENSOR=cube.metadata.get('sensor', 'Unknown'),
            PRODUCT='METRIC_ET_multiband',
            BANDS=','.join(bands_dict.keys())
        )


def write_netcdf(
    path: str,
    cube: DataCube,
    variables: Optional[Dict[str, xr.DataArray]] = None
) -> None:
    """Write DataCube to NetCDF with CF compliance.
    
    Args:
        path: Output file path
        cube: DataCube containing the data
        variables: Optional dictionary of additional variables to include
    """
    import xarray as xr
    from datetime import datetime
    
    # Build dataset
    ds = xr.Dataset()
    
    # Add band data
    for name, data in cube.data.items():
        ds[name] = data
        
        # Add CF-compliant attributes
        ds[name].attrs.update({
            'units': cube.metadata.get(f'{name}_units', '1'),
            'long_name': cube.metadata.get(f'{name}_long_name', name),
            'standard_name': cube.metadata.get(f'{name}_standard_name', None)
        })
    
    # Add additional variables if provided
    if variables:
        for name, data in variables.items():
            ds[name] = data
    
    # Add coordinate reference info
    if cube.transform is not None:
        # Store transform as attributes
        transform = cube.transform
        if hasattr(transform, 'a'):
            ds.attrs['transform_a'] = transform.a
            ds.attrs['transform_b'] = transform.b
            ds.attrs['transform_c'] = transform.c
            ds.attrs['transform_d'] = transform.d
            ds.attrs['transform_e'] = transform.e
            ds.attrs['transform_f'] = transform.f
    
    # Global attributes for CF compliance
    ds.attrs.update({
        'title': 'METRIC Evapotranspiration Data',
        'institution': 'METRIC Remote Sensing',
        'source': cube.metadata.get('sensor', 'Landsat'),
        'history': f'Created {datetime.now().isoformat()}',
        'references': 'METRIC model for evapotranspiration estimation',
        'Conventions': 'CF-1.8'
    })
    
    # Write to NetCDF
    ds.to_netcdf(path)
    ds.close()


def write_statistics_csv(
    path: str,
    stats_dict: Dict[str, Any]
) -> None:
    """Write summary statistics to CSV.
    
    Args:
        path: Output file path
        stats_dict: Dictionary of statistics to write
    """
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['Statistic', 'Value', 'Unit'])
        
        # Write statistics
        for key, value in stats_dict.items():
            if isinstance(value, (int, float)):
                unit = stats_dict.get(f'{key}_unit', '')
                writer.writerow([key, value, unit])


def _convert_to_serializable(value):
    """Convert values to JSON-serializable types."""
    import xarray as xr
    
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_convert_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _convert_to_serializable(v) for k, v in value.items()}
    if isinstance(value, xr.DataArray):
        # Extract scalar value if possible
        vals = value.values
        if np.isscalar(vals):
            return _convert_to_serializable(float(vals))
        return None  # Skip arrays
    if hasattr(value, 'item'):
        return value.item()
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


def write_metadata(
    path: str,
    cube: DataCube,
    calibration: CalibrationResult,
    input_files: Optional[Dict[str, str]] = None,
    processing_params: Optional[Dict[str, Any]] = None,
    quality_info: Optional[Dict[str, Any]] = None
) -> None:
    """Write processing metadata to JSON.
    
    Args:
        path: Output file path
        cube: DataCube containing the data
        calibration: CalibrationResult from dT calibration
        input_files: Dictionary of input files used
        processing_params: Dictionary of processing parameters
        quality_info: Dictionary of quality assessment information
    """
    # Extract anchor pixel coordinates with proper handling for None values
    cold_pixel_x = getattr(calibration, 'cold_pixel_x', None)
    cold_pixel_y = getattr(calibration, 'cold_pixel_y', None)
    hot_pixel_x = getattr(calibration, 'hot_pixel_x', None)
    hot_pixel_y = getattr(calibration, 'hot_pixel_y', None)
    
    # Handle NaN values - convert to None for JSON
    def _to_float(val):
        if val is None:
            return None
        if isinstance(val, (int, float, np.integer, np.floating)):
            if np.isnan(val):
                return None
            return float(val)
        return None
    
    # Convert extent to serializable format
    extent = cube.extent
    if extent is not None:
        extent_serializable = _convert_to_serializable(extent)
    else:
        extent_serializable = None
    
    metadata = {
        'scene_id': _convert_to_serializable(cube.metadata.get('scene_id', 'Unknown')),
        'acquisition_time': cube.acquisition_time.isoformat() if cube.acquisition_time else None,
        'calibration': {
            'a_coefficient': _to_float(calibration.a_coefficient),
            'b_coefficient': _to_float(calibration.b_coefficient),
            'dT_cold': _to_float(calibration.dT_cold),
            'dT_hot': _to_float(calibration.dT_hot),
            'Ts_cold': _to_float(calibration.ts_cold),
            'Ts_hot': _to_float(calibration.ts_hot),
            'air_temperature': _to_float(calibration.air_temperature),
            'Rn_cold': _to_float(getattr(calibration, 'rn_cold', None)),
            'Rn_hot': _to_float(getattr(calibration, 'rn_hot', None)),
            'G_cold': _to_float(getattr(calibration, 'g_cold', None)),
            'G_hot': _to_float(getattr(calibration, 'g_hot', None)),
            'valid': calibration.valid,
            'errors': [str(e) for e in calibration.errors] if calibration.errors else []
        },
        'anchor_pixels': {
            'cold_pixel': {
                'x': _to_float(cold_pixel_x),
                'y': _to_float(cold_pixel_y),
                'ndvi': _to_float(getattr(calibration, 'cold_pixel_ndvi', None)),
                'albedo': _to_float(getattr(calibration, 'cold_pixel_albedo', None)),
                'Ts': _to_float(calibration.ts_cold),
                'dT': _to_float(calibration.dT_cold),
                'LAI': _to_float(getattr(calibration, 'cold_pixel_lai', None)),
                'emissivity': _to_float(getattr(calibration, 'cold_pixel_emissivity', None)),
                'Rn': _to_float(getattr(calibration, 'rn_cold', None)),
                'G': _to_float(getattr(calibration, 'g_cold', None)),
                'H': _to_float(getattr(calibration, 'h_cold', None)),
                'LE': _to_float(getattr(calibration, 'le_cold', None)),
                'ETrF': _to_float(getattr(calibration, 'cold_pixel_etrf', None))
            },
            'hot_pixel': {
                'x': _to_float(hot_pixel_x),
                'y': _to_float(hot_pixel_y),
                'ndvi': _to_float(getattr(calibration, 'hot_pixel_ndvi', None)),
                'albedo': _to_float(getattr(calibration, 'hot_pixel_albedo', None)),
                'Ts': _to_float(calibration.ts_hot),
                'dT': _to_float(calibration.dT_hot),
                'LAI': _to_float(getattr(calibration, 'hot_pixel_lai', None)),
                'emissivity': _to_float(getattr(calibration, 'hot_pixel_emissivity', None)),
                'Rn': _to_float(getattr(calibration, 'rn_hot', None)),
                'G': _to_float(getattr(calibration, 'g_hot', None)),
                'H': _to_float(getattr(calibration, 'h_hot', None)),
                'LE': _to_float(getattr(calibration, 'le_hot', None)),
                'ETrF': _to_float(getattr(calibration, 'hot_pixel_etrf', None))
            }
        },
        'quality': _convert_to_serializable(quality_info) if quality_info else {
            'scene_quality': _convert_to_serializable(cube.metadata.get('scene_quality', 'UNKNOWN')),
            'cloud_coverage': _to_float(cube.metadata.get('cloud_cover', None)),
            'valid_pixels': _to_float(cube.metadata.get('valid_pixels', None)),
            'ndvi_range': {
                'min': _to_float(cube.metadata.get('ndvi_min', None)),
                'max': _to_float(cube.metadata.get('ndvi_max', None)),
                'mean': _to_float(cube.metadata.get('ndvi_mean', None))
            },
            'temperature_range': {
                'min': _to_float(cube.metadata.get('ts_min', None)),
                'max': _to_float(cube.metadata.get('ts_max', None)),
                'mean': _to_float(cube.metadata.get('ts_mean', None))
            }
        },
        'scene_info': {
            'extent': extent_serializable,
            'shape': {
                'y': _convert_to_serializable(cube.y_dim),
                'x': _convert_to_serializable(cube.x_dim)
            },
            'crs': _convert_to_serializable(str(cube.crs)) if cube.crs else None,
            'sensor': _convert_to_serializable(cube.metadata.get('sensor', 'Unknown')),
            'sun_elevation': _to_float(cube.metadata.get('sun_elevation', None)),
            'sun_azimuth': _to_float(cube.metadata.get('sun_azimuth', None))
        },
        'input_files': _convert_to_serializable(input_files) if input_files else {},
        'processing_parameters': _convert_to_serializable(processing_params) if processing_params else {},
        'processing_info': {
            'timestamp': datetime.now().isoformat(),
            'software': 'METRIC ETa Pipeline',
            'version': '1.0.0'
        }
    }
    
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)


class OutputWriter:
    """Main output writer class for METRIC ETa pipeline.
    
    This class provides convenient methods for writing all output products
    from the METRIC processing pipeline.
    
    Attributes:
        output_dir: Base output directory for all files
        compression: Default compression for GeoTIFF files
        nodata: Default no-data value
        output_products: List of products to write (None = all products)
    """
    
    # Default output product definitions
    DEFAULT_PRODUCTS = {
        'required': [
            ('ETa_daily', 'ET_daily', 'float32'),
            ('ET_inst', 'ET_inst', 'float32'),
            ('ETrF', 'ETrF', 'float32'),
            ('LE', 'LE', 'float32'),
            ('quality', 'quality_mask', 'uint8')
        ],
        'optional': [
            ('Rn', 'R_n', 'float32'),
            ('G', 'G', 'float32'),
            ('H', 'H', 'float32')
        ],
        'quality': [
            ('ET_quality_class', 'ET_quality_class', 'uint8'),
            ('ETa_classified', 'ETa_class', 'uint8'),
            ('CWSI', 'CWSI', 'float32')
        ],
        'surface': [
            ('NDVI', 'ndvi', 'float32'),
            ('Albedo', 'albedo', 'float32'),
            ('LST', 'lst', 'float32'),
            ('LAI', 'lai', 'float32')
        ]
    }
    
    def __init__(
        self,
        output_dir: str = ".",
        compression: str = "LZW",
        nodata: float = np.nan,
        output_products: Optional[list] = None,
        include_surface_properties: bool = False
    ):
        """Initialize the OutputWriter.
        
        Args:
            output_dir: Base output directory for all files
            compression: Default compression for GeoTIFF files
            nodata: Default no-data value
            output_products: List of products to write as tuples (output_name, band_name, dtype).
                            Example: [('ETa_daily', 'ET_daily', 'float32'), ('ETrF', 'ETrF', 'float32')]
                            If None, writes all default products.
            include_surface_properties: Whether to include surface property outputs (NDVI, Albedo, LST, LAI)
        """
        self.output_dir = Path(output_dir)
        self.compression = compression
        self.nodata = nodata
        self.output_files = []
        self._include_surface = include_surface_properties
        
        # Build output products list
        if output_products is not None:
            # Use custom product list
            self._output_products = output_products
        else:
            # Use default products
            self._output_products = (
                self.DEFAULT_PRODUCTS['required'] +
                self.DEFAULT_PRODUCTS['optional'] +
                self.DEFAULT_PRODUCTS['quality']
            )
            if include_surface_properties:
                self._output_products += self.DEFAULT_PRODUCTS['surface']
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _make_filename(
        self,
        product: str,
        scene_id: str,
        date: str,
        extension: str
    ) -> Path:
        """Generate output filename following naming convention.
        
        Args:
            product: Product name (e.g., 'ETa_daily', 'ETrF')
            scene_id: Scene identifier (e.g., 'LC08')
            date: Date string (YYYYMMDD)
            extension: File extension (e.g., 'tif', 'csv')
            
        Returns:
            Full output file path
        """
        filename = f"{product}_{scene_id}_{date}.{extension}"
        return self.output_dir / filename
    
    def write_et_products(
        self,
        cube: DataCube,
        scene_id: str,
        date: str,
        calibration: Optional[CalibrationResult] = None,
        products: Optional[list] = None
    ) -> Dict[str, str]:
        """Write ET products to GeoTIFF files with configurable output selection.
        
        Args:
            cube: DataCube containing ET products
            scene_id: Scene identifier
            date: Date string (YYYYMMDD)
            calibration: CalibrationResult from dT calibration (optional for custom products)
            products: Override list of products to write. If None, uses instance defaults.
                     Format: [(output_name, band_name, dtype), ...]
                     Example: [('ETa_daily', 'ET_daily', 'float32'), ('ETrF', 'ETrF', 'float32')]
        
        Returns:
            Dictionary mapping product names to file paths
        """
        output_files = {}
        
        # Use provided products or fall back to instance defaults
        product_list = products if products is not None else self._output_products
        
        for product_name, band_name, dtype in product_list:
            if band_name in cube.data:
                filepath = self._make_filename(product_name, scene_id, date, 'tif')
                write_geotiff(
                    str(filepath),
                    cube.data[band_name],
                    cube,
                    compression=self.compression,
                    nodata=self.nodata,
                    dtype=dtype
                )
                output_files[product_name] = str(filepath)
                self.output_files.append(filepath)
        
        return output_files
    
    def write_custom_products(
        self,
        cube: DataCube,
        scene_id: str,
        date: str,
        products: list
    ) -> Dict[str, str]:
        """Write custom products to GeoTIFF files.
        
        Args:
            cube: DataCube containing the data
            scene_id: Scene identifier
            date: Date string (YYYYMMDD)
            products: List of products as tuples (output_name, band_name, dtype)
        
        Returns:
            Dictionary mapping product names to file paths
        """
        return self.write_et_products(cube, scene_id, date, products=products)
    
    write_et_products_csv = write_statistics_csv
    
    def compute_statistics(self, cube: DataCube) -> Dict[str, Any]:
        """Compute scene statistics for ET products.

        Args:
            cube: DataCube containing the data

        Returns:
            Dictionary of statistics
        """
        stats = {}

        # Compute statistics for ET products
        for band_name in ['ET_daily', 'ET_inst', 'ETrF', 'LE']:
            if band_name in cube.data:
                data = cube.data[band_name]
                if hasattr(data, 'values'):
                    data = data.values
                valid_data = data[~np.isnan(data)]

                if len(valid_data) > 0:
                    stats[f'{band_name}_mean'] = float(np.nanmean(valid_data))
                    stats[f'{band_name}_std'] = float(np.nanstd(valid_data))
                    stats[f'{band_name}_min'] = float(np.nanmin(valid_data))
                    stats[f'{band_name}_max'] = float(np.nanmax(valid_data))
                    stats[f'{band_name}_count'] = int(np.sum(~np.isnan(data)))

        return stats

    def write_scene_statistics(
        self,
        cube: DataCube,
        scene_id: str,
        date: str
    ) -> str:
        """Write scene statistics to CSV.

        Args:
            cube: DataCube containing the data
            scene_id: Scene identifier
            date: Date string (YYYYMMDD)

        Returns:
            Path to output file
        """
        stats = self.compute_statistics(cube)

        filepath = self._make_filename('statistics', scene_id, date, 'csv')
        write_statistics_csv(str(filepath), stats)
        self.output_files.append(filepath)

        return str(filepath)
    
    def write_metadata_file(
        self,
        cube: DataCube,
        calibration: CalibrationResult,
        scene_id: str,
        date: str,
        input_files: Optional[Dict[str, str]] = None,
        processing_params: Optional[Dict[str, Any]] = None,
        quality_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Write processing metadata to JSON.
        
        Args:
            cube: DataCube containing the data
            calibration: CalibrationResult from dT calibration
            scene_id: Scene identifier
            date: Date string (YYYYMMDD)
            input_files: Dictionary of input files used
            processing_params: Dictionary of processing parameters
            quality_info: Dictionary of quality assessment information
            
        Returns:
            Path to output file
        """
        filepath = self._make_filename('metadata', scene_id, date, 'json')
        write_metadata(
            str(filepath),
            cube,
            calibration,
            input_files,
            processing_params,
            quality_info
        )
        self.output_files.append(filepath)
        
        return str(filepath)
    
    def write_multiband_product(
        self,
        cube: DataCube,
        scene_id: str,
        date: str,
        bands: list,
        product_name: str = 'ET_products'
    ) -> str:
        """Write multi-band GeoTIFF with selected bands.
        
        Args:
            cube: DataCube containing the data
            scene_id: Scene identifier
            date: Date string (YYYYMMDD)
            bands: List of band names to include
            product_name: Name for the output product
            
        Returns:
            Path to output file
        """
        bands_dict = {}
        for band_name in bands:
            if band_name in cube.data:
                bands_dict[band_name] = cube.data[band_name].values
        
        if not bands_dict:
            raise ValueError("None of the specified bands are available")
        
        filepath = self._make_filename(product_name, scene_id, date, 'tif')
        write_multiband_geotiff(str(filepath), bands_dict, cube, self.compression)
        self.output_files.append(filepath)
        
        return str(filepath)
    
    def write_netcdf_output(
        self,
        cube: DataCube,
        scene_id: str,
        date: str,
        variables: Optional[Dict[str, xr.DataArray]] = None
    ) -> str:
        """Write DataCube to NetCDF format.
        
        Args:
            cube: DataCube containing the data
            scene_id: Scene identifier
            date: Date string (YYYYMMDD)
            variables: Optional dictionary of additional variables
            
        Returns:
            Path to output file
        """
        filepath = self._make_filename(f'{scene_id}_ET', date, 'nc')
        write_netcdf(str(filepath), cube, variables)
        self.output_files.append(filepath)
        
        return str(filepath)
    
    def get_output_files(self) -> list:
        """Get list of all output files generated.
        
        Returns:
            List of output file paths
        """
        return self.output_files
