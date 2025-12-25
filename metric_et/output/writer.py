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
    if np.issubdtype(dtype, np.integer):
        # For integer types, replace NaN with 0
        data = np.where(np.isnan(data), 0, data)
        nodata = 0
    else:
        data = np.where(np.isnan(data), nodata, data)

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
        nodata=nodata
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


def write_metadata(
    path: str,
    cube: DataCube,
    calibration: CalibrationResult,
    input_files: Optional[Dict[str, str]] = None,
    processing_params: Optional[Dict[str, Any]] = None
) -> None:
    """Write processing metadata to JSON.
    
    Args:
        path: Output file path
        cube: DataCube containing the data
        calibration: CalibrationResult from dT calibration
        input_files: Dictionary of input files used
        processing_params: Dictionary of processing parameters
    """
    metadata = {
        'processing_info': {
            'timestamp': datetime.now().isoformat(),
            'software': 'METRIC ETa Pipeline',
            'version': '1.0.0'
        },
        'scene_info': {
            'acquisition_time': cube.acquisition_time.isoformat() if cube.acquisition_time else None,
            'sensor': cube.metadata.get('sensor', 'Unknown'),
            'scene_id': cube.metadata.get('scene_id', 'Unknown'),
            'extent': cube.extent,
            'shape': {
                'y': cube.y_dim,
                'x': cube.x_dim
            },
            'crs': str(cube.crs) if cube.crs else None
        },
        'calibration': {
            'a_coefficient': float(calibration.a_coefficient) if isinstance(calibration.a_coefficient, (int, float, np.integer, np.floating)) else float(np.nanmean(calibration.a_coefficient)) if hasattr(calibration.a_coefficient, 'values') else float(calibration.a_coefficient),
            'b_coefficient': float(calibration.b_coefficient) if isinstance(calibration.b_coefficient, (int, float, np.integer, np.floating)) else float(np.nanmean(calibration.b_coefficient)) if hasattr(calibration.b_coefficient, 'values') else float(calibration.b_coefficient),
            'dT_cold': float(calibration.dT_cold) if isinstance(calibration.dT_cold, (int, float, np.integer, np.floating)) else float(np.nanmean(calibration.dT_cold)) if hasattr(calibration.dT_cold, 'values') else float(calibration.dT_cold),
            'dT_hot': float(calibration.dT_hot) if isinstance(calibration.dT_hot, (int, float, np.integer, np.floating)) else float(np.nanmean(calibration.dT_hot)) if hasattr(calibration.dT_hot, 'values') else float(calibration.dT_hot),
            'Ts_cold': float(calibration.ts_cold) if isinstance(calibration.ts_cold, (int, float, np.integer, np.floating)) else float(np.nanmean(calibration.ts_cold)) if hasattr(calibration.ts_cold, 'values') else float(calibration.ts_cold),
            'Ts_hot': float(calibration.ts_hot) if isinstance(calibration.ts_hot, (int, float, np.integer, np.floating)) else float(np.nanmean(calibration.ts_hot)) if hasattr(calibration.ts_hot, 'values') else float(calibration.ts_hot),
            'Ta': float(calibration.air_temperature) if isinstance(calibration.air_temperature, (int, float, np.integer, np.floating)) else float(np.nanmean(calibration.air_temperature)) if hasattr(calibration.air_temperature, 'values') else float(calibration.air_temperature),
            'ETr_inst': float(calibration.etr_inst) if isinstance(calibration.etr_inst, (int, float, np.integer, np.floating)) else float(np.nanmean(calibration.etr_inst)) if hasattr(calibration.etr_inst, 'values') else float(calibration.etr_inst),
            'valid': calibration.valid,
            'errors': calibration.errors
        },
        'input_files': input_files or {},
        'processing_parameters': processing_params or {}
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
    """
    
    def __init__(
        self,
        output_dir: str = ".",
        compression: str = "LZW",
        nodata: float = np.nan
    ):
        """Initialize the OutputWriter.
        
        Args:
            output_dir: Base output directory for all files
            compression: Default compression for GeoTIFF files
            nodata: Default no-data value
        """
        self.output_dir = Path(output_dir)
        self.compression = compression
        self.nodata = nodata
        self.output_files = []
        
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
        calibration: CalibrationResult
    ) -> Dict[str, str]:
        """Write all ET products to GeoTIFF files.
        
        Args:
            cube: DataCube containing ET products
            scene_id: Scene identifier
            date: Date string (YYYYMMDD)
            calibration: CalibrationResult from dT calibration
            
        Returns:
            Dictionary mapping product names to file paths
        """
        output_files = {}
        
        # Define product mappings
        products = [
            ('ETa_daily', 'ET_daily', 'float32'),
            ('ET_inst', 'ET_inst', 'float32'),
            ('ETrF', 'ETrF', 'float32'),
            ('LE', 'LE', 'float32'),
            ('quality', 'quality_mask', 'uint8')
        ]
        
        # Optional energy balance products
        optional_products = [
            ('Rn', 'Rn', 'float32'),
            ('G', 'G', 'float32'),
            ('H', 'H', 'float32')
        ]
        
        # Write required products
        for product_name, band_name, dtype in products:
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
        
        # Write optional products if available
        for product_name, band_name, dtype in optional_products:
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
        processing_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Write processing metadata to JSON.
        
        Args:
            cube: DataCube containing the data
            calibration: CalibrationResult from dT calibration
            scene_id: Scene identifier
            date: Date string (YYYYMMDD)
            input_files: Dictionary of input files used
            processing_params: Dictionary of processing parameters
            
        Returns:
            Path to output file
        """
        filepath = self._make_filename('metadata', scene_id, date, 'json')
        write_metadata(
            str(filepath),
            cube,
            calibration,
            input_files,
            processing_params
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
