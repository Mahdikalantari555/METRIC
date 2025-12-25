"""DataCube class for storing multi-band raster data."""

from typing import Any, Dict, Optional, Union
import numpy as np
import xarray as xr
from datetime import datetime
from dataclasses import dataclass, field
import rasterio.crs


@dataclass
class DataCube:
    """
    A data structure for storing multi-band raster data with metadata.
    
    This class provides a container for storing and manipulating
    multi-band raster data from satellite imagery and other sources.
    
    Attributes:
        data: Dictionary storing band name -> xarray.DataArray mappings
        crs: Coordinate reference system (rasterio format)
        transform: Affine transform for the data
        acquisition_time: datetime of data acquisition
        metadata: Additional metadata dictionary
        extent: Geographic extent (xmin, ymin, xmax, ymax)
    """
    
    data: Dict[str, xr.DataArray] = field(default_factory=dict)
    crs: Optional[rasterio.crs.CRS] = None
    transform: Optional[np.ndarray] = None
    acquisition_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extent: Optional[tuple] = None
    
    def add(self, name: str, data: Union[xr.DataArray, np.ndarray, float]) -> 'DataCube':
        """
        Add a band or scalar value to the DataCube.
        
        Args:
            name: Name of the band/scalar
            data: xarray.DataArray, numpy array, or scalar value
            
        Returns:
            self for method chaining
        """
        if isinstance(data, xr.DataArray):
            self.data[name] = data
        elif isinstance(data, (int, float)):
            # Store as scalar - just add to metadata
            self.metadata[name] = data
        else:
            # Convert numpy array to DataArray
            da = xr.DataArray(data, dims=['y', 'x'])
            self.data[name] = da
        
        return self
    
    def get(self, name: str) -> Union[xr.DataArray, Any, None]:
        """
        Get a band or scalar value by name.

        Args:
            name: Name of the band/scalar

        Returns:
            DataArray, scalar value, or None if not found
        """
        if name in self.data:
            return self.data[name]
        elif name in self.metadata:
            return self.metadata[name]
        elif name == 'acquisition_time':
            return self.acquisition_time
        return None
    
    def bands(self) -> list:
        """Return list of band names stored in data (not metadata)."""
        return list(self.data.keys())
    
    def scalars(self) -> list:
        """Return list of scalar names stored in metadata."""
        return list(self.metadata.keys())
    
    def shape(self, band_name: str) -> tuple:
        """
        Get the shape of a specific band.
        
        Args:
            band_name: Name of the band
            
        Returns:
            Tuple of (y, x) dimensions
            
        Raises:
            KeyError: If band not found
        """
        if band_name not in self.data:
            raise KeyError(f"Band '{band_name}' not found in DataCube")
        return self.data[band_name].shape
    
    @property
    def y_dim(self) -> int:
        """Get the y-dimension size (first dimension of bands)."""
        if not self.data:
            raise ValueError("No data in DataCube")
        first_band = next(iter(self.data.values()))
        return first_band.shape[0]
    
    @property
    def x_dim(self) -> int:
        """Get the x-dimension size (second dimension of bands)."""
        if not self.data:
            raise ValueError("No data in DataCube")
        first_band = next(iter(self.data.values()))
        return first_band.shape[1]
    
    def update_crs(self, crs: rasterio.crs.CRS, transform: np.ndarray) -> 'DataCube':
        """
        Update CRS and transform for the DataCube.
        
        Args:
            crs: Coordinate reference system
            transform: Affine transform array
            
        Returns:
            self for method chaining
        """
        self.crs = crs
        self.transform = transform
        return self
    
    def to_dict(self) -> dict:
        """
        Convert DataCube to dictionary for serialization.
        
        Returns:
            Dictionary representation of the DataCube
        """
        return {
            'bands': {k: v.values.tolist() for k, v in self.data.items()},
            'scalars': self.metadata,
            'crs': str(self.crs) if self.crs else None,
            'acquisition_time': self.acquisition_time.isoformat() if self.acquisition_time else None,
            'extent': self.extent,
            'shape': {'y': self.y_dim, 'x': self.x_dim}
        }
    
    def __repr__(self) -> str:
        band_count = len(self.data)
        scalar_count = len(self.metadata)
        return (f"DataCube(bands={band_count}, scalars={scalar_count}, "
                f"shape=({self.y_dim}, {self.x_dim}), "
                f"crs={self.crs})")
