"""
METRIC ETa Model - Python implementation of the METRIC algorithm.

METRIC (Mapping Evapotranspiration with Internalized Calibration) is a 
remote sensing-based model for estimating evapotranspiration (ET) from 
satellite imagery.

This package provides tools for:
- Loading and preprocessing satellite data (Landsat)
- Calculating surface energy balance components
- Calibrating using anchor pixels
- Computing instantaneous and daily ET

Author: METRIC Development Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "METRIC Development Team"

# Core modules
from metric_et.core import (
    DataCube,
    constants
)

# IO modules
from metric_et.io import (
    LandsatReader,
    MeteoReader
)

# Preprocessing modules
from metric_et.preprocess import (
    CloudMask,
    Resampling
)

# Surface properties
from metric_et.surface import (
    Albedo,
    VegetationIndices,
    Emissivity,
    RoughnessLength
)

# Radiation balance
from metric_et.radiation import (
    ShortwaveRadiation,
    LongwaveRadiation,
    NetRadiation
)

# Energy balance
from metric_et.energy_balance import (
    SoilHeatFlux,
    SensibleHeatFlux,
    LatentHeatFlux,
    SoilHeatFluxConfig,
    SensibleHeatFluxConfig,
    LatentHeatFluxConfig
)

# Pipeline
from metric_et.pipeline import METRICPipeline

__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Core
    'DataCube',
    'constants',
    
    # IO
    'LandsatReader',
    'MeteoReader',
    
    # Preprocessing
    'CloudMask',
    'Resampling',
    
    # Surface
    'Albedo',
    'VegetationIndices',
    'Emissivity',
    'RoughnessLength',
    
    # Radiation
    'ShortwaveRadiation',
    'LongwaveRadiation',
    'NetRadiation',
    
    # Energy Balance
    'SoilHeatFlux',
    'SensibleHeatFlux',
    'LatentHeatFlux',
    'SoilHeatFluxConfig',
    'SensibleHeatFluxConfig',
    'LatentHeatFluxConfig',
    
    # Pipeline
    'METRICPipeline',
]
