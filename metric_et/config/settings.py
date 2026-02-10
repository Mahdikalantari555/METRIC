"""Configuration settings for METRIC ETa model."""

import os
from pathlib import Path

# ============================================================================
# DATA PATHS
# ============================================================================

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input data directories
DATA_DIR = BASE_DIR / "data"
LANDSAT_DIR = DATA_DIR / "landsat_data"
METEO_DIR = DATA_DIR / "meteo_data"
DEM_DIR = DATA_DIR / "dem"

# Output directories
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = OUTPUT_DIR / "results"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"

# ============================================================================
# BANDS CONFIGURATION
# ============================================================================

# Reflectance bands for Landsat 8/9
REFLECTANCE_BANDS = [
    "blue",
    "green", 
    "red",
    "nir08",
    "swir16",
    "swir22"
]

# Thermal infrared band
THERMAL_BAND = "lwir11"

# Quality assurance bands
QA_BANDS = ["qa", "qa_pixel"]

# All satellite bands
ALL_BANDS = REFLECTANCE_BANDS + [THERMAL_BAND] + QA_BANDS

# ============================================================================
# METEOROLOGICAL VARIABLES
# ============================================================================

# Weather variables expected from meteorological data
METEO_VARIABLES = [
    "temperature_2m",           # Air temperature at 2m (K or °C)
    "relative_humidity_2m",     # Relative humidity at 2m (%)
    "wind_speed_10m",           # Wind speed at 10m (m/s)
    "surface_pressure",         # Surface pressure (Pa or hPa)
    "shortwave_radiation",      # Incoming shortwave radiation (W/m²)
    "longwave_radiation",       # Incoming longwave radiation (W/m²)
    "et0_fao_evapotranspiration"  # Reference ET (mm/day)
]

# Optional meteorological variables
OPTIONAL_METEO_VARIABLES = [
    "dewpoint_temperature",
    "vapor_pressure_deficit",
    "cloud_cover",
    "solar_zenith_angle"
]

# ============================================================================
# SPATIAL RESOLUTION
# ============================================================================

# Landsat resolution (meters)
LANDSAT_RESOLUTION = 30  # 30m for reflectance and thermal (resampled)

# Thermal native resolution (meters)
THERMAL_NATIVE_RESOLUTION = 100  # 100m for Landsat 8/9 TIRS

# DEM resolution (meters)
DEM_RESOLUTION = 30  # Match Landsat resolution

# ============================================================================
# PHYSICAL CONSTANTS (from core/constants.py)
# ============================================================================

from ..core.constants import (
    STEFAN_BOLTZMANN,
    VON_KARMAN,
    AIR_DENSITY,
    AIR_SPECIFIC_HEAT,
    LATENT_HEAT_VAPORIZATION,
    SOLAR_CONSTANT,
    GRAVITATIONAL_ACCELERATION
)

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Anchor pixel selection parameters
ANCHOR_PIXEL = {
    "min_count_hot": 5,
    "max_count_hot": 10,
    "min_count_cold": 5,
    "max_count_cold": 10,
    "hot_fraction_range": (0.01, 0.05),
    "cold_fraction_range": (0.90, 1.00),
    "ndvi_hot_range": (0.1, 0.4),
    "ndvi_cold_range": (0.7, 0.95),
    "min_lai_hot": 0.5,
    "min_lai_cold": 3.0
}

# dT calibration parameters
DT_CALIBRATION = {
    "min_dT": -5.0,
    "max_dT": 5.0,
    "default_iterations": 3,
    "convergence_threshold": 0.01,
    "roughness_ratio": 0.5
}

# Energy balance parameters
ENERGY_BALANCE = {
    "soil_heat_flux_ratio": {
        "vegetated": 0.05,
        "bare_soil": 0.15,
        "water": 0.02
    },
    "aerodynamic_resistance_params": {
        "height_displacement_factor": 0.67,
        "roughness_length_momentum": 0.1,
        "roughness_length_heat": 0.01
    }
}

# ET calculation parameters
ET_PARAMETERS = {
    "instantaneous_to_daily": "coefficient_method",  # or "integral_method"
    "daytime_fraction": 0.75,  # Typical fraction of daily ET occurring during daylight
    "nighttime_correction": True
}

# ============================================================================
# FILE FORMATS
# ============================================================================

# Input file formats
INPUT_FORMATS = {
    "landsat": {
        "extension": ".tif",
        "driver": "GTiff",
        "dtype": "float32"
    },
    "dem": {
        "extension": ".tif",
        "driver": "GTiff",
        "dtype": "float32"
    },
    "meteo": {
        "extension": ".csv",
        "format": "pandas_dataframe"
    }
}

# Output file formats
OUTPUT_FORMATS = {
    "et_instantaneous": {
        "extension": ".tif",
        "driver": "GTiff",
        "dtype": "float32",
        "units": "W/m²"
    },
    "et_daily": {
        "extension": ".tif",
        "driver": "GTiff",
        "dtype": "float32",
        "units": "mm/day"
    },
    "energy_balance": {
        "extension": ".nc",
        "driver": "netCDF4",
        "dtype": "float32"
    }
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING = {
    "level": "INFO",
    "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    "file_log": True,
    "log_file": BASE_DIR / "logs" / "metric_et.log"
}

# ============================================================================
# VALIDATION RANGES
# ============================================================================

VALIDATION_RANGES = {
    "ndvi": (0.0, 1.0),
    "albedo": (0.0, 0.6),
    "lst": (250.0, 350.0),  # Kelvin
    "emissivity": (0.90, 1.0),
    "et_instantaneous": (0.0, 700.0),  # W/m²
    "et_daily": (0.0, 30.0),  # mm/day
    "dT": (-15.0, 15.0),  # Kelvin
    "rn": (0.0, 1100.0),  # W/m²
    "g": (-50.0, 200.0),  # W/m²
    "h": (-50.0, 500.0),  # W/m²
    "le": (-50.0, 700.0)  # W/m²
}

# ============================================================================
# NDVI THRESHOLDS FOR LAND COVER
# ============================================================================

NDVI_THRESHOLDS = {
    "water": (0.0, 0.1),
    "urban": (0.1, 0.2),
    "bare_soil": (0.2, 0.35),
    "sparse_vegetation": (0.35, 0.5),
    "dense_vegetation": (0.5, 0.7),
    "forest": (0.7, 1.0)
}

# ============================================================================
# DEFAULT OUTPUT VARIABLES
# ============================================================================

# Output variable configurations
OUTPUT_VARIABLES = {
    "required": [
        "et_instantaneous",
        "et_daily",
        "lst",
        "ndvi",
        "albedo",
        "emissivity"
    ],
    "optional": [
        "net_radiation",
        "soil_heat_flux",
        "sensible_heat_flux",
        "latent_heat_flux",
        "dT",
        "rn_g_ratio",
        "aerodynamic_resistance",
        "bulk_aerodynamic_resistance"
    ]
}

# ============================================================================
# CONFIGURABLE OUTPUT PRODUCTS
# ============================================================================
# These define the output products that can be written by the OutputWriter.
# Each product is a tuple: (output_name, band_name_in_cube, dtype)
# - output_name: Name used in the output filename
# - band_name_in_cube: Key name of the band in the DataCube
# - dtype: Output data type (float32, uint8, etc.)

OUTPUT_PRODUCTS = {
    # Required ET products (always included by default)
    "required": [
        ("ETa_daily", "ET_daily", "float32"),
        ("ET_inst", "ET_inst", "float32"),
        ("ETrF", "ETrF", "float32"),
        ("LE", "LE", "float32"),
        ("quality", "quality_mask", "uint8")
    ],
    # Optional energy balance products
    "energy_balance": [
        ("Rn", "R_n", "float32"),
        ("G", "G", "float32"),
        ("H", "H", "float32")
    ],
    # Quality layer products
    "quality": [
        ("ET_quality_class", "ET_quality_class", "uint8"),
        ("ETa_classified", "ETa_class", "uint8"),
        ("CWSI", "CWSI", "float32")
    ],
    # Surface property products
    "surface": [
        ("NDVI", "ndvi", "float32"),
        ("Albedo", "albedo", "float32"),
        ("LST", "lst", "float32"),
        ("LAI", "lai", "float32"),
        ("Emissivity", "emissivity", "float32"),
        ("FVC", "fvc", "float32"),
        ("SAVI", "savi", "float32")
    ],
    # Radiation products
    "radiation": [
        ("Rns", "R_ns", "float32"),
        ("Rnl", "R_nl", "float32"),
        ("Rs_down", "Rs_down", "float32"),
        ("Rl_down", "R_l_down", "float32"),
        ("Rl_up", "R_l_up", "float32")
    ]
}

# Helper function to create custom output product list
def get_output_products(
    include_required: bool = True,
    include_energy: bool = True,
    include_quality: bool = True,
    include_surface: bool = False,
    include_radiation: bool = False,
    custom_products: list = None
) -> list:
    """Generate a custom output product list based on inclusion flags.
    
    Args:
        include_required: Include required ET products
        include_energy: Include energy balance products (Rn, G, H)
        include_quality: Include quality layer products
        include_surface: Include surface property products
        include_radiation: Include radiation products
        custom_products: Additional custom products to include
    
    Returns:
        List of product tuples (output_name, band_name, dtype)
    """
    products = []
    
    if include_required:
        products.extend(OUTPUT_PRODUCTS['required'])
    if include_energy:
        products.extend(OUTPUT_PRODUCTS['energy_balance'])
    if include_quality:
        products.extend(OUTPUT_PRODUCTS['quality'])
    if include_surface:
        products.extend(OUTPUT_PRODUCTS['surface'])
    if include_radiation:
        products.extend(OUTPUT_PRODUCTS['radiation'])
    
    if custom_products:
        products.extend(custom_products)
    
    return products


# Predefined output product presets
OUTPUT_PRESETS = {
    # Minimal: Only essential ET products
    "minimal": [
        ("ETa_daily", "ET_daily", "float32"),
        ("ETrF", "ETrF", "float32")
    ],
    # Standard: ET products with energy balance
    "standard": [
        ("ETa_daily", "ET_daily", "float32"),
        ("ET_inst", "ET_inst", "float32"),
        ("ETrF", "ETrF", "float32"),
        ("Rn", "R_n", "float32"),
        ("G", "G", "float32"),
        ("H", "H", "float32"),
        ("LE", "LE", "float32")
    ],
    # Full: All available products
    "full": None,  # None means all products
    
    # ET only: Instantaneous and daily ET
    "et_only": [
        ("ETa_daily", "ET_daily", "float32"),
        ("ET_inst", "ET_inst", "float32"),
        ("ETrF", "ETrF", "float32")
    ],
    
    # With quality: ET products with quality layers
    "with_quality": [
        ("ETa_daily", "ET_daily", "float32"),
        ("ET_inst", "ET_inst", "float32"),
        ("ETrF", "ETrF", "float32"),
        ("ET_quality_class", "ET_quality_class", "uint8"),
        ("ETa_classified", "ETa_class", "uint8"),
        ("CWSI", "CWSI", "float32")
    ],
    
    # Research: All products including surface and radiation
    "research": None  # All products
}
