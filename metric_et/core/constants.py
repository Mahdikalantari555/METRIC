"""Physical constants for METRIC ETa model."""

import numpy as np

# ============================================================================
# RADIATION CONSTANTS
# ============================================================================

# Stefan-Boltzmann constant (W/m²/K⁴)
STEFAN_BOLTZMANN = 5.670374419e-8

# Solar constant (W/m²)
SOLAR_CONSTANT = 1361.0

# ============================================================================
# ATMOSPHERIC CONSTANTS
# ============================================================================

# Von Karman constant (dimensionless)
VON_KARMAN = 0.41

# Air density at sea level (kg/m³)
AIR_DENSITY = 1.225

# Air density at reference pressure (kg/m³)
AIR_DENSITY_REF = 1.225

# Specific heat capacity of air at constant pressure (J/kg/K)
AIR_SPECIFIC_HEAT = 1013.0

# Latent heat of vaporization at 25°C (J/kg)
LATENT_HEAT_VAPORIZATION = 2.45e6

def latent_heat_vaporization(temperature_k: float) -> float:
    """
    Calculate latent heat of vaporization as a function of temperature.
    
    Args:
        temperature_k: Air temperature in Kelvin
        
    Returns:
        Latent heat of vaporization in J/kg
    """
    t_celsius = temperature_k - 273.15
    return 2.501e6 - 2361.0 * t_celsius

# ============================================================================
# GRAVITATIONAL CONSTANTS
# ============================================================================

GRAVITATIONAL_ACCELERATION = 9.81

# ============================================================================
# WATER PROPERTIES
# ============================================================================

WATER_DENSITY = 1000.0
WATER_THERMAL_CONDUCTIVITY = 0.6

# ============================================================================
# SOIL PROPERTIES
# ============================================================================

SOIL_BULK_DENSITY = 1300.0
SOIL_HEAT_CAPACITY = 1.2e6

# ============================================================================
# VEGETATION PROPERTIES
# ============================================================================

NDVI_VEGETATION_THRESHOLD = 0.2
MAX_FRACTIONAL_COVER = 1.0
MIN_LAI = 0.0
MAX_LAI = 6.0

# ============================================================================
# TEMPERATURE CONSTANTS
# ============================================================================

FREEZING_POINT = 273.15
CELSIUS_TO_KELVIN = 273.15

# ============================================================================
# TIME CONSTANTS
# ============================================================================

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
HOURS_PER_DAY = 24.0

# ============================================================================
# ANGLE CONVERSIONS
# ============================================================================

DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi

# ============================================================================
# CONVERSION FACTORS
# ============================================================================

W_TO_MM_PER_DAY = 0.0864
MM_PER_DAY_TO_W = 11.574
MJ_M2_DAY_TO_W = 1e6 / 86400
W_TO_MJ_M2_DAY = 86400 / 1e6

# ============================================================================
# PRESSURE CONVERSIONS
# ============================================================================

PA_TO_HPA = 0.01
HPA_TO_PA = 100.0

# ============================================================================
# WIND SPEED CONVERSIONS
# ============================================================================

WIND_2M_TO_10M = 1.13
WIND_10M_TO_2M = 0.8846

# ============================================================================
# ATMOSPHERIC PROFILE
# ============================================================================

ENVIRONMENTAL_LAPSE_RATE = -0.0065
STANDARD_PRESSURE = 101325.0
STANDARD_TEMPERATURE = 288.15

__all__ = [
    'STEFAN_BOLTZMANN', 'SOLAR_CONSTANT', 'VON_KARMAN', 'AIR_DENSITY',
    'AIR_DENSITY_REF', 'AIR_SPECIFIC_HEAT', 'LATENT_HEAT_VAPORIZATION',
    'latent_heat_vaporization', 'GRAVITATIONAL_ACCELERATION', 'WATER_DENSITY',
    'WATER_THERMAL_CONDUCTIVITY', 'SOIL_BULK_DENSITY', 'SOIL_HEAT_CAPACITY',
    'NDVI_VEGETATION_THRESHOLD', 'MAX_FRACTIONAL_COVER', 'MIN_LAI', 'MAX_LAI',
    'FREEZING_POINT', 'CELSIUS_TO_KELVIN', 'SECONDS_PER_HOUR', 'SECONDS_PER_DAY',
    'HOURS_PER_DAY', 'DEG_TO_RAD', 'RAD_TO_DEG', 'W_TO_MM_PER_DAY',
    'MM_PER_DAY_TO_W', 'MJ_M2_DAY_TO_W', 'W_TO_MJ_M2_DAY', 'PA_TO_HPA',
    'HPA_TO_PA', 'WIND_2M_TO_10M', 'WIND_10M_TO_2M', 'ENVIRONMENTAL_LAPSE_RATE',
    'STANDARD_PRESSURE', 'STANDARD_TEMPERATURE'
]
