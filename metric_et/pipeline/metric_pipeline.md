# METRIC ET Pipeline Documentation

## Overview

The `METRICPipeline` class implements the complete Mapping Evapotranspiration at high Resolution with Internalized Calibration (METRIC) algorithm for estimating evapotranspiration (ET) from Landsat satellite imagery. This pipeline processes Landsat data through a series of steps to produce high-resolution ET maps.

## Pipeline Architecture

The pipeline follows a sequential processing approach with the following main steps:

1. **Data Loading** - Load Landsat imagery and meteorological data
2. **Preprocessing** - Apply cloud masking and quality control
3. **Surface Properties** - Calculate vegetation indices, albedo, emissivity, and roughness
4. **Radiation Balance** - Compute shortwave and longwave radiation components
5. **Energy Balance** - Calculate energy fluxes using surface temperature
6. **Calibration** - Apply METRIC calibration using anchor pixels
7. **ET Calculation** - Compute instantaneous and daily evapotranspiration
8. **Output** - Save results and create visualizations

## Key Components

### Data Loading (`load_data`)

Loads Landsat data from a directory and fetches spatially-varying meteorological data from Open-Meteo API. The method:

- Reads Landsat bands using `LandsatReader`
- Applies ROI clipping using GeoJSON boundaries
- Fetches weather data dynamically based on scene extent
- Converts temperature to Kelvin and stores all data in a `DataCube`

### Radiation Balance Calculations (RS Calculations)

The `calculate_radiation_balance()` method is central to the METRIC algorithm and computes all radiation components. This is where the **RS (shortwave radiation) calculations** are performed.

#### Shortwave Radiation (`ShortwaveRadiation`)

Calculates incoming shortwave radiation (Rs↓) and reflected shortwave radiation (Rs↑):

- **Rs↓ (Downward Shortwave)**: Uses solar geometry and atmospheric transmittance
- **Rs↑ (Upward Shortwave)**: Calculated as `Rs↑ = Rs↓ × albedo`
- **Net Shortwave (Rns)**: `Rns = Rs↓ - Rs↑ = Rs↓ × (1 - albedo)`

**Key Formulas:**
```
Rs↓ = S0 × cos(θ) × τ_sw × τ_atm
Rns = Rs↓ × (1 - albedo)
```

Where:
- `S0`: Solar constant (1367 W/m²)
- `θ`: Solar zenith angle
- `τ_sw`: Atmospheric transmittance for shortwave
- `τ_atm`: Broadband atmospheric transmittance
- `albedo`: Surface broadband albedo

#### Longwave Radiation (`LongwaveRadiation`)

Calculates thermal infrared radiation components:

- **Rl↓ (Downward Longwave)**: Atmospheric emission using air temperature and emissivity
- **Rl↑ (Upward Longwave)**: Surface emission using surface temperature and emissivity
- **Net Longwave (Rnl)**: `Rnl = Rl↓ - Rl↑`

**Key Formulas:**
```
Rl↓ = ε_atm × σ × T_air⁴
Rl↑ = ε_surface × σ × T_surface⁴
Rnl = Rl↓ - Rl↑
```

Where:
- `ε_atm`: Atmospheric emissivity (function of air temperature and humidity)
- `ε_surface`: Surface emissivity (calculated from NDVI and albedo)
- `σ`: Stefan-Boltzmann constant (5.67 × 10⁻⁸ W/m²/K⁴)
- `T_air`, `T_surface`: Temperatures in Kelvin

#### Net Radiation (`NetRadiation`)

Combines shortwave and longwave components:

**Total Net Radiation:**
```
Rn = Rns + Rnl = Rs↓×(1-albedo) + (Rl↓ - Rl↑)
```

**Daytime Net Radiation:**
```
Rn_daytime = Rn × (daytime_fraction)
```

Where `daytime_fraction` is calculated based on sunrise/sunset times or solar geometry, representing the fraction of daylight hours when net radiation is positive.

### Energy Balance (`calculate_energy_balance`)

Uses the radiation balance to compute energy fluxes:

- **Soil Heat Flux (G)**: Estimated from Rn and vegetation cover
- **Sensible Heat Flux (H)**: Calculated using temperature differences (may be negative for cold pixels during calibration, which is normal)
- **Latent Heat Flux (LE)**: `LE = Rn - G - H`

### METRIC Calibration (`calibrate`)

The core of METRIC methodology involves calibrating the sensible heat flux using anchor pixels:

#### Anchor Pixel Selection

Selects extreme pixels representing:
- **Cold pixel**: Fully vegetated, well-watered (minimum surface temperature)
- **Hot pixel**: Bare soil, dry conditions (maximum surface temperature)

#### dT Calibration

Calibrates the temperature difference relationship:
```
dT = a + b × (Ts - T_air)
```

Where `dT` is the aerodynamic temperature difference used in sensible heat flux calculations.

#### Instantaneous ET0 Calculation

Critical for RS calculations - converts daily ET0 to instantaneous ET0 at overpass time:

**Energy Ratio Method (Primary):**
```
ET0_inst = ET0_daily × (Rs_inst / Rs_daily_hourly)
```

Where:
- `Rs_inst`: Instantaneous shortwave radiation at overpass (W/m²)
- `Rs_daily_hourly`: Daily shortwave radiation converted to hourly rate (MJ/m²/hr)

**Conversion Details:**
- `Rs_inst_MJ = Rs_inst_W × 0.0036` (W/m² to MJ/m²/hr)
- `Rs_daily_hourly = Rs_daily_MJ/day ÷ 24` (MJ/m²/day to MJ/m²/hr)

**Fallback Method:**
```
ET0_inst = ET0_daily ÷ 12
```

Used when radiation data is unavailable or the energy ratio falls outside typical bounds (0.01-0.5). The divisor of 12 represents the typical fraction of daily ET occurring during Landsat overpass hours.

### ET Calculation (`calculate_et`)

#### Instantaneous ET
```
ET_inst = LE / (ρ × λ)
```

Where:
- `ET_inst`: Instantaneous evapotranspiration (mm/hr)
- `LE`: Latent heat flux (W/m²)
- `ρ`: Air density (kg/m³)
- `λ`: Latent heat of vaporization (J/kg)

#### Reference ET Fraction (ETrF)
```
ETrF = ET_inst / ETr_inst
```

Where:
- `ETrF`: Reference ET fraction (dimensionless, 0-1.5 typical range)
- `ETr_inst`: Instantaneous alfalfa reference ET (mm/hr)

#### Daily ET
```
ET_daily = ETrF × ETr_daily
```

Where:
- `ET_daily`: Daily evapotranspiration (mm/day)
- `ETr_daily = ET0_daily × 1.15` (FAO grass to alfalfa conversion)

## RS Calculations in Detail

The RS (shortwave radiation) calculations are fundamental to METRIC accuracy:

### 1. Instantaneous Shortwave Radiation (Rs↓)
- Calculated using solar geometry and atmospheric conditions
- Units: W/m²
- Used for ET0_inst conversion and energy balance

### 2. Daily Shortwave Radiation Sum
- Fetched from Open-Meteo API
- Units: MJ/m²/day
- Used for scaling ET0 to instantaneous values

### 3. Radiation Ratio for ET0 Scaling
```
ratio = (Rs_inst_W × 0.0036) / (Rs_daily_MJ/day ÷ 24)
ET0_inst = ET0_daily × ratio
```

This ratio represents the fraction of daily ET that occurs at satellite overpass time.

### 4. Net Shortwave Radiation
```
Rns = Rs↓ × (1 - albedo)
```

Critical for energy balance closure.

## Configuration and Parameters

The pipeline uses a configuration dictionary for customizable parameters:

```python
config = {
    'calibration': {
        'method': 'triangle',  # or 'manual'
    },
    'et0': {
        'default_instantaneous': 0.65,
        'fallback_divisor': 12.0,
    },
    'temperature': {
        'default_kelvin': 293.15,
    }
}
```

## Output Products

The pipeline produces:

- **ET_inst**: Instantaneous ET (mm/hr)
- **ET_daily**: Daily ET (mm/day)
- **ETrF**: Reference ET fraction (dimensionless)
- **Surface properties**: NDVI, albedo, emissivity, LAI
- **Energy fluxes**: Rn, G, H, LE
- **Quality metrics**: Confidence and quality classes

## Usage Example

```python
from metric_et.pipeline.metric_pipeline import METRICPipeline

# Initialize pipeline
pipeline = METRICPipeline(config=config, roi_path='roi.geojson')

# Run processing
results = pipeline.run(
    landsat_dir='landsat_data/landsat_20251221/',
    meteo_data={},
    output_dir='output/',
    roi_path='roi.geojson'
)

# Access results
et_daily = results['ET_daily']
et_inst = results['ET_inst']
```

## Validation and Quality Control

The pipeline includes comprehensive validation:

- **Radiation balance closure**: Rn = Rns + Rnl
- **Energy balance closure**: Rn = G + H + LE
- **ET reasonableness checks**: 0 ≤ ETrF ≤ 1.5
- **Spatial consistency**: Weather data varies with location

## References

- Allen, R.G., et al. (2007). METRIC: Mapping Evapotranspiration at High Resolution.
- FAO-56: Crop Evapotranspiration Guidelines.

## Notes on RS Calculations

The RS calculations are particularly critical because:

1. **ET0 Scaling**: Accurate Rs_inst/Rs_daily ratio ensures proper calibration
2. **Energy Balance**: Rs↓ drives the entire energy partitioning
3. **Temporal Integration**: Proper conversion from instantaneous to daily ET
4. **Spatial Variation**: Weather data varies across the scene extent

The pipeline uses multiple fallback methods for RS calculations to ensure robustness when data is missing or unreliable.