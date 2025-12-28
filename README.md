# METRIC ETa Model

## Mapping Evapotranspiration with a Residual-Based Calibration

A Python implementation of the METRIC (Mapping Evapotranspiration with a Residual-Based Calibration) model for calculating evapotranspiration (ETa) from Landsat satellite imagery.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Scientific Basis](#scientific-basis)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Usage Guide](#usage-guide)
7. [Input Data Requirements](#input-data-requirements)
8. [Output Products](#output-products)
9. [Algorithm Documentation](#algorithm-documentation)
10. [API Reference](#api-reference)
11. [Examples](#examples)
12. [FAQ](#faq)
13. [References](#references)

---

## Project Overview

The METRIC ETa model is a remote sensing-based method for calculating field-scale evapotranspiration (ET) using thermal infrared imagery from satellites. This implementation processes Landsat Collection 2 Level-2 data to produce instantaneous and daily ET estimates at 30m resolution.

### What is METRIC?

METRIC (Mapping Evapotranspiration with a Residual-Based Calibration) was developed at the University of Idaho and is widely used for:

- **Agricultural water management**: Quantifying crop water use
- **Irrigation scheduling**: Optimizing water application
- **Water balance studies**: Regional water resource assessment
- **Drought monitoring**: Tracking vegetation water stress

### Key Features

- **High-resolution ET mapping**: 30m pixel size from Landsat
- **Anchor pixel calibration**: Uses hot/cold reference pixels for accurate calibration
- **Complete energy balance**: Computes Rn, G, H, and LE components
- **Multi-scene processing**: Supports time series analysis
- **Flexible configuration**: YAML-based configuration system
- **Multiple output formats**: GeoTIFF, NetCDF, and visualizations
- **CLI interface**: Easy-to-use command line tools

### Scientific Basis

METRIC applies the surface energy balance equation:

$$R_n - G = H + LE$$

Where:
- $R_n$ = Net radiation (W/m²)
- $G$ = Soil heat flux (W/m²)
- $H$ = Sensible heat flux (W/m²)
- $LE$ = Latent heat flux (W/m²)

The model uses thermal band data and meteorological inputs to compute ET as the residual of the energy balance.

---

## Installation

### System Requirements

- **Python**: 3.9 or higher
- **Operating System**: Windows 10+, macOS, or Linux
- **Memory**: Minimum 8GB RAM (16GB recommended for large scenes)
- **Storage**: 2GB for installation + space for input/output data

### Python Dependencies

```
numpy>=1.21.0
xarray>=0.18.0
rasterio>=1.2.0
loguru>=0.5.0
pyyaml>=5.4.0
matplotlib>=3.4.0
pandas>=1.3.0
scipy>=1.7.0
```

### Installation Methods

#### Method 1: Conda Environment (Recommended)

Create a conda environment from the provided environment file and install the package:

```bash
conda env create -f environment.yml
conda activate metric-env
pip install -e .
```

This will create an environment named `metric-env` with all required dependencies installed and make the METRIC ETa package importable.

#### Method 2: Pip Installation (Development Mode)

For development or if you prefer to manage dependencies manually:

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd METRIC

# Create conda environment and install dependencies
conda env create -f environment.yml
conda activate metric-env

# Install in development mode
pip install -e .
```

#### Method 3: Manual Installation

If you prefer to install dependencies manually:

```bash
pip install -r metric_et/requirements.txt
pip install -e .
```

### Verification Steps

```python
# Verify installation
from metric_et import METRICPipeline
from metric_et.core.datacube import DataCube
from metric_et.calibration import DTCalibration

print("METRIC ETa installed successfully!")
print(f"DataCube available: {DataCube is not None}")
print(f"Pipeline available: {METRICPipeline is not None}")
```

---

## Quick Start

### Minimal Example

Process a single Landsat scene with default settings:

```python
from metric_et import METRICPipeline
from metric_et.io.landsat_reader import read_landsat_scene
from metric_et.io.meteo_reader import load_weather_data

# Initialize the pipeline
pipeline = METRICPipeline()

# Load input data
landsat_dir = "data/landsat_20251204_166_038/"

# Run the pipeline
results = pipeline.run(
    landsat_dir=landsat_dir,
    meteo_data=[], #Pipe line fetch weather data
    output_dir="output/"
)

print("Processing complete!")
print(f"Results saved to output/")
```

### Command Line Interface

```bash
# Process a single scene
metric-et process --scene data/landsat_20251204_166_038/ --output output/

# Process multiple scenes
metric-et batch --input data/ --output output/

# Generate visualization
metric-et visualize --input output/ET_daily.tif --colormap viridis
```

---

## Usage Guide

### CLI Commands

#### Process Single Scene

```bash
metric-et process \
    --scene /path/to/landsat_scene \
    --weather /path/to/weather.csv \
    --output /path/to/output \
    --config /path/to/config.yaml
```

#### Batch Processing

```bash
metric-et batch \
    --input /path/to/scenes/ \
    --weather /path/to/weather.csv \
    --output /path/to/output \
    --threads 4
```

#### Generate Reports

```bash
metric-et report \
    --input /path/to/results/ \
    --format pdf \
    --output /path/to/report.pdf
```

### Configuration File Options

```yaml
# config.yaml

# Input/Output directories
input_dir: data/
output_dir: output/

# Date range for processing
date_range:
  start: 2025-09-15
  end: 2025-12-04

# Processing parameters
cloud_threshold: 30

# Calibration settings
calibration:
  method: auto          # auto, hot-cold, manual
  cold_etrf: 1.05       # Expected ETrF for cold pixel
  hot_etrf: 0.05        # Expected ETrF for hot pixel

# Output settings
output:
  format: GeoTIFF       # GeoTIFF, NetCDF
  bands:
    - ET_daily
    - ETrF
    - LE
    - dT

# Visualization settings
visualization:
  colormap: viridis
  dpi: 300
  show_plots: false

# Logging settings
logging:
  level: INFO
  file: metric_et.log
```

### Parameter Descriptions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cloud_threshold` | int | 30 | Maximum cloud cover percentage to process |
| `calibration.method` | string | auto | Calibration method (auto, hot-cold, manual) |
| `calibration.cold_etrf` | float | 1.05 | Expected ETrF for cold (wet) pixel |
| `calibration.hot_etrf` | float | 0.05 | Expected ETrF for hot (dry) pixel |
| `output.format` | string | GeoTIFF | Output format (GeoTIFF, NetCDF) |
| `visualization.colormap` | string | viridis | Colormap for visualizations |

### Common Workflows

#### 1. Single Scene Processing

```python
from metric_et import METRICPipeline

pipeline = METRICPipeline(config={
    'calibration': {
        'method': 'hot-cold',
        'cold_etrf': 1.05,
        'hot_etrf': 0.05
    }
})

results = pipeline.run(
    landsat_dir="data/landsat_20251204_166_038/",
    meteo_data={"temperature_2m": 25.0, "wind_speed": 3.0},
    output_dir="output/single_scene/"
)
```

#### 2. Time Series Processing

```python
from metric_et import METRICPipeline
import os
from datetime import datetime, timedelta

# Process multiple scenes
scene_dates = [
    "2025-09-15",
    "2025-09-23",
    "2025-10-01",
    "2025-10-09",
]

pipeline = METRICPipeline()

for date in scene_dates:
    scene_dir = f"data/landsat_{date.replace('-', '')}_166_038/"
    
    if os.path.exists(scene_dir):
        results = pipeline.run(
            landsat_dir=scene_dir,
            meteo_data={"temperature_2m": 25.0, "wind_speed": 3.0},
            output_dir=f"output/{date}/"
        )
        print(f"Processed: {date}")
```

---

## Input Data Requirements

### Landsat Data Format

**Required Format**: Landsat Collection 2 Level-2

**Required Bands**:
| Band | Filename | Description | Resolution |
|------|----------|-------------|------------|
| Blue | `blue.tif` | Coastal/Aerosol band | 30m |
| Green | `green.tif` | Green band | 30m |
| Red | `red.tif` | Red band | 30m |
| NIR | `nir08.tif` | Near-infrared | 30m |
| SWIR-1 | `swir16.tif` | Shortwave IR-1 | 30m |
| SWIR-2 | `swir22.tif` | Shortwave IR-2 | 30m |
| Thermal | `lwir11.tif` | Thermal infrared | 100m (resampled) |
| QA | `qa.tif` | Quality assessment | 30m |
| QA Pixel | `qa_pixel.tif` | Pixel quality | 30m |

**Metadata File**: `MTL.json` (required)

**Scaling**:
- Reflective bands: Scale factor 0.000275, Offset -0.2
- Thermal band: Scale factor 0.0001, No offset (Kelvin)

### Weather Data Format

CSV format with the following columns:

```csv
datetime,temperature_2m,relative_humidity,wind_speed,pressure,solar_radiation
2025-09-15T10:30:00,25.3,45,3.2,1013,850
2025-09-23T10:30:00,24.8,50,2.8,1015,820
```

**Required Fields**:
- `datetime`: ISO format timestamp
- `temperature_2m`: Air temperature at 2m (°C)
- `relative_humidity`: Relative humidity (%)
- `wind_speed`: Wind speed at 10m (m/s)
- `pressure`: Atmospheric pressure (hPa)
- `solar_radiation`: Incoming solar radiation (W/m²)

### DEM Requirements

**Optional**: Digital Elevation Model for terrain correction

**Format**: GeoTIFF with the same CRS and resolution as Landsat data

**Fields**:
- Elevation (meters above sea level)
- Slope (degrees)
- Aspect (degrees)

### Coordinate Systems

**Supported CRS**:
- UTM zones (EPSG:326XX, EPSG:327XX)
- WGS84 (EPSG:4326) - requires reprojection
- Custom projections supported via PROJ4 string

**Recommended**: UTM zone for the study area

---

## Output Products

### GeoTIFF Files

| File | Description | Units | Data Type |
|------|-------------|-------|-----------|
| `ET_daily.tif` | Daily evapotranspiration | mm/day | Float32 |
| `ET_inst.tif` | Instantaneous ET at overpass | mm/hr | Float32 |
| `ETrF.tif` | Reference ET fraction | - | Float32 |
| `LE.tif` | Latent heat flux | W/m² | Float32 |
| `H.tif` | Sensible heat flux | W/m² | Float32 |
| `Rn.tif` | Net radiation | W/m² | Float32 |
| `G.tif` | Soil heat flux | W/m² | Float32 |
| `dT.tif` | Temperature difference | K | Float32 |

### Statistics CSV

CSV file with pixel statistics for each output band:

```csv
band,mean,std,min,max,median
ET_daily,4.25,1.23,0.15,8.92,4.1
ETrF,0.65,0.18,0.02,1.28,0.62
LE,180.5,52.3,6.4,378.2,175.2
```

### Visualizations

- **ET maps**: Color-coded maps of daily ET
- **ETrF spatial patterns**: Vegetation stress visualization
- **Time series plots**: Multi-date ET trends
- **Scatter plots**: ET vs. NDVI, ET vs. temperature

### Metadata

- Processing parameters used
- Calibration coefficients
- Input data quality metrics
- Timestamp and software version

---

## Algorithm Documentation

### Energy Balance Equation

The surface energy balance is the foundation of the METRIC model:

$$R_n - G = H + LE$$

Where:
- $R_n$ = Net radiation (energy available)
- $G$ = Soil heat flux (energy into ground)
- $H$ = Sensible heat flux (energy heating the air)
- $LE$ = Latent heat flux (energy used for evaporation)

ET is computed from LE:

$$ET = \frac{LE}{\lambda}$$

Where $\lambda$ is the latent heat of vaporization (~2.45 MJ/kg).

### Anchor Pixel Methodology

METRIC uses two reference pixels to calibrate the model:

**Cold Pixel (Wet)**
- Well-watered vegetation or open water
- Expected ETrF ≈ 1.05 (slightly above reference)
- Minimum sensible heat flux
- Surface temperature close to air temperature

**Hot Pixel (Dry)**
- Dry bare soil or stressed vegetation
- Expected ETrF ≈ 0.05 (near zero)
- Maximum sensible heat flux
- Surface temperature much higher than air temperature

The calibrated dT relationship:

$$dT = a \cdot T_s + b$$

Where:
- $dT = T_s - T_a$ (temperature difference)
- $a$ and $b$ are calibrated from anchor pixels

### Calibration Process

1. Identify anchor pixel locations
2. Extract surface temperatures
3. Compute expected dT from ETrF targets
4. Solve linear calibration equation:

$$a = \frac{dT_{hot} - dT_{cold}}{T_{s,hot} - T_{s,cold}}$$

$$b = dT_{cold} - a \cdot T_{s,cold}$$

### ET Computation

**Step 1**: Calculate instantaneous LE from energy balance residual

$$LE = R_n - G - H$$

**Step 2**: Convert to instantaneous ET rate

$$ET_{inst} = \frac{LE}{\lambda} \times 3600 \quad (\text{mm/hr})$$

**Step 3**: Compute reference ET fraction

$$ETrF = \frac{ET_{inst}}{ETr_{inst}}$$

**Step 4**: Scale to daily ET

$$ET_{daily} = ETrF \times ETr_{daily}$$

---

## API Reference

### Core Classes

#### DataCube

```python
from metric_et.core.datacube import DataCube

# Create empty DataCube
cube = DataCube()

# Add band data
cube.add('ndvi', ndvi_array)
cube.add('albedo', albedo_array)

# Add scalar value
cube.add('air_temperature', 25.0)

# Retrieve data
ndvi = cube.get('ndvi')
temp = cube.get('air_temperature')

# List bands and scalars
bands = cube.bands()      # ['ndvi', 'albedo']
scalars = cube.scalars()  # ['air_temperature']
```

**Methods**:
- `add(name, data)`: Add band or scalar
- `get(name)`: Retrieve by name
- `bands()`: List all bands
- `scalars()`: List all scalars
- `shape(band_name)`: Get dimensions
- `update_crs(crs, transform)`: Set spatial reference

#### METRICPipeline

```python
from metric_et import METRICPipeline

# Initialize with config
pipeline = METRICPipeline(config={
    'calibration': {
        'method': 'hot-cold',
        'cold_etrf': 1.05,
        'hot_etrf': 0.05
    }
})

# Run complete pipeline
results = pipeline.run(
    landsat_dir='data/landsat_scene/',
    meteo_data={'temperature_2m': 25.0, 'wind_speed': 3.0},
    dem_path='data/dem.tif',
    output_dir='output/'
)
```

**Methods**:
- `run(...)`: Execute full processing pipeline
- `load_data(...)`: Load input data
- `preprocess()`: Apply preprocessing
- `calculate_surface_properties()`: Compute albedo, NDVI, etc.
- `calculate_radiation_balance()`: Compute Rn components
- `calculate_energy_balance()`: Compute G, H, LE
- `calibrate()`: Apply anchor pixel calibration
- `calculate_et()`: Compute ET and ETrF
- `get_results()`: Retrieve output arrays
- `save_results()`: Write output files

### Module Functions

#### Radiation Module

```python
from metric_et.radiation import NetRadiation

net_rad = NetRadiation(clip_negative=True)
cube = net_rad.compute(cube)  # Adds R_n, R_n_daytime
```

#### Energy Balance Module

```python
from metric_et.energy_balance import SensibleHeatFlux

h_flux = SensibleHeatFlux()
result = h_flux.calculate(
    rn=net_radiation,
    ts=surface_temp,
    ta=air_temperature,
    u=wind_speed,
    z0m=roughness_length
)
# result['H']: Sensible heat flux
# result['rah']: Aerodynamic resistance
# result['dT']: Temperature difference
```

#### Calibration Module

```python
from metric_et.calibration import DTCalibration

calib = DTCalibration(et0_inst=0.5)  # mm/hr at overpass
result = calib.calibrate(
    ts_cold=300.0,  # Cold pixel Ts (K)
    ts_hot=320.0,   # Hot pixel Ts (K)
    air_temperature=298.0  # Ta (K)
)
# result.a_coefficient: Slope
# result.b_coefficient: Intercept
```

#### ET Calculation Module

```python
from metric_et.et import InstantaneousET

et_calc = InstantaneousET()
result = et_calc.calculate(
    le=latent_heat,
    etr_inst=etr_instantaneous
)
# result['ET_inst']: Instantaneous ET (mm/hr)
# result['ETrF']: Reference ET fraction
```

---

## Examples

### Single Scene Processing

See [`examples/single_scene.py`](examples/single_scene.py)

### Time Series Processing

See [`examples/time_series.py`](examples/time_series.py)

### Custom Configuration

See [`examples/custom_config.py`](examples/custom_config.py)

### Complete Workflow

See [`examples/complete_workflow.py`](examples/complete_workflow.py)

---

## FAQ

### Common Issues

**Q: How do I handle scenes with high cloud cover?**
A: Set `cloud_threshold` in config.yaml. Scenes exceeding the threshold are skipped. Use the QA band to identify and mask cloud-contaminated pixels.

**Q: Why are my ET values negative?**
A: Negative ET values indicate energy balance closure issues. Check:
- Weather data quality
- Anchor pixel selection
- Cloud masking effectiveness

**Q: How do I reproject data to a different CRS?**
A: METRIC automatically handles CRS if input data is in consistent CRS. For reprojection, use GDAL or rasterio before processing.

### Best Practices

1. **Preprocess data**: Verify Landsat data quality before processing
2. **Check weather data**: Ensure meteorological inputs are accurate
3. **Validate anchor pixels**: Visually confirm hot/cold pixel locations
4. **Quality control**: Review ETrF spatial patterns for reasonableness
5. **Time series consistency**: Use consistent calibration settings across dates

### Performance Tips

- Use SSD storage for large time series
- Increase `max_workers` for parallel processing
- Reduce output resolution for rapid testing
- Enable cloud masking to skip invalid pixels

---

## References

### Scientific Papers

1. Allen, R.G., Tasumi, M., Trezza, R. (2007). "Satellite-based energy balance for mapping evapotranspiration with internalized calibration (METRIC) - Model." Journal of Irrigation and Drainage Engineering, 133(4), 380-394.

2. Allen, R.G., Tasumi, M., Morse, A., Trezza, R. (2007). "Satellite-based energy balance for mapping evapotranspiration with internalized calibration (METRIC) - Applications." Journal of Irrigation and Drainage Engineering, 133(4), 395-406.

3. Bastiaanssen, W.G.M., Menenti, M., Feddes, R.A., Holtslag, A.A.M. (1998). "A remote sensing surface energy balance algorithm for land (SEBAL) 1. Formulation." Journal of Hydrology, 212-213, 198-212.

### Related Resources

- [ASCE Standardized Reference Evapotranspiration Equation](https://ascelibrary.org/doi/10.1061/%28ASCE%290733-4744%282005%29131%3A1%282%29%29)
- [Landsat Collection 2 Product Documentation](https://www.usgs.gov/landsat/landsat-collection-2)
- [PySEBAL Repository](https://github.com/ethanson/pySEBAL)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- University of Idaho for the original METRIC methodology
- USGS for Landsat data products
- Open source community for the Python ecosystem

---

**Version**: 1.0.0  
**Last Updated**: 2025-12-23
