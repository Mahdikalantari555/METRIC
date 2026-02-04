# METRIC Evapotranspiration Modeling - Comprehensive Workflow Documentation

## Table of Contents

1. [Introduction](#1-introduction)
2. [Scientific Foundation](#2-scientific-foundation)
3. [Input Variables](#3-input-variables)
4. [Workflow Pipeline](#4-workflow-pipeline)
5. [Formulas and Calculations](#5-formulas-and-calculations)
6. [Anchor Pixel Calibration](#6-anchor-pixel-calibration)
7. [Energy Balance Components](#7-energy-balance-components)
8. [Evapotranspiration Calculation](#8-evapotranspiration-calculation)
9. [Constraints and Validation](#9-constraints-and-validation)
10. [Assumptions](#10-assumptions)
11. [Output Products](#11-output-products)
12. [Quality Assessment](#12-quality-assessment)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Introduction

### 1.1 What is METRIC?

METRIC (Mapping Evapotranspiration with a Residual-Based Calibration) is a remote sensing-based model developed by the University of Idaho for calculating actual evapotranspiration (ETa) from satellite imagery. The model uses the surface energy balance approach, where ET is computed as the residual of the energy balance equation.

### 1.2 Core Energy Balance Equation

The fundamental principle of METRIC is the surface energy balance:

```
Rn - G = H + LE
```

Where:
- **Rn** = Net radiation (W/m²)
- **G** = Soil heat flux (W/m²)
- **H** = Sensible heat flux (W/m²)
- **LE** = Latent heat flux (W/m²)

### 1.3 Key Concepts

1. **Anchor Pixel Calibration**: METRIC uses "anchor pixels" (hot and cold) to calibrate the sensible heat flux calculation
2. **Reference ET Fraction (ETrF)**: The ratio of actual ET to reference ET, used for daily ET scaling
3. **Instantaneous to Daily Extrapolation**: Converting satellite overpass-time ET to daily totals

---

## 2. Scientific Foundation

### 2.1 Energy Balance Components

#### Net Radiation (Rn)

Net radiation represents the balance between incoming and outgoing radiation:

```
Rn = Rns - Rnl
```

Where:
- **Rns** = Net shortwave radiation (incoming - reflected)
- **Rnl** = Net longwave radiation (incoming - outgoing)

#### Soil Heat Flux (G)

Heat transferred into or out of the soil:

```
G = Rn × [0.05 + 0.18 × exp(-0.5 × LAI)]  (vegetated surfaces)
G = 0.3 × Rn  (bare soil)
G = 0.4 × Rn  (wet bare soil)
```

#### Sensible Heat Flux (H)

Heat transferred between surface and air due to temperature difference:

```
H = ρ × Cp × (dT / rah)
```

Where:
- **ρ** = Air density (~1.2 kg/m³)
- **Cp** = Specific heat of air (~1004 J/kg/K)
- **dT** = Temperature difference (Ts - Ta)
- **rah** = Aerodynamic resistance (s/m)

#### Latent Heat Flux (LE)

Energy used for evaporation (the residual from energy balance):

```
LE = Rn - G - H
```

### 2.2 From Latent Heat to Evapotranspiration

```
ET_inst = LE / λ
```

Where:
- **λ** = Latent heat of vaporization (~2.45 × 10⁶ J/kg)
- **ET_inst** = Instantaneous ET rate (mm/hr)

---

## 3. Input Variables

### 3.1 Remote Sensing Data (Landsat)

| Variable | Band | Description | Unit |
|----------|------|-------------|------|
| Blue | Band 2 | 450-520 nm | Reflectance |
| Green | Band 3 | 520-600 nm | Reflectance |
| Red | Band 4 | 630-690 nm | Reflectance |
| NIR | Band 5 | 760-900 nm | Reflectance |
| SWIR1 | Band 6 | 1550-1750 nm | Reflectance |
| SWIR2 | Band 7 | 2080-2300 nm | Reflectance |
| Thermal | Band 10/11 | 10,300-12,500 nm | Brightness Temp (K) |
| QA Pixel | - | Quality assessment | Bit mask |

### 3.2 Meteorological Data

#### Hourly Variables (at satellite overpass ~10:30)

| Variable | Source | Description | Unit | Used In |
|----------|--------|-------------|------|---------|
| Air Temperature (Ta) | Open-Meteo | 2m air temperature at overpass | K | dT calculation, LE flux |
| Wind Speed (u) | Open-Meteo | 10m wind speed at overpass | m/s | Aerodynamic resistance (rah) |
| Shortwave Radiation (Rs) | Open-Meteo | Incoming shortwave at overpass | W/m² | Net radiation (Rn) |
| Relative Humidity (RH) | Open-Meteo | Relative humidity at overpass | % | Vapor pressure, LE flux |
| Surface Pressure | Open-Meteo | Surface pressure at overpass | Pa | Air density calculations |

#### Daily Variables (whole day totals)

| Variable | Source | Description | Unit | Used In |
|----------|--------|-------------|------|---------|
| ET0 FAO | Open-Meteo | Daily reference ET (grass) | mm/day | ETrF scaling, ET_daily calculation |
| Shortwave Radiation Sum | Open-Meteo | Daily total incoming shortwave | MJ/m²/day | ET0_inst conversion |

**Usage in METRIC:**
```python
# Daily ET calculation
ET_daily = ETrF × ET0_daily

# Instantaneous reference ET (for calibration)
ET0_inst = ET0_daily × (Rs_inst / Rs_daily)
```

The **daily ET0** is the key variable for scaling instantaneous ETrF to daily ET totals. The daily shortwave radiation is used to convert daily ET0 to instantaneous values for calibration purposes.

### 3.3 Derived Surface Properties

| Variable | Formula | Description | Range |
|----------|---------|-------------|-------|
| NDVI | (NIR - Red) / (NIR + Red) | Vegetation index | -1 to 1 |
| Albedo | Weighted sum of bands | Surface reflectance | 0 to 1 |
| Emissivity | Function of NDVI/LAI | Thermal emissivity | 0.9 to 1.0 |
| LAI | f(NDVI) | Leaf Area Index | 0 to 8 |
| Surface Temp (Ts) | Calibrated thermal | Kinetic temperature | 250-400 K |

### 3.4 Configuration Parameters

```python
@dataclass
class METRICConfig:
    # QA thresholds
    cloud_reject_threshold: float = 0.70      # Reject if >70% cloud
    cloud_low_quality_threshold: float = 0.30  # Flag if >30% cloud
    
    # Anchor pixel settings
    cluster_size: int = 20                     # Number of pixels in cluster
    min_temperature: float = 200.0             # Valid Ts minimum (K)
    max_temperature: float = 400.0             # Valid Ts maximum (K)
    
    # ET constraints
    min_et_daily: float = 0.0                  # Minimum daily ET (mm/day)
    max_et_daily: float = 30.0                 # Maximum daily ET (mm/day)
    min_etrf: float = 0.0                      # Minimum ETrF
    max_etrf: float = 2.0                      # Maximum ETrF
    
    # Regional adaptations
    region: str = None                         # Region preset
```

---

## 4. Workflow Pipeline

### 4.1 Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                    METRIC Processing Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: Data Loading & Preprocessing                          │
│  ├── Load Landsat bands (6 reflective + 1 thermal)              │
│  ├── Load meteorological data from Open-Meteo API               │
│  ├── Clip to ROI (Region of Interest)                           │
│  └── Apply cloud masking                                        │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: Surface Properties Calculation                        │
│  ├── Calculate vegetation indices (NDVI, LAI, SAVI)             │
│  ├── Calculate broadband albedo                                 │
│  ├── Calculate emissivity                                       │
│  └── Calculate land surface temperature (LST)                   │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: Radiation Balance                                     │
│  ├── Calculate incoming shortwave radiation                     │
│  ├── Calculate incoming longwave radiation                      │
│  └── Calculate net radiation (Rn)                               │
├─────────────────────────────────────────────────────────────────┤
│  Stage 4: Soil Heat Flux (G)                                    │
│  └── Calculate G from Rn, NDVI, and surface temperature         │
├─────────────────────────────────────────────────────────────────┤
│  Stage 5: Anchor Pixel Calibration                              │
│  ├── Select hot and cold anchor pixels                          │
│  ├── Calculate calibration coefficients (a, b)                  │
│  └── Validate calibration quality                               │
├─────────────────────────────────────────────────────────────────┤
│  Stage 6: Energy Balance (H and LE)                             │
│  ├── Calculate sensible heat flux (H)                           │
│  └── Calculate latent heat flux (LE)                            │
├─────────────────────────────────────────────────────────────────┤
│  Stage 7: Evapotranspiration                                    │
│  ├── Calculate instantaneous ET                                 │
│  ├── Calculate ETrF (reference ET fraction)                     │
│  └── Scale to daily ET                                          │
├─────────────────────────────────────────────────────────────────┤
│  Stage 8: Output Generation                                      │
│  ├── Save ET products (GeoTIFF)                                 │
│  ├── Create visualization maps                                  │
│  └── Write metadata file                                        │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Detailed Stage Descriptions

#### Stage 1: Data Loading & Preprocessing

```python
def load_data(self, landsat_dir: str, meteo_data: Dict, roi_path: str) -> None:
    """
    Load input data into DataCube.
    
    Steps:
    1. Initialize DataCube
    2. Load Landsat bands using LandsatReader
    3. Load ROI geometry
    4. Fetch meteorological data from Open-Meteo API
    5. Clip to ROI boundary
    6. Apply cloud masking
    """
```

**Input Requirements:**
- Landsat directory with MTL.json metadata file
- GeoJSON file defining ROI boundary
- Internet connection for Open-Meteo API

**Quality Checks:**
- Verify all required bands are present
- Check for cloud coverage >70% (HARD REJECT)
- Flag scenes with >30% cloud (LOW QUALITY)

#### Stage 2: Surface Properties Calculation

```python
def calculate_surface_properties(self) -> None:
    """
    Calculate all surface properties from Landsat bands.
    
    Vegetation Indices:
    - NDVI = (NIR - Red) / (NIR + Red)
    - EVI = 2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)
    - LAI = -ln((0.69 - NDVI)/0.59) / 0.91
    
    Albedo:
    - Broadband albedo from weighted band combination
    """
```

#### Stage 3: Radiation Balance

```python
def calculate_radiation_balance(self) -> None:
    """
    Calculate radiation balance components.
    
    Shortwave Radiation:
    - Rs_down = Extraterrestrial radiation × Atmospheric transmissivity
    - Rns = (1 - Albedo) × Rs_down
    
    Longwave Radiation:
    - Rl_down = εa × σ × Ta^4
    - Rl_up = εs × σ × Ts^4
    - Rnl = Rl_down - Rl_up
    
    Net Radiation:
    - Rn = Rns - Rnl
    """
```

#### Stage 4: Soil Heat Flux

```python
def calculate_soil_heat_flux(self) -> None:
    """
    Calculate soil heat flux (G).
    
    Methods:
    - Automatic: G = Rn × f(NDVI, LAI)
    - Simplified: G = 0.3 × Rn
    - Advanced: G = f(Rn, Ts, Ta, NDVI)
    """
```

#### Stage 5: Anchor Pixel Calibration

```python
def calibrate(self) -> None:
    """
    Perform METRIC anchor pixel calibration.
    
    Steps:
    1. Select cold pixel (well-watered vegetation)
    2. Select hot pixel (dry bare soil)
    3. Calculate dT calibration coefficients
    4. Validate calibration quality
    """
```

#### Stage 6: Energy Balance

```python
def calculate_energy_balance(self) -> None:
    """
    Calculate H and LE using calibrated dT relationship.
    
    Sensible Heat Flux:
    - dT = Ts - Ta
    - dT_calibrated = a × dT + b
    - H = ρ × Cp × dT_calibrated / rah
    
    Latent Heat Flux (residual):
    - LE = Rn - G - H
    
    Evaporative Fraction:
    - EF = LE / (Rn - G)
    """
```

#### Stage 7: Evapotranspiration

```python
def calculate_et(self) -> None:
    """
    Calculate instantaneous and daily ET.
    
    Instantaneous ET:
    - ET_inst = LE / λ (mm/hr)
    - ETrF = ET_inst / ETr_inst
    
    Daily ET:
    - ET_daily = ETrF × ETr_daily
    """
```

---

## 5. Formulas and Calculations

### 5.1 Vegetation Indices

#### Normalized Difference Vegetation Index (NDVI)

```python
NDVI = (NIR - Red) / (NIR + Red)
```

**Formula Explanation:**
- Measures vegetation greenness
- Range: -1 to 1
- Healthy vegetation: 0.6 to 0.9
- Water: < 0
- Soil: 0.1 to 0.2

**Example:**
```
NIR = 0.45, Red = 0.12
NDVI = (0.45 - 0.12) / (0.45 + 0.12) = 0.33 / 0.57 = 0.579
```

#### Leaf Area Index (LAI)

```python
LAI = -ln((0.69 - NDVI) / 0.59) / 0.91
```

**Physical Interpretation:**
- LAI > 6: Dense forest, full canopy closure
- LAI 3-6: Agricultural crops, dense vegetation
- LAI 1-3: Sparse vegetation, shrubs
- LAI < 1: Bare soil, desert

### 5.2 Surface Temperature

```python
# Thermal radiance to brightness temperature
Ts = K2 / ln(K1 / Lλ + 1) - 273.15  # Celsius

# Where K1, K2 are band-specific constants from MTL.json
# Landsat 8/9 Band 10: K1 = 774.89, K2 = 1321.08
```

### 5.3 Albedo Calculation

```python
# Narrowband to broadband conversion (Landsat 8/9)
Albedo = 0.25 × B2 + 0.15 × B4 + 0.13 × B5 + 0.25 × B6 + 0.07 × B7 + 0.15 × B10

# Where B2-B7 are top-of-atmosphere reflectance
# B10 is thermal band contribution
```

### 5.4 Net Radiation

```python
# Net shortwave radiation
Rns = (1 - α) × Rs↓

# Net longwave radiation
Rl↓ = εa × σ × Ta^4
Rl↑ = εs × σ × Ts^4
Rnl = Rl↓ - Rl↑

# Net radiation
Rn = Rns - Rnl
```

**Constants:**
- σ = 5.67 × 10⁻⁸ W/m²/K⁴ (Stefan-Boltzmann constant)
- α = Surface albedo
- εs = Surface emissivity
- εa = Atmospheric emissivity

### 5.5 Soil Heat Flux

```python
# METRIC standard method (Kustas & Norman, 1999)
G/Rn = 0.05 + 0.18 × exp(-0.5 × LAI)

# Alternative: G = Rn × (0.3 + 1.0 × NDVI - NDVI²) × (1 - 0.9 × NDVI²)
```

### 5.6 Sensible Heat Flux

```python
# Temperature difference
dT = Ts - Ta

# Calibrated dT using anchor pixels
dT_calibrated = a × dT + b

# Sensible heat flux
H = ρ × Cp × dT_calibrated / rah

# Aerodynamic resistance (simplified)
rah = (ln((z - d) / z0m) × ln((z - d) / z0t)) / (k² × u)

# Simplified form
H = dt_a × dT
```

**METRIC Calibration:**
```python
# Anchor pixel calibration - METRIC-consistent approach
# 
# Step 1: Calculate a coefficient from hot pixel energy balance
a = dt_a = (Rn_hot - G_hot) / dT_hot
# 
# Step 2: Calculate b coefficient to enforce cold pixel constraint (dT_cold ≈ 0)
# dT_cold = a * (Ts_cold - Ta_cold) + b = 0
# Therefore: b = -a * (Ts_cold - Ta_cold)
# 
# This ensures well-watered vegetation (cold pixel) has near-zero sensible heat
a = dt_a = (Rn_hot - G_hot) / dT_hot
b = -a * (Ts_cold - Ta_cold)
```

### 5.7 Latent Heat Flux

```python
# Residual from energy balance
LE = Rn - G - H

# Alternative: Using evaporative fraction
EF = LE / (Rn - G)
LE = EF × (Rn - G)
```

### 5.8 Instantaneous Evapotranspiration

```python
# Latent heat to ET conversion
ET_inst = LE / λ

# Where λ = 2.45 × 10⁶ J/kg (latent heat of vaporization)

# Example calculation
LE = 300 W/m² = 300 J/m²/s
λ = 2,450,000 J/kg
ET_inst = 300 / 2,450,000 = 0.000122 kg/m²/s
ET_inst = 0.000122 × 3600 = 0.44 mm/hr
```

### 5.9 Reference ET Fraction (ETrF)

```python
# Reference ET for alfalfa
ETr = ET0 × 1.15

# Reference ET fraction
ETrF = ET_inst / ETr_inst

# Where ETr_inst = ETr_daily × (Rs_inst / Rs_daily)
```

### 5.10 Daily Evapotranspiration

```python
# METRIC standard scaling (NO daylight fraction - ETrF already accounts for daily ET)
ET_daily = ETrF × ETr_daily

# Note: The daylight fraction approach (ET_daily = ETrF × ETr_daily × f_daylight) 
# is NOT used in METRIC. ETrF is a dimensionless ratio that represents the
# fraction of reference ET for the entire day, not just the instantaneous period.
```

---

## 6. Anchor Pixel Calibration

### 6.1 Cold Pixel Selection

**Criteria:**
- High NDVI (≥ P90)
- Low temperature (≤ P15 normalized Ts)
- Low albedo (≤ P30)
- High LAI (≥ P80)

**Physical Meaning:**
- Represents well-watered vegetation
- ET ≈ Reference ET (ETrF ≈ 1.0)
- H ≈ 0 (energy goes to evaporation)

### 6.2 Hot Pixel Selection

**Criteria:**
- Low NDVI (≤ P10)
- High temperature (≥ P90 normalized Ts)
- High albedo (≥ P70)
- Low LAI (≤ P20)

**Physical Meaning:**
- Represents dry bare soil
- ET ≈ 0 (no water available)
- H ≈ Rn - G (all energy goes to sensible heat)

### 6.3 Calibration Coefficients

```python
# dT calibration: dT_calibrated = a × dT + b
# 
# Step 1: Calculate 'a' coefficient from hot pixel energy balance
a = (Rn_hot - G_hot) / dT_hot
# 
# Step 2: Calculate 'b' coefficient to enforce cold pixel constraint
# The cold pixel should have dT_cold ≈ 0 (well-watered vegetation, H ≈ 0)
# From: dT_cold = a × (Ts_cold - Ta) + b
# We solve: b = -a × (Ts_cold - Ta)
# 
# Where:
# Rn_hot = Net radiation at hot pixel
# G_hot = Soil heat flux at hot pixel
# dT_hot = Ts_hot - Ta
# Ts_cold = Surface temperature at cold pixel
# Ta = Air temperature
```

### 6.4 Cluster-Based Selection

Instead of single pixels, METRIC selects N pixels (cluster) for robustness:

```python
N = 20  # Default cluster size

# Calculate median statistics from cluster
Ts_cold = median(Ts[cold_pixels])
Ts_hot = median(Ts[hot_pixels])

# Reduces impact of individual pixel errors
# Improves calibration robustness
```

### 6.5 Calibration Validation

```python
# Check 1: Temperature difference
dT = Ts_hot - Ts_cold
if dT < 15 K:
    raise Warning("Temperature difference too small")

# Check 2: dT at cold pixel
dT_cold = Ts_cold - Ta
if dT_cold > 2.0:
    raise Warning("Cold pixel not well-watered")

# Check 3: Energy balance at hot pixel
H_ratio = H_hot / (Rn_hot - G_hot)
if H_ratio < 0.5:
    raise Warning("Hot pixel energy dominance issue")
```

---

## 7. Energy Balance Components

### 7.1 Complete Energy Balance Calculation

```python
def calculate_energy_balance(rn, g, ts, ta, u, z0m, ndvi, a, b):
    """
    Calculate all energy balance components.
    
    Returns:
        G, H, LE, rah, EF, ET_inst
    """
    # Step 1: Calculate dT
    dT = ts - ta
    
    # Step 2: Apply calibration
    dT_calibrated = a * dT + b
    
    # Step 3: Calculate H
    rho = 1.2  # kg/m³
    cp = 1004  # J/kg/K
    rah = 50   # s/m (typical)
    H = rho * cp * dT_calibrated / rah
    
    # Step 4: Calculate LE
    LE = rn - g - H
    
    # Step 5: Calculate EF
    available_energy = rn - g
    EF = LE / available_energy
    
    # Step 6: Calculate ET_inst
    lambda_v = 2.45e6  # J/kg
    ET_inst = LE / lambda_v * 3600  # mm/hr
    
    return G, H, LE, rah, EF, ET_inst
```

### 7.2 Energy Balance Closure

```python
def check_energy_balance_closure(rn, g, h, le):
    """
    Check if energy balance closes (Rn - G = H + LE).
    
    Returns:
        closure_ratio, residual
    """
    available = rn - g
    used = h + le
    residual = available - used
    closure = (used / available) * 100  # Percentage
    
    # Expected: 85-95% closure for remote sensing
    return closure, residual
```

### 7.3 Expected Energy Partitioning

| Surface Type | Rn | G | H | LE | LE/(Rn-G) |
|--------------|----|----|----|----|-----------|
| Wet vegetation | 500 | 50 | 50 | 400 | 0.89 |
| Dry bare soil | 600 | 80 | 480 | 40 | 0.08 |
| Mixed agriculture | 550 | 60 | 250 | 240 | 0.49 |
| Urban | 450 | 100 | 300 | 50 | 0.14 |

---

## 8. Evapotranspiration Calculation

### 8.1 Instantaneous ET

```python
def calculate_et_inst(le):
    """
    Convert latent heat flux to instantaneous ET.
    
    ET_inst = LE / λ
    
    Units:
        LE: W/m² = J/m²/s
        λ: J/kg
        ET_inst: kg/m²/s = mm/s
    """
    lambda_v = 2.45e6  # J/kg
    et_inst = le / lambda_v * 3600  # mm/hr
    return et_inst
```

### 8.2 Reference ET Fraction

```python
def calculate_etrf(et_inst, etr_inst):
    """
    Calculate reference ET fraction.
    
    ETrF = ET_inst / ETr_inst
    
    Interpretation:
        ETrF > 1.0: ET > reference (overnight ET, advection)
        ETrF = 1.0: Well-watered vegetation
        ETrF < 1.0: Water stress
        ETrF = 0.0: No ET (dry surface)
    """
    etrf = et_inst / etr_inst
    return np.clip(etrf, 0.0, 2.0)  # Physical bounds
```

### 8.3 Daily ET Scaling

```python
def calculate_et_daily(etrf, etr_daily):
    """
    Scale instantaneous ETrF to daily ET.
    
    ET_daily = ETrF × ETr_daily
    
    Assumptions:
    - ETrF is constant throughout day
    - Energy partitioning is stable
    """
    et_daily = etrf * etr_daily
    return np.clip(et_daily, 0.0, 30.0)  # Physical bounds
```

### 8.4 ET Time Series Extrapolation

For multi-date analysis, interpolate ETrF between clear-sky dates:

```python
def extrapolate_et(etrf_dates, etrf_values, target_date):
    """
    Interpolate ETrF for dates without satellite data.
    
    Methods:
        - Linear interpolation
        - Fitted curve (spline)
        - ETr-based scaling
    """
    # Simple linear interpolation
    etrf_interp = np.interp(target_date, etrf_dates, etrf_values)
    return etrf_interp
```

---

## 9. Constraints and Validation

### 9.1 Scene-Level Validation

```python
@dataclass
class SceneValidation:
    """Scene-level validation checks."""
    
    # QA coverage
    qa_valid_percent: float       # Must be > 70%
    
    # NDVI dynamic range
    ndvi_p05: float              # Must be < NDVI_p95 - 0.30
    ndvi_p95: float
    
    # Radiation sanity
    rn_median: float             # Must be > 300 W/m²
    
    # ET0 sanity
    et0_inst: float              # Must be 0.1-1.5 mm/hr
```

### 9.2 Physical Constraints

| Variable | Minimum | Maximum | Unit | Reason |
|----------|---------|---------|------|--------|
| ET_daily | 0.0 | 30.0 | mm/day | Physical limit |
| ETrF | 0.0 | 2.0 | - | Physical bounds |
| EF | 0.0 | 2.0 | - | Energy partition |
| H | -50 | 600 | W/m² | Physical range |
| LE | 0 | 800 | W/m² | Energy limit |
| G | 0 | 300 | W/m² | Energy fraction |
| dT | -5 | 45 | K | Physical range |
| EF | 0.0 | 1.0 | - | Energy partition |
| H | -50 | 600 | W/m² | Physical range |
| LE | 0 | 800 | W/m² | Energy limit |
| G | 0 | 300 | W/m² | Energy fraction |
| dT | -5 | 45 | K | Physical range |

### 9.3 Quality Flags

```python
class SceneQuality(Enum):
    GOOD = "GOOD"           # All validation passed
    DEGRADED = "DEGRADED"   # Used fallback calibration
    LOW_QUALITY = "LOW_QUALITY"  # QA issues, still processed
    REJECTED = "REJECTED"   # Severe issues, no output
```

### 9.4 Anchor Pixel Validation

```python
def validate_anchor_pixels(cold_pixel, hot_pixel):
    """
    Validate selected anchor pixels.
    
    Checks:
    1. Temperature difference > 15 K
    2. Cold pixel dT < 2 K (well-watered)
    3. Hot pixel dT 15-30 K (dry surface)
    4. Energy balance constraints
    """
    issues = []
    
    # Check temperature difference
    dT = hot_pixel.ts - cold_pixel.ts
    if dT < 15:
        issues.append(f"Temperature difference {dT:.1f}K < 15K")
    
    # Check cold pixel dT
    dT_cold = cold_pixel.ts - air_temperature
    if dT_cold > 2.0:
        issues.append(f"Cold pixel dT {dT_cold:.1f}K > 2K")
    
    # Check hot pixel
    dT_hot = hot_pixel.ts - air_temperature
    if dT_hot < 5 or dT_hot > 45:
        issues.append(f"Hot pixel dT {dT_hot:.1f}K outside [5, 45]K")
    
    return issues
```

---

## 10. Assumptions

### 10.1 Atmospheric Assumptions

| Assumption | Description | Impact |
|------------|-------------|--------|
| Clear sky | No significant cloud cover | Affects radiation calculation |
| Well-mixed atmosphere | Uniform air temperature | Simplifies H calculation |
| Stable conditions | No strong advection | Affects rah calculation |
| Standard pressure | 101.3 kPa | Used in density calculation |

### 10.2 Surface Assumptions

| Assumption | Description | Impact |
|------------|-------------|--------|
| Horizontal surface | No slope effects | Simplified radiation |
| Uniform within pixel | Mixed pixels handled | Affects LST, albedo |
| Steady-state energy | Instantaneous balance | Valid at satellite overpass |
| Clear vegetation view | No topographic shading | Affects Rs calculation |

### 10.3 METRIC-Specific Assumptions

1. **Anchoring Assumption**: Well-chosen anchor pixels represent the full range of ET conditions
2. **Stability Assumption**: Atmospheric stability corrections are adequate
3. **Fraction Constancy**: ETrF is constant between satellite overpasses
4. **No Advection**: Horizontal advection of heat is negligible

### 10.4 Calibration Assumptions

```python
# Key METRIC assumptions in code
METRIC_ASSUMPTIONS = {
    "cold_pixel": {
        "dT": "~0 K",           # Near energy balance
        "ETrF": "1.0-1.2",      # Well-watered
        "H": "< 50 W/m²"        # Minimal sensible heat
    },
    "hot_pixel": {
        "dT": "15-30 K",        # Dry surface
        "ETrF": "< 0.1",        # Minimal ET
        "H": "~Rn-G"            # All energy to sensible heat
    }
}
```

---

## 11. Output Products

### 12.1 Standard ET Products

| Product | Description | Unit | Data Type |
|---------|-------------|------|-----------|
| ET_inst | Instantaneous ET at overpass | mm/hr | Float32 |
| ET_daily | Daily ET total | mm/day | Float32 |
| ETrF | Reference ET fraction | - | Float32 |
| LE | Latent heat flux | W/m² | Float32 |
| H | Sensible heat flux | W/m² | Float32 |
| G | Soil heat flux | W/m² | Float32 |
| Rn | Net radiation | W/m² | Float32 |

### 12.2 Surface Property Products

| Product | Description | Range |
|---------|-------------|-------|
| NDVI | Normalized Difference Vegetation Index | -1 to 1 |
| Albedo | Surface broadband albedo | 0 to 1 |
| LST | Land surface temperature | 250-400 K |
| LAI | Leaf Area Index | 0 to 8 |
| Emissivity | Surface emissivity | 0.9 to 1.0 |

### 12.3 Quality Products

| Product | Description | Values |
|---------|-------------|--------|
| ET_quality_class | Quality assessment | 0-4 (Poor to Excellent) |
| ET_confidence | Valid pixel fraction | 0 to 1 |

### 12.4 Metadata File

```json
{
    "scene_id": "LC08_L2SP_166038_20230427_20230509_02_T1",
    "acquisition_time": "2023-04-27T10:30:00Z",
    "calibration": {
        "a_coefficient": 25.4,
        "b_coefficient": -293.15,
        "dT_cold": 0.5,
        "dT_hot": 18.2
    },
    "anchor_pixels": {
        "cold_pixel": {"x": 150, "y": 200, "ndvi": 0.75},
        "hot_pixel": {"x": 300, "y": 400, "ndvi": 0.08}
    },
    "quality": {
        "scene_quality": "GOOD",
        "cloud_coverage": 0.15,
        "valid_pixels": 0.92
    }
}
```

---

## 12. Quality Assessment

### 13.1 ET Quality Classes

```python
class ETQualityClass(IntEnum):
    UNCERTAIN = 0      # ETrF < 0 or > 1.5 - Requires review
    POOR = 1           # ETrF 0.0-0.3 - Extreme stress or bare soil
    ACCEPTABLE = 2     # ETrF 0.3-1.4 - Some water stress
    GOOD = 3           # ETrF 0.6-1.3 - Normal conditions
    EXCELLENT = 4      # ETrF 0.8-1.2 - Well-watered vegetation
```

### 13.2 Physical Bounds Checking

```python
def check_physical_bounds(et_daily, etrf):
    """
    Check if ET values are physically realistic.
    
    Bounds:
        - ET_daily: 0-30 mm/day
        - ETrF: 0.0-2.0
    """
    valid_et = (et_daily >= 0) & (et_daily <= 30)
    valid_etrf = (etrf >= 0) & (etrf <= 2.0)
    
    return valid_et & valid_etrf
```

### 13.3 Spatial Consistency

```python
def check_spatial_consistency(et_daily, window_size=3):
    """
    Check for spatial outliers.
    
    Pixels deviating >50% from neighbors are flagged.
    """
    # Calculate neighborhood mean
    neighbor_mean = convolve(et_daily, kernel)
    
    # Calculate deviation
    deviation = np.abs(et_daily - neighbor_mean) / neighbor_mean
    
    # Flag outliers
    is_outlier = deviation > 0.5
    
    return is_outlier
```

### 13.4 Quality Statistics

```python
def calculate_quality_stats(et_daily, etrf, quality_class):
    """
    Calculate quality statistics for scene.
    
    Returns:
        Dictionary with:
        - et_mean, et_std, et_min, et_max
        - etrf_mean, etrf_std
        - quality_distribution (count per class)
        - valid_fraction
    """
    stats = {
        "et_mean": np.nanmean(et_daily),
        "et_std": np.nanstd(et_daily),
        "etrf_mean": np.nanmean(etrf),
        "quality_distribution": {
            "excellent": np.sum(quality_class == 4),
            "good": np.sum(quality_class == 3),
            "acceptable": np.sum(quality_class == 2),
            "poor": np.sum(quality_class == 1),
            "uncertain": np.sum(quality_class == 0)
        }
    }
    return stats
```

---

## 13. Troubleshooting

### 15.1 Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "No valid pixels found" | Complete cloud cover | Use different date |
| "NDVI dynamic range too low" | Uniform land cover | Check scene quality |
| "Net radiation too low" | Nighttime scene | Use daytime image |
| "Temperature difference too small" | Cloudy conditions | Use different scene |
| "Calibration failed" | Poor anchor pixels | Manual anchor selection |

### 15.2 Quality Issues

#### Issue: ETrF > 1.5

**Cause**: Overestimation of ET
**Solutions**:
1. Check cold pixel selection
2. Verify ET0 calculation
3. Review cloud masking

#### Issue: Negative ET

**Cause**: LE calculation error
**Solutions**:
1. Check H calculation
2. Verify Rn and G values
3. Review calibration coefficients

#### Issue: Uniform ET across scene

**Cause**: Calibration issue
**Solutions**:
1. Check anchor pixel temperature difference
2. Verify dT calculation
3. Review scene pre-validation

### 15.3 Performance Optimization

```python
# For large scenes, use chunked processing
def process_large_scene(landsat_dir, chunk_size=1000):
    """
    Process scene in chunks to reduce memory usage.
    """
    for chunk in chunk_scene(landsat_dir, chunk_size):
        yield process_chunk(chunk)
```

### 15.4 Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Key debug messages:
# [ENERGY BALANCE DEBUG] Rn: min=..., max=..., mean=...
# [ENERGY BALANCE DEBUG] G/Rn ratio: ...
# [ENERGY BALANCE DEBUG] H: ...
# [ENERGY BALANCE DEBUG] LE: ...
```

---

## Appendix A: Constants

| Constant | Symbol | Value | Unit |
|----------|--------|-------|------|
| Stefan-Boltzmann | σ | 5.67 × 10⁻⁸ | W/m²/K⁴ |
| Latent heat of vaporization | λ | 2.45 × 10⁶ | J/kg |
| Specific heat of air | Cp | 1004 | J/kg/K |
| Air density | ρ | 1.2 | kg/m³ |
| von Karman constant | k | 0.41 | - |
| Gravity | g | 9.81 | m/s² |

## Appendix B: Landsat Band Specifications

| Band | Wavelength (μm) | Spatial Resolution | Use |
|------|-----------------|-------------------|-----|
| Band 2 (Blue) | 0.45-0.52 | 30 m | NDVI, cloud detection |
| Band 3 (Green) | 0.52-0.60 | 30 m | NDVI |
| Band 4 (Red) | 0.63-0.69 | 30 m | NDVI |
| Band 5 (NIR) | 0.85-0.90 | 30 m | NDVI, LAI |
| Band 6 (SWIR1) | 1.57-1.65 | 30 m | LAI, albedo |
| Band 7 (SWIR2) | 2.11-2.29 | 30 m | Albedo |
| Band 10 (TIRS1) | 10.60-11.19 | 100 m | LST |

## Appendix C: References

1. Allen, R.G., Tasumi, M., Morse, A., Trezza, R., Wright, J.L., Bastiaanssen, W., Kramber, W., Lorite, I., and Robison, C.W. (2007). Satellite-based energy balance for mapping evapotranspiration with internalized calibration (METRIC)-Applications. Journal of Irrigation and Drainage Engineering, 133(4), 395-406.

2. Bastiaanssen, W.G.M., Menenti, M., Feddes, R.A., and Holtslag, A.A.M. (1998). A remote sensing surface energy balance algorithm for land (SEBAL): 1. Formulation. Journal of Hydrology, 212-213, 198-212.

3. Tasumi, M., Allen, R.G., Trezza, R., and Wright, J.L. (2005). Satellite-based energy balance to assess within-population variance of crop coefficient curves. Journal of Irrigation and Drainage Engineering, 131(1), 94-109.

---

*Document Version: 1.0*
*Last Updated: 2025-01-28*
*METRIC Model Version: 2.0+*
