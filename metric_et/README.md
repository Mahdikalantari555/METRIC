# METRIC ETa Pipeline

METRIC (Mapping Evapotranspiration with Internalized Calibration) Evapotranspiration estimation pipeline for Landsat data.

## Installation

```bash
pip install metric-et
```

## Usage

```bash
# Show help
metric --help

# Process scenes
metric process --input-dir data/ --output-dir output/ --start-date 2025-09-15 --end-date 2025-12-04

# Process single scene
metric process-scene data/landsat_20251204_166_038/ --output output/

# Manage anchor pixels
metric anchors data/landsat_20251204_166_038/ --output output/

# Export results
metric export --input output/ETa_20251204.tif --format GeoTIFF --output output/

# Create visualization
metric visualize --input output/ETa_20251204.tif --type map --output output/

# Show summary
metric summary output/
```

## License

MIT
