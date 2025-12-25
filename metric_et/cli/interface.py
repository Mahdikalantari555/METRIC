"""
METRIC ETa Pipeline CLI Interface

Command-line interface for the METRIC evapotranspiration mapping pipeline.
"""

import os
import sys
import json
import click
from datetime import datetime
from pathlib import Path
from typing import Optional
from functools import wraps

import yaml
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions and Decorators
# ============================================================================

def validate_date(ctx, param, value):
    """Validate date format YYYY-MM-DD."""
    if value is None:
        return None
    try:
        return datetime.strptime(value, '%Y-%m-%d')
    except ValueError:
        raise click.BadParameter(
            f'Invalid date format: {value}. Use YYYY-MM-DD format.',
            ctx=ctx,
            param=param
        )


def validate_path(ctx, param, value, must_exist=False, create_if_missing=False):
    """Validate and return path."""
    if value is None:
        return None
    
    path = Path(value)
    
    if must_exist and not path.exists():
        raise click.BadParameter(
            f'Path does not exist: {value}',
            ctx=ctx,
            param=param
        )
    
    if create_if_missing:
        path.mkdir(parents=True, exist_ok=True)
    
    return path


def validate_scene_path(ctx, param, value):
    """Validate scene directory path."""
    path = validate_path(ctx, param, value, must_exist=True)
    
    # Check for required files (MTL.json or band files)
    mtl_file = path / 'MTL.json'
    if not mtl_file.exists():
        # Also check for band files
        band_extensions = ['.tif', '.tiff']
        has_bands = any(
            f.endswith(tuple(band_extensions)) 
            for f in path.iterdir()
        )
        if not has_bands:
            raise click.BadParameter(
                f'Scene directory must contain MTL.json or band files: {value}',
                ctx=ctx,
                param=param
            )
    
    return path


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML or JSON file."""
    if config_path is None:
        return {}
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise click.ClickException(f'Configuration file not found: {config_path}')
    
    try:
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            raise click.ClickException(
                f'Unsupported config format: {config_path.suffix}. Use .yaml or .json'
            )
    except Exception as e:
        raise click.ClickException(f'Error loading config: {str(e)}')


# ============================================================================
# Main Command Group
# ============================================================================

@click.group()
@click.option('--verbose', '-v', is_flag=True, default=False, help='Enable verbose logging')
@click.option('--log-file', type=click.Path(path_type=Path), help='Custom log file path')
@click.option('--config', type=click.Path(exists=True, path_type=Path), help='Configuration file path (YAML or JSON)')
@click.version_option(version='1.0.0', prog_name='METRIC ETa Pipeline')
@click.pass_context
def cli(ctx, verbose, log_file, config):
    """
    METRIC ETa Pipeline - Evapotranspiration mapping using remote sensing.
    
    This tool processes satellite imagery to compute actual evapotranspiration
    using the METRIC (Mapping Evapotranspiration with Internalized Calibration)
    model.
    
    \b
    Common commands:
      \b
      metric process        Process multiple scenes
      metric process-scene  Process a single scene
      metric anchors        Manage anchor pixels
      metric export         Export results
      metric visualize      Create visualizations
      metric summary        Show processing summary
    
    For help on a specific command, run: metric COMMAND --help
    """
    # Store context for subcommands
    ctx.ensure_object(dict)
    
    # Configure logging
    if verbose:
        logging.getLogger('metric_et').setLevel(logging.DEBUG)
        logger.debug('Verbose logging enabled')
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger('metric_et').addHandler(file_handler)
        logger.info(f'Logging to file: {log_file}')
    
    # Load configuration
    if config:
        ctx.obj['config'] = load_config(config)
        logger.debug(f'Loaded configuration from: {config}')
    else:
        ctx.obj['config'] = {}
    
    ctx.obj['verbose'] = verbose
    ctx.obj['log_file'] = log_file


# ============================================================================
# Process Command
# ============================================================================

@cli.command()
@click.option('--input-dir', '-i', type=click.Path(exists=True, path_type=Path), required=True, help='Input data directory containing Landsat scenes')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), required=True, help='Output directory for processed results')
@click.option('--start-date', '-s', type=str, callback=validate_date, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', '-e', type=str, callback=validate_date, help='End date (YYYY-MM-DD)')
@click.option('--scene', type=str, help='Process specific scene ID (e.g., LC08_L1TP_166038_2025_09_15)')
@click.option('--cloud-threshold', type=int, default=30, help='Maximum cloud cover percentage (default: 30)')
@click.option('--calibration', type=click.Choice(['auto', 'hot-cold', 'manual'], case_sensitive=False), default='auto', help='Calibration mode (default: auto)')
@click.option('--dry-run', is_flag=True, default=False, help='Preview without processing')
@click.option('--config', type=click.Path(exists=True, path_type=Path), help='Configuration file path (overrides default)')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Enable verbose logging')
@click.option('--log-file', type=click.Path(path_type=Path), help='Custom log file path')
@click.pass_context
def process(ctx, input_dir, output_dir, start_date, end_date, scene,
            cloud_threshold, calibration, dry_run, config, verbose, log_file):
    """
    Process Landsat scenes for ETa estimation.
    
    \b
    Examples:
      \b
      metric process -i data/ -o output/ -s 2025-09-15 -e 2025-12-04
      metric process -i data/ -o output/ --cloud-threshold 20 --dry-run
      metric process -i data/ -o output/ --scene LC08_L1TP_166038_2025_09_15
    
    This command processes all scenes in the input directory (or a specific
    scene) within the specified date range and computes daily ETa using
    the METRIC model.
    """
    # Configure logging
    if verbose:
        logging.getLogger('metric_et').setLevel(logging.DEBUG)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger('metric_et').addHandler(file_handler)
    
    # Merge config with CLI args
    cfg = ctx.obj.get('config', {})
    if config:
        cfg = load_config(config)
    
    # Validate and create output directory
    output_dir = validate_path(None, None, output_dir, create_if_missing=True)
    
    # Find scenes to process
    scenes_to_process = []
    
    if scene:
        # Find specific scene
        scene_patterns = [
            f'*_{scene}*',
            f'*{scene}*',
            f'*{scene}'
        ]
        for pattern in scene_patterns:
            matches = list(input_dir.glob(pattern))
            if matches:
                scenes_to_process.extend(matches)
                break
        else:
            raise click.ClickException(f'Scene not found: {scene}')
    else:
        # Get all scene directories
        scene_dirs = sorted([
            d for d in input_dir.iterdir()
            if d.is_dir() and ('landsat' in d.name.lower() or 'lc08' in d.name.lower() or 'lt05' in d.name.lower())
        ])
        
        for scene_dir in scene_dirs:
            # Extract date from scene name
            scene_date_str = None
            for part in scene_dir.name.split('_'):
                if len(part) == 8 and part.isdigit():
                    scene_date_str = part
                    break
            
            if scene_date_str:
                try:
                    scene_date = datetime.strptime(scene_date_str, '%Y%m%d')
                    
                    # Filter by date range
                    if start_date and scene_date < start_date:
                        continue
                    if end_date and scene_date > end_date:
                        continue
                    
                    scenes_to_process.append(scene_dir)
                except ValueError:
                    continue
    
    if not scenes_to_process:
        raise click.ClickException('No scenes found matching criteria')
    
    click.echo(f'Found {len(scenes_to_process)} scene(s) to process')
    
    if dry_run:
        click.echo('\nDry run - scenes that would be processed:')
        for scene_dir in scenes_to_process:
            click.echo(f'  - {scene_dir.name}')
        click.echo(f'\nOutput directory: {output_dir}')
        click.echo(f'Date range: {start_date or "all"} to {end_date or "all"}')
        click.echo(f'Cloud threshold: {cloud_threshold}%')
        click.echo(f'Calibration: {calibration}')
        return
    
    # Process scenes
    try:
        from metric_et.pipeline.metric_pipeline import METRICPipeline
        from metric_et.config.settings import PipelineSettings
        
        settings = PipelineSettings(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            cloud_threshold=cloud_threshold,
            calibration_method=calibration
        )
        
        pipeline = METRICPipeline(settings)
        
        with click.progressbar(
            scenes_to_process,
            label='Processing scenes',
            show_percent=True,
            show_pos=True
        ) as scenes:
            for scene_path in scenes:
                try:
                    click.echo(f'\nProcessing: {scene_path.name}')
                    result = pipeline.process_scene(scene_path)
                    click.echo(f'  [OK] Completed: {result.output_file}')
                except Exception as e:
                    click.echo(f'  [FAIL] Failed: {str(e)}', err=True)
                    logger.exception(f'Error processing {scene_path.name}')
        
        click.echo(f'\nProcessing complete. Results saved to: {output_dir}')
        
    except ImportError as e:
        click.echo(f'\nProcessing module not available: {e}')
        click.echo('Pipeline implementation pending.')
        
        # Save dry run results
        run_info = {
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'start_date': start_date.isoformat() if start_date else None,
            'end_date': end_date.isoformat() if end_date else None,
            'scenes': [str(s.name) for s in scenes_to_process],
            'cloud_threshold': cloud_threshold,
            'calibration': calibration,
            'status': 'pending'
        }
        
        info_file = output_dir / 'processing_run.json'
        with open(info_file, 'w') as f:
            json.dump(run_info, f, indent=2)
        
        click.echo(f'Processing task saved to: {info_file}')


# ============================================================================
# Process Scene Command
# ============================================================================

@cli.command()
@click.argument('scene-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--output', '-o', type=click.Path(path_type=Path), required=True, help='Output directory')
@click.option('--weather', type=click.Path(exists=True, path_type=Path), help='Weather data file (CSV)')
@click.option('--skip-calibration', is_flag=True, default=False, help='Skip METRIC calibration')
@click.option('--dry-run', is_flag=True, default=False, help='Preview without processing')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Enable verbose logging')
@click.option('--log-file', type=click.Path(path_type=Path), help='Custom log file path')
@click.pass_context
def process_scene(ctx, scene_path, output, weather, skip_calibration, dry_run, verbose, log_file):
    """
    Process a single Landsat scene.
    
    \b
    Examples:
      \b
      metric process-scene data/landsat_20251204_166_038/ -o output/
      metric process-scene data/landsat_20251204_166_038/ -o output/ --weather weather.csv
      metric process-scene data/landsat_20251204_166_038/ -o output/ --skip-calibration
    
    This command processes a single Landsat scene directory and computes
    the evapotranspiration fraction (ETf) and daily ETa.
    """
    # Configure logging
    if verbose:
        logging.getLogger('metric_et').setLevel(logging.DEBUG)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger('metric_et').addHandler(file_handler)
    
    # Validate scene path
    scene_path = validate_scene_path(None, None, scene_path)
    
    # Create output directory
    output = validate_path(None, None, output, create_if_missing=True)
    
    if dry_run:
        click.echo('Dry run - scene parameters:')
        click.echo(f'  Scene: {scene_path.name}')
        click.echo(f'  Output: {output}')
        click.echo(f'  Weather data: {weather or "default"}')
        click.echo(f'  Skip calibration: {skip_calibration}')
        return
    
    # Load weather data if provided
    weather_data = None
    if weather:
        weather_data = load_config(weather)
    
    try:
        from metric_et.pipeline.metric_pipeline import METRICPipeline
        from metric_et.config.settings import PipelineSettings
        
        settings = PipelineSettings(
            input_dir=str(scene_path.parent),
            output_dir=str(output),
            skip_calibration=skip_calibration
        )
        
        pipeline = METRICPipeline(settings)
        
        with click.progressbar(
            ['Preprocessing', 'Radiation', 'Energy Balance', 'Calibration', 'ET Calculation'],
            label='Processing stages',
            show_percent=True
        ) as stages:
            result = pipeline.process_scene(scene_path)
        
        click.echo(f'\n[OK] Scene processed successfully')
        click.echo(f'  Output file: {result.output_file}')
        if hasattr(result, 'statistics'):
            click.echo(f'  Mean ETa: {result.statistics.get("mean_eta", "N/A"):.2f} mm/day')
            
    except ImportError:
        click.echo('\nScene processing module not available.')
        click.echo('Pipeline implementation pending.')
        
        scene_info = {
            'scene': str(scene_path),
            'output': str(output),
            'weather_data': str(weather) if weather else None,
            'skip_calibration': skip_calibration,
            'status': 'pending'
        }
        
        info_file = output / 'scene_processing.json'
        with open(info_file, 'w') as f:
            json.dump(scene_info, f, indent=2)
        
        click.echo(f'Processing task saved to: {info_file}')


# ============================================================================
# Anchor Pixel Command
# ============================================================================

@cli.command()
@click.argument('scene-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output directory for anchor results')
@click.option('--method', type=click.Choice(['auto', 'manual'], case_sensitive=False), default='auto', help='Anchor selection method (default: auto)')
@click.option('--cold-eta', type=float, default=1.05, help='Cold pixel expected ETrF (default: 1.05)')
@click.option('--hot-eta', type=float, default=0.05, help='Hot pixel expected ETrF (default: 0.05)')
@click.option('--interactive', is_flag=True, default=False, help='Interactive anchor pixel selection')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Enable verbose logging')
@click.option('--log-file', type=click.Path(path_type=Path), help='Custom log file path')
@click.pass_context
def anchors(ctx, scene_path, output, method, cold_eta, hot_eta, interactive, verbose, log_file):
    """
    Manage METRIC anchor pixels (hot and cold references).
    
    \b
    Examples:
      \b
      metric anchors data/landsat_20251204_166_038/
      metric anchors data/landsat_20251204_166_038/ --method manual
      metric anchors data/landsat_20251204_166_038/ --cold-eta 1.10 --hot-eta 0.10
    
    Anchor pixels are used to calibrate the METRIC model. The cold pixel
    represents a well-watered area (high ET), and the hot pixel represents
    a dry, stressed area (low ET).
    """
    # Configure logging
    if verbose:
        logging.getLogger('metric_et').setLevel(logging.DEBUG)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger('metric_et').addHandler(file_handler)
    
    # Validate scene path
    scene_path = validate_scene_path(None, None, scene_path)
    
    if output is None:
        output = scene_path / 'anchors'
    
    output = validate_path(None, None, output, create_if_missing=True)
    
    if interactive:
        click.echo('Interactive mode requires GUI environment')
        click.echo('Falling back to automatic anchor selection...')
        method = 'auto'
    
    try:
        from metric_et.calibration.anchor_pixels import AnchorPixelManager
        
        manager = AnchorPixelManager(
            scene_path=scene_path,
            output_dir=output,
            method=method,
            cold_etrf=cold_eta,
            hot_etrf=hot_eta
        )
        
        with click.progressbar(
            ['Finding cold pixel', 'Finding hot pixel', 'Validating anchors'],
            label='Anchor selection',
            show_percent=True
        ) as steps:
            cold_pixel, hot_pixel = manager.find_anchor_pixels()
        
        # Save anchor results
        anchor_results = {
            'scene': scene_path.name,
            'cold_pixel': {
                'x': cold_pixel.x,
                'y': cold_pixel.y,
                'etrf': cold_pixel.etrf,
                'temperature': getattr(cold_pixel, 'temperature', None)
            },
            'hot_pixel': {
                'x': hot_pixel.x,
                'y': hot_pixel.y,
                'etrf': hot_pixel.etrf,
                'temperature': getattr(hot_pixel, 'temperature', None)
            },
            'dT_cold': getattr(cold_pixel, 'dT', None),
            'dT_hot': getattr(hot_pixel, 'dT', None)
        }
        
        results_file = output / 'anchor_results.json'
        with open(results_file, 'w') as f:
            json.dump(anchor_results, f, indent=2)
        
        click.echo(f'\n[OK] Anchor pixel analysis complete')
        click.echo(f'  Cold pixel: ({cold_pixel.x}, {cold_pixel.y})')
        click.echo(f'  Hot pixel: ({hot_pixel.x}, {hot_pixel.y})')
        click.echo(f'  Results saved to: {results_file}')
        
    except ImportError:
        click.echo('Anchor pixel module not yet implemented')
        click.echo('Saving placeholder anchor configuration...')
        
        anchor_config = {
            'scene': scene_path.name,
            'method': method,
            'cold_etrf': cold_eta,
            'hot_etrf': hot_eta,
            'note': 'Placeholder - implement anchor pixel module first'
        }
        
        results_file = output / 'anchor_config.json'
        with open(results_file, 'w') as f:
            json.dump(anchor_config, f, indent=2)
        
        click.echo(f'  Configuration saved to: {results_file}')


# ============================================================================
# Export Command
# ============================================================================

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True, path_type=Path), required=True, help='Input file to export')
@click.option('--format', type=click.Choice(['GeoTIFF', 'NetCDF', 'CSV'], case_sensitive=False), required=True, help='Output format')
@click.option('--output', '-o', type=click.Path(path_type=Path), required=True, help='Output directory')
@click.option('--bands', type=str, help='Comma-separated band names (e.g., ET_daily,ETrF,LE)')
@click.option('--crs', type=str, help='Target CRS (e.g., EPSG:4326)')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Enable verbose logging')
@click.option('--log-file', type=click.Path(path_type=Path), help='Custom log file path')
@click.pass_context
def export(ctx, input_file, format, output, bands, crs, verbose, log_file):
    """
    Export ETa results to various formats.
    
    \b
    Examples:
      \b
      metric export -i output/ETa_20251204.tif --format GeoTIFF -o output/
      metric export -i output/ETa_20251204.tif --format NetCDF -o output/
      metric export -i output/ETa_20251204.tif --format CSV --bands ET_daily,ETrF -o output/
    
    Supported output formats:
      - GeoTIFF: Geo-referenced TIFF image
      - NetCDF: NetCDF4 scientific data format
      - CSV: Comma-separated values for point data
    """
    # Configure logging
    if verbose:
        logging.getLogger('metric_et').setLevel(logging.DEBUG)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger('metric_et').addHandler(file_handler)
    
    # Validate inputs
    input_file = validate_path(None, None, input_file, must_exist=True)
    output = validate_path(None, None, output, create_if_missing=True)
    
    # Parse bands
    band_list = None
    if bands:
        band_list = [b.strip() for b in bands.split(',')]
    
    click.echo(f'Exporting: {input_file.name}')
    click.echo(f'  Format: {format}')
    click.echo(f'  Output: {output}')
    if band_list:
        click.echo(f'  Bands: {", ".join(band_list)}')
    if crs:
        click.echo(f'  Target CRS: {crs}')
    
    # Check if export module is available
    try:
        from metric_et.io.export import export_raster
        
        result = export_raster(
            input_file=str(input_file),
            output_dir=str(output),
            format=format,
            bands=band_list,
            target_crs=crs
        )
        
        click.echo(f'\n[OK] Export complete: {result}')
        
    except ImportError:
        click.echo('Export module not yet implemented')
        click.echo(f'Would export {input_file} to {format} format')
        
        # Create placeholder
        export_info = {
            'input': str(input_file),
            'format': format,
            'output_dir': str(output),
            'bands': band_list,
            'crs': crs,
            'status': 'pending'
        }
        
        info_file = output / f'export_{input_file.stem}.json'
        with open(info_file, 'w') as f:
            json.dump(export_info, f, indent=2)
        
        click.echo(f'Export task saved to: {info_file}')


# ============================================================================
# Visualization Command
# ============================================================================

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True, path_type=Path), required=True, help='Input file to visualize')
@click.option('--type', type=click.Choice(['map', 'histogram', 'timeseries'], case_sensitive=False), required=True, help='Visualization type')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output directory for saving figure')
@click.option('--title', type=str, help='Plot title')
@click.option('--colormap', type=click.Choice(['viridis', 'jet', 'terrain', 'RdYlGn', 'Blues'], case_sensitive=False), default='viridis', help='Colormap for visualization (default: viridis)')
@click.option('--show', is_flag=True, default=False, help='Display plot interactively')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Enable verbose logging')
@click.option('--log-file', type=click.Path(path_type=Path), help='Custom log file path')
@click.pass_context
def visualize(ctx, input_file, viz_type, output, title, colormap, show, verbose, log_file):
    """
    Create visualizations of ETa results.
    
    \b
    Examples:
      \b
      metric visualize -i output/ETa_20251204.tif --type map --output output/
      metric visualize -i output/ETa_20251204.tif --type histogram --title "ETa Distribution"
      metric visualize -i output/ETa_20251204.tif --type timeseries --colormap jet
    
    Visualization types:
      - map: Geographic map of ETa values
      - histogram: Histogram of ETa distribution
      - timeseries: Time series of mean ETa (requires multiple dates)
    """
    # Configure logging
    if verbose:
        logging.getLogger('metric_et').setLevel(logging.DEBUG)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger('metric_et').addHandler(file_handler)
    
    # Validate inputs
    input_file = validate_path(None, None, input_file, must_exist=True)
    
    if output is None:
        output = input_file.parent
    
    output = validate_path(None, None, output, create_if_missing=True)
    
    # Set default title
    if title is None:
        title = f'{input_file.stem} - {viz_type.capitalize()}'
    
    click.echo(f'Creating {viz_type} visualization')
    click.echo(f'  Input: {input_file.name}')
    click.echo(f'  Title: {title}')
    click.echo(f'  Colormap: {colormap}')
    
    # Check if visualization module is available
    try:
        from metric_et.output.visualization import Visualization
        import rasterio
        import numpy as np
        
        # Load the input file
        viz = Visualization(output_dir=str(output))
        
        with rasterio.open(str(input_file)) as src:
            data = src.read(1)
            crs = src.crs
            transform = src.transform
        
        # Get metadata from filename
        title = title or f'{input_file.stem}'
        
        if viz_type == 'map':
            # Create ET map
            from metric_et.core.datacube import DataCube
            cube = DataCube()
            cube.update_crs(crs, transform)
            cube.add('data', data)
            viz.plot_et_map(data, cube, title=title, colormap=colormap, output_path=str(output / f'{input_file.stem}_{viz_type}.png'), show=show)
            
        elif viz_type == 'histogram':
            viz.plot_et_histogram(data, title=title, output_path=str(output / f'{input_file.stem}_{viz_type}.png'), show=show)
            
        elif viz_type == 'timeseries':
            click.echo('Time series visualization requires multiple dates')
            click.echo('Please provide multiple input files for time series')
            
        click.echo(f'\n[OK] Visualization saved')
        
    except ImportError as e:
        click.echo(f'Visualization module error: {e}')


# ============================================================================
# Summary Command
# ============================================================================

@cli.command()
@click.argument('output-dir', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--json', 'output_json', is_flag=True, default=False, help='Output summary as JSON')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Enable verbose logging')
@click.option('--log-file', type=click.Path(path_type=Path), help='Custom log file path')
@click.pass_context
def summary(ctx, output_dir, output_json, verbose, log_file):
    """
    Show processing summary and statistics.
    
    \b
    Examples:
      \b
      metric summary output/
      metric summary output/ --json
    
    Displays information about processed scenes, statistics, and
    calibration results stored in the output directory.
    """
    # Configure logging
    if verbose:
        logging.getLogger('metric_et').setLevel(logging.DEBUG)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger('metric_et').addHandler(file_handler)
    
    # Look for processing results
    results = {
        'scenes': [],
        'statistics': {},
        'calibration': {}
    }
    
    # Find processed scene directories
    scene_dirs = [
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name.startswith('ETa_')
    ]
    
    for scene_dir in sorted(scene_dirs):
        scene_info = {
            'name': scene_dir.name,
            'files': [],
            'date_processed': datetime.fromtimestamp(scene_dir.stat().st_mtime).isoformat()
        }
        
        # Find output files
        for f in scene_dir.glob('*'):
            if f.is_file():
                scene_info['files'].append(f.name)
        
        results['scenes'].append(scene_info)
    
    # Look for calibration results
    calibration_files = list(output_dir.glob('*calibration*.json'))
    for cf in calibration_files:
        try:
            with open(cf, 'r') as f:
                cal_data = json.load(f)
                results['calibration'][cf.stem] = cal_data
        except Exception:
            pass
    
    # Look for statistics
    stats_files = list(output_dir.glob('*statistics*.json'))
    for sf in stats_files:
        try:
            with open(sf, 'r') as f:
                stats_data = json.load(f)
                results['statistics'][sf.stem] = stats_data
        except Exception:
            pass
    
    # Output summary
    if output_json:
        click.echo(json.dumps(results, indent=2))
    else:
        click.secho('\n============================================================', fg='cyan')
        click.secho('             METRIC ETa Pipeline Summary', fg='cyan', bold=True)
        click.secho('============================================================\n', fg='cyan')
        
        click.secho(f'Output Directory: {output_dir}', fg='white', bold=True)
        click.echo()
        
        if results['scenes']:
            click.secho(f'Processed Scenes ({len(results["scenes"])})', fg='green', bold=True)
            for scene in results['scenes']:
                click.echo(f'  * {scene["name"]}')
                if scene['files']:
                    for f in scene['files'][:3]:
                        click.echo(f'      - {f}')
                    if len(scene['files']) > 3:
                        click.echo(f'      ... and {len(scene["files"]) - 3} more')
            click.echo()
        else:
            click.secho('No processed scenes found', fg='yellow')
            click.echo()
        
        if results['calibration']:
            click.secho('Calibration Results', fg='green', bold=True)
            for cal_name, cal_data in results['calibration'].items():
                click.echo(f'  * {cal_name}')
            click.echo()
        
        if results['statistics']:
            click.secho('Statistics', fg='green', bold=True)
            for stats_name, stats_data in results['statistics'].items():
                click.echo(f'  * {stats_name}')
            click.echo()
        
        click.secho('============================================================', fg='cyan')


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    cli()
