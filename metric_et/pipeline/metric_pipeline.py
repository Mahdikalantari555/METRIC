"""METRIC ETa processing pipeline."""

from typing import Dict, Optional, Tuple, Union, Any
import numpy as np
import xarray as xr
from loguru import logger
import os

# Import decision logic from calibration
from ..calibration.dt_calibration import CalibrationStatus


class METRICPipeline:
    """Main processing pipeline for METRIC ETa model."""

    def __init__(self, config: Optional[Dict] = None, roi_path: Optional[str] = None):
        """Initialize METRIC processing pipeline."""
        self.config = config or {}
        self.roi_path = roi_path
        self.data = None
        self._calibration_result = None
        self._anchor_result = None       # Store anchor pixel result for visualization
        self._eb_manager = None  # Store EnergyBalanceManager instance for reuse
        self._scene_quality = "GOOD"  # Track scene quality based on calibration decision
        self._scene_id = "unknown"  # Track current scene ID
        self._qa_coverage_issue = False  # Track QA coverage issues for quality flagging
        self._original_extent = None     # Store original scene extent before clipping
        self._roi_extent = None          # Store ROI extent after clipping
        self._roi_mask = None            # Store boolean mask for ROI boundaries
        logger.info("Initialized METRICPipeline")
    
    def run(
        self, landsat_dir: str, meteo_data: Dict,
        output_dir: Optional[str] = None, roi_path: Optional[str] = None
    ) -> Dict[str, xr.DataArray]:
        """Run complete METRIC ETa processing pipeline."""
        import logging

        logger = logging.getLogger(__name__)

        try:
            logger.info("Starting METRIC ETa processing pipeline")

            # Step 1: Load and preprocess data
            logger.info("Step 1: Loading and preprocessing data")
            self.load_data(landsat_dir, meteo_data, roi_path=roi_path)
            self.preprocess()

            # Extract scene ID from DataCube metadata (from MTL.json)
            scene_id = self.data.metadata.get('scene_id', os.path.basename(landsat_dir.strip('/\\')))
            self._scene_id = scene_id
            logger.info(f"Processing scene: {scene_id}")

            # Step 2: Calculate surface properties
            logger.info("Step 2: Calculating surface properties")
            self.calculate_surface_properties()

            # Step 3: Calculate radiation balance (Rn)
            logger.info("Step 3: Calculating radiation balance")
            self.calculate_radiation_balance()

            # Step 4: Calculate soil heat flux (G) - calibration-free
            logger.info("Step 4: Calculating soil heat flux")
            self.calculate_soil_heat_flux()

            # Step 5: Apply unified METRIC calibration pipeline
            logger.info("Step 5: Applying unified METRIC calibration pipeline")
            self.calibrate()

            # Check if scene was rejected by decision logic
            # NOTE: We now allow all scenes to proceed to ET calculation
            # Only severe QA issues (< 0.30) should reject scenes
            if self._scene_quality == "REJECTED":
                logger.warning(
                    f"Scene {self._scene_id} was rejected during calibration. "
                    "However, proceeding with ET calculation for quality assessment."
                )
                # Continue to ET calculation instead of returning early
            
            # Check for QA coverage issues and flag as LOW QUALITY if needed
            if hasattr(self, '_qa_coverage_issue') and self._qa_coverage_issue:
                if self._scene_quality == "GOOD":
                    self._scene_quality = "LOW_QUALITY"
                    logger.info(f"Scene {self._scene_id} flagged as LOW QUALITY due to QA coverage issues")

            # Step 6: Calculate final ET
            logger.info("Step 6: Calculating evapotranspiration")
            self.calculate_et()

            # Step 7: Save results if output directory provided
            if output_dir:
                logger.info("Step 7: Saving results")
                self.save_results(output_dir)

            logger.info("METRIC ETa processing pipeline completed successfully")

            # Return results
            return self.get_results()

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def load_data(self, landsat_dir: str, meteo_data: Dict, dem_path: Optional[str] = None, roi_path: Optional[str] = None) -> None:
        """Load input data into DataCube."""
        from ..io import LandsatReader, MeteoReader
        from ..core.datacube import DataCube
        import os
        import logging
        from datetime import datetime

        logger = logging.getLogger(__name__)

        try:
            # Initialize DataCube
            self.data = DataCube()

            # Load Landsat data
            logger.info(f"Loading Landsat data from {landsat_dir}")
            landsat_reader = LandsatReader()
            landsat_cube = landsat_reader.load(landsat_dir)

            # Load ROI geometry for later use in final clipping (not for initial data loading)
            roi_path = roi_path or self.roi_path or "amirkabir.geojson"
            import geopandas as gpd
            roi_gdf = gpd.read_file(roi_path)
            # Reproject ROI to match raster CRS from DataCube
            roi_gdf = roi_gdf.to_crs(landsat_cube.crs)
            self._roi_geom = roi_gdf.geometry.iloc[0]  # Store ROI geometry for later use
            logger.info("Loaded ROI geometry for final clipping")

            # Copy Landsat data to main cube
            for band_name in landsat_cube.bands():
                band_data = landsat_cube.get(band_name)
                # Ensure rioxarray integration is available for spatial operations
                if not hasattr(band_data, 'rio'):
                    import rioxarray
                    band_data = band_data.rio.write_crs(landsat_cube.crs)
                self.data.add(band_name, band_data)

            # Copy metadata
            self.data.metadata.update(landsat_cube.metadata)
            self.data.crs = landsat_cube.crs
            self.data.transform = landsat_cube.transform
            self.data.extent = landsat_cube.extent
            self.data.acquisition_time = landsat_cube.acquisition_time

            # Fix acquisition time if it's at midnight (no time info in MTL)
            if self.data.acquisition_time and self.data.acquisition_time.time() == datetime.min.time():
                # Assume Landsat overpass time of 10:30 (typical for Landsat 8/9)
                overpass_time = datetime.strptime("10:30", "%H:%M").time()
                self.data.acquisition_time = datetime.combine(self.data.acquisition_time.date(), overpass_time)
                logger.info(f"Adjusted acquisition time to {self.data.acquisition_time}")

            # Load weather data dynamically from Open-Meteo API
            logger.info("Fetching meteorological data from Open-Meteo API")

            # Get target coordinates from Landsat data
            sample_band = next(iter(self.data.data.values()))
            target_coords = {dim: sample_band.coords[dim] for dim in sample_band.dims}

            # Get full scene extent and convert to lat/lon for weather fetching
            full_scene_bounds = self.data.extent  # (min_x, min_y, max_x, max_y) in projected CRS
            from pyproj import Transformer
            # Create transformer from data CRS to WGS84
            transformer = Transformer.from_crs(self.data.crs, "EPSG:4326", always_xy=True)
            # Transform corners
            min_lon, min_lat = transformer.transform(full_scene_bounds[0], full_scene_bounds[1])
            max_lon, max_lat = transformer.transform(full_scene_bounds[2], full_scene_bounds[3])
            full_scene_extent = (min_lon, min_lat, max_lon, max_lat)

            # Initialize dynamic weather fetcher
            from ..io.dynamic_weather_fetcher import DynamicWeatherFetcher
            weather_config = self.config.get('weather', {})
            grid_spacing = weather_config.get('grid_spacing_km', 9.0)
            weather_fetcher = DynamicWeatherFetcher(grid_spacing_km=grid_spacing)

            try:
                # Fetch spatially varying weather data using full scene extent
                weather_arrays = weather_fetcher.fetch_weather_for_scene(
                    landsat_dir, target_coords, full_scene_extent
                )

                # Convert temperature from Celsius to Kelvin
                if "temperature_2m" in weather_arrays:
                    weather_arrays["temperature_2m"] = weather_arrays["temperature_2m"] + 273.15

                # Add weather data to cube
                for var_name, array in weather_arrays.items():
                    self.data.add(var_name, array)

                logger.info(f"Spatially varying weather data loaded for {len(weather_arrays)} variables using full scene extent")

            except Exception as e:
                logger.error(f"Failed to fetch dynamic weather data: {e}")
                raise

            # TODO: Load DEM if provided
            if dem_path:
                logger.warning("DEM loading not implemented yet")

            logger.info("Data loading completed successfully")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess(self) -> None:
        """Apply preprocessing."""
        import logging

        logger = logging.getLogger(__name__)

        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            logger.info("Starting preprocessing")

            # Step 1.5: Clip to ROI first
            logger.info("Step 1.5: Clipping scene bands to ROI boundary")
            self.clip_to_roi()

            # Step 1.6: Apply cloud masking to clipped scene
            logger.info("Step 1.6: Applying cloud masking to clipped scene")
            qa_pixel = self.data.get('qa_pixel')
            if qa_pixel is not None:
                from ..preprocess.cloud_mask import CloudMasker
                masker = CloudMasker(
                    cloud_confidence_threshold=CloudMasker.CONFIDENCE_HIGH,
                    dilate_pixels=3,
                    include_snow=False,
                    include_water=True
                )
                cloud_mask = masker.create_mask(qa_pixel)

                # Apply mask to all bands
                masked_cube = masker.apply_mask(self.data, cloud_mask, fill_value=np.nan)

                # Replace original data with masked data
                self.data = masked_cube
                logger.info("Cloud masking applied to clipped scene")
            else:
                logger.warning("No QA pixel band found, skipping cloud masking")

            # Step 1.7: Calculate cloud coverage fraction on clipped scene
            logger.info("Step 1.7: Calculating cloud coverage fraction on clipped scene")
            if qa_pixel is not None:
                masked_pixels = np.sum(cloud_mask)
                total_pixels = cloud_mask.size
                self._cloud_coverage = masked_pixels / total_pixels
                self._masked_pixels = masked_pixels
                self._total_pixels = total_pixels
                logger.info(f"Cloud coverage calculated on clipped scene: {masked_pixels}/{total_pixels} = {self._cloud_coverage:.3f}")
            else:
                logger.warning("No QA pixel band found for cloud coverage calculation")

            # TODO: Add additional preprocessing steps as needed
            # - Resampling
            # - Atmospheric correction
            # - Geometric correction

            logger.info("Preprocessing completed")

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise
    
    def clip_to_roi(self) -> None:
        """Clip spatial scene bands to ROI boundary during preprocessing.
        
        This method clips only the spatial Landsat bands (similar to cloud masking) to the ROI boundary
        early in the preprocessing workflow, ensuring all subsequent calculations
        run on the smaller, clipped dataset for computational efficiency.
        
        Weather data (spatially uniform) is not clipped since it doesn't have spatial extent.
        """
        import logging
        
        logger = logging.getLogger(__name__)
        
        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            
            if not hasattr(self, '_roi_geom'):
                logger.warning("No ROI geometry available, skipping ROI clipping")
                return
            
            logger.info("Clipping spatial scene bands to ROI boundary")
            
            # Get list of all bands in the data cube
            all_bands = self.data.bands()
            clipped_bands = []
            skipped_bands = []
            
            for band_name in all_bands:
                band_data = self.data.get(band_name)
                
                # Skip bands that don't have CRS (spatially uniform data like weather)
                if not hasattr(band_data, 'rio') or band_data.rio.crs is None:
                    logger.debug(f"Skipping {band_name} - no spatial CRS (spatially uniform data)")
                    skipped_bands.append(band_name)
                    continue
                
                # Ensure rioxarray integration is available for spatial operations
                if not hasattr(band_data, 'rio'):
                    import rioxarray
                    band_data = band_data.rio.write_crs(self.data.crs)
                
                # Check if CRS is available
                if band_data.rio.crs is None:
                    logger.debug(f"Skipping {band_name} - no CRS available")
                    skipped_bands.append(band_name)
                    continue
                
                try:
                    # Clip to ROI geometry
                    clipped_band = band_data.rio.clip(
                        [self._roi_geom], 
                        crs=band_data.rio.crs, 
                        drop=False
                    )
                    
                    # Replace original band with clipped version
                    self.data.add(band_name, clipped_band)
                    clipped_bands.append(band_name)
                    
                    logger.debug(f"Clipped {band_name} to ROI")
                except Exception as e:
                    logger.warning(f"Failed to clip {band_name}: {e}")
                    skipped_bands.append(band_name)
                    continue
            
            if clipped_bands:
                logger.info(f"Clipped {len(clipped_bands)} spatial bands to ROI: {clipped_bands}")
                
                if skipped_bands:
                    logger.info(f"Skipped {len(skipped_bands)} non-spatial bands: {skipped_bands}")
                
                # Update data extent and transform based on clipped data
                sample_clipped = self.data.get(clipped_bands[0])
                if hasattr(sample_clipped, 'rio'):
                    self.data.extent = sample_clipped.rio.bounds()
                    self.data.transform = sample_clipped.rio.transform()
                else:
                    # Fallback if rioxarray is not available
                    logger.warning("rioxarray not available for extent update")
                
                logger.info(f"Updated data extent to ROI boundaries: {self.data.extent}")
                
                # Calculate actual pixel counts from first clipped band
                sample_band = self.data.get(clipped_bands[0])
                if hasattr(sample_band, 'shape'):
                    clipped_pixels = np.prod(sample_band.shape)
                    logger.info(f"ROI clipping completed: working with {clipped_pixels} pixels in clipped area")
                
                # Store clipping information for enhanced QA coverage calculation
                # Store the original extent before clipping (from initial data loading)
                if not hasattr(self, '_original_extent') or self._original_extent is None:
                    self._original_extent = self.data.extent  # This is the original extent before clipping
                self._roi_extent = self.data.extent
                self._roi_mask = self._create_roi_mask_from_clipped_data(sample_band)
                
            else:
                logger.warning("No spatial bands found to clip")
                if skipped_bands:
                    logger.info(f"All bands were non-spatial: {skipped_bands}")
                
        except Exception as e:
            logger.error(f"Error clipping to ROI: {e}")
            raise
    
    def _create_roi_mask_from_clipped_data(self, sample_band) -> np.ndarray:
        """
        Create boolean mask indicating which pixels are within ROI boundaries.
        
        Args:
            sample_band: Sample band data to extract coordinate information
            
        Returns:
            Boolean mask array where True indicates pixels within ROI boundaries
        """
        try:
            # Log the shape of the input band to verify clipping worked
            if hasattr(sample_band, 'shape'):
                logger.debug(f"ROI mask input band shape: {sample_band.shape}")
            
            # Create mask based on non-NaN values in the clipped band
            # This represents the actual ROI area after clipping
            if hasattr(sample_band, 'values'):
                roi_mask = ~np.isnan(sample_band.values)
            else:
                roi_mask = ~np.isnan(sample_band)
            
            # Log valid pixels vs total pixels in the ROI-clipped area
            valid_count = np.sum(roi_mask)
            total_count = roi_mask.size
            logger.debug(f"Created ROI mask: {valid_count} valid pixels out of {total_count} pixels in ROI area")
            
            # Warn if the total count matches full scene size (clipping may have failed)
            if total_count > 500000:  # Threshold for typical scene size
                logger.warning(f"ROI pixel count ({total_count}) suggests clipping may have failed - this is full scene size")
            
            return roi_mask
            
        except Exception as e:
            logger.warning(f"Failed to create ROI mask: {e}")
            # Fallback: create full mask if coordinates not available
            if hasattr(sample_band, 'shape'):
                return np.ones(sample_band.shape, dtype=bool)
            else:
                return np.array([True])
    
    def calculate_qa_coverage_enhanced(self, ndvi: xr.DataArray) -> Tuple[float, int, int]:
        """
        Calculate QA coverage counting only cloud-masked pixels as loss.
        
        This method counts only pixels removed by cloud masking as pixel loss,
        not clipped boundary regions. The calculation is:
        - Total pixels: All pixels in the scene
        - Valid pixels: Pixels not masked by clouds (NaN values from cloud masking)
        
        Args:
            ndvi: NDVI array for QA coverage calculation
            
        Returns:
            Tuple of (valid_pixel_fraction, valid_pixels, total_pixels)
            where only cloud-masked pixels count as loss
        """
        try:
            # Count valid pixels (not NaN from cloud masking)
            valid_pixels = np.sum(~np.isnan(ndvi.values))
            total_pixels = ndvi.size
            
            # Calculate QA coverage
            qa_coverage = valid_pixels / total_pixels
            
            logger.debug(f"Cloud-mask QA calculation: {valid_pixels}/{total_pixels} = {qa_coverage:.3f}")
            return qa_coverage, valid_pixels, total_pixels
            
        except Exception as e:
            logger.error(f"Error in cloud-mask QA calculation: {e}")
            # Fallback to original behavior on error
            valid_pixels = np.sum(~np.isnan(ndvi.values))
            total_pixels = ndvi.size
            return valid_pixels / total_pixels, valid_pixels, total_pixels
    
    def calculate_surface_properties(self) -> None:
        """Calculate surface properties."""
        from ..surface import VegetationIndices, AlbedoCalculator, EmissivityCalculator, RoughnessCalculator, LSTCalculator
        import logging

        logger = logging.getLogger(__name__)

        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            logger.info("Calculating surface properties")

            # Calculate vegetation indices (NDVI, EVI, LAI, SAVI, FVC)
            veg_indices = VegetationIndices()
            veg_indices.compute(self.data)
            logger.info("Vegetation indices calculated")

            # Calculate broadband albedo
            albedo_calc = AlbedoCalculator()
            albedo_calc.compute(self.data)
            logger.info("Albedo calculation completed")

            # Check if albedo was added
            if "albedo" in self.data.bands():
                albedo_data = self.data.get("albedo")
                logger.info(f"Albedo added to DataCube: shape={albedo_data.shape}, mean={np.nanmean(albedo_data.values):.3f}")
            else:
                logger.error("Albedo not found in DataCube after calculation!")

            # Calculate emissivity
            emissivity_calc = EmissivityCalculator()
            emissivity_calc.compute(self.data)
            logger.info("Emissivity calculation completed")

            # Calculate land surface temperature (LST)
            lst_calc = LSTCalculator()
            lst_calc.compute(self.data)
            logger.info("LST calculation completed")

            # Calculate roughness parameters
            roughness_calc = RoughnessCalculator()
            roughness_calc.compute(self.data)
            logger.info("Roughness calculation completed")

            logger.info("Surface properties calculation completed")

        except Exception as e:
            logger.error(f"Error calculating surface properties: {e}")
            raise
    
    def calculate_radiation_balance(self) -> None:
        """Calculate radiation balance components."""
        from ..radiation import ShortwaveRadiation, LongwaveRadiation, NetRadiation
        import logging

        logger = logging.getLogger(__name__)

        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            logger.info("Calculating radiation balance")

            # Calculate shortwave radiation components
            shortwave_calc = ShortwaveRadiation()
            shortwave_calc.compute(self.data)

            # Calculate longwave radiation components
            longwave_calc = LongwaveRadiation()
            longwave_calc.compute(self.data)

            # Calculate net radiation
            net_radiation_calc = NetRadiation()
            net_radiation_calc.compute(self.data)

            logger.info("Radiation balance calculation completed")

        except Exception as e:
            logger.error(f"Error calculating radiation balance: {e}")
            raise

    def validate_scene(self) -> Tuple[bool, str]:
        """Perform scene-level pre-validation (HARD REJECT) checks.

        Returns:
            Tuple of (rejected: bool, reason: str)
            If rejected is True, the scene should be rejected with the given reason.
        """
        import logging
        from ..core.constants import MJ_M2_DAY_TO_W

        logger = logging.getLogger(__name__)

        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            logger.info("Performing scene-level pre-validation checks")

            rejected = False
            reason = ""

            # Get required data
            ndvi = self.data.get("ndvi")
            rn = self.data.get("R_n")
            et0_daily_array = self.data.get("et0_fao_evapotranspiration")
            rs_inst_array = self.data.get("shortwave_radiation")
            rs_daily_array = self.data.get("shortwave_radiation_sum")

            if ndvi is None:
                return True, "NDVI data not available for validation"
            if rn is None:
                return True, "Net radiation (R_n) not available for validation"
            if et0_daily_array is None or rs_inst_array is None or rs_daily_array is None:
                return True, "Weather data not available for ET0_inst calculation"

            # 1. Cloud coverage check: use cloud coverage fraction instead of QA coverage
            # This uses the cloud coverage calculated during preprocessing
            if not hasattr(self, '_cloud_coverage'):
                logger.warning("Cloud coverage not calculated during preprocessing, skipping cloud coverage check")
                cloud_coverage = 0.0
            else:
                cloud_coverage = self._cloud_coverage

            # Get cloud coverage thresholds from config
            # Reject if >70% cloud coverage (cloud_reject_threshold = 0.70)
            # Flag as LOW QUALITY if >30% cloud coverage (cloud_low_quality_threshold = 0.30)
            cloud_reject_threshold = self.config.get('cloud_reject_threshold', 0.70)
            cloud_low_quality_threshold = self.config.get('cloud_low_quality_threshold', 0.30)

            logger.info(f"Cloud coverage: {cloud_coverage:.3f}")
            logger.info(f"Thresholds - Reject: >{cloud_reject_threshold:.2f}, Low Quality: >{cloud_low_quality_threshold:.2f}")

            if cloud_coverage > cloud_reject_threshold:
                # Reject if more than 70% cloud coverage
                rejected = True
                reason = f"Cloud coverage too high: {cloud_coverage:.3f} > {cloud_reject_threshold:.2f}"
                logger.error(reason)
                return rejected, reason
            elif cloud_coverage > cloud_low_quality_threshold:
                # Flag as LOW QUALITY if more than 30% cloud coverage
                logger.warning(f"Cloud coverage indicates moderate cloud cover: {cloud_coverage:.3f} > {cloud_low_quality_threshold:.2f} - flagging as LOW QUALITY")
                # Store this information for later quality flagging
                self._qa_coverage_issue = True
                # Don't return here - continue with processing but will be flagged as LOW QUALITY later

            # 2. NDVI dynamic range check: NDVI_p95 - NDVI_p05 < 0.30
            ndvi_values = ndvi.values[~np.isnan(ndvi.values)]
            if len(ndvi_values) == 0:
                return True, "No valid NDVI values for dynamic range check"

            ndvi_p5 = np.percentile(ndvi_values, 5)
            ndvi_p95 = np.percentile(ndvi_values, 95)
            ndvi_range = ndvi_p95 - ndvi_p5

            logger.info(f"NDVI dynamic range: P95={ndvi_p95:.3f}, P5={ndvi_p5:.3f}, range={ndvi_range:.3f}")

            if ndvi_range < 0.30:
                rejected = True
                reason = f"NDVI dynamic range too low: {ndvi_range:.3f} < 0.30"
                logger.warning(reason)
                return rejected, reason

            # 3. Net radiation sanity check: median(Rn) < 300 W/m²
            rn_values = rn.values[~np.isnan(rn.values)]
            if len(rn_values) == 0:
                return True, "No valid Rn values for median check"

            rn_median = np.median(rn_values)

            logger.info(f"Net radiation median: {rn_median:.1f} W/m²")

            if rn_median < 300:
                rejected = True
                reason = f"Net radiation median too low: {rn_median:.1f} < 300 W/m²"
                logger.warning(reason)
                return rejected, reason

            # 4. ET0_inst sanity check: ET0_inst < 0.2 or > 1.0 mm/hr
            # Calculate ET0_inst similar to calibration
            et0_daily = float(np.nanmean(et0_daily_array.values))
            rs_inst = float(np.nanmean(rs_inst_array.values))
            rs_daily = float(np.nanmean(rs_daily_array.values))

            rs_daily_avg_w = rs_daily * MJ_M2_DAY_TO_W  # MJ/m²/day -> W/m²

            if rs_daily_avg_w <= 0:
                return True, "Invalid daily shortwave radiation for ET0_inst calculation"

            radiation_ratio = rs_inst / rs_daily_avg_w
            et0_inst = et0_daily * radiation_ratio  # mm/day-equivalent

            logger.info(f"ET0_inst: {et0_inst:.3f} mm/day-equiv")

            if et0_inst < 5.0 or et0_inst > 40.0:
                rejected = True
                reason = f"ET0_inst out of range: {et0_inst:.3f} mm/day-equiv (must be 5-40)"
                logger.warning(reason)
                return rejected, reason

            logger.info("Scene-level pre-validation passed")
            return False, ""

        except Exception as e:
            logger.error(f"Error in scene validation: {e}")
            return True, f"Validation error: {str(e)}"
    
    def calculate_soil_heat_flux(self) -> None:
        """Calculate soil heat flux (G) independently of calibration.
        
        G depends only on:
        - Net radiation (Rn)
        - NDVI or LAI
        - Surface temperature (Ts)
        
        This method computes G BEFORE calibration so it can be used
        for anchor pixel selection (Rn-G optimization).
        """
        from ..energy_balance import SoilHeatFlux, SoilHeatFluxConfig
        import logging

        logger = logging.getLogger(__name__)

        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            logger.info("Calculating soil heat flux (calibration-free)")

            # Check if Rn is available
            rn = self.data.get("R_n")
            if rn is None:
                raise ValueError("Net radiation (R_n) not found. Calculate radiation balance first.")

            # Get required inputs
            ndvi = self.data.get("ndvi")
            ts_kelvin = self.data.get("lst")
            ta_kelvin = self.data.get("temperature_2m")

            if ts_kelvin is None:
                raise ValueError("Surface temperature (lst) not found")
            if ta_kelvin is None:
                raise ValueError("Air temperature (temperature_2m) not found")

            # Initialize soil heat flux calculator
            g_config = SoilHeatFluxConfig(method="automatic")
            g_calculator = SoilHeatFlux(g_config)

            # Calculate G
            logger.info("Computing soil heat flux using Rn, NDVI, Ts, Ta")
            g_result = g_calculator.calculate(
                rn=rn.values,
                ndvi=ndvi.values if ndvi is not None else None,
                ts_kelvin=ts_kelvin.values,
                ta_kelvin=ta_kelvin.values
            )

            # Add G to data cube
            self.data.add("G", g_result['G'])

            # Log statistics
            g_values = g_result['G']
            valid_g = g_values[~np.isnan(g_values)]
            logger.info(f"Soil heat flux calculated: {len(valid_g)} valid pixels")
            logger.info(f"  Range: [{np.min(valid_g):.1f}, {np.max(valid_g):.1f}] W/m²")
            logger.info(f"  Mean: {np.mean(valid_g):.1f} W/m², Std: {np.std(valid_g):.1f} W/m²")

            # Energy balance check: G should be 10-30% of Rn
            if 'G_Rn_ratio' in g_result:
                ratio = g_result['G_Rn_ratio']
                valid_ratio = ratio[~np.isnan(ratio)]
                if len(valid_ratio) > 0:
                    logger.info(f"  G/Rn ratio: mean={np.mean(valid_ratio):.3f}, range=[{np.min(valid_ratio):.3f}, {np.max(valid_ratio):.3f}]")

            logger.info("Soil heat flux calculation completed")

        except Exception as e:
            logger.error(f"Error calculating soil heat flux: {e}")
            raise
    
    def calculate_energy_balance(self) -> None:
        """Calculate energy balance components (H and LE) with calibration.
        
        Assumes Rn and G are already computed.
        Uses anchor pixel calibration from METRIC to compute H and LE.
        """
        from ..energy_balance import EnergyBalanceManager
        import logging

        logger = logging.getLogger(__name__)

        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            logger.info("Calculating energy balance (H and LE with calibration)")

            # Check if Rn and G are available
            rn = self.data.get("R_n")
            g_flux = self.data.get("G")
            
            if rn is None:
                raise ValueError("Net radiation (R_n) not found. Calculate radiation balance first.")
            if g_flux is None:
                raise ValueError("Soil heat flux (G) not found. Calculate soil heat flux first.")

            # Initialize or reuse energy balance manager
            if self._eb_manager is None:
                self._eb_manager = EnergyBalanceManager()

            # Calculate energy balance components (H and LE)
            # This will use the anchor pixel calibration set during calibrate() step
            eb_results = self._eb_manager.calculate(self.data)

            logger.info("Energy balance calculation completed")

        except Exception as e:
            logger.error(f"Error calculating energy balance: {e}")
            raise

    def _safe_nanmean(self, data, data_name: str) -> float:
        """Safely calculate nanmean with proper error handling and logging.
        
        Args:
            data: Input data (xarray DataArray or numpy array)
            data_name: Name of the data for logging purposes
            
        Returns:
            Float value of the mean, or raises ValueError if data is invalid
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if data is None:
            logger.error(f"{data_name} data is None")
            raise ValueError(f"{data_name} data is None")
            
        # Check if data has values attribute (xarray DataArray)
        if hasattr(data, 'values'):
            values = data.values
        else:
            values = data
            
        # Check for empty arrays
        if values.size == 0:
            logger.error(f"{data_name} has empty array")
            raise ValueError(f"{data_name} has empty array")
            
        # Check if all values are NaN
        if np.all(np.isnan(values)):
            logger.error(f"{data_name} contains only NaN values")
            raise ValueError(f"{data_name} contains only NaN values")
            
        return float(np.nanmean(values))

    def _get_air_temperature(self) -> float:
        """Get air temperature with improved spatial handling.
        
        Returns:
            Air temperature in Kelvin as float
            
        This method handles both spatially varying and uniform temperature data,
        with proper fallback to default values.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        temp_2m = self.data.get("temperature_2m")
        config_temp = self.config.get('temperature', {})
        default_temp = config_temp.get('default_kelvin', 293.15)  # 20°C in Kelvin
        
        if temp_2m is not None:
            try:
                # Handle both spatially varying and uniform temperature data
                air_temperature = self._safe_nanmean(temp_2m, "temperature_2m")
                logger.info(f"Air temperature: {air_temperature:.2f} K ({air_temperature-273.15:.2f} °C)")
                return air_temperature
            except Exception as e:
                logger.warning(f"Failed to process temperature data: {e}, using default")
        
        logger.warning(f"Using default air temperature: {default_temp} K ({default_temp-273.15:.2f} °C)")
        return default_temp

    def _get_et0_daily(self) -> float:
        """Get daily ET0 value for METRIC scaling.

        Returns:
            Daily ET0 value in mm/day

        METRIC uses ET0_daily directly for final scaling: ET_daily = EF * ET0_daily
        """
        import logging
        logger = logging.getLogger(__name__)

        et0_daily = self.data.get("et0_fao_evapotranspiration")

        if et0_daily is not None:
            et0_value = self._safe_nanmean(et0_daily, "ET0_daily")
            logger.info(f"Using ET0_daily = {et0_value:.3f} mm/day for METRIC scaling")
            return et0_value

        # Fallback to default value if no ET0 data available
        default_et0 = 5.0  # mm/day
        logger.warning(f"No ET0 data available, using default ET0_daily = {default_et0} mm/day")
        return default_et0
    
    def calibrate(self) -> None:
        """Apply unified METRIC calibration pipeline with decision logic and logging."""
        from ..calibration import AnchorPixelSelector, DTCalibration
        from ..energy_balance import EnergyBalanceManager
        import logging

        logger = logging.getLogger(__name__)

        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            logger.info("Starting unified METRIC calibration pipeline")

            # Initialize or reuse energy balance manager
            if self._eb_manager is None:
                self._eb_manager = EnergyBalanceManager()

            # Select anchor pixel selector
            calibration_config = self.config.get('calibration', {})
            method = calibration_config.get('method', 'automatic')
            
            if method == 'automatic':
                # Use standard AnchorSelector with built-in automatic method
                anchor_selector = AnchorPixelSelector(method='automatic')
            else:
                # Use standard selector with specified method
                anchor_selector = AnchorPixelSelector(method=method)

            # Execute unified calibration pipeline
            calibrator = DTCalibration.create()
            calibration, anchor_result = calibrator.unified_calibration_pipeline(
                cube=self.data,
                scene_id=self._scene_id,
                energy_balance_manager=self._eb_manager,
                anchor_pixel_selector=anchor_selector,
                validation_config=calibration_config
            )

            # Store calibration result for later use
            self._calibration_result = calibration
            self._anchor_result = anchor_result
            self._scene_quality = calibration.scene_quality

            # Log cluster information
            import logging
            logger = logging.getLogger(__name__)
            logger.info("=== CLUSTER STORAGE LOGGING ===")
            logger.info(f"anchor_result is None: {anchor_result is None}")
            if anchor_result:
                logger.info(f"anchor_result has cold_cluster: {hasattr(anchor_result, 'cold_cluster')}")
                logger.info(f"anchor_result has hot_cluster: {hasattr(anchor_result, 'hot_cluster')}")
                if hasattr(anchor_result, 'cold_cluster') and anchor_result.cold_cluster:
                    logger.info(f"Stored cold cluster with {len(anchor_result.cold_cluster.pixel_indices)} pixels")
                if hasattr(anchor_result, 'hot_cluster') and anchor_result.hot_cluster:
                    logger.info(f"Stored hot cluster with {len(anchor_result.hot_cluster.pixel_indices)} pixels")
            logger.info("=== END CLUSTER STORAGE LOGGING ===")

            # Check decision result
            if calibration.status == CalibrationStatus.REJECTED:
                # Case 3: No valid calibration and no fallback - reject scene
                logger.error(
                    f"Scene {self._scene_id} REJECTED: {calibration.rejection_reason}. "
                    "No ET outputs will be produced."
                )
                return  # Exit without producing ET outputs

            # For ACCEPTED or REUSED status, ensure energy balance is calculated
            # The unified pipeline should have already calculated it, but ensure it's available
            if self.data.get("H") is None or self.data.get("LE") is None:
                logger.info("Recalculating energy balance with final calibration")
                self._eb_manager.set_anchor_pixel_calibration(
                    calibration.a_coefficient, calibration.b_coefficient
                )
                eb_results = self._eb_manager.calculate(self.data)

            logger.info(f"Unified METRIC calibration completed with status: {calibration.status.value}")
            logger.info(f"  Pre-validation: {calibration.prevalidation_passed}")
            logger.info(f"  Anchor physics: {calibration.anchor_physics_valid}")
            logger.info(f"  Global validation: {calibration.global_validation_passed}")
            if calibration.global_violations:
                logger.warning(f"  Global violations: {calibration.global_violations}")

        except Exception as e:
            logger.error(f"Error in unified METRIC calibration: {e}")
            raise
    
    def calculate_et(self) -> None:
        """Calculate evapotranspiration."""
        from ..et import InstantaneousET, DailyET, ETQuality
        import logging

        logger = logging.getLogger(__name__)

        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            logger.info("=== STARTING EVAPOTRANSPIRATION CALCULATION ===")

            # Log all available input data
            logger.info("=== INPUT DATA SUMMARY ===")
            all_bands = self.data.bands()
            logger.info(f"Available bands: {all_bands}")

            # Log key surface properties
            ndvi = self.data.get("ndvi")
            albedo = self.data.get("albedo")
            emissivity = self.data.get("emissivity")
            lai = self.data.get("lai")

            if ndvi is not None:
                valid_ndvi = ndvi.values[~np.isnan(ndvi.values)]
                logger.info(f"NDVI: {len(valid_ndvi)} valid pixels, range [{np.min(valid_ndvi):.3f}, {np.max(valid_ndvi):.3f}], mean={np.mean(valid_ndvi):.3f}")

            if albedo is not None:
                valid_albedo = albedo.values[~np.isnan(albedo.values)]
                logger.info(f"Albedo: {len(valid_albedo)} valid pixels, range [{np.min(valid_albedo):.3f}, {np.max(valid_albedo):.3f}], mean={np.mean(valid_albedo):.3f}")

            if emissivity is not None:
                valid_emiss = emissivity.values[~np.isnan(emissivity.values)]
                logger.info(f"Emissivity: {len(valid_emiss)} valid pixels, range [{np.min(valid_emiss):.3f}, {np.max(valid_emiss):.3f}], mean={np.mean(valid_emiss):.3f}")

            # Log radiation components
            rn = self.data.get("R_n")
            rs_down = self.data.get("Rs_down")
            rl_up = self.data.get("R_l_up")
            rl_down = self.data.get("R_l_down")

            if rn is not None:
                valid_rn = rn.values[~np.isnan(rn.values)]
                logger.info(f"Net Radiation (Rn): {len(valid_rn)} valid pixels, range [{np.min(valid_rn):.1f}, {np.max(valid_rn):.1f}] W/m², mean={np.mean(valid_rn):.1f} W/m²")

            if rs_down is not None:
                valid_rs = rs_down.values[~np.isnan(rs_down.values)]
                logger.info(f"Shortwave Down (Rs↓): {len(valid_rs)} valid pixels, range [{np.min(valid_rs):.1f}, {np.max(valid_rs):.1f}] W/m², mean={np.mean(valid_rs):.1f} W/m²")

            # Log energy balance components
            g_flux = self.data.get("G")
            h_flux = self.data.get("H")
            le_flux = self.data.get("LE")

            if g_flux is not None:
                valid_g = g_flux.values[~np.isnan(g_flux.values)]
                logger.info(f"Soil Heat Flux (G): {len(valid_g)} valid pixels, range [{np.min(valid_g):.1f}, {np.max(valid_g):.1f}] W/m², mean={np.mean(valid_g):.1f} W/m²")

            if h_flux is not None:
                valid_h = h_flux.values[~np.isnan(h_flux.values)]
                logger.info(f"Sensible Heat Flux (H): {len(valid_h)} valid pixels, range [{np.min(valid_h):.1f}, {np.max(valid_h):.1f}] W/m², mean={np.mean(valid_h):.1f} W/m²")

            if le_flux is not None:
                valid_le = le_flux.values[~np.isnan(le_flux.values)]
                logger.info(f"Latent Heat Flux (LE): {len(valid_le)} valid pixels, range [{np.min(valid_le):.1f}, {np.max(valid_le):.1f}] W/m², mean={np.mean(valid_le):.1f} W/m²")

            # Log weather data
            temp_2m = self.data.get("temperature_2m")
            et0_daily = self.data.get("et0_fao_evapotranspiration")
            rs_daily = self.data.get("shortwave_radiation_sum")

            if temp_2m is not None:
                valid_temp = temp_2m.values[~np.isnan(temp_2m.values)]
                logger.info(f"Air Temperature (Tₐ): {len(valid_temp)} valid pixels, range [{np.min(valid_temp)-273.15:.1f}, {np.max(valid_temp)-273.15:.1f}] °C, mean={np.mean(valid_temp)-273.15:.1f} °C")

            if et0_daily is not None:
                valid_et0 = et0_daily.values[~np.isnan(et0_daily.values)]
                logger.info(f"Daily ET₀ (FAO): {len(valid_et0)} valid pixels, range [{np.min(valid_et0):.3f}, {np.max(valid_et0):.3f}] mm/day, mean={np.mean(valid_et0):.3f} mm/day")

            if rs_daily is not None:
                valid_rs_daily = rs_daily.values[~np.isnan(rs_daily.values)]
                logger.info(f"Daily Shortwave Sum: {len(valid_rs_daily)} valid pixels, range [{np.min(valid_rs_daily):.1f}, {np.max(valid_rs_daily):.1f}] MJ/m²/day, mean={np.mean(valid_rs_daily):.1f} MJ/m²/day")

            # Log surface temperature
            ts_kelvin = self.data.get("lst")
            if ts_kelvin is not None:
                valid_ts = ts_kelvin.values[~np.isnan(ts_kelvin.values)]
                logger.info(f"Surface Temperature (Ts): {len(valid_ts)} valid pixels, range [{np.min(valid_ts)-273.15:.1f}, {np.max(valid_ts)-273.15:.1f}] °C, mean={np.mean(valid_ts)-273.15:.1f} °C")

            logger.info("=== STARTING INSTANTANEOUS ET CALCULATION ===")

            # Calculate instantaneous ET
            inst_et_calc = InstantaneousET()

            # Get ET0_daily for METRIC scaling
            et0_daily_value = self._get_et0_daily()

            logger.info("Step 1: Calculating instantaneous ET from energy balance")
            logger.info("Formula: ET_inst = LE / (ρ × λ) where LE is latent heat flux")

            # Calculate instantaneous ET from LE
            inst_et_result = inst_et_calc.calculate(le=self.data.get("LE"))

            # Log instantaneous ET results
            if "ET_inst" in inst_et_result:
                et_inst_values = inst_et_result["ET_inst"]
                valid_et_inst = et_inst_values[~np.isnan(et_inst_values)]
                logger.info(f"ET_inst results: {len(valid_et_inst)} valid pixels")
                logger.info(f"  Range: [{np.min(valid_et_inst):.6f}, {np.max(valid_et_inst):.6f}] mm/hr")
                logger.info(f"  Mean: {np.mean(valid_et_inst):.6f} mm/hr, Std: {np.std(valid_et_inst):.6f} mm/hr")

            # Add instantaneous ET to data
            self.data.add("ET_inst", inst_et_result["ET_inst"])

            # Set ETrF = EF (METRIC standard: ETrF ≈ EF)
            ef = self.data.get("EF")
            if ef is not None:
                self.data.add("ETrF", ef)
                etrf_values = ef.values
                valid_etrf = etrf_values[~np.isnan(etrf_values)]
                logger.info(f"ETrF set to EF: {len(valid_etrf)} valid pixels")
                logger.info(f"  Range: [{np.min(valid_etrf):.6f}, {np.max(valid_etrf):.6f}]")
                logger.info(f"  Mean: {np.mean(valid_etrf):.6f}, Std: {np.std(valid_etrf):.6f}")

                # Check for unrealistic ETrF values
                if np.max(valid_etrf) > 1.5:
                    logger.warning(f"WARNING: ETrF maximum ({np.max(valid_etrf):.3f}) exceeds 1.5")
                if np.min(valid_etrf) < 0:
                    logger.warning(f"WARNING: ETrF minimum ({np.min(valid_etrf):.3f}) is negative")
            else:
                logger.warning("EF not available, cannot set ETrF")

            logger.info("=== STARTING DAILY ET CALCULATION ===")

            # Calculate daily ET using METRIC standard approach
            daily_et_calc = DailyET()
            logger.info("Step 3: Calculating daily ET using METRIC methodology")
            logger.info("Formula: ET_daily = EF × ET0_daily")
            logger.info("Note: METRIC uses evaporative fraction directly with daily reference ET")

            # Use ET0_daily directly for METRIC scaling
            et0_daily_data = self.data.get("et0_fao_evapotranspiration")

            daily_et_result = daily_et_calc.calculate(
                etrf=self.data.get("ETrF"),  # EF from instantaneous calculation
                etr_daily=et0_daily_data   # Use ET0_daily directly (not converted to ETr)
            )

            # Log daily ET results
            if "ET_daily" in daily_et_result:
                et_daily_values = daily_et_result["ET_daily"]
                valid_et_daily = et_daily_values[~np.isnan(et_daily_values)]
                logger.info(f"ET_daily results: {len(valid_et_daily)} valid pixels")
                logger.info(f"  Range: [{np.min(valid_et_daily):.6f}, {np.max(valid_et_daily):.6f}] mm/day")
                logger.info(f"  Mean: {np.mean(valid_et_daily):.6f} mm/day, Std: {np.std(valid_et_daily):.6f} mm/day")

                # Check for unrealistic values
                if np.max(valid_et_daily) > 15:
                    logger.warning(f"WARNING: ET_daily maximum ({np.max(valid_et_daily):.3f} mm/day) seems high")
                if np.min(valid_et_daily) < 0:
                    logger.warning(f"WARNING: ET_daily minimum ({np.min(valid_et_daily):.3f} mm/day) is negative")

            # Add daily ET to data
            self.data.add("ET_daily", daily_et_result["ET_daily"])

            logger.info("=== ET QUALITY ASSESSMENT ===")

            # Calculate ET quality assessment
            quality_calc = ETQuality()
            quality_result = quality_calc.assess(
                et_daily=self.data.get("ET_daily"),
                etrf=self.data.get("ETrF")
            )

            # Add quality metrics to data
            self.data.add("ET_quality_class", quality_result["quality_class"])

            logger.info("=== ETa CLASSIFICATION ===")
            
            # Create single ETa classification layer with values 1-7
            et_daily = self.data.get("ET_daily")
            if et_daily is not None:
                et_values = et_daily.values
                
                # Initialize with 0 (no class / NaN)
                eta_class = np.zeros(et_values.shape, dtype=np.uint8)
                
                # Assign class values based on ETa range
                # Class 1: 0-3 mm/day
                eta_class[(et_values >= 0) & (et_values < 3)] = 1
                # Class 2: 3-6 mm/day
                eta_class[(et_values >= 3) & (et_values < 6)] = 2
                # Class 3: 6-9 mm/day
                eta_class[(et_values >= 6) & (et_values < 9)] = 3
                # Class 4: 9-12 mm/day
                eta_class[(et_values >= 9) & (et_values < 12)] = 4
                # Class 5: 12-15 mm/day
                eta_class[(et_values >= 12) & (et_values < 15)] = 5
                # Class 6: 15-20 mm/day
                eta_class[(et_values >= 15) & (et_values < 20)] = 6
                # Class 7: 20+ mm/day
                eta_class[et_values >= 20] = 7
                
                self.data.add("ETa_class", eta_class)
                
                logger.info("ETa classification layer created: 1=0-3, 2=3-6, 3=6-9, 4=9-12, 5=12-15, 6=15-20, 7=20+")
            
            logger.info("=== CWSI CALCULATION ===")
            
            # Calculate CWSI = 1 - (ETa / ET0)
            et_daily = self.data.get("ET_daily")
            et0_daily = self.data.get("et0_fao_evapotranspiration")
            
            if et_daily is not None and et0_daily is not None:
                et_values = et_daily.values
                et0_values = et0_daily.values
                
                # Handle division - avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    cwsi = 1.0 - (et_values / et0_values)
                    # Set invalid values (NaN, inf) to NaN
                    cwsi = np.where(np.isfinite(cwsi), cwsi, np.nan)
                
                self.data.add("CWSI", cwsi)
                
                # Log statistics
                valid_cwsi = cwsi[~np.isnan(cwsi)]
                if len(valid_cwsi) > 0:
                    logger.info(f"CWSI calculated: {len(valid_cwsi)} valid pixels")
                    logger.info(f"  Range: [{np.nanmin(valid_cwsi):.3f}, {np.nanmax(valid_cwsi):.3f}]")
                    logger.info(f"  Mean: {np.nanmean(valid_cwsi):.3f}")
            else:
                logger.warning("ET_daily or ET0 data not available for CWSI calculation")

            logger.info(f"ET Quality: {quality_result.get('valid_fraction', 'N/A')} valid pixels")

            logger.info("=== EVAPOTRANSPIRATION CALCULATION COMPLETED ===")

        except Exception as e:
            logger.error(f"Error calculating ET: {e}")
            raise
    
    def get_results(self) -> Dict[str, xr.DataArray]:
        """Get processing results."""
        if self.data is None:
            raise ValueError("No data available. Run the pipeline first.")

        # Return key results
        results = {}

        # Surface properties
        surface_keys = ['ndvi', 'albedo', 'emissivity', 'lst', 'lai', 'z0m', 'd']
        for key in surface_keys:
            if key in self.data.bands():
                results[key] = self.data.get(key)

        # Radiation balance
        radiation_keys = ['R_n', 'R_n_daytime', 'R_ns', 'R_nl']
        for key in radiation_keys:
            if key in self.data.bands():
                results[key] = self.data.get(key)

        # Energy balance
        energy_keys = ['G', 'H', 'LE', 'rah', 'EF']
        for key in energy_keys:
            if key in self.data.bands():
                results[key] = self.data.get(key)

        # ET results
        et_keys = ['ET_inst', 'ET_daily', 'ETrF', 'ET_quality_class', 'ET_confidence']
        for key in et_keys:
            if key in self.data.bands():
                results[key] = self.data.get(key)

        return results

    def get_scene_quality(self) -> Dict[str, Any]:
        """
        Get scene quality information based on calibration decision.

        Returns:
            Dictionary with scene quality and calibration status information
        """
        if self._calibration_result is None:
            return {
                "quality": "UNKNOWN",
                "status": "NO_CALIBRATION",
                "scene_id": self._scene_id
            }

        return {
            "quality": self._scene_quality,
            "status": self._calibration_result.status.value,
            "scene_id": self._scene_id,
            "a_coefficient": self._calibration_result.a_coefficient,
            "b_coefficient": self._calibration_result.b_coefficient,
            "timestamp": self._calibration_result.timestamp,
            "rejection_reason": self._calibration_result.rejection_reason,
            "reuse_source": self._calibration_result.reuse_source,
            "valid": self._calibration_result.valid
        }

    def save_results(self, output_dir: str) -> None:
        """Save results to output directory."""
        from ..output import OutputWriter, Visualization
        import os
        import logging
        from datetime import datetime

        logger = logging.getLogger(__name__)

        try:
            if self.data is None:
                raise ValueError("No data available. Run the pipeline first.")

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            logger.info(f"Saving results to {output_dir}")

            # Use OutputWriter to save products
            writer = OutputWriter(output_dir=output_dir)
            
            # Get scene information for naming
            scene_id = self.data.metadata.get('scene_id', 'METRIC')
            date_str = self.data.acquisition_time.strftime('%Y%m%d') if self.data.acquisition_time else 'unknown'
            
            # Use the actual calibration result from the calibration step
            # The calibration result should be stored during the calibrate() step
            if hasattr(self, '_calibration_result'):
                actual_calibration = self._calibration_result
            else:
                # Fallback to mock calibration if no actual result is available
                from ..calibration.dt_calibration import CalibrationResult
                actual_calibration = CalibrationResult(
                    a_coefficient=0.0, b_coefficient=0.0,
                    dT_cold=0.0, dT_hot=0.0,
                    ts_cold=0.0, ts_hot=0.0,
                    air_temperature=293.15,
                    valid=True, errors=[],
                    # NEW: Enhanced anchor pixel metadata fields with defaults
                    et0_inst=0.0,
                    le_cold=0.0, h_cold=0.0, rn_cold=0.0, g_cold=0.0,
                    cold_pixel_ndvi=np.nan, cold_pixel_albedo=np.nan,
                    cold_pixel_lai=np.nan, cold_pixel_emissivity=np.nan, cold_pixel_etrf=np.nan,
                    cold_pixel_x=0, cold_pixel_y=0,
                    hot_pixel_ndvi=np.nan, hot_pixel_albedo=np.nan,
                    hot_pixel_lai=np.nan, hot_pixel_emissivity=np.nan, hot_pixel_etrf=np.nan,
                    hot_pixel_x=0, hot_pixel_y=0,
                    rn_hot=np.nan, g_hot=np.nan, h_hot=np.nan, le_hot=np.nan
                )
            
            # Write ET products
            output_files = writer.write_et_products(
                self.data, scene_id, date_str, actual_calibration
            )
            
            logger.info(f"ET products saved: {list(output_files.keys())}")

            # Create visualizations
            viz = Visualization(output_dir=output_dir)
            overview_filename = f"overview_{date_str}.png"
            viz.create_summary_figure(self.data, os.path.join(output_dir, overview_filename), calibration_result=actual_calibration, anchor_result=self._anchor_result)
            if self.data.get('ET_daily') is not None:
                et_map_filename = f"et_map_{date_str}.png"
                viz.plot_et_map(
                    self.data.get('ET_daily'),
                    self.data,
                    output_path=os.path.join(output_dir, et_map_filename)
                )

            # Save metadata
            metadata_path = os.path.join(output_dir, "processing_metadata.json")
            
            # Build quality information from scene statistics
            quality_info = {
                'scene_quality': str(self._scene_quality),
                'cloud_coverage': float(getattr(self, '_cloud_coverage', 0)) if hasattr(self, '_cloud_coverage') else None,
                'valid_pixels': None,  # Will be calculated from data
                'ndvi_range': {
                    'min': None,
                    'max': None,
                    'mean': None
                },
                'temperature_range': {
                    'min': None,
                    'max': None,
                    'mean': None
                }
            }
            
            # Calculate statistics for quality info
            ndvi = self.data.get('ndvi')
            if ndvi is not None:
                ndvi_values = ndvi.values[~np.isnan(ndvi.values)]
                if len(ndvi_values) > 0:
                    quality_info['ndvi_range']['min'] = float(np.min(ndvi_values))
                    quality_info['ndvi_range']['max'] = float(np.max(ndvi_values))
                    quality_info['ndvi_range']['mean'] = float(np.mean(ndvi_values))
            
            ts = self.data.get('lst')
            if ts is not None:
                ts_values = ts.values[~np.isnan(ts.values)]
                if len(ts_values) > 0:
                    quality_info['temperature_range']['min'] = float(np.min(ts_values))
                    quality_info['temperature_range']['max'] = float(np.max(ts_values))
                    quality_info['temperature_range']['mean'] = float(np.mean(ts_values))
            
            # Calculate valid pixel fraction
            if ndvi is not None:
                valid_pixels = np.sum(~np.isnan(ndvi.values))
                total_pixels = ndvi.size
                quality_info['valid_pixels'] = float(valid_pixels / total_pixels)
            
            writer.write_metadata_file(
                self.data, actual_calibration, scene_id, date_str,
                quality_info=quality_info
            )

            logger.info("Results saved successfully")

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def clip_outputs_to_aoi(self) -> None:
        """Clip final ET outputs to AOI boundary.
        
        DEPRECATED: This method is no longer used since ROI clipping is now performed
        during preprocessing (Step 1.5) using clip_to_roi(). All subsequent calculations
        run on the already-clipped data, so no final output clipping is needed.
        
        This method is kept for backward compatibility but is now a no-op.
        """
        import logging
        
        logger = logging.getLogger(__name__)
        
        try:
            if self.data is None:
                raise ValueError("No data available. Run the pipeline first.")
            
            logger.info("ROI clipping is now performed during preprocessing - no final output clipping needed")
            logger.info("All data is already clipped to ROI boundaries from preprocessing step")
            
        except Exception as e:
            logger.error(f"Error in clip_outputs_to_aoi: {e}")
            # Don't raise exception to avoid breaking the pipeline
            logger.warning("Continuing pipeline execution")
