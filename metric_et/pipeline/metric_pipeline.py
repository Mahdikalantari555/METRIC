"""METRIC ETa processing pipeline."""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import xarray as xr
from loguru import logger


class METRICPipeline:
    """Main processing pipeline for METRIC ETa model."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize METRIC processing pipeline."""
        self.config = config or {}
        self.data = None
        logger.info("Initialized METRICPipeline")
    
    def run(
        self, landsat_dir: str, meteo_data: Dict,
        output_dir: Optional[str] = None
    ) -> Dict[str, xr.DataArray]:
        """Run complete METRIC ETa processing pipeline."""
        import logging

        logger = logging.getLogger(__name__)

        try:
            logger.info("Starting METRIC ETa processing pipeline")

            # Step 1: Load and preprocess data
            logger.info("Step 1: Loading and preprocessing data")
            self.load_data(landsat_dir, meteo_data)
            self.preprocess()

            # Step 2: Calculate surface properties
            logger.info("Step 2: Calculating surface properties")
            self.calculate_surface_properties()

            # Step 3: Calculate radiation balance
            logger.info("Step 3: Calculating radiation balance")
            self.calculate_radiation_balance()

            # Step 4: Calculate initial energy balance (before calibration)
            logger.info("Step 4: Calculating initial energy balance")
            self.calculate_energy_balance()

            # Step 5: Apply METRIC calibration
            logger.info("Step 5: Applying METRIC calibration")
            self.calibrate()

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
    
    def load_data(self, landsat_dir: str, meteo_data: Dict, dem_path: Optional[str] = None) -> None:
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

            # Load ROI and clip Landsat data to ROI
            roi_path = "amirkabir.geojson"
            import geopandas as gpd
            import rioxarray
            roi_gdf = gpd.read_file(roi_path)
            # Reproject ROI to match raster CRS
            sample_band = next(iter(landsat_cube.data.values()))
            roi_gdf = roi_gdf.to_crs(sample_band.rio.crs)
            roi_geom = roi_gdf.geometry.iloc[0]  # assuming single feature
            logger.info("Loaded and reprojected ROI from amirkabir.geojson")

            # Clip all Landsat bands to ROI
            for band_name in landsat_cube.bands():
                band_data = landsat_cube.get(band_name)
                clipped = band_data.rio.clip([roi_geom], crs=band_data.rio.crs, drop=False)
                landsat_cube.add(band_name, clipped)

            # Update extent and transform based on clipped data
            sample_clipped = landsat_cube.get(next(iter(landsat_cube.bands())))
            landsat_cube.extent = sample_clipped.rio.bounds()
            landsat_cube.transform = sample_clipped.rio.transform()
            logger.info("Clipped Landsat data to ROI")

            # Copy Landsat data to main cube
            for band_name in landsat_cube.bands():
                self.data.add(band_name, landsat_cube.get(band_name))

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

            # Get actual processed extent (clipped to ROI) and convert to lat/lon
            projected_bounds = self.data.extent  # (min_x, min_y, max_x, max_y) in projected CRS
            from pyproj import Transformer
            # Create transformer from data CRS to WGS84
            transformer = Transformer.from_crs(self.data.crs, "EPSG:4326", always_xy=True)
            # Transform corners
            min_lon, min_lat = transformer.transform(projected_bounds[0], projected_bounds[1])
            max_lon, max_lat = transformer.transform(projected_bounds[2], projected_bounds[3])
            actual_extent = (min_lon, min_lat, max_lon, max_lat)

            # Initialize dynamic weather fetcher
            from ..io.dynamic_weather_fetcher import DynamicWeatherFetcher
            weather_fetcher = DynamicWeatherFetcher()

            try:
                # Fetch spatially varying weather data using actual extent
                weather_arrays = weather_fetcher.fetch_weather_for_scene(
                    landsat_dir, target_coords, actual_extent
                )

                # Convert temperature from Celsius to Kelvin
                if "temperature_2m" in weather_arrays:
                    weather_arrays["temperature_2m"] = weather_arrays["temperature_2m"] + 273.15

                # Add weather data to cube
                for var_name, array in weather_arrays.items():
                    self.data.add(var_name, array)

                logger.info(f"Spatially varying weather data loaded for {len(weather_arrays)} variables using ROI extent")

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

            # Apply cloud masking
            logger.info("Applying cloud masking")
            from ..preprocess.cloud_mask import CloudMasker
            
            # Use high confidence threshold to be conservative
            masker = CloudMasker(
                cloud_confidence_threshold=CloudMasker.CONFIDENCE_HIGH,
                dilate_pixels=3,
                include_snow=False,
                include_water=True
            )
            
            # Create cloud mask from QA pixel band
            qa_pixel = self.data.get('qa_pixel')
            if qa_pixel is not None:
                cloud_mask = masker.create_mask(qa_pixel)
                logger.info(f"Cloud mask created: {np.sum(cloud_mask)} clear pixels out of {cloud_mask.size}")
                
                # Apply mask to all bands
                masked_cube = masker.apply_mask(self.data, cloud_mask, fill_value=np.nan)
                
                # Replace original data with masked data
                self.data = masked_cube
                logger.info("Cloud masking applied successfully")
            else:
                logger.warning("No QA pixel band found, skipping cloud masking")

            # TODO: Add additional preprocessing steps as needed
            # - Resampling
            # - Atmospheric correction
            # - Geometric correction

            logger.info("Preprocessing completed")

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise
    
    def calculate_surface_properties(self) -> None:
        """Calculate surface properties."""
        from ..surface import VegetationIndices, AlbedoCalculator, EmissivityCalculator, RoughnessCalculator
        import logging

        logger = logging.getLogger(__name__)

        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            logger.info("Calculating surface properties")

            # Calculate vegetation indices (NDVI, EVI, LAI, SAVI, FVC)
            veg_indices = VegetationIndices()
            veg_indices.compute(self.data)

            # Calculate broadband albedo
            albedo_calc = AlbedoCalculator()
            albedo_calc.compute(self.data)

            # Calculate emissivity
            emissivity_calc = EmissivityCalculator()
            emissivity_calc.compute(self.data)

            # Calculate roughness parameters
            roughness_calc = RoughnessCalculator()
            roughness_calc.compute(self.data)

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
    
    def calculate_energy_balance(self) -> None:
        """Calculate energy balance components."""
        from ..energy_balance import EnergyBalanceManager
        import logging

        logger = logging.getLogger(__name__)

        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            logger.info("Calculating energy balance")

            # Initialize energy balance manager
            eb_manager = EnergyBalanceManager()

            # Calculate energy balance components
            # Note: This will be updated after calibration with anchor pixels
            eb_results = eb_manager.calculate(self.data)

            logger.info("Energy balance calculation completed")

        except Exception as e:
            logger.error(f"Error calculating energy balance: {e}")
            raise
    
    def calibrate(self) -> None:
        """Apply METRIC calibration."""
        from ..calibration import AnchorPixelSelector, DTCalibration
        from ..energy_balance import EnergyBalanceManager
        import logging

        logger = logging.getLogger(__name__)

        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            logger.info("Starting METRIC calibration")

            # Select anchor pixels using Triangle method
            anchor_selector = AnchorPixelSelector(method="triangle")
            anchor_result = anchor_selector.select_anchor_pixels(
                ts=self.data.get("lwir11"),
                ndvi=self.data.get("ndvi"),
                albedo=self.data.get("albedo")
            )

            # Debug logging for anchor result
            logger.info(f"anchor_result type: {type(anchor_result)}")
            logger.info(f"cold_pixel type: {type(anchor_result.cold_pixel)}")
            logger.info(f"cold_pixel attributes: {dir(anchor_result.cold_pixel)}")
            if hasattr(anchor_result.cold_pixel, 'temperature'):
                logger.info(f"cold_pixel.temperature: {anchor_result.cold_pixel.temperature}")
            else:
                logger.info("cold_pixel has no 'temperature' attribute")
            if hasattr(anchor_result.cold_pixel, 'ts'):
                logger.info(f"cold_pixel.ts: {anchor_result.cold_pixel.ts}")
            else:
                logger.info("cold_pixel has no 'ts' attribute")

            # Calibrate dT relationship
            # Convert daily ET0 to instantaneous ET0 at overpass time using METRIC energy ratios
            et0_daily = self.data.get("et0_fao_evapotranspiration")
            rs_inst = self.data.get("Rs_down")  # Instantaneous shortwave radiation from radiation module
            rs_daily = self.data.get("shortwave_radiation_sum")  # Daily shortwave radiation sum from Open-Meteo (MJ/m²/day)

            if et0_daily is not None and rs_inst is not None and rs_daily is not None:
                # METRIC approach: Use energy ratios directly
                # Convert instantaneous radiation from W/m² to MJ/m²/hr: Rs_inst_MJ = Rs_inst_W * 0.0036
                if hasattr(rs_inst, 'values'):
                    rs_inst_scalar = float(np.nanmean(rs_inst.values))
                else:
                    rs_inst_scalar = float(np.nanmean(rs_inst))
                
                # Get daily radiation sum
                if hasattr(rs_daily, 'values'):
                    rs_daily_scalar = float(np.nanmean(rs_daily.values))
                else:
                    rs_daily_scalar = float(np.nanmean(rs_daily))
                
                # Convert instantaneous radiation to energy units (MJ/m²/hr)
                rs_inst_mj = rs_inst_scalar * 0.0036
                
                # Calculate ET0_inst using METRIC energy ratio: ET0_inst = ET0_daily * (Rs_inst_MJ / Rs_daily_MJ)
                # NO time division - the ratio already accounts for temporal scaling
                rs_ratio = rs_inst_mj / rs_daily_scalar
                et0_inst = et0_daily * rs_ratio
                
                logger.info(f"Rs_inst: {rs_inst_scalar:.2f} W/m² = {rs_inst_mj:.4f} MJ/m²/hr, Rs_daily: {rs_daily_scalar:.2f} MJ/m²/day, Ratio: {rs_ratio:.4f}")
            elif et0_daily is not None:
                # METRIC-safe fallback: Use typical overpass fraction when radiation data missing
                # Typical Landsat overpass represents ~15% of daily ET0
                et0_inst = et0_daily * 0.15
                logger.warning("Using METRIC fallback: ET0_inst = ET0_daily × 0.15 (radiation data missing)")
            else:
                et0_inst = 0.65  # Default value
                logger.warning("Using default ET0_inst = 0.65 mm/hr")

            # Extract scalar air temperature from weather data
            temp_2m = self.data.get("temperature_2m")
            if temp_2m is not None:
                # Get the scalar value (temperature is spatially uniform)
                if hasattr(temp_2m, 'values'):
                    air_temperature = float(np.nanmean(temp_2m.values))
                else:
                    air_temperature = float(np.nanmean(temp_2m))
            else:
                air_temperature = 293.15  # Default 20°C in Kelvin

            calibrator = DTCalibration(et0_inst=et0_inst)
            calibration = calibrator.calibrate_from_anchors(self.data, anchor_result)

            # Store calibration result for later use in output writing
            self._calibration_result = calibration

            # Update energy balance manager with calibration coefficients
            eb_manager = EnergyBalanceManager()
            eb_manager.set_anchor_pixel_calibration(calibration.a_coefficient, calibration.b_coefficient)

            # Recalculate energy balance with calibration
            eb_results = eb_manager.calculate(self.data)

            logger.info("METRIC calibration completed")

        except Exception as e:
            logger.error(f"Error in METRIC calibration: {e}")
            raise
    
    def calculate_et(self) -> None:
        """Calculate evapotranspiration."""
        from ..et import InstantaneousET, DailyET, ETQuality
        import logging

        logger = logging.getLogger(__name__)

        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            logger.info("Calculating evapotranspiration")

            # Calculate instantaneous ET
            inst_et_calc = InstantaneousET()

            # DEBUG: Log LE spatial variation before ET calculation
            le_data = self.data.get("LE")
            if le_data is not None:
                if hasattr(le_data, 'values'):
                    le_values = le_data.values
                else:
                    le_values = le_data
                valid_le = le_values[~np.isnan(le_values)]
                if len(valid_le) > 0:
                    logger.info(f"DEBUG ET_calc - LE spatial stats: Min={np.min(valid_le):.2f}, Max={np.max(valid_le):.2f}, "
                               f"Mean={np.mean(valid_le):.2f}, Std={np.std(valid_le):.2f}, Unique={len(np.unique(valid_le))}")

            # Use the same instantaneous ETr calculation as in calibration
            # Convert daily ET0 to instantaneous ETr at overpass time using METRIC energy ratios
            et0_daily = self.data.get("et0_fao_evapotranspiration")
            rs_inst = self.data.get("Rs_down")  # Instantaneous shortwave radiation from radiation module
            rs_daily = self.data.get("shortwave_radiation_sum")  # Daily shortwave radiation sum from Open-Meteo (MJ/m²/day)

            if et0_daily is not None and rs_inst is not None and rs_daily is not None:
                # METRIC approach: Use energy ratios directly
                # Convert instantaneous radiation from W/m² to MJ/m²/hr: Rs_inst_MJ = Rs_inst_W * 0.0036
                if hasattr(rs_inst, 'values'):
                    rs_inst_scalar = float(np.nanmean(rs_inst.values))
                else:
                    rs_inst_scalar = float(np.nanmean(rs_inst))
                
                # Get daily radiation sum
                if hasattr(rs_daily, 'values'):
                    rs_daily_scalar = float(np.nanmean(rs_daily.values))
                else:
                    rs_daily_scalar = float(np.nanmean(rs_daily))
                
                # Convert instantaneous radiation to energy units (MJ/m²/hr)
                rs_inst_mj = rs_inst_scalar * 0.0036
                
                # Calculate ET0_inst using METRIC energy ratio: ET0_inst = ET0_daily * (Rs_inst_MJ / Rs_daily_MJ)
                # NO time division - the ratio already accounts for temporal scaling
                rs_ratio = rs_inst_mj / rs_daily_scalar
                et0_inst = et0_daily * rs_ratio
                etr_inst = et0_inst * 1.15  # Convert to alfalfa reference ET
                
                # Get scalar values for logging
                et0_inst_scalar = float(np.nanmean(et0_inst.values)) if hasattr(et0_inst, 'values') else float(np.nanmean(et0_inst))
                etr_inst_scalar = float(np.nanmean(etr_inst.values)) if hasattr(etr_inst, 'values') else float(np.nanmean(etr_inst))
                logger.info(f"DEBUG ET_calc - ETr_inst calculated with METRIC energy ratio: {etr_inst_scalar:.6f} mm/hr (ET0_inst={et0_inst_scalar:.6f}, Rs_ratio={rs_ratio:.4f})")
            elif et0_daily is not None:
                # METRIC-safe fallback: Use typical overpass fraction when radiation data missing
                # Typical Landsat overpass represents ~15% of daily ET0
                et0_inst = et0_daily * 0.15
                etr_inst = et0_inst * 1.15
                
                # Get scalar values for logging
                et0_daily_scalar = float(np.nanmean(et0_daily.values)) if hasattr(et0_daily, 'values') else float(np.nanmean(et0_daily))
                etr_inst_scalar = float(np.nanmean(etr_inst.values)) if hasattr(etr_inst, 'values') else float(np.nanmean(etr_inst))
                logger.info(f"DEBUG ET_calc - ETr_inst calculated with METRIC fallback: {etr_inst_scalar:.6f} mm/hr (ET0_daily={et0_daily_scalar:.6f} × 0.15)")
            else:
                etr_inst = None
                logger.warning("DEBUG ET_calc - No ET0 data available, ETr_inst = None")

            inst_et_result = inst_et_calc.calculate(
                le=self.data.get("LE"),
                etr_inst=etr_inst,
                temperature_k=self.data.get("lwir11")
            )

            # DEBUG: Log ET_inst and ETrF spatial variation after calculation
            if "ET_inst" in inst_et_result:
                et_inst_values = inst_et_result["ET_inst"]
                valid_et_inst = et_inst_values[~np.isnan(et_inst_values)]
                if len(valid_et_inst) > 0:
                    logger.info(f"DEBUG ET_calc - ET_inst spatial stats: Min={np.min(valid_et_inst):.6f}, Max={np.max(valid_et_inst):.6f}, "
                               f"Mean={np.mean(valid_et_inst):.6f}, Std={np.std(valid_et_inst):.6f}, Unique={len(np.unique(valid_et_inst))}")

            if "ETrF" in inst_et_result:
                etrf_values = inst_et_result["ETrF"]
                valid_etrf = etrf_values[~np.isnan(etrf_values)]
                if len(valid_etrf) > 0:
                    logger.info(f"DEBUG ET_calc - ETrF spatial stats: Min={np.min(valid_etrf):.6f}, Max={np.max(valid_etrf):.6f}, "
                               f"Mean={np.mean(valid_etrf):.6f}, Std={np.std(valid_etrf):.6f}, Unique={len(np.unique(valid_etrf))}")
                    if np.max(valid_etrf) - np.mean(valid_etrf) < 0.01:
                        logger.warning("DEBUG ET_calc - CRITICAL: ETrF has minimal spatial variation!")

            # Add instantaneous ET to data
            self.data.add("ET_inst", inst_et_result["ET_inst"])
            if "ETrF" in inst_et_result:
                self.data.add("ETrF", inst_et_result["ETrF"])

            # Calculate daily ET
            daily_et_calc = DailyET()
            # Convert daily ET0 to ETr (alfalfa reference)
            et0_daily = self.data.get("et0_fao_evapotranspiration")
            if et0_daily is not None:
                etr_daily = et0_daily * 1.15  # Convert ET0 to ETr
                # Get scalar values for logging
                et0_daily_scalar = float(np.nanmean(et0_daily.values)) if hasattr(et0_daily, 'values') else float(np.nanmean(et0_daily))
                etr_daily_scalar = float(np.nanmean(etr_daily.values)) if hasattr(etr_daily, 'values') else float(np.nanmean(etr_daily))
                logger.info(f"DEBUG ET_calc - ETr_daily calculated as spatially uniform: {etr_daily_scalar:.6f} mm/day (ET0_daily={et0_daily_scalar:.6f})")
            else:
                etr_daily = None
                logger.warning("DEBUG ET_calc - No ET0 data available, ETr_daily = None")

            # DEBUG: Log ETrF before daily calculation
            etrf_data = self.data.get("ETrF")
            if etrf_data is not None:
                if hasattr(etrf_data, 'values'):
                    etrf_values = etrf_data.values
                else:
                    etrf_values = etrf_data
                valid_etrf = etrf_values[~np.isnan(etrf_values)]
                if len(valid_etrf) > 0:
                    logger.info(f"DEBUG ET_calc - ETrF for daily calc: Min={np.min(valid_etrf):.6f}, Max={np.max(valid_etrf):.6f}, "
                               f"Mean={np.mean(valid_etrf):.6f}, Std={np.std(valid_etrf):.6f}, Unique={len(np.unique(valid_etrf))}")

            daily_et_result = daily_et_calc.calculate(
                etrf=self.data.get("ETrF"),
                etr_daily=etr_daily
            )

            # DEBUG: Log final ET_daily spatial variation
            if "ET_daily" in daily_et_result:
                et_daily_values = daily_et_result["ET_daily"]
                valid_et_daily = et_daily_values[~np.isnan(et_daily_values)]
                if len(valid_et_daily) > 0:
                    logger.info(f"DEBUG ET_calc - ET_daily final stats: Min={np.min(valid_et_daily):.6f}, Max={np.max(valid_et_daily):.6f}, "
                               f"Mean={np.mean(valid_et_daily):.6f}, Std={np.std(valid_et_daily):.6f}, Unique={len(np.unique(valid_et_daily))}")
                    max_mean_diff = abs(np.max(valid_et_daily) - np.mean(valid_et_daily))
                    if max_mean_diff < 0.01:
                        logger.error(f"DEBUG ET_calc - CRITICAL: ET_daily has minimal spatial variation (Max-Mean = {max_mean_diff:.6f})")
                    else:
                        logger.info(f"DEBUG ET_calc - OK: ET_daily has spatial variation (Max-Mean = {max_mean_diff:.6f})")

            # Add daily ET to data
            self.data.add("ET_daily", daily_et_result["ET_daily"])

            # Calculate ET quality assessment
            quality_calc = ETQuality()
            quality_result = quality_calc.assess(
                et_daily=self.data.get("ET_daily"),
                etrf=self.data.get("ETrF")
            )

            # Add quality metrics to data
            self.data.add("ET_quality_class", quality_result["quality_class"])
            # Note: confidence_score is not in the assess() result, using valid_fraction instead
            self.data.add("ET_confidence", quality_result["valid_fraction"])

            logger.info("ET calculation completed")

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
        surface_keys = ['ndvi', 'albedo', 'emissivity', 'lai', 'z0m', 'd']
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
                    etr_inst=0.0,
                    valid=True, errors=[]
                )
            
            # Write ET products
            output_files = writer.write_et_products(
                self.data, scene_id, date_str, actual_calibration
            )
            
            logger.info(f"ET products saved: {list(output_files.keys())}")

            # Create visualizations
            viz = Visualization(output_dir=output_dir)
            viz.create_summary_figure(self.data, os.path.join(output_dir, "overview.png"))
            if self.data.get('ET_daily') is not None:
                viz.plot_et_map(
                    self.data.get('ET_daily'),
                    self.data,
                    output_path=os.path.join(output_dir, "et_map.png")
                )

            # Save metadata
            metadata_path = os.path.join(output_dir, "processing_metadata.json")
            writer.write_metadata_file(
                self.data, actual_calibration, scene_id, date_str
            )

            logger.info("Results saved successfully")

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
