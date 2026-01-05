"""METRIC ETa processing pipeline."""

from typing import Dict, Optional, Tuple, Union, Any
import numpy as np
import xarray as xr
from loguru import logger


class METRICPipeline:
    """Main processing pipeline for METRIC ETa model."""

    def __init__(self, config: Optional[Dict] = None, roi_path: Optional[str] = None):
        """Initialize METRIC processing pipeline."""
        self.config = config or {}
        self.roi_path = roi_path
        self.data = None
        self._calibration_result = None
        self._eb_manager = None  # Store EnergyBalanceManager instance for reuse
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

            # Step 2: Calculate surface properties
            logger.info("Step 2: Calculating surface properties")
            self.calculate_surface_properties()

            # Step 3: Calculate radiation balance (Rn)
            logger.info("Step 3: Calculating radiation balance")
            self.calculate_radiation_balance()

            # Step 4: Calculate soil heat flux (G) - calibration-free
            logger.info("Step 4: Calculating soil heat flux")
            self.calculate_soil_heat_flux()

            # Step 5: Apply METRIC calibration (needs Rn-G for anchor selection)
            logger.info("Step 5: Applying METRIC calibration")
            self.calibrate()

            # Step 6: Calculate final energy balance (H, LE) with calibration
            logger.info("Step 6: Calculating energy balance with calibration")
            self.calculate_energy_balance()

            # Step 7: Calculate final ET
            logger.info("Step 7: Calculating evapotranspiration")
            self.calculate_et()

            # Step 8: Save results if output directory provided
            if output_dir:
                logger.info("Step 8: Saving results")
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

            # Load ROI and clip Landsat data to ROI
            roi_path = roi_path or self.roi_path or "amirkabir.geojson"
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
        """Apply METRIC calibration."""
        from ..calibration import AnchorPixelSelector, DTCalibration
        from ..energy_balance import EnergyBalanceManager
        import logging

        logger = logging.getLogger(__name__)

        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            logger.info("Starting METRIC calibration")

            # Select anchor pixels using enhanced physical method with Rn-G optimization
            calibration_config = self.config.get('calibration', {})
            method = calibration_config.get('method', 'enhanced_physical')
            
            # Check if Rn and G are available for energy-based selection
            rn = self.data.get("R_n")
            g_flux = self.data.get("G")
            
            if rn is None or g_flux is None:
                raise ValueError("Rn and G must be computed before anchor pixel selection. "
                               "Check radiation balance and soil heat flux calculations.")
            
            if method == 'enhanced_physical':
                # Use enhanced selector which needs DataCube with Rn and G
                from ..calibration.anchor_pixels_enhanced import EnhancedAnchorPixelSelector
                anchor_selector = EnhancedAnchorPixelSelector()
                
                # Perform anchor pixel selection with energy-based optimization
                anchor_result = anchor_selector.select_anchor_pixels(
                    cube=self.data,
                    rn=rn.values,
                    g_flux=g_flux.values
                )
            else:
                # Use standard selector with arrays
                anchor_selector = AnchorPixelSelector(method=method)
                anchor_result = anchor_selector.select_anchor_pixels(
                    ts=self.data.get("lst").values,
                    ndvi=self.data.get("ndvi").values if self.data.get("ndvi") is not None else None,
                    albedo=self.data.get("albedo").values if self.data.get("albedo") is not None else None
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

            # Get ET0_daily for METRIC calibration reference
            # METRIC uses ET0_daily directly - no instantaneous ET0 estimation needed
            et0_daily_value = self._get_et0_daily()
            
            # Use improved method for air temperature
            air_temperature = self._get_air_temperature()

            calibrator = DTCalibration.create()
            calibration = calibrator.calibrate_from_anchors(self.data, anchor_result)

            # Store calibration result for later use in output writing
            self._calibration_result = calibration

            # Reuse the same energy balance manager instance for state consistency
            if self._eb_manager is None:
                self._eb_manager = EnergyBalanceManager()
                
            self._eb_manager.set_anchor_pixel_calibration(calibration.a_coefficient, calibration.b_coefficient)

            # Recalculate energy balance with calibration
            eb_results = self._eb_manager.calculate(self.data)

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

            # Calculate instantaneous ET - still need ETrF for daily scaling
            # Use ET0_daily as reference for ETrF calculation
            et0_daily_data = self.data.get("et0_fao_evapotranspiration")
            # Convert ET0_daily to ET0_inst for ETrF calculation
            et0_inst_for_etrf = et0_daily_data / 12.0  # mm/day → mm/hr
            inst_et_result = inst_et_calc.calculate(
                le=self.data.get("LE"),
                etr_inst=et0_inst_for_etrf    # Use ET0_daily in mm/hr for ETrF calculation 
            )

            # Log instantaneous ET results
            if "ET_inst" in inst_et_result:
                et_inst_values = inst_et_result["ET_inst"]
                valid_et_inst = et_inst_values[~np.isnan(et_inst_values)]
                logger.info(f"ET_inst results: {len(valid_et_inst)} valid pixels")
                logger.info(f"  Range: [{np.min(valid_et_inst):.6f}, {np.max(valid_et_inst):.6f}] mm/hr")
                logger.info(f"  Mean: {np.mean(valid_et_inst):.6f} mm/hr, Std: {np.std(valid_et_inst):.6f} mm/hr")

            if "ETrF" in inst_et_result:
                etrf_values = inst_et_result["ETrF"]
                valid_etrf = etrf_values[~np.isnan(etrf_values)]
                logger.info(f"ETrF results: {len(valid_etrf)} valid pixels")
                logger.info(f"  Range: [{np.min(valid_etrf):.6f}, {np.max(valid_etrf):.6f}]")
                logger.info(f"  Mean: {np.mean(valid_etrf):.6f}, Std: {np.std(valid_etrf):.6f}")

                # Check for unrealistic ETrF values
                if np.max(valid_etrf) > 1.5:
                    logger.warning(f"WARNING: ETrF maximum ({np.max(valid_etrf):.3f}) exceeds typical range [0, 1.2]")
                if np.min(valid_etrf) < 0:
                    logger.warning(f"WARNING: ETrF minimum ({np.min(valid_etrf):.3f}) is negative")

            # Add instantaneous ET to data
            self.data.add("ET_inst", inst_et_result["ET_inst"])
            if "ETrF" in inst_et_result:
                self.data.add("ETrF", inst_et_result["ETrF"])

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
            self.data.add("ET_confidence", quality_result["valid_fraction"])

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
            overview_filename = f"overview_{date_str}.png"
            viz.create_summary_figure(self.data, os.path.join(output_dir, overview_filename))
            if self.data.get('ET_daily') is not None:
                et_map_filename = f"et_map_{date_str}.png"
                viz.plot_et_map(
                    self.data.get('ET_daily'),
                    self.data,
                    output_path=os.path.join(output_dir, et_map_filename)
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
