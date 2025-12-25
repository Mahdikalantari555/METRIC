"""Anchor pixel selection module for METRIC calibration."""

from typing import Optional, Dict, Any
import numpy as np


class AnchorPixel:
    """Represents a hot or cold anchor pixel selection."""

    def __init__(
        self,
        x: int,
        y: int,
        temperature: Optional[float] = None,
        ts: Optional[float] = None,
        ndvi: Optional[float] = None,
        albedo: Optional[float] = None
    ):
        self.x = x
        self.y = y
        self.temperature = temperature
        self.ts = ts if ts is not None else temperature
        self.ndvi = ndvi
        self.albedo = albedo

    def to_dict(self) -> Dict[str, Any]:
        return {
            'x': self.x,
            'y': self.y,
            'temperature': self.temperature,
            'ts': self.ts,
            'ndvi': self.ndvi,
            'albedo': self.albedo
        }


class AnchorPixelsResult:
    """Result of anchor pixel selection."""

    def __init__(
        self,
        cold_pixel: AnchorPixel,
        hot_pixel: AnchorPixel,
        method: str,
        confidence: float
    ):
        self.cold_pixel = cold_pixel
        self.hot_pixel = hot_pixel
        self.method = method
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cold_pixel': self.cold_pixel.to_dict(),
            'hot_pixel': self.hot_pixel.to_dict(),
            'method': self.method,
            'confidence': self.confidence
        }


class AnchorPixelSelector:
    """
    Select hot and cold anchor pixels for METRIC calibration.
    """

    def __init__(
        self,
        min_temperature: float = 200.0,
        max_temperature: float = 400.0,
        method: str = "max_temp"
    ):
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.method = method
    
    def select(self, cube):
        """
        Select anchor pixels from DataCube.

        Args:
            cube: DataCube with required bands

        Returns:
            AnchorPixelsResult
        """
        from ..core.datacube import DataCube

        # Get required data
        ts = cube.get("lwir11")  # Surface temperature
        ndvi = cube.get("ndvi")
        albedo = cube.get("albedo")
        qa_pixel = cube.get("qa_pixel")  # QA pixel for debugging

        if ts is None:
            raise ValueError("Surface temperature (lwir11) not found in DataCube")

        # Call the selection method
        # Note: Cloud masking should already be applied to the data cube,
        # so we don't need a separate cloud_mask parameter
        return self.select_anchor_pixels(
            ts=ts.values,
            ndvi=ndvi.values if ndvi is not None else None,
            albedo=albedo.values if albedo is not None else None,
            qa_pixel=qa_pixel.values if qa_pixel is not None else None
        )

    def select_anchor_pixels(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray] = None,
        albedo: Optional[np.ndarray] = None,
        qa_pixel: Optional[np.ndarray] = None,
        **kwargs
    ) -> AnchorPixelsResult:
        """
        Select hot and cold anchor pixels using the specified method.

        Args:
            ts: Surface temperature array (Kelvin)
            ndvi: NDVI array (optional)
            albedo: Albedo array (optional)

        Returns:
            AnchorPixelsResult with selected anchor pixels
        """
        import logging
        from scipy.spatial import ConvexHull

        logger = logging.getLogger(__name__)

        # Convert xarray DataArrays to numpy arrays if needed
        if hasattr(ts, 'values'):
            ts = ts.values
        if ndvi is not None and hasattr(ndvi, 'values'):
            ndvi = ndvi.values
        if albedo is not None and hasattr(albedo, 'values'):
            albedo = albedo.values

        logger.info(f"Anchor selection: ts shape {ts.shape}, min {np.nanmin(ts):.2f}, max {np.nanmax(ts):.2f}")
        if ndvi is not None:
            logger.info(f"NDVI shape {ndvi.shape}, min {np.nanmin(ndvi):.3f}, max {np.nanmax(ndvi):.3f}")
        if albedo is not None:
            logger.info(f"Albedo shape {albedo.shape}, min {np.nanmin(albedo):.3f}, max {np.nanmax(albedo):.3f}")

        # Create valid mask - cloud masking should already be applied (NaN values)
        valid = np.ones(ts.shape, dtype=bool)
        valid &= ~np.isnan(ts)
        valid &= (ts > 0)  # Exclude fill values
        valid &= (ts >= self.min_temperature) & (ts <= self.max_temperature)

        # Also exclude pixels where NDVI is NaN (if NDVI is available)
        if ndvi is not None:
            valid &= ~np.isnan(ndvi)

        # METRIC Cold Pixel Constraints: Apply NDVI and albedo filters for cold pixel selection
        cold_valid = valid.copy()
        if ndvi is not None:
            cold_valid &= (ndvi > 0.70)  # NDVI > 0.70 for well-watered vegetation
        if albedo is not None:
            cold_valid &= (albedo < 0.20)  # Albedo < 0.20 for well-watered vegetation
            cold_valid &= ~np.isnan(albedo)  # Also exclude NaN albedo

        logger.info(f"Valid pixels: {np.sum(valid)} out of {valid.size}")

        # Debug: Check masking statistics
        total_pixels = valid.size
        nan_ts = np.sum(np.isnan(ts))
        nan_ndvi = np.sum(np.isnan(ndvi)) if ndvi is not None else 0
        nan_albedo = np.sum(np.isnan(albedo)) if albedo is not None else 0
        logger.info(f"NaN statistics: ts={nan_ts}/{total_pixels}, ndvi={nan_ndvi}/{total_pixels}, albedo={nan_albedo}/{total_pixels}")

        # Check temperature range filtering
        temp_valid = (ts >= self.min_temperature) & (ts <= self.max_temperature) & ~np.isnan(ts)
        logger.info(f"Temperature range valid: {np.sum(temp_valid)}/{total_pixels} (T >= {self.min_temperature}K and T <= {self.max_temperature}K)")

        if not np.any(valid):
            raise ValueError("No valid pixels found for anchor selection")

        # Dispatch to appropriate method
        method = self.method.lower() if self.method else "max_temp"
        logger.info(f"QA pixel available: {qa_pixel is not None}")

        if method == "max_temp":
            result = self._select_max_temp(ts, ndvi, albedo, valid, qa_pixel)
        elif method == "triangle":
            result = self._select_triangle(ts, ndvi, albedo, valid, ConvexHull, qa_pixel)
        elif method == "percentile":
            result = self._select_percentile(ts, ndvi, albedo, valid, qa_pixel)
        else:
            logger.warning(f"Unknown method '{method}', falling back to 'max_temp'")
            result = self._select_max_temp(ts, ndvi, albedo, valid, qa_pixel)

        return result

    def _select_max_temp(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        valid: np.ndarray,
        qa_pixel: Optional[np.ndarray] = None
    ) -> AnchorPixelsResult:
        """Select anchor pixels using maximum temperature method."""
        import logging
        logger = logging.getLogger(__name__)

        # Use METRIC-constrained mask for cold pixel, general valid mask for hot pixel
        cold_valid = valid.copy()
        if ndvi is not None:
            cold_valid &= (ndvi > 0.70)  # NDVI > 0.70 for well-watered vegetation
        if albedo is not None:
            cold_valid &= (albedo < 0.20)  # Albedo < 0.20 for well-watered vegetation
            cold_valid &= ~np.isnan(albedo)  # Also exclude NaN albedo

        # Check if we have enough METRIC-constrained cold pixels
        if not np.any(cold_valid):
            logger.warning("No pixels meet METRIC cold pixel constraints (NDVI > 0.70, Albedo < 0.20). Falling back to temperature-only selection.")
            cold_valid = valid.copy()  # Fallback to original valid mask

        valid_ts = ts[valid]
        cold_valid_ts = ts[cold_valid]
        valid_coords = np.where(valid)
        cold_valid_coords = np.where(cold_valid)

        # Cold pixel (minimum temperature from METRIC-constrained pixels)
        cold_idx = np.argmin(cold_valid_ts)
        cold_x, cold_y = cold_valid_coords[1][cold_idx], cold_valid_coords[0][cold_idx]
        cold_pixel = AnchorPixel(
            x=cold_x,
            y=cold_y,
            temperature=cold_valid_ts[cold_idx],
            ts=cold_valid_ts[cold_idx],
            ndvi=ndvi[cold_valid][cold_idx] if ndvi is not None else None,
            albedo=albedo[cold_valid][cold_idx] if albedo is not None else None
        )

        # METRIC Cold Pixel Validation Logging
        logger.info("=== COLD PIXEL VALIDATION CHECKS (METRIC-CONSTRAINED SELECTION) ===")
        cold_ndvi = ndvi[cold_valid][cold_idx] if ndvi is not None else None
        cold_albedo = albedo[cold_valid][cold_idx] if albedo is not None else None

        # Check NDVI constraint (should be > 0.70 for well-watered vegetation)
        if cold_ndvi is not None:
            ndvi_valid = cold_ndvi > 0.70
            logger.info(f"Cold pixel NDVI: {cold_ndvi:.3f} (METRIC requirement: > 0.70) - {'PASS' if ndvi_valid else 'FAIL'}")
        else:
            logger.warning("Cold pixel NDVI: Not available for validation")

        # Check albedo constraint (should be < 0.20 for well-watered vegetation)
        if cold_albedo is not None:
            albedo_valid = cold_albedo < 0.20
            logger.info(f"Cold pixel Albedo: {cold_albedo:.3f} (METRIC requirement: < 0.20) - {'PASS' if albedo_valid else 'FAIL'}")
        else:
            logger.warning("Cold pixel Albedo: Not available for validation")

        # Check LST percentile (should be in lowest 5-10%)
        cold_temp_percentile = np.sum(valid_ts <= valid_ts[cold_idx]) / len(valid_ts) * 100
        lst_percentile_valid = cold_temp_percentile <= 10  # Allow up to 10th percentile
        logger.info(f"Cold pixel LST percentile: {cold_temp_percentile:.1f}% (METRIC requirement: ≤ 10%) - {'PASS' if lst_percentile_valid else 'FAIL'}")

        # Check for over-constrained selection
        constraints_met = 0
        total_constraints = 0

        if cold_ndvi is not None:
            total_constraints += 1
            if cold_ndvi > 0.70:
                constraints_met += 1

        if cold_albedo is not None:
            total_constraints += 1
            if cold_albedo < 0.20:
                constraints_met += 1

        total_constraints += 1  # LST percentile
        if lst_percentile_valid:
            constraints_met += 1

        constraint_satisfaction = constraints_met / total_constraints if total_constraints > 0 else 0
        logger.info(f"Cold pixel constraint satisfaction: {constraints_met}/{total_constraints} ({constraint_satisfaction:.1%})")

        if constraint_satisfaction < 0.5:
            logger.warning("WARNING: Cold pixel meets less than 50% of METRIC constraints - may be over-constrained or mis-located")

        # Debug: Check if selected pixels are in masked areas
        logger.info(f"Cold pixel selected at (x={cold_x}, y={cold_y}), Ts={valid_ts[cold_idx]:.2f}K")
        if ndvi is not None:
            logger.info(f"  Cold pixel NDVI: {ndvi[valid][cold_idx]:.3f}")
        if albedo is not None:
            logger.info(f"  Cold pixel albedo: {albedo[valid][cold_idx]:.3f}")
        # Check original arrays for NaN at this location
        if np.isnan(ts[cold_y, cold_x]):
            logger.warning(f"  WARNING: Cold pixel at ({cold_x}, {cold_y}) is NaN in original ts array!")
        if ndvi is not None and np.isnan(ndvi[cold_y, cold_x]):
            logger.warning(f"  WARNING: Cold pixel at ({cold_x}, {cold_y}) is NaN in original ndvi array!")
        if albedo is not None and np.isnan(albedo[cold_y, cold_x]):
            logger.warning(f"  WARNING: Cold pixel at ({cold_x}, {cold_y}) is NaN in original albedo array!")
        # Check QA pixel value
        if qa_pixel is not None:
            qa_value = qa_pixel[cold_y, cold_x]
            logger.info(f"  Cold pixel QA value: {qa_value}")
            # Check if it would be masked by cloud detection
            from ..preprocess.cloud_mask import CloudMasker
            masker = CloudMasker()
            # Simple check: if QA indicates cloud
            cloud_bit = (qa_value >> masker.QA_CLOUD_BIT) & 1
            shadow_bit = (qa_value >> masker.QA_CLOUD_SHADOW_BIT) & 1
            dilated_bit = (qa_value >> masker.QA_DILATED_CLOUD_BIT) & 1
            if cloud_bit or shadow_bit or dilated_bit:
                logger.warning(f"  WARNING: Cold pixel at ({cold_x}, {cold_y}) has cloud/shadow flags in QA!")

        # Hot pixel (maximum temperature)
        hot_idx = np.argmax(valid_ts)
        hot_x, hot_y = valid_coords[1][hot_idx], valid_coords[0][hot_idx]
        hot_pixel = AnchorPixel(
            x=hot_x,
            y=hot_y,
            temperature=valid_ts[hot_idx],
            ts=valid_ts[hot_idx],
            ndvi=ndvi[valid][hot_idx] if ndvi is not None else None,
            albedo=albedo[valid][hot_idx] if albedo is not None else None
        )

        # Debug: Check if selected pixels are in masked areas
        logger.info(f"Hot pixel selected at (x={hot_x}, y={hot_y}), Ts={valid_ts[hot_idx]:.2f}K")
        if ndvi is not None:
            logger.info(f"  Hot pixel NDVI: {ndvi[valid][hot_idx]:.3f}")
        if albedo is not None:
            logger.info(f"  Hot pixel albedo: {albedo[valid][hot_idx]:.3f}")
        # Check original arrays for NaN at this location
        if np.isnan(ts[hot_y, hot_x]):
            logger.warning(f"  WARNING: Hot pixel at ({hot_x}, {hot_y}) is NaN in original ts array!")
        if ndvi is not None and np.isnan(ndvi[hot_y, hot_x]):
            logger.warning(f"  WARNING: Hot pixel at ({hot_x}, {hot_y}) is NaN in original ndvi array!")
        if albedo is not None and np.isnan(albedo[hot_y, hot_x]):
            logger.warning(f"  WARNING: Hot pixel at ({hot_x}, {hot_y}) is NaN in original albedo array!")
        # Check QA pixel value
        if qa_pixel is not None:
            qa_value = qa_pixel[hot_y, hot_x]
            logger.info(f"  Hot pixel QA value: {qa_value}")
            # Check if it would be masked by cloud detection
            from ..preprocess.cloud_mask import CloudMasker
            masker = CloudMasker()
            # Simple check: if QA indicates cloud
            cloud_bit = (qa_value >> masker.QA_CLOUD_BIT) & 1
            shadow_bit = (qa_value >> masker.QA_CLOUD_SHADOW_BIT) & 1
            dilated_bit = (qa_value >> masker.QA_DILATED_CLOUD_BIT) & 1
            if cloud_bit or shadow_bit or dilated_bit:
                logger.warning(f"  WARNING: Hot pixel at ({hot_x}, {hot_y}) has cloud/shadow flags in QA!")

        confidence = min(1.0, (hot_pixel.temperature - cold_pixel.temperature) / 20.0)
        logger.info(f"Max_temp method: cold T={cold_pixel.temperature:.2f}K, hot T={hot_pixel.temperature:.2f}K, confidence={confidence:.3f}")

        return AnchorPixelsResult(
            cold_pixel=cold_pixel,
            hot_pixel=hot_pixel,
            method="max_temp",
            confidence=confidence
        )

    def _select_triangle(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        valid: np.ndarray,
        ConvexHull,
        qa_pixel: Optional[np.ndarray] = None
    ) -> AnchorPixelsResult:
        """
        Select anchor pixels using the Triangle method (convex hull).

        This method finds the upper envelope (dry edge) of the Ts-NDVI scatterplot
        using the convex hull, then selects the hot pixel from this envelope.
        This provides better representation of the surface-air temperature gradient
        in heterogeneous landscapes.
        """
        import logging
        logger = logging.getLogger(__name__)

        if ndvi is None:
            logger.warning("NDVI not available, falling back to max_temp method")
            return self._select_max_temp(ts, ndvi, albedo, valid)

        valid_mask = valid & ~np.isnan(ndvi)
        if not np.any(valid_mask):
            logger.warning("No valid pixels with NDVI, falling back to max_temp method")
            return self._select_max_temp(ts, ndvi, albedo, valid)

        # Apply METRIC cold pixel constraints for Percentile method
        cold_valid_mask = valid_mask.copy()
        if albedo is not None:
            cold_valid_mask &= (albedo < 0.20)  # Albedo < 0.20 for well-watered vegetation
            cold_valid_mask &= ~np.isnan(albedo)  # Also exclude NaN albedo
        cold_valid_mask &= (ndvi > 0.70)  # NDVI > 0.70 for well-watered vegetation

        if not np.any(cold_valid_mask):
            logger.warning("No pixels meet METRIC cold pixel constraints in Percentile method. Using NDVI-only constraint.")
            cold_valid_mask = valid_mask & (ndvi > 0.70)  # Fallback to NDVI only
            if not np.any(cold_valid_mask):
                logger.warning("No pixels meet even NDVI constraint. Falling back to max_temp method")
                return self._select_max_temp(ts, ndvi, albedo, valid)

        # Apply METRIC cold pixel constraints for Triangle method
        cold_valid_mask = valid_mask.copy()
        if albedo is not None:
            cold_valid_mask &= (albedo < 0.20)  # Albedo < 0.20 for well-watered vegetation
            cold_valid_mask &= ~np.isnan(albedo)  # Also exclude NaN albedo
        cold_valid_mask &= (ndvi > 0.70)  # NDVI > 0.70 for well-watered vegetation

        if not np.any(cold_valid_mask):
            logger.warning("No pixels meet METRIC cold pixel constraints in Triangle method. Using NDVI-only constraint.")
            cold_valid_mask = valid_mask & (ndvi > 0.70)  # Fallback to NDVI only
            if not np.any(cold_valid_mask):
                logger.warning("No pixels meet even NDVI constraint. Falling back to max_temp method")
                return self._select_max_temp(ts, ndvi, albedo, valid)

        valid_ts = ts[valid_mask]
        valid_ndvi = ndvi[valid_mask]
        valid_coords = np.where(valid_mask)

        cold_valid_ts = ts[cold_valid_mask]
        cold_valid_ndvi = ndvi[cold_valid_mask]
        cold_valid_coords = np.where(cold_valid_mask)

        cold_valid_ts = ts[cold_valid_mask]
        cold_valid_ndvi = ndvi[cold_valid_mask]
        cold_valid_coords = np.where(cold_valid_mask)

        n_points = len(valid_ts)
        logger.info(f"Triangle method: {n_points} valid points for convex hull")

        if n_points < 10:
            logger.warning(f"Too few points ({n_points}) for convex hull, falling back to max_temp")
            return self._select_max_temp(ts, ndvi, albedo, valid)

        # Stack points for convex hull (NDVI, Ts)
        points = np.column_stack([valid_ndvi, valid_ts])

        try:
            hull = ConvexHull(points)
            # Get vertices of the convex hull
            hull_vertices = hull.vertices

            # Extract hull points and sort by NDVI
            hull_points = points[hull_vertices]
            hull_ndvi = hull_points[:, 0]
            hull_ts = hull_points[:, 1]

            # Sort by NDVI for upper envelope
            sorted_indices = np.argsort(hull_ndvi)
            hull_ndvi_sorted = hull_ndvi[sorted_indices]
            hull_ts_sorted = hull_ts[sorted_indices]

            # Find the upper envelope (max Ts for each NDVI region)
            # The upper envelope is the upper boundary of the hull
            # In Ts-NDVI space, this is the "dry edge"
            upper_envelope_mask = np.zeros(len(hull_ndvi_sorted), dtype=bool)

            # Use monotonic decreasing approach for upper envelope
            max_ts_so_far = -np.inf
            for i in range(len(hull_ndvi_sorted) - 1, -1, -1):
                if hull_ts_sorted[i] > max_ts_so_far:
                    upper_envelope_mask[i] = True
                    max_ts_so_far = hull_ts_sorted[i]

            # Get upper envelope points
            upper_ndvi = hull_ndvi_sorted[upper_envelope_mask]
            upper_ts = hull_ts_sorted[upper_envelope_mask]

            if len(upper_ts) == 0:
                logger.warning("No upper envelope points found, falling back to max_temp")
                return self._select_max_temp(ts, ndvi, albedo, valid)

            # Hot pixel: max temperature on upper envelope (typically low NDVI)
            hot_hull_idx = np.argmax(upper_ts)
            hot_hull_ndvi = upper_ndvi[hot_hull_idx]
            hot_hull_ts = upper_ts[hot_hull_idx]

            # Find actual pixel closest to this hull point
            hot_candidates = valid_mask & (np.abs(ndvi - hot_hull_ndvi) < 0.05)
            if np.any(hot_candidates):
                hot_candidate_ts = ts[hot_candidates]
                hot_candidate_coords = np.where(hot_candidates)
                hot_idx = np.argmax(hot_candidate_ts)
                hot_pixel = AnchorPixel(
                    x=hot_candidate_coords[1][hot_idx],
                    y=hot_candidate_coords[0][hot_idx],
                    temperature=hot_candidate_ts[hot_idx],
                    ts=hot_candidate_ts[hot_idx],
                    ndvi=ndvi[hot_candidates][hot_idx],
                    albedo=albedo[hot_candidates][hot_idx] if albedo is not None else None
                )
            else:
                # Fall back to max temperature overall
                hot_idx = np.argmax(valid_ts)
                hot_pixel = AnchorPixel(
                    x=valid_coords[1][hot_idx],
                    y=valid_coords[0][hot_idx],
                    temperature=valid_ts[hot_idx],
                    ts=valid_ts[hot_idx],
                    ndvi=valid_ndvi[hot_idx],
                    albedo=albedo[valid_mask][hot_idx] if albedo is not None else None
                )

            # Cold pixel: min temperature from METRIC-constrained pixels (typically high NDVI, low albedo)
            cold_idx = np.argmin(cold_valid_ts)
            cold_pixel = AnchorPixel(
                x=cold_valid_coords[1][cold_idx],
                y=cold_valid_coords[0][cold_idx],
                temperature=cold_valid_ts[cold_idx],
                ts=cold_valid_ts[cold_idx],
                ndvi=cold_valid_ndvi[cold_idx],
                albedo=albedo[cold_valid_mask][cold_idx] if albedo is not None else None
            )

            # METRIC Cold Pixel Validation Logging for Triangle method
            logger.info("=== COLD PIXEL VALIDATION CHECKS (TRIANGLE METHOD - METRIC CONSTRAINED) ===")
            cold_ndvi = cold_valid_ndvi[cold_idx]
            cold_albedo = albedo[cold_valid_mask][cold_idx] if albedo is not None else None

            # Check NDVI constraint (should be > 0.70 for well-watered vegetation)
            if cold_ndvi is not None:
                ndvi_valid = cold_ndvi > 0.70
                logger.info(f"Cold pixel NDVI: {cold_ndvi:.3f} (METRIC requirement: > 0.70) - {'PASS' if ndvi_valid else 'FAIL'}")
            else:
                logger.warning("Cold pixel NDVI: Not available for validation")

            # Check albedo constraint (should be < 0.20 for well-watered vegetation)
            if cold_albedo is not None:
                albedo_valid = cold_albedo < 0.20
                logger.info(f"Cold pixel Albedo: {cold_albedo:.3f} (METRIC requirement: < 0.20) - {'PASS' if albedo_valid else 'FAIL'}")
            else:
                logger.warning("Cold pixel Albedo: Not available for validation")

            # Check LST percentile (should be in lowest 5-10%)
            cold_temp_percentile = np.sum(valid_ts <= valid_ts[cold_idx]) / len(valid_ts) * 100
            lst_percentile_valid = cold_temp_percentile <= 10  # Allow up to 10th percentile
            logger.info(f"Cold pixel LST percentile: {cold_temp_percentile:.1f}% (METRIC requirement: ≤ 10%) - {'PASS' if lst_percentile_valid else 'FAIL'}")

            # Check for over-constrained selection
            constraints_met = 0
            total_constraints = 0

            if cold_ndvi is not None:
                total_constraints += 1
                if cold_ndvi > 0.70:
                    constraints_met += 1

            if cold_albedo is not None:
                total_constraints += 1
                if cold_albedo < 0.20:
                    constraints_met += 1

            total_constraints += 1  # LST percentile
            if lst_percentile_valid:
                constraints_met += 1

            constraint_satisfaction = constraints_met / total_constraints if total_constraints > 0 else 0
            logger.info(f"Cold pixel constraint satisfaction: {constraints_met}/{total_constraints} ({constraint_satisfaction:.1%})")

            if constraint_satisfaction < 0.5:
                logger.warning("WARNING: Cold pixel meets less than 50% of METRIC constraints - may be over-constrained or mis-located")

            confidence = min(1.0, (hot_pixel.temperature - cold_pixel.temperature) / 20.0)
            logger.info(f"Triangle method: cold T={cold_pixel.temperature:.2f}K, hot T={hot_pixel.temperature:.2f}K, confidence={confidence:.3f}")
            logger.info(f"  Hot pixel on hull: NDVI={hot_hull_ndvi:.3f}, Ts={hot_hull_ts:.2f}K")

            # Debug: Check selected pixels
            logger.info(f"Final hot pixel selected at (x={hot_pixel.x}, y={hot_pixel.y}), Ts={hot_pixel.temperature:.2f}K")
            logger.info(f"Final cold pixel selected at (x={cold_pixel.x}, y={cold_pixel.y}), Ts={cold_pixel.temperature:.2f}K")

            # Check QA pixel values
            if qa_pixel is not None:
                hot_qa = qa_pixel[hot_pixel.y, hot_pixel.x]
                cold_qa = qa_pixel[cold_pixel.y, cold_pixel.x]
                logger.info(f"  Hot pixel QA value: {hot_qa}")
                logger.info(f"  Cold pixel QA value: {cold_qa}")

                # Check cloud flags
                from ..preprocess.cloud_mask import CloudMasker
                masker = CloudMasker()
                hot_cloud = (hot_qa >> masker.QA_CLOUD_BIT) & 1
                hot_shadow = (hot_qa >> masker.QA_CLOUD_SHADOW_BIT) & 1
                hot_dilated = (hot_qa >> masker.QA_DILATED_CLOUD_BIT) & 1
                cold_cloud = (cold_qa >> masker.QA_CLOUD_BIT) & 1
                cold_shadow = (cold_qa >> masker.QA_CLOUD_SHADOW_BIT) & 1
                cold_dilated = (cold_qa >> masker.QA_DILATED_CLOUD_BIT) & 1

                if hot_cloud or hot_shadow or hot_dilated:
                    logger.warning(f"  WARNING: Hot pixel has cloud/shadow flags!")
                if cold_cloud or cold_shadow or cold_dilated:
                    logger.warning(f"  WARNING: Cold pixel has cloud/shadow flags!")

            return AnchorPixelsResult(
                cold_pixel=cold_pixel,
                hot_pixel=hot_pixel,
                method="triangle",
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Convex hull computation failed: {e}, falling back to max_temp")
            return self._select_max_temp(ts, ndvi, albedo, valid)

    def _select_percentile(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        valid: np.ndarray
    ) -> AnchorPixelsResult:
        """
        Select anchor pixels using percentile-based method.

        This method uses NDVI and Ts percentiles to identify:
        - Hot pixel: from lower NDVI percentiles (dry/bare soil) combined with upper Ts percentiles
        - Cold pixel: from upper NDVI percentiles (dense vegetation) combined with lower Ts percentiles
        """
        import logging
        logger = logging.getLogger(__name__)

        if ndvi is None:
            logger.warning("NDVI not available, falling back to max_temp method")
            return self._select_max_temp(ts, ndvi, albedo, valid)

        valid_mask = valid & ~np.isnan(ndvi)
        if not np.any(valid_mask):
            logger.warning("No valid pixels with NDVI, falling back to max_temp method")
            return self._select_max_temp(ts, ndvi, albedo, valid)

        valid_ts = ts[valid_mask]
        valid_ndvi = ndvi[valid_mask]
        valid_coords = np.where(valid_mask)

        n_points = len(valid_ts)
        logger.info(f"Percentile method: {n_points} valid points")

        if n_points < 10:
            logger.warning(f"Too few points ({n_points}), falling back to max_temp")
            return self._select_max_temp(ts, ndvi, albedo, valid)

        # Define percentile thresholds
        # Hot pixel: low NDVI (dry areas) with high temperature
        ndvi_hot_low = np.percentile(valid_ndvi, 10)  # Lower 10% NDVI
        ndvi_hot_high = np.percentile(valid_ndvi, 30)  # Lower 30% NDVI

        # Cold pixel: high NDVI (vegetated areas) with low temperature
        ndvi_cold_low = np.percentile(valid_ndvi, 70)  # Upper 70% NDVI
        ndvi_cold_high = np.percentile(valid_ndvi, 100)  # Upper 100% NDVI

        # Temperature percentiles for refined selection
        ts_hot_threshold = np.percentile(valid_ts, 80)  # Upper 20% temperature
        ts_cold_threshold = np.percentile(valid_ts, 20)  # Lower 20% temperature

        logger.info(f"NDVI hot range: [{ndvi_hot_low:.3f}, {ndvi_hot_high:.3f}]")
        logger.info(f"NDVI cold range: [{ndvi_cold_low:.3f}, {ndvi_cold_high:.3f}]")
        logger.info(f"Ts hot threshold: {ts_hot_threshold:.2f}K, cold: {ts_cold_threshold:.2f}K")

        # Hot pixel candidate: low NDVI AND high temperature
        hot_candidates = valid_mask & (valid_ndvi.reshape(ts.shape) >= ndvi_hot_low) & (valid_ndvi.reshape(ts.shape) <= ndvi_hot_high)

        if np.any(hot_candidates):
            hot_candidate_ts = ts[hot_candidates]
            hot_candidate_coords = np.where(hot_candidates)

            # Among candidates, select the one with highest temperature
            hot_idx = np.argmax(hot_candidate_ts)
            hot_pixel = AnchorPixel(
                x=hot_candidate_coords[1][hot_idx],
                y=hot_candidate_coords[0][hot_idx],
                temperature=hot_candidate_ts[hot_idx],
                ts=hot_candidate_ts[hot_idx],
                ndvi=ndvi[hot_candidates][hot_idx],
                albedo=albedo[hot_candidates][hot_idx] if albedo is not None else None
            )
        else:
            # Fall back to max temperature in low NDVI areas
            low_ndvi_mask = valid_mask & (valid_ndvi.reshape(ts.shape) <= ndvi_hot_high)
            if np.any(low_ndvi_mask):
                hot_ts = ts[low_ndvi_mask]
                hot_coords = np.where(low_ndvi_mask)
                hot_idx = np.argmax(hot_ts)
                hot_pixel = AnchorPixel(
                    x=hot_coords[1][hot_idx],
                    y=hot_coords[0][hot_idx],
                    temperature=hot_ts[hot_idx],
                    ts=hot_ts[hot_idx],
                    ndvi=ndvi[low_ndvi_mask][hot_idx],
                    albedo=albedo[low_ndvi_mask][hot_idx] if albedo is not None else None
                )
            else:
                # Final fallback
                hot_idx = np.argmax(valid_ts)
                hot_pixel = AnchorPixel(
                    x=valid_coords[1][hot_idx],
                    y=valid_coords[0][hot_idx],
                    temperature=valid_ts[hot_idx],
                    ts=valid_ts[hot_idx],
                    ndvi=valid_ndvi[hot_idx],
                    albedo=albedo[valid_mask][hot_idx] if albedo is not None else None
                )

        # Cold pixel candidate: METRIC-constrained pixels (NDVI > 0.70, Albedo < 0.20) with low temperature
        cold_valid_mask = valid_mask.copy()
        if albedo is not None:
            cold_valid_mask &= (albedo < 0.20)  # Albedo < 0.20 for well-watered vegetation
            cold_valid_mask &= ~np.isnan(albedo)  # Also exclude NaN albedo
        cold_valid_mask &= (ndvi > 0.70)  # NDVI > 0.70 for well-watered vegetation

        if not np.any(cold_valid_mask):
            logger.warning("No pixels meet METRIC cold pixel constraints in Percentile method. Using NDVI-only constraint.")
            cold_valid_mask = valid_mask & (ndvi > 0.70)  # Fallback to NDVI only
            if not np.any(cold_valid_mask):
                logger.warning("No pixels meet even NDVI constraint. Falling back to max_temp method")
                return self._select_max_temp(ts, ndvi, albedo, valid)

        cold_candidates = cold_valid_mask

        if np.any(cold_candidates):
            cold_candidate_ts = ts[cold_candidates]
            cold_candidate_coords = np.where(cold_candidates)

            # Among METRIC-constrained candidates, select the one with lowest temperature
            cold_idx = np.argmin(cold_candidate_ts)
            cold_pixel = AnchorPixel(
                x=cold_candidate_coords[1][cold_idx],
                y=cold_candidate_coords[0][cold_idx],
                temperature=cold_candidate_ts[cold_idx],
                ts=cold_candidate_ts[cold_idx],
                ndvi=ndvi[cold_candidates][cold_idx],
                albedo=albedo[cold_candidates][cold_idx] if albedo is not None else None
            )
        else:
            # No METRIC-constrained pixels available - this should not happen due to earlier checks
            logger.error("Unexpected: No METRIC-constrained cold pixels available in Percentile method")
            cold_idx = np.argmin(valid_ts)
            cold_pixel = AnchorPixel(
                x=valid_coords[1][cold_idx],
                y=valid_coords[0][cold_idx],
                temperature=valid_ts[cold_idx],
                ts=valid_ts[cold_idx],
                ndvi=valid_ndvi[cold_idx],
                albedo=albedo[valid_mask][cold_idx] if albedo is not None else None
            )

        confidence = min(1.0, (hot_pixel.temperature - cold_pixel.temperature) / 20.0)
        logger.info(f"Percentile method: cold T={cold_pixel.temperature:.2f}K, hot T={hot_pixel.temperature:.2f}K, confidence={confidence:.3f}")
        logger.info(f"  Hot pixel: NDVI={hot_pixel.ndvi:.3f}, Cold pixel: NDVI={cold_pixel.ndvi:.3f}")

        return AnchorPixelsResult(
            cold_pixel=cold_pixel,
            hot_pixel=hot_pixel,
            method="percentile",
            confidence=confidence
        )


__all__ = ['AnchorPixel', 'AnchorPixelsResult', 'AnchorPixelSelector']
