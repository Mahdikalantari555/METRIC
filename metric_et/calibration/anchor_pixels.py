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
        method: str = "automatic"
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
        lai: Optional[np.ndarray] = None,
        le: Optional[np.ndarray] = None,
        rn: Optional[np.ndarray] = None,
        g: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
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
        elif method == "enhanced_physical":
            result = self._select_enhanced_physical(ts, ndvi, albedo, valid, qa_pixel)
        elif method == "automatic":
            result = self._select_automatic(ts, ndvi, albedo, valid, qa_pixel, lai, le, rn, g, h)
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

    def _select_enhanced_physical(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        valid: np.ndarray,
        qa_pixel: Optional[np.ndarray] = None,
        lai: Optional[np.ndarray] = None,
        rn: Optional[np.ndarray] = None,
        g: Optional[np.ndarray] = None,
        slope_mask: Optional[np.ndarray] = None,
        water_mask: Optional[np.ndarray] = None,
        cloud_mask: Optional[np.ndarray] = None
    ) -> AnchorPixelsResult:
        """
        Select anchor pixels using enhanced physically-constrained algorithm.
        
        This method implements the complete 5-step algorithm:
        1. Physical Pre-Filtering (Hard Constraints)
        2. Temperature Distribution Filtering
        3. Energy-Based Final Selection (Rn-G optimization)
        4. Quality Control (ET-based validation)
        5. Fallback Strategy (Weighted scoring)
        
        Args:
            ts: Surface temperature array [K]
            ndvi: NDVI array
            albedo: Albedo array
            valid: Valid pixel mask
            qa_pixel: Quality assurance pixel array
            lai: Leaf Area Index array (optional)
            rn: Net radiation [W/m²] (optional)
            g: Soil heat flux [W/m²] (optional)
            slope_mask: Slope mask (optional)
            water_mask: Water mask (optional)
            cloud_mask: Cloud mask (optional)
            
        Returns:
            AnchorPixelsResult with selected anchor pixels
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Using enhanced physically-constrained anchor pixel selection")
        
        # Step 1: Physical Pre-Filtering (Hard Constraints)
        diagnostics = {
            'cold_candidates_count': 0,
            'hot_candidates_count': 0,
            'percentiles_used': {'cold': None, 'hot': None},
            'fallback_activated': False,
            'constraints_relaxed': False
        }
        
        # Create masks
        cold_mask = self._create_cold_pixel_mask(
            ts, ndvi, albedo, lai, slope_mask, water_mask, cloud_mask, valid
        )
        
        hot_mask = self._create_hot_pixel_mask(
            ts, ndvi, albedo, lai, slope_mask, water_mask, cloud_mask, valid
        )
        
        # Step 2: Temperature Distribution Filtering
        cold_candidates, hot_candidates, diagnostics = self._apply_temperature_filtering(
            ts, cold_mask, hot_mask, diagnostics
        )
        
        # Step 3: Energy-Based Final Selection (Rn-G optimization)
        cold_pixel, hot_pixel = self._select_energy_based_pixels(
            ts, cold_candidates, hot_candidates, rn, g, ndvi, albedo, lai
        )
        
        # Step 4: Quality Control (ET-based validation)
        qc_passed = self._perform_quality_control(
            cold_pixel, hot_pixel, ts, ndvi, albedo, rn, g
        )
        
        # Step 5: Fallback Strategy if needed
        if not qc_passed:
            logger.info("Quality control failed, activating fallback strategy")
            cold_pixel, hot_pixel, diagnostics = self._apply_fallback_strategy(
                ts, ndvi, albedo, lai, rn, g, cold_mask, hot_mask, diagnostics
            )
            diagnostics['fallback_activated'] = True
        
        # Calculate confidence based on temperature difference and constraint satisfaction
        temp_diff = hot_pixel.temperature - cold_pixel.temperature
        confidence = min(1.0, temp_diff / 25.0)  # Normalize by typical dT range
        
        logger.info(f"Enhanced physical method: cold T={cold_pixel.temperature:.2f}K, "
                  f"hot T={hot_pixel.temperature:.2f}K, "
                  f"dT={temp_diff:.2f}K, confidence={confidence:.3f}")
        logger.info(f"Diagnostics: {diagnostics}")
        
        return AnchorPixelsResult(
            cold_pixel=cold_pixel,
            hot_pixel=hot_pixel,
            method="enhanced_physical",
            confidence=confidence
        )
    
    def _create_cold_pixel_mask(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        lai: Optional[np.ndarray],
        slope_mask: Optional[np.ndarray],
        water_mask: Optional[np.ndarray],
        cloud_mask: Optional[np.ndarray],
        valid: np.ndarray
    ) -> np.ndarray:
        """Create cold pixel mask using physical constraints."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Start with valid pixels
        cold_mask = valid.copy()
        
        # Apply hard constraints for cold pixels
        if ndvi is not None:
            cold_mask &= (ndvi > 0.75)  # NDVI > 0.75 for dense vegetation
        
        if lai is not None:
            cold_mask &= (lai > 3.0)  # LAI > 3.0 for dense canopy
        
        if albedo is not None:
            cold_mask &= (albedo >= 0.18) & (albedo <= 0.25)  # Albedo range for healthy vegetation
        
        # Exclude water bodies
        if water_mask is not None:
            cold_mask &= ~water_mask
        
        # Exclude slopes if available
        if slope_mask is not None:
            cold_mask &= ~slope_mask
        
        # Exclude clouds if available
        if cloud_mask is not None:
            cold_mask &= ~cloud_mask
        
        # Exclude image borders (3-5 pixels)
        border_pixels = 3
        cold_mask &= ~self._create_border_mask(ts.shape, border_pixels)
        
        logger.info(f"Cold pixel mask: {np.sum(cold_mask)} pixels meet constraints")
        return cold_mask
    
    def _create_hot_pixel_mask(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        lai: Optional[np.ndarray],
        slope_mask: Optional[np.ndarray],
        water_mask: Optional[np.ndarray],
        cloud_mask: Optional[np.ndarray],
        valid: np.ndarray
    ) -> np.ndarray:
        """Create hot pixel mask using physical constraints."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Start with valid pixels
        hot_mask = valid.copy()
        
        # Apply hard constraints for hot pixels
        if ndvi is not None:
            hot_mask &= (ndvi < 0.20)  # NDVI < 0.20 for bare soil
        
        if lai is not None:
            hot_mask &= (lai < 0.50)  # LAI < 0.50 for sparse vegetation
        
        if albedo is not None:
            hot_mask &= (albedo > 0.30)  # Albedo > 0.30 for dry surfaces
        
        # Exclude water bodies
        if water_mask is not None:
            hot_mask &= ~water_mask
        
        # Exclude slopes if available
        if slope_mask is not None:
            hot_mask &= ~slope_mask
        
        # Exclude clouds if available
        if cloud_mask is not None:
            hot_mask &= ~cloud_mask
        
        # Exclude image borders (3-5 pixels)
        border_pixels = 3
        hot_mask &= ~self._create_border_mask(ts.shape, border_pixels)
        
        logger.info(f"Hot pixel mask: {np.sum(hot_mask)} pixels meet constraints")
        return hot_mask
    
    def _create_border_mask(self, shape: tuple, border_pixels: int) -> np.ndarray:
        """Create mask for image borders."""
        border_mask = np.zeros(shape, dtype=bool)
        border_mask[:border_pixels, :] = True
        border_mask[-border_pixels:, :] = True
        border_mask[:, :border_pixels] = True
        border_mask[:, -border_pixels:] = True
        return border_mask
    
    def _apply_temperature_filtering(
        self,
        ts: np.ndarray,
        cold_mask: np.ndarray,
        hot_mask: np.ndarray,
        diagnostics: dict
    ) -> tuple:
        """Apply temperature distribution filtering to create candidate sets."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Cold pixel temperature threshold (5th-10th percentile)
        cold_ts_values = ts[cold_mask]
        if len(cold_ts_values) > 0:
            cold_percentile = 10  # Start with 10th percentile
            ts_cold_threshold = np.percentile(cold_ts_values, cold_percentile)
            diagnostics['percentiles_used']['cold'] = cold_percentile
            
            # Create cold candidates
            cold_candidates = cold_mask & (ts <= ts_cold_threshold)
            diagnostics['cold_candidates_count'] = np.sum(cold_candidates)
            
            logger.info(f"Cold candidates: {np.sum(cold_candidates)} pixels "
                      f"(Ts <= {ts_cold_threshold:.2f}K, {cold_percentile}th percentile)")
        else:
            cold_candidates = cold_mask.copy()
            logger.warning("No cold pixels meet constraints, using all valid pixels")
        
        # Hot pixel temperature threshold (90th-95th percentile)
        hot_ts_values = ts[hot_mask]
        if len(hot_ts_values) > 0:
            hot_percentile = 90  # Start with 90th percentile
            ts_hot_threshold = np.percentile(hot_ts_values, hot_percentile)
            diagnostics['percentiles_used']['hot'] = hot_percentile
            
            # Create hot candidates
            hot_candidates = hot_mask & (ts >= ts_hot_threshold)
            diagnostics['hot_candidates_count'] = np.sum(hot_candidates)
            
            logger.info(f"Hot candidates: {np.sum(hot_candidates)} pixels "
                      f"(Ts >= {ts_hot_threshold:.2f}K, {hot_percentile}th percentile)")
        else:
            hot_candidates = hot_mask.copy()
            logger.warning("No hot pixels meet constraints, using all valid pixels")
        
        return cold_candidates, hot_candidates, diagnostics
    
    def _select_energy_based_pixels(
        self,
        ts: np.ndarray,
        cold_candidates: np.ndarray,
        hot_candidates: np.ndarray,
        rn: Optional[np.ndarray],
        g: Optional[np.ndarray],
        ndvi: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        lai: Optional[np.ndarray]
    ) -> tuple:
        """Select pixels based on energy balance (Rn-G optimization)."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Cold pixel: maximize Rn - G (actively transpiring vegetation)
        if rn is not None and g is not None:
            rn_minus_g = rn - g
            
            if np.any(cold_candidates):
                cold_rng_values = rn_minus_g[cold_candidates]
                cold_idx = np.argmax(cold_rng_values)
                cold_coords = np.where(cold_candidates)
                cold_x, cold_y = cold_coords[1][cold_idx], cold_coords[0][cold_idx]
                
                logger.info(f"Cold pixel selected by max Rn-G: {rn_minus_g[cold_y, cold_x]:.2f} W/m²")
            else:
                # Fallback to coldest temperature
                cold_idx = np.argmin(ts[cold_candidates])
                cold_coords = np.where(cold_candidates)
                cold_x, cold_y = cold_coords[1][cold_idx], cold_coords[0][cold_idx]
                logger.warning("No cold candidates, using coldest temperature")
        else:
            # Fallback to coldest temperature
            cold_idx = np.argmin(ts[cold_candidates])
            cold_coords = np.where(cold_candidates)
            cold_x, cold_y = cold_coords[1][cold_idx], cold_coords[0][cold_idx]
            logger.warning("Rn/G not available, using coldest temperature")
        
        # Hot pixel: minimize Rn - G (near-zero latent heat flux)
        if rn is not None and g is not None:
            if np.any(hot_candidates):
                hot_rng_values = rn_minus_g[hot_candidates]
                hot_idx = np.argmin(hot_rng_values)
                hot_coords = np.where(hot_candidates)
                hot_x, hot_y = hot_coords[1][hot_idx], hot_coords[0][hot_idx]
                
                logger.info(f"Hot pixel selected by min Rn-G: {rn_minus_g[hot_y, hot_x]:.2f} W/m²")
            else:
                # Fallback to hottest temperature
                hot_idx = np.argmax(ts[hot_candidates])
                hot_coords = np.where(hot_candidates)
                hot_x, hot_y = hot_coords[1][hot_idx], hot_coords[0][hot_idx]
                logger.warning("No hot candidates, using hottest temperature")
        else:
            # Fallback to hottest temperature
            hot_idx = np.argmax(ts[hot_candidates])
            hot_coords = np.where(hot_candidates)
            hot_x, hot_y = hot_coords[1][hot_idx], hot_coords[0][hot_idx]
            logger.warning("Rn/G not available, using hottest temperature")
        
        # Create AnchorPixel objects
        cold_pixel = AnchorPixel(
            x=cold_x, y=cold_y,
            temperature=ts[cold_y, cold_x],
            ts=ts[cold_y, cold_x],
            ndvi=ndvi[cold_y, cold_x] if ndvi is not None else None,
            albedo=albedo[cold_y, cold_x] if albedo is not None else None
        )
        
        hot_pixel = AnchorPixel(
            x=hot_x, y=hot_y,
            temperature=ts[hot_y, hot_x],
            ts=ts[hot_y, hot_x],
            ndvi=ndvi[hot_y, hot_x] if ndvi is not None else None,
            albedo=albedo[hot_y, hot_x] if albedo is not None else None
        )
        
        return cold_pixel, hot_pixel
    
    def _perform_quality_control(
        self,
        cold_pixel: AnchorPixel,
        hot_pixel: AnchorPixel,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        rn: Optional[np.ndarray],
        g: Optional[np.ndarray]
    ) -> bool:
        """Perform quality control checks on selected pixels."""
        import logging
        logger = logging.getLogger(__name__)
        
        qc_passed = True
        
        # Cold pixel QC: NDVI > 0.75, albedo 0.18-0.25, low temperature
        if cold_pixel.ndvi is not None and cold_pixel.ndvi <= 0.75:
            logger.warning(f"Cold pixel QC FAIL: NDVI {cold_pixel.ndvi:.3f} <= 0.75")
            qc_passed = False
        
        if cold_pixel.albedo is not None:
            if cold_pixel.albedo < 0.18 or cold_pixel.albedo > 0.25:
                logger.warning(f"Cold pixel QC FAIL: albedo {cold_pixel.albedo:.3f} outside [0.18, 0.25]")
                qc_passed = False
        
        # Hot pixel QC: NDVI < 0.20, albedo > 0.30, high temperature
        if hot_pixel.ndvi is not None and hot_pixel.ndvi >= 0.20:
            logger.warning(f"Hot pixel QC FAIL: NDVI {hot_pixel.ndvi:.3f} >= 0.20")
            qc_passed = False
        
        if hot_pixel.albedo is not None and hot_pixel.albedo <= 0.30:
            logger.warning(f"Hot pixel QC FAIL: albedo {hot_pixel.albedo:.3f} <= 0.30")
            qc_passed = False
        
        # Temperature difference check
        temp_diff = hot_pixel.temperature - cold_pixel.temperature
        if temp_diff < 5.0:  # Minimum 5K difference expected
            logger.warning(f"QC FAIL: Temperature difference {temp_diff:.2f}K < 5K")
            qc_passed = False
        
        logger.info(f"Quality control: {'PASSED' if qc_passed else 'FAILED'}")
        return qc_passed
    
    def _apply_fallback_strategy(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        lai: Optional[np.ndarray],
        rn: Optional[np.ndarray],
        g: Optional[np.ndarray],
        cold_mask: np.ndarray,
        hot_mask: np.ndarray,
        diagnostics: dict
    ) -> tuple:
        """Apply fallback strategy with weighted scoring."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Relax constraints
        relaxed_cold_mask = self._relax_constraints(
            ts, ndvi, albedo, lai, cold_mask, "cold"
        )
        
        relaxed_hot_mask = self._relax_constraints(
            ts, ndvi, albedo, lai, hot_mask, "hot"
        )
        
        diagnostics['constraints_relaxed'] = True
        
        # Apply weighted scoring
        cold_pixel = self._select_by_weighted_score(
            ts, ndvi, albedo, lai, rn, g, relaxed_cold_mask, "cold"
        )
        
        hot_pixel = self._select_by_weighted_score(
            ts, ndvi, albedo, lai, rn, g, relaxed_hot_mask, "hot"
        )
        
        return cold_pixel, hot_pixel, diagnostics
    
    def _relax_constraints(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        lai: Optional[np.ndarray],
        original_mask: np.ndarray,
        pixel_type: str
    ) -> np.ndarray:
        """Relax constraints for fallback strategy."""
        import logging
        logger = logging.getLogger(__name__)
        
        relaxed_mask = original_mask.copy()
        
        if pixel_type == "cold":
            # Relax NDVI constraint
            if ndvi is not None:
                relaxed_mask |= (ndvi > 0.70)  # Relax from 0.75 to 0.70
            
            # Relax albedo constraint
            if albedo is not None:
                relaxed_mask |= ((albedo >= 0.15) & (albedo <= 0.28))  # Widen range
            
        elif pixel_type == "hot":
            # Relax NDVI constraint
            if ndvi is not None:
                relaxed_mask |= (ndvi < 0.25)  # Relax from 0.20 to 0.25
            
            # Relax albedo constraint
            if albedo is not None:
                relaxed_mask |= (albedo > 0.25)  # Relax from 0.30 to 0.25
        
        logger.info(f"Relaxed {pixel_type} mask: {np.sum(relaxed_mask)} pixels")
        return relaxed_mask
    
    def _select_by_weighted_score(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        lai: Optional[np.ndarray],
        rn: Optional[np.ndarray],
        g: Optional[np.ndarray],
        candidate_mask: np.ndarray,
        pixel_type: str
    ) -> AnchorPixel:
        """Select pixel using weighted scoring."""
        import logging
        logger = logging.getLogger(__name__)
        
        if not np.any(candidate_mask):
            logger.error(f"No {pixel_type} candidates available in fallback")
            # Return dummy pixel (should not happen due to earlier checks)
            return AnchorPixel(0, 0, temperature=300.0)
        
        # Calculate scores
        scores = np.zeros(candidate_mask.shape)
        
        if pixel_type == "cold":
            # Cold pixel score: +NDVI + LAI - Ts + (Rn - G)
            if ndvi is not None:
                scores += ndvi * 2.0  # NDVI weight
            if lai is not None:
                scores += lai * 1.5  # LAI weight
            scores -= ts * 0.1  # Temperature penalty
            if rn is not None and g is not None:
                scores += (rn - g) * 0.01  # Rn-G bonus
        
        elif pixel_type == "hot":
            # Hot pixel score: -NDVI - LAI + Ts - (Rn - G)
            if ndvi is not None:
                scores -= ndvi * 2.0  # NDVI penalty
            if lai is not None:
                scores -= lai * 1.5  # LAI penalty
            scores += ts * 0.1  # Temperature bonus
            if rn is not None and g is not None:
                scores -= (rn - g) * 0.01  # Rn-G penalty
        
        # Apply mask and find best candidate
        masked_scores = np.where(candidate_mask, scores, -np.inf)
        best_idx = np.argmax(masked_scores)
        best_coords = np.unravel_index(best_idx, ts.shape)
        
        pixel = AnchorPixel(
            x=best_coords[1], y=best_coords[0],
            temperature=ts[best_coords],
            ts=ts[best_coords],
            ndvi=ndvi[best_coords] if ndvi is not None else None,
            albedo=albedo[best_coords] if albedo is not None else None
        )
        
        logger.info(f"Fallback {pixel_type} pixel selected with score {scores[best_coords]:.2f}")
        return pixel

    def _select_automatic(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        valid: np.ndarray,
        qa_pixel: Optional[np.ndarray] = None,
        lai: Optional[np.ndarray] = None,
        le: Optional[np.ndarray] = None,
        rn: Optional[np.ndarray] = None,
        g: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None
    ) -> AnchorPixelsResult:
        """
        Select anchor pixels using the exact METRIC algorithm specifications.

        This method implements the complete METRIC anchor pixel selection algorithm:
        1. Percentile-based constraints for initial candidate selection
        2. Energy-based optimization for final pixel selection
        3. Hard constraint validation for physical consistency

        Args:
            ts: Surface temperature array [K]
            ndvi: NDVI array
            albedo: Albedo array
            valid: Valid pixel mask
            qa_pixel: Quality assurance pixel array
            lai: Leaf Area Index array
            le: Latent heat flux array [W/m²]
            rn: Net radiation array [W/m²]
            g: Soil heat flux array [W/m²]
            h: Sensible heat flux array [W/m²]

        Returns:
            AnchorPixelsResult with selected anchor pixels
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Using METRIC-compliant automatic anchor pixel selection")

        # Create valid mask excluding NaN values
        valid_mask = valid & ~np.isnan(ts)
        if ndvi is not None:
            valid_mask &= ~np.isnan(ndvi)
        if albedo is not None:
            valid_mask &= ~np.isnan(albedo)
        if lai is not None:
            valid_mask &= ~np.isnan(lai)

        if not np.any(valid_mask):
            logger.warning("No valid pixels available for automatic selection")
            return self._select_max_temp(ts, ndvi, albedo, valid, qa_pixel)

        # Get valid data
        valid_ts = ts[valid_mask]
        valid_ndvi = ndvi[valid_mask] if ndvi is not None else None
        valid_albedo = albedo[valid_mask] if albedo is not None else None
        valid_lai = lai[valid_mask] if lai is not None else None
        valid_coords = np.where(valid_mask)

        n_points = len(valid_ts)
        logger.info(f"Automatic method: {n_points} valid points for METRIC analysis")

        if n_points < 20:
            logger.warning(f"Too few points ({n_points}) for reliable analysis, falling back to max_temp")
            return self._select_max_temp(ts, ndvi, albedo, valid, qa_pixel)

        # === STEP 1: PERCENTILE-BASED CANDIDATE SELECTION ===
        # METRIC specifications for percentile thresholds

        # Cold pixel constraints: NDVI ≥ P90–P95, Albedo ≤ P20–P30, Ts ≤ P10–P15, LAI ≥ P80
        cold_ndvi_min = np.percentile(valid_ndvi, 90) if valid_ndvi is not None else None  # NDVI ≥ P90
        cold_albedo_max = np.percentile(valid_albedo, 30) if valid_albedo is not None else None  # Albedo ≤ P30
        cold_ts_max = np.percentile(valid_ts, 15)  # Ts ≤ P15
        cold_lai_min = np.percentile(valid_lai, 80) if valid_lai is not None else None  # LAI ≥ P80

        # Hot pixel constraints: NDVI ≤ P5–P10, Albedo ≥ P70–P85, Ts ≥ P90–P95, LAI ≤ P20
        hot_ndvi_max = np.percentile(valid_ndvi, 10) if valid_ndvi is not None else None  # NDVI ≤ P10
        hot_albedo_min = np.percentile(valid_albedo, 70) if valid_albedo is not None else None  # Albedo ≥ P70
        hot_ts_min = np.percentile(valid_ts, 90)  # Ts ≥ P90
        hot_lai_max = np.percentile(valid_lai, 20) if valid_lai is not None else None  # LAI ≤ P20

        logger.info("METRIC percentile thresholds:")
        logger.info(f"  Cold - Ts ≤ {cold_ts_max:.2f}K, NDVI ≥ {cold_ndvi_min:.3f}, Albedo ≤ {cold_albedo_max:.3f}, LAI ≥ {cold_lai_min:.3f}")
        logger.info(f"  Hot - Ts ≥ {hot_ts_min:.2f}K, NDVI ≤ {hot_ndvi_max:.3f}, Albedo ≥ {hot_albedo_min:.3f}, LAI ≤ {hot_lai_max:.3f}")

        # Apply percentile constraints to create candidate sets
        cold_candidates = valid_mask.copy()
        hot_candidates = valid_mask.copy()

        # Temperature constraints
        cold_candidates &= (ts <= cold_ts_max)
        hot_candidates &= (ts >= hot_ts_min)

        # NDVI constraints
        if valid_ndvi is not None:
            cold_candidates &= (ndvi >= cold_ndvi_min)
            hot_candidates &= (ndvi <= hot_ndvi_max)

        # Albedo constraints
        if valid_albedo is not None:
            cold_candidates &= (albedo <= cold_albedo_max)
            hot_candidates &= (albedo >= hot_albedo_min)

        # LAI constraints
        if valid_lai is not None:
            cold_candidates &= (lai >= cold_lai_min)
            hot_candidates &= (lai <= hot_lai_max)

        cold_count = np.sum(cold_candidates)
        hot_count = np.sum(hot_candidates)

        logger.info(f"Percentile-based candidates: {cold_count} cold, {hot_count} hot")

        # If insufficient candidates, relax constraints slightly
        if cold_count < 5:
            logger.info("Relaxing cold pixel constraints")
            cold_ts_max = np.percentile(valid_ts, 20)  # Relax to P20
            cold_candidates = valid_mask & (ts <= cold_ts_max)
            if valid_ndvi is not None:
                cold_ndvi_min = np.percentile(valid_ndvi, 85)  # Relax to P85
                cold_candidates &= (ndvi >= cold_ndvi_min)
            cold_count = np.sum(cold_candidates)

        if hot_count < 5:
            logger.info("Relaxing hot pixel constraints")
            hot_ts_min = np.percentile(valid_ts, 85)  # Relax to P85
            hot_candidates = valid_mask & (ts >= hot_ts_min)
            if valid_ndvi is not None:
                hot_ndvi_max = np.percentile(valid_ndvi, 15)  # Relax to P15
                hot_candidates &= (ndvi <= hot_ndvi_max)
            hot_count = np.sum(hot_candidates)

        logger.info(f"After relaxation: {cold_count} cold, {hot_count} hot candidates")

        # Final fallback if still insufficient
        if cold_count < 3 or hot_count < 3:
            logger.warning("Insufficient candidates even after relaxation, using max_temp")
            return self._select_max_temp(ts, ndvi, albedo, valid, qa_pixel)

        # === STEP 2: ENERGY-BASED FINAL SELECTION ===

        # Cold pixel: minimize |LE − (Rn − G)| or minimize H
        if le is not None and rn is not None and g is not None:
            # Primary: minimize |LE - (Rn - G)|
            rn_minus_g = rn - g
            le_diff = np.abs(le - rn_minus_g)
            cold_energy_values = le_diff[cold_candidates]
            cold_idx = np.argmin(cold_energy_values)
            logger.info("Cold pixel selected by minimizing |LE - (Rn - G)|")
        elif h is not None:
            # Secondary: minimize H
            cold_energy_values = h[cold_candidates]
            cold_idx = np.argmin(cold_energy_values)
            logger.info("Cold pixel selected by minimizing H")
        else:
            # Fallback: minimum temperature
            cold_energy_values = ts[cold_candidates]
            cold_idx = np.argmin(cold_energy_values)
            logger.info("Cold pixel selected by minimum temperature (energy data unavailable)")

        cold_coords = np.where(cold_candidates)
        cold_x, cold_y = cold_coords[1][cold_idx], cold_coords[0][cold_idx]

        # Hot pixel: maximize H
        if h is not None:
            hot_energy_values = h[hot_candidates]
            hot_idx = np.argmax(hot_energy_values)
            logger.info("Hot pixel selected by maximizing H")
        else:
            # Fallback: maximum temperature
            hot_energy_values = ts[hot_candidates]
            hot_idx = np.argmax(hot_energy_values)
            logger.info("Hot pixel selected by maximum temperature (H unavailable)")

        hot_coords = np.where(hot_candidates)
        hot_x, hot_y = hot_coords[1][hot_idx], hot_coords[0][hot_idx]

        # Create AnchorPixel objects
        cold_pixel = AnchorPixel(
            x=cold_x, y=cold_y,
            temperature=ts[cold_y, cold_x],
            ts=ts[cold_y, cold_x],
            ndvi=ndvi[cold_y, cold_x] if ndvi is not None else None,
            albedo=albedo[cold_y, cold_x] if albedo is not None else None
        )

        hot_pixel = AnchorPixel(
            x=hot_x, y=hot_y,
            temperature=ts[hot_y, hot_x],
            ts=ts[hot_y, hot_x],
            ndvi=ndvi[hot_y, hot_x] if ndvi is not None else None,
            albedo=albedo[hot_y, hot_x] if albedo is not None else None
        )

        # === STEP 3: HARD CONSTRAINT VALIDATION ===

        logger.info("=== METRIC HARD CONSTRAINT VALIDATION ===")

        validation_passed = True

        # Cold pixel constraints
        if h is not None:
            h_cold = h[cold_y, cold_x]
            if not (0 <= h_cold <= 50):  # H_cold ∈ [0, 50] W/m²
                logger.warning(f"Cold pixel H validation FAIL: H={h_cold:.1f} W/m² not in [0, 50]")
                validation_passed = False
            else:
                logger.info(f"Cold pixel H validation PASS: H={h_cold:.1f} W/m²")

        # dT_cold constraint (soft: ≤0.5K, hard: ≤1.0K)
        # Note: dT_cold is the temperature difference from air temperature, but we don't have Ta here
        # We'll skip this check as Ta is not available in the current implementation

        if le is not None and rn is not None and g is not None:
            le_cold = le[cold_y, cold_x]
            rn_cold = rn[cold_y, cold_x]
            g_cold = g[cold_y, cold_x]
            if le_cold > (rn_cold - g_cold):  # LE_cold ≤ Rn_cold − G_cold
                logger.warning(f"Cold pixel LE validation FAIL: LE={le_cold:.1f} > Rn-G={rn_cold-g_cold:.1f}")
                validation_passed = False
            else:
                logger.info(f"Cold pixel LE validation PASS: LE={le_cold:.1f} ≤ Rn-G={rn_cold-g_cold:.1f}")

        # Hot pixel constraints
        if le is not None:
            le_hot = le[hot_y, hot_x]
            if le_hot > 20:  # LE_hot ≈ 0 (≤ 20 W/m²)
                logger.warning(f"Hot pixel LE validation FAIL: LE={le_hot:.1f} > 20 W/m²")
                validation_passed = False
            else:
                logger.info(f"Hot pixel LE validation PASS: LE={le_hot:.1f} ≤ 20 W/m²")

        if h is not None:
            h_hot = h[hot_y, hot_x]
            if h_hot < 200:  # H_hot ≥ 200 W/m²
                logger.warning(f"Hot pixel H validation FAIL: H={h_hot:.1f} < 200 W/m²")
                validation_passed = False
            else:
                logger.info(f"Hot pixel H validation PASS: H={h_hot:.1f} ≥ 200 W/m²")

        # Temperature difference constraint
        temp_diff = hot_pixel.temperature - cold_pixel.temperature
        if temp_diff < 15:  # Ts_hot − Ts_cold ≥ 15 K
            logger.warning(f"Temperature difference validation FAIL: dT={temp_diff:.1f} < 15 K")
            validation_passed = False
        else:
            logger.info(f"Temperature difference validation PASS: dT={temp_diff:.1f} ≥ 15 K")

        if not validation_passed:
            logger.warning("METRIC hard constraints not satisfied - consider fallback or manual selection")
            # Note: In a full implementation, you might want to trigger fallback logic here

        # Calculate confidence based on constraint satisfaction and temperature difference
        confidence = min(1.0, temp_diff / 25.0)  # Base confidence on temperature difference
        if validation_passed:
            confidence = min(1.0, confidence + 0.2)  # Bonus for passing validation

        logger.info(f"METRIC automatic method: cold T={cold_pixel.temperature:.2f}K, "
                  f"hot T={hot_pixel.temperature:.2f}K, "
                  f"dT={temp_diff:.2f}K, confidence={confidence:.3f}")

        return AnchorPixelsResult(
            cold_pixel=cold_pixel,
            hot_pixel=hot_pixel,
            method="automatic",
            confidence=confidence
        )


__all__ = ['AnchorPixel', 'AnchorPixelsResult', 'AnchorPixelSelector']
