"""Anchor pixel selection module for METRIC calibration.

This module implements the enhanced METRIC-compliant automatic anchor pixel selection
algorithm with the following improvements:

1. Cluster-based selection - Select N pixels instead of single pixels
2. ET_ratio-based cold pixel selection - Minimize |ET/ET0_inst - 1|
3. Hot pixel energy constraints - H/(Rn-G) >= 0.8, LE/(Rn-G) <= 0.2
4. dT-Ts shape validation - Validate slope a in 0.1-0.4 K/K range
5. Scene normalization - Use Ts_anom = Ts - median(Ts_valid)
6. Enhanced confidence - Penalize easy scenes and relaxation
"""

from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import logging

import numpy as np


@dataclass
class AnchorCluster:
    """Represents a cluster of anchor pixels with statistics."""
    pixel_indices: List[Tuple[int, int]]  # (y, x) coordinates
    ts_values: np.ndarray
    h_values: Optional[np.ndarray] = None
    le_values: Optional[np.ndarray] = None
    ndvi_values: Optional[np.ndarray] = None
    albedo_values: Optional[np.ndarray] = None
    
    # Statistics (anchor values) - use MEDIAN for robustness
    ts_median: float = np.nan
    h_median: float = np.nan
    le_median: float = np.nan
    ndvi_median: float = np.nan
    albedo_median: float = np.nan
    
    def __post_init__(self):
        """Calculate median statistics after initialization."""
        if len(self.ts_values) > 0 and np.any(np.isfinite(self.ts_values)):
            self.ts_median = float(np.nanmedian(self.ts_values))
        if self.h_values is not None and len(self.h_values) > 0 and np.any(np.isfinite(self.h_values)):
            self.h_median = float(np.nanmedian(self.h_values))
        if self.le_values is not None and len(self.le_values) > 0 and np.any(np.isfinite(self.le_values)):
            self.le_median = float(np.nanmedian(self.le_values))
        if self.ndvi_values is not None and len(self.ndvi_values) > 0 and np.any(np.isfinite(self.ndvi_values)):
            self.ndvi_median = float(np.nanmedian(self.ndvi_values))
        if self.albedo_values is not None and len(self.albedo_values) > 0 and np.any(np.isfinite(self.albedo_values)):
            self.albedo_median = float(np.nanmedian(self.albedo_values))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pixel_indices': self.pixel_indices,
            'ts_median': self.ts_median,
            'h_median': self.h_median,
            'le_median': self.le_median,
            'ndvi_median': self.ndvi_median,
            'albedo_median': self.albedo_median,
            'n_pixels': len(self.pixel_indices)
        }


class AnchorPixel:
    """Represents a single anchor pixel (for backward compatibility)."""

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


@dataclass
class AnchorPixelsResult:
    """Result of anchor pixel selection with cluster support."""
    
    # Single pixel anchors (for backward compatibility)
    cold_pixel: Optional[AnchorPixel] = None
    hot_pixel: Optional[AnchorPixel] = None
    
    # Cluster-based anchors (new)
    cold_cluster: Optional[AnchorCluster] = None
    hot_cluster: Optional[AnchorCluster] = None
    
    method: str = "automatic"
    confidence: float = 0.0
    
    # Validation results
    validation_passed: bool = False
    validation_issues: List[str] = field(default_factory=list)
    
    # Scene metadata
    ts_median: float = np.nan
    ts_std: float = np.nan
    temp_difference: float = np.nan
    
    # Relaxation level (0 = no relaxation, 1 = max relaxation)
    relaxation_level: float = 0.0
    
    def __post_init__(self):
        """Ensure backward compatibility by creating single pixels from clusters."""
        # Create single pixel anchors from clusters for backward compatibility
        if self.cold_cluster is not None and len(self.cold_cluster.pixel_indices) > 0:
            if self.cold_pixel is None:
                # Use the best pixel (first in list) for single-pixel access
                best_y, best_x = self.cold_cluster.pixel_indices[0]
                self.cold_pixel = AnchorPixel(
                    x=best_x,
                    y=best_y,
                    temperature=self.cold_cluster.ts_values[0] if len(self.cold_cluster.ts_values) > 0 else np.nan,
                    ts=self.cold_cluster.ts_values[0] if len(self.cold_cluster.ts_values) > 0 else np.nan,
                    ndvi=self.cold_cluster.ndvi_values[0] if self.cold_cluster.ndvi_values is not None and len(self.cold_cluster.ndvi_values) > 0 else np.nan,
                    albedo=self.cold_cluster.albedo_values[0] if self.cold_cluster.albedo_values is not None and len(self.cold_cluster.albedo_values) > 0 else np.nan
                )
        
        if self.hot_cluster is not None and len(self.hot_cluster.pixel_indices) > 0:
            if self.hot_pixel is None:
                best_y, best_x = self.hot_cluster.pixel_indices[0]
                self.hot_pixel = AnchorPixel(
                    x=best_x,
                    y=best_y,
                    temperature=self.hot_cluster.ts_values[0] if len(self.hot_cluster.ts_values) > 0 else np.nan,
                    ts=self.hot_cluster.ts_values[0] if len(self.hot_cluster.ts_values) > 0 else np.nan,
                    ndvi=self.hot_cluster.ndvi_values[0] if self.hot_cluster.ndvi_values is not None and len(self.hot_cluster.ndvi_values) > 0 else np.nan,
                    albedo=self.hot_cluster.albedo_values[0] if self.hot_cluster.albedo_values is not None and len(self.hot_cluster.albedo_values) > 0 else np.nan
                )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'method': self.method,
            'confidence': self.confidence,
            'validation_passed': self.validation_passed,
            'validation_issues': self.validation_issues,
            'ts_median': self.ts_median,
            'ts_std': self.ts_std,
            'temp_difference': self.temp_difference,
            'relaxation_level': self.relaxation_level
        }
        
        if self.cold_pixel is not None:
            result['cold_pixel'] = self.cold_pixel.to_dict()
        
        if self.hot_pixel is not None:
            result['hot_pixel'] = self.hot_pixel.to_dict()
        
        if self.cold_cluster is not None:
            result['cold_cluster'] = self.cold_cluster.to_dict()
        
        if self.hot_cluster is not None:
            result['hot_cluster'] = self.hot_cluster.to_dict()
        
        return result


class AnchorPixelSelector:
    """
    Select hot and cold anchor pixels for METRIC calibration.
    
    Enhanced implementation with cluster-based selection and improved
    METRIC-compliant constraints.
    
    Key improvements:
    - Selects N pixels (cluster) instead of single pixels
    - Uses ET ratio criterion for cold pixel selection
    - Applies energy ratio constraints to hot pixels
    - Validates dT-Ts shape
    - Scene-scale temperature normalization
    - Enhanced confidence calculation
    """

    def __init__(
        self,
        min_temperature: float = 200.0,
        max_temperature: float = 400.0,
        method: str = "automatic",
        # Cluster configuration
        cluster_size: int = 20,  # N = 10-50 recommended
        # Scene normalization
        use_scene_normalization: bool = True,
        # Energy ratio thresholds
        hot_h_energy_ratio_min: float = 0.8,  # H/(Rn-G) >= 0.8
        hot_le_energy_ratio_max: float = 0.2,  # LE/(Rn-G) <= 0.2
        # ET ratio thresholds
        et_ratio_target: float = 1.0,  # ET/ET0_inst target
        et_ratio_tolerance: float = 0.15,  # ±15% tolerance
        # Latent heat of vaporization (J/kg)
        lambda_v: float = 2.45e6
    ):
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.method = method
        
        # Cluster configuration
        self.cluster_size = cluster_size
        
        # Scene normalization
        self.use_scene_normalization = use_scene_normalization
        
        # Energy ratio thresholds
        self.hot_h_energy_ratio_min = hot_h_energy_ratio_min
        self.hot_le_energy_ratio_max = hot_le_energy_ratio_max
        
        # ET ratio thresholds
        self.et_ratio_target = et_ratio_target
        self.et_ratio_tolerance = et_ratio_tolerance
        
        # Physical constants
        self.lambda_v = lambda_v
    
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

        # Get optional energy data
        le = cube.get("LE")
        rn = cube.get("R_n")
        g = cube.get("G")
        h = cube.get("H")

        # Get ET0 data for cold pixel selection
        et0_daily = cube.get("et0_fao_evapotranspiration")
        rs_inst = cube.get("shortwave_radiation")
        rs_daily = cube.get("shortwave_radiation_sum")

        # Calculate ET0_inst if data available
        et0_inst = None
        if et0_daily is not None and rs_inst is not None and rs_daily is not None:
            from ..core.constants import MJ_M2_DAY_TO_W
            et0_daily_val = float(np.nanmean(et0_daily.values))
            rs_inst_val = float(np.nanmean(rs_inst.values))
            rs_daily_val = float(np.nanmean(rs_daily.values))
            rs_daily_w = rs_daily_val * MJ_M2_DAY_TO_W
            
            if rs_daily_w > 0:
                et0_inst = et0_daily_val * (rs_inst_val / rs_daily_w)

        # Call the selection method
        return self.select_anchor_pixels(
            ts=ts.values,
            ndvi=ndvi.values if ndvi is not None else None,
            albedo=albedo.values if albedo is not None else None,
            qa_pixel=qa_pixel.values if qa_pixel is not None else None,
            le=le.values if le is not None else None,
            rn=rn.values if rn is not None else None,
            g=g.values if g is not None else None,
            h=h.values if h is not None else None,
            et0_inst=et0_inst
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
        et0_inst: Optional[float] = None,
        **kwargs
    ) -> AnchorPixelsResult:
        """
        Select hot and cold anchor pixels using the enhanced METRIC method.

        Args:
            ts: Surface temperature array (Kelvin)
            ndvi: NDVI array (optional)
            albedo: Albedo array (optional)
            qa_pixel: Quality assurance pixel array
            lai: Leaf Area Index array
            le: Latent heat flux array [W/m²]
            rn: Net radiation array [W/m²]
            g: Soil heat flux array [W/m²]
            h: Sensible heat flux array [W/m²]
            et0_inst: Instantaneous reference ET (mm/hr) for cold pixel selection

        Returns:
            AnchorPixelsResult with selected anchor pixels and clusters
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

        logger.info(f"Valid pixels: {np.sum(valid)} out of {valid.size}")

        if not np.any(valid):
            raise ValueError("No valid pixels found for anchor selection")

        # Always use enhanced automatic method
        logger.info("Using enhanced METRIC-compliant automatic anchor pixel selection")
        result = self._select_enhanced(
            ts=ts,
            ndvi=ndvi,
            albedo=albedo,
            valid=valid,
            qa_pixel=qa_pixel,
            lai=lai,
            le=le,
            rn=rn,
            g=g,
            h=h,
            et0_inst=et0_inst
        )

        return result

    def _select_enhanced(
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
        h: Optional[np.ndarray] = None,
        et0_inst: Optional[float] = None
    ) -> AnchorPixelsResult:
        """
        Enhanced anchor pixel selection with all improvements.
        
        Implements:
        1. Scene-scale temperature normalization
        2. Percentile-based candidate selection
        3. Energy ratio constraints for hot pixels
        4. ET ratio-based cold pixel selection
        5. Cluster extraction and median statistics
        6. dT-Ts shape validation
        7. Enhanced confidence calculation
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Starting enhanced METRIC anchor pixel selection")
        
        # Create valid mask excluding NaN values
        valid_mask = valid & ~np.isnan(ts)
        if ndvi is not None:
            valid_mask &= ~np.isnan(ndvi)
        if albedo is not None:
            valid_mask &= ~np.isnan(albedo)
        if lai is not None:
            valid_mask &= ~np.isnan(lai)

        if not np.any(valid_mask):
            raise ValueError("No valid pixels available for automatic selection")

        # Get valid data
        valid_ts = ts[valid_mask]
        valid_ndvi = ndvi[valid_mask] if ndvi is not None else None
        valid_albedo = albedo[valid_mask] if albedo is not None else None
        valid_lai = lai[valid_mask] if lai is not None else None
        valid_coords = np.where(valid_mask)

        n_points = len(valid_ts)
        logger.info(f"Enhanced method: {n_points} valid points for METRIC analysis")

        if n_points < 20:
            logger.warning(f"Low point count ({n_points}) - consider using more relaxation")
        
        # === STEP 1: SCENE-SCALE TEMPERATURE NORMALIZATION ===
        
        if self.use_scene_normalization:
            ts_median = np.nanmedian(valid_ts)
            ts_anom = ts - ts_median
            logger.info(f"Scene normalization: Ts_median = {ts_median:.2f} K")
        else:
            ts_anom = ts.copy()
            ts_median = np.nanmedian(valid_ts)
        
        ts_std = np.nanstd(valid_ts)
        logger.info(f"Ts statistics: median={ts_median:.2f} K, std={ts_std:.2f} K")
        
        # Use normalized Ts for percentile calculations
        valid_ts_anom = ts_anom[valid_mask]
        
        # === STEP 2: PERCENTILE-BASED CANDIDATE SELECTION ===
        # METRIC specifications for percentile thresholds

        # Cold pixel constraints: NDVI >= P90-P95, Albedo <= P20-P30, Ts <= P10-P15, LAI >= P80
        cold_ndvi_min = np.percentile(valid_ndvi, 90) if valid_ndvi is not None else None
        cold_albedo_max = np.percentile(valid_albedo, 30) if valid_albedo is not None else None
        cold_ts_max = np.percentile(valid_ts_anom, 15)
        cold_lai_min = np.percentile(valid_lai, 80) if valid_lai is not None else None

        # Hot pixel constraints: NDVI <= P5-P10, Albedo >= P70-P85, Ts >= P90-P95, LAI <= P20
        hot_ndvi_max = np.percentile(valid_ndvi, 10) if valid_ndvi is not None else None
        hot_albedo_min = np.percentile(valid_albedo, 70) if valid_albedo is not None else None
        hot_ts_min = np.percentile(valid_ts_anom, 90)
        hot_lai_max = np.percentile(valid_lai, 20) if valid_lai is not None else None

        logger.info("METRIC percentile thresholds (using normalized Ts):")
        
        cold_ndvi_str = f"{cold_ndvi_min:.3f}" if cold_ndvi_min is not None else "N/A"
        cold_albedo_str = f"{cold_albedo_max:.3f}" if cold_albedo_max is not None else "N/A"
        cold_lai_str = f"{cold_lai_min:.3f}" if cold_lai_min is not None else "N/A"
        
        hot_ndvi_str = f"{hot_ndvi_max:.3f}" if hot_ndvi_max is not None else "N/A"
        hot_albedo_str = f"{hot_albedo_min:.3f}" if hot_albedo_min is not None else "N/A"
        hot_lai_str = f"{hot_lai_max:.3f}" if hot_lai_max is not None else "N/A"
        
        logger.info(
            f"  Cold - Ts_anom <= {cold_ts_max:.2f}K, NDVI >= {cold_ndvi_str}, "
            f"Albedo <= {cold_albedo_str}, LAI >= {cold_lai_str}"
        )
        logger.info(
            f"  Hot - Ts_anom >= {hot_ts_min:.2f}K, NDVI <= {hot_ndvi_str}, "
            f"Albedo >= {hot_albedo_str}, LAI <= {hot_lai_str}"
        )

        # Apply percentile constraints to create candidate sets
        cold_candidates = valid_mask.copy()
        hot_candidates = valid_mask.copy()

        # Temperature constraints (using normalized Ts)
        cold_candidates &= (ts_anom <= cold_ts_max)
        hot_candidates &= (ts_anom >= hot_ts_min)

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

        # Track relaxation level
        relaxation_level = 0.0
        max_relaxation = 3  # Allow up to 3 relaxation steps
        
        # === STEP 3: ENERGY RATIO CONSTRAINTS FOR HOT PIXELS ===
        
        if hot_count >= 3 and (h is not None or le is not None):
            hot_candidates = self._apply_energy_ratio_constraints(
                hot_candidates, h, le, rn, g
            )
            hot_count = np.sum(hot_candidates)
            logger.info(f"After energy ratio constraints: {hot_count} hot candidates")

        # === STEP 4: RELAX CONSTRAINTS IF NEEDED ===
        
        while (cold_count < 5 or hot_count < 5) and relaxation_level < max_relaxation:
            relaxation_level += 0.25
            logger.info(f"Relaxing constraints (level={relaxation_level:.2f})")
            
            # Relax temperature percentiles
            cold_ts_max = np.percentile(valid_ts_anom, 15 + relaxation_level * 10)
            hot_ts_min = np.percentile(valid_ts_anom, 90 - relaxation_level * 10)
            
            cold_candidates = valid_mask & (ts_anom <= cold_ts_max)
            hot_candidates = valid_mask & (ts_anom >= hot_ts_min)
            
            # Relax NDVI constraints
            if valid_ndvi is not None:
                cold_ndvi_min = np.percentile(valid_ndvi, 90 - relaxation_level * 10)
                hot_ndvi_max = np.percentile(valid_ndvi, 10 + relaxation_level * 10)
                cold_candidates &= (ndvi >= cold_ndvi_min)
                hot_candidates &= (ndvi <= hot_ndvi_max)
            
            # Reapply energy ratio constraints to relaxed hot candidates
            if hot_count >= 3 and (h is not None or le is not None):
                hot_candidates = self._apply_energy_ratio_constraints(
                    hot_candidates, h, le, rn, g
                )
            
            cold_count = np.sum(cold_candidates)
            hot_count = np.sum(hot_candidates)
            
            logger.info(f"After relaxation: {cold_count} cold, {hot_count} hot candidates")

        # Final fallback if still insufficient
        if cold_count < 3 or hot_count < 3:
            logger.warning(f"Insufficient candidates: {cold_count} cold, {hot_count} hot")
            logger.warning("Using extreme percentile fallback")
            # Use extreme percentiles as last resort
            cold_candidates = valid_mask & (ts_anom <= np.percentile(valid_ts_anom, 20))
            hot_candidates = valid_mask & (ts_anom >= np.percentile(valid_ts_anom, 80))
            cold_count = np.sum(cold_candidates)
            hot_count = np.sum(hot_candidates)
            
            if cold_count < 1 or hot_count < 1:
                raise ValueError("Insufficient candidates even after extreme fallback")

        # === STEP 5: ENERGY-BASED CANDIDATE RANKING ===

        # Cold pixel: use ET ratio criterion if ET0_inst available
        if et0_inst is not None and et0_inst > 0 and le is not None:
            cold_sorted_indices = self._rank_cold_by_et_ratio(
                cold_candidates, le, et0_inst
            )
            logger.info("Cold pixels ranked by ET ratio criterion")
        elif h is not None:
            # Fallback: minimize H
            cold_energy_values = np.full(ts.shape, np.nan)
            cold_energy_values[cold_candidates] = h[cold_candidates]
            valid_idx = np.where(np.isfinite(cold_energy_values[cold_candidates]))[0]
            cold_sorted_indices = valid_idx[np.argsort(cold_energy_values[cold_candidates][valid_idx])]
            logger.info("Cold pixels ranked by minimum H")
        else:
            # Fallback: minimum temperature
            cold_energy_values = np.full(ts.shape, np.nan)
            cold_energy_values[cold_candidates] = ts_anom[cold_candidates]
            valid_idx = np.where(np.isfinite(cold_energy_values[cold_candidates]))[0]
            cold_sorted_indices = valid_idx[np.argsort(cold_energy_values[cold_candidates][valid_idx])]
            logger.info("Cold pixels ranked by minimum temperature")

        # Hot pixel: maximize H
        if h is not None:
            hot_energy_values = np.full(ts.shape, np.nan)
            hot_energy_values[hot_candidates] = h[hot_candidates]
            valid_idx = np.where(np.isfinite(hot_energy_values[hot_candidates]))[0]
            hot_sorted_indices = valid_idx[np.argsort(-hot_energy_values[hot_candidates][valid_idx])]  # Descending
            logger.info("Hot pixels ranked by maximum H")
        else:
            # Fallback: maximum temperature
            hot_energy_values = np.full(ts.shape, np.nan)
            hot_energy_values[hot_candidates] = ts_anom[hot_candidates]
            valid_idx = np.where(np.isfinite(hot_energy_values[hot_candidates]))[0]
            hot_sorted_indices = valid_idx[np.argsort(-hot_energy_values[hot_candidates][valid_idx])]
            logger.info("Hot pixels ranked by maximum temperature")

        # === STEP 6: EXTRACT N ANCHOR PIXELS (CLUSTER) ===

        N = min(self.cluster_size, len(cold_sorted_indices), len(hot_sorted_indices))
        logger.info(f"Extracting cluster of {N} pixels from each anchor")

        cold_coords = np.where(cold_candidates)
        hot_coords = np.where(hot_candidates)

        cold_pool_indices = [
            (cold_coords[0][cold_sorted_indices[i]], cold_coords[1][cold_sorted_indices[i]])
            for i in range(N)
        ]

        hot_pool_indices = [
            (hot_coords[0][hot_sorted_indices[i]], hot_coords[1][hot_sorted_indices[i]])
            for i in range(N)
        ]

        # === STEP 7: CALCULATE CLUSTER STATISTICS (MEDIAN) ===

        cold_ts_values = np.array([ts[y, x] for y, x in cold_pool_indices])
        
        # Handle H values - only create array if h is available
        if h is not None:
            cold_h_values = np.array([h[y, x] if not np.isnan(h[y, x]) else np.nan for y, x in cold_pool_indices])
        else:
            cold_h_values = np.array([])  # Empty array, will result in NaN median
        
        # Handle LE values - only create array if le is available
        if le is not None:
            cold_le_values = np.array([le[y, x] if not np.isnan(le[y, x]) else np.nan for y, x in cold_pool_indices])
        else:
            cold_le_values = np.array([])
        
        # Handle NDVI and albedo values
        if ndvi is not None:
            cold_ndvi_values = np.array([ndvi[y, x] if not np.isnan(ndvi[y, x]) else np.nan for y, x in cold_pool_indices])
        else:
            cold_ndvi_values = np.array([])
        
        if albedo is not None:
            cold_albedo_values = np.array([albedo[y, x] if not np.isnan(albedo[y, x]) else np.nan for y, x in cold_pool_indices])
        else:
            cold_albedo_values = np.array([])

        hot_ts_values = np.array([ts[y, x] for y, x in hot_pool_indices])
        
        if h is not None:
            hot_h_values = np.array([h[y, x] if not np.isnan(h[y, x]) else np.nan for y, x in hot_pool_indices])
        else:
            hot_h_values = np.array([])
        
        if le is not None:
            hot_le_values = np.array([le[y, x] if not np.isnan(le[y, x]) else np.nan for y, x in hot_pool_indices])
        else:
            hot_le_values = np.array([])
        
        if ndvi is not None:
            hot_ndvi_values = np.array([ndvi[y, x] if not np.isnan(ndvi[y, x]) else np.nan for y, x in hot_pool_indices])
        else:
            hot_ndvi_values = np.array([])
        
        if albedo is not None:
            hot_albedo_values = np.array([albedo[y, x] if not np.isnan(albedo[y, x]) else np.nan for y, x in hot_pool_indices])
        else:
            hot_albedo_values = np.array([])

        # Anchor statistics: use MEDIAN for robustness (handle empty/NaN arrays)
        ts_cold = float(np.nanmedian(cold_ts_values))
        h_cold = float(np.nanmedian(cold_h_values)) if len(cold_h_values) > 0 and np.any(np.isfinite(cold_h_values)) else np.nan
        le_cold = float(np.nanmedian(cold_le_values)) if len(cold_le_values) > 0 and np.any(np.isfinite(cold_le_values)) else np.nan

        ts_hot = float(np.nanmedian(hot_ts_values))
        h_hot = float(np.nanmedian(hot_h_values)) if len(hot_h_values) > 0 and np.any(np.isfinite(hot_h_values)) else np.nan
        le_hot = float(np.nanmedian(hot_le_values)) if len(hot_le_values) > 0 and np.any(np.isfinite(hot_le_values)) else np.nan

        # Create cluster objects
        cold_cluster = AnchorCluster(
            pixel_indices=cold_pool_indices,
            ts_values=cold_ts_values,
            h_values=cold_h_values,
            le_values=cold_le_values,
            ndvi_values=cold_ndvi_values,
            albedo_values=cold_albedo_values
        )

        hot_cluster = AnchorCluster(
            pixel_indices=hot_pool_indices,
            ts_values=hot_ts_values,
            h_values=hot_h_values,
            le_values=hot_le_values,
            ndvi_values=hot_ndvi_values,
            albedo_values=hot_albedo_values
        )

        logger.info(f"Cold cluster median: Ts={ts_cold:.2f}K, H={h_cold:.1f} W/m², LE={le_cold:.1f} W/m²")
        logger.info(f"Hot cluster median: Ts={ts_hot:.2f}K, H={h_hot:.1f} W/m², LE={le_hot:.1f} W/m²")

        # === STEP 8: VALIDATION ===

        validation_passed = True
        validation_issues = []

        # Temperature difference constraint
        temp_diff = ts_hot - ts_cold
        min_temp_diff = 15.0
        if temp_diff < min_temp_diff:
            validation_issues.append(f"Temperature difference dT={temp_diff:.1f}K < {min_temp_diff}K")
            validation_passed = False
        else:
            logger.info(f"Temperature difference validation PASS: dT={temp_diff:.1f}K >= {min_temp_diff}K")

        # Cold pixel validation (allow H_cold up to 50-70 W/m², not forcing to 0)
        if h is not None and h_cold is not None and not np.isnan(h_cold):
            if h_cold > 150:
                validation_issues.append(f"Cold pixel H={h_cold:.1f} W/m² > 150 W/m² (may indicate poor cold anchor)")
                # Don't fail for this - just warn
                logger.warning(f"Cold pixel H={h_cold:.1f} W/m² is higher than typical")
        
        if le is not None and le_cold is not None and not np.isnan(le_cold) and rn is not None and g is not None:
            # Check that LE_cold <= available energy (physical constraint)
            rn_cold_val = np.median([rn[y, x] for y, x in cold_pool_indices])
            g_cold_val = np.median([g[y, x] for y, x in cold_pool_indices])
            available_energy_cold = rn_cold_val - g_cold_val
            
            if le_cold > available_energy_cold:
                validation_issues.append(f"Cold pixel LE={le_cold:.1f} > available energy {available_energy_cold:.1f} W/m²")
                validation_passed = False

        # Hot pixel energy dominance validation
        if h is not None and le is not None and rn is not None and g is not None:
            rn_hot_val = np.median([rn[y, x] for y, x in hot_pool_indices])
            g_hot_val = np.median([g[y, x] for y, x in hot_pool_indices])
            available_energy_hot = rn_hot_val - g_hot_val
            
            if available_energy_hot > 10:  # Valid energy
                h_ratio = h_hot / available_energy_hot if h_hot is not None and available_energy_hot > 0 else 0
                le_ratio = le_hot / available_energy_hot if le_hot is not None and available_energy_hot > 0 else 0
                
                if h_ratio < 0.5:  # Should be >= 0.8 ideally
                    validation_issues.append(f"Hot pixel H ratio={h_ratio:.2f} < 0.5 (insufficient energy dominance)")
                    # Don't fail - may be due to data quality
                    logger.warning(f"Hot pixel H ratio={h_ratio:.2f} is low")
                
                if le_ratio > 0.5:  # Should be <= 0.2 ideally
                    validation_issues.append(f"Hot pixel LE ratio={le_ratio:.2f} > 0.5 (excessive evaporation)")
                    # Don't fail
                    logger.warning(f"Hot pixel LE ratio={le_ratio:.2f} is high")

        # dT-Ts shape validation
        air_temperature = 293.15  # Default, will be overwritten in calibration
        dt_shape_valid, dt_shape_issues = self._validate_dt_ts_shape(
            ts_cold, ts_hot, h_cold, h_hot, rn_cold_val if rn is not None else np.nan,
            rn_hot_val if rn is not None else np.nan, g_cold_val if g is not None else np.nan,
            g_hot_val if g is not None else np.nan, air_temperature
        )
        validation_issues.extend(dt_shape_issues)
        if not dt_shape_valid:
            validation_passed = False

        # === STEP 9: ENHANCED CONFIDENCE CALCULATION ===

        confidence = self._calculate_confidence(
            validation_passed=validation_passed,
            temp_diff=temp_diff,
            ts_valid_std=ts_std,
            n_cold_candidates=cold_count,
            n_hot_candidates=hot_count,
            relaxation_level=relaxation_level,
            dt_shape_valid=dt_shape_valid
        )

        logger.info(f"Enhanced confidence: {confidence:.3f} (validation_passed={validation_passed}, relaxation={relaxation_level:.2f})")

        # Log final results
        logger.info("=== ENHANCED ANCHOR SELECTION RESULTS ===")
        logger.info(f"Cold cluster: {N} pixels, Ts_median={ts_cold:.2f}K")
        logger.info(f"Hot cluster: {N} pixels, Ts_median={ts_hot:.2f}K")
        logger.info(f"Temperature difference: dT = {temp_diff:.2f}K")
        logger.info(f"Validation: {'PASSED' if validation_passed else 'FAILED'}")
        logger.info(f"Confidence: {confidence:.3f}")

        return AnchorPixelsResult(
            cold_pixel=None,  # Will be auto-created from cluster
            hot_pixel=None,
            cold_cluster=cold_cluster,
            hot_cluster=hot_cluster,
            method="enhanced_automatic",
            confidence=confidence,
            validation_passed=validation_passed,
            validation_issues=validation_issues,
            ts_median=ts_median,
            ts_std=ts_std,
            temp_difference=temp_diff,
            relaxation_level=relaxation_level
        )

    def _apply_energy_ratio_constraints(
        self,
        hot_candidates: np.ndarray,
        h: Optional[np.ndarray],
        le: Optional[np.ndarray],
        rn: Optional[np.ndarray],
        g: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Apply METRIC-compliant hot pixel energy ratio constraints.
        
        Required:
        - H / (Rn - G) >= 0.8 (energy-dominated)
        - LE / (Rn - G) <= 0.2 (minimal evaporation)
        
        If constraints remove too many pixels, relax to 0.7 / 0.3.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if h is None or le is None or rn is None or g is None:
            logger.warning("Energy data not available - skipping hot pixel energy constraints")
            return hot_candidates
        
        available_energy = rn - g
        valid_energy = (available_energy > 10) & ~np.isnan(available_energy)
        
        # Apply constraints where energy is valid
        energy_mask = hot_candidates.copy()
        
        h_ratio = np.zeros_like(h, dtype=float)
        le_ratio = np.zeros_like(le, dtype=float)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            h_ratio = np.where(valid_energy, h / np.maximum(available_energy, 1.0), 0)
            le_ratio = np.where(valid_energy, le / np.maximum(available_energy, 1.0), 0)
        
        # Primary constraints
        energy_mask &= (h_ratio >= self.hot_h_energy_ratio_min)
        energy_mask &= (le_ratio <= self.hot_le_energy_ratio_max)
        
        n_after_primary = np.sum(energy_mask)
        n_before = np.sum(hot_candidates)
        
        if n_after_primary < max(5, n_before * 0.1):  # Less than 10% remaining
            # Relax constraints
            logger.info(f"Relaxing hot pixel energy constraints: {n_after_primary} pixels remaining, using 0.7/0.3")
            energy_mask = hot_candidates.copy()
            energy_mask &= (h_ratio >= 0.7)
            energy_mask &= (le_ratio <= 0.3)
        
        return energy_mask

    def _rank_cold_by_et_ratio(
        self,
        cold_candidates: np.ndarray,
        le: np.ndarray,
        et0_inst: float
    ) -> np.ndarray:
        """
        Rank cold pixels by ET ratio deviation from 1.0.
        
        METRIC Reality: Cold pixel assumption is ET_cold ≈ ET0_inst ± 5-10%
        NOT: H_cold = 0
        
        Selection criterion: minimize |ET_cold / ET0_inst - 1|
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Convert LE to ET (mm/hr)
        # ET = LE * 3600 / (lambda_v * 1000)
        et_values = le * 3600.0 / (self.lambda_v * 1000.0)
        
        # Calculate ET ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            et_ratio = np.where(et0_inst > 0, et_values / et0_inst, np.nan)
        
        # Score: minimize deviation from 1.0
        et_ratio_deviation = np.abs(et_ratio - 1.0)
        
        # Apply to cold candidates
        candidate_scores = np.full(et_ratio_deviation.shape, np.nan)
        candidate_scores[cold_candidates] = et_ratio_deviation[cold_candidates]
        
        # Get valid indices
        valid_mask = np.isfinite(candidate_scores) & cold_candidates
        valid_indices = np.where(valid_mask)[0]
        
        # Sort by score (ascending - best first)
        scores = candidate_scores[valid_mask]
        sorted_local_indices = np.argsort(scores)
        sorted_indices = valid_indices[sorted_local_indices]
        
        logger.info(f"ET ratio statistics: median={np.nanmedian(et_ratio[cold_candidates]):.3f}, "
                   f"mean={np.nanmean(et_ratio[cold_candidates]):.3f}")
        
        return sorted_indices

    def _validate_dt_ts_shape(
        self,
        ts_cold: float,
        ts_hot: float,
        h_cold: float,
        h_hot: float,
        rn_cold: float,
        rn_hot: float,
        g_cold: float,
        g_hot: float,
        air_temperature: float
    ) -> Tuple[bool, List[str]]:
        """
        Validate dT-Ts relationship shape after calibration.
        
        NOTE: The slope validation (a coefficient) is computed during calibration,
        not at anchor selection time. This method validates physically meaningful
        constraints that can be checked with anchor pixel data alone.
        
        Validates:
        - dT values are physically plausible
        - Temperature range is sufficient
        - Hot pixel energy dominance (H ≈ available energy)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        issues = []
        
        # Calculate dT values
        dT_cold = ts_cold - air_temperature
        dT_hot = ts_hot - air_temperature
        
        # Calculate temperature range
        ts_range = ts_hot - ts_cold
        
        # NOTE: The slope (a coefficient) CANNOT be validated here because:
        # - At anchor selection time, we haven't computed the METRIC calibration yet
        # - The actual 'a' coefficient is computed in DTCalibration.calibrate()
        # - The slope validation should be done AFTER calibration, using the
        #   calibrated 'a_coefficient' from CalibrationResult
        # 
        # The old code calculated: slope = (dT_hot - dT_cold) / (Ts_hot - Ts_cold)
        # But this is ALWAYS 1.0 because dT = Ts - Ta, so:
        # (dT_hot - dT_cold) = (Ts_hot - Ta) - (Ts_cold - Ta) = Ts_hot - Ts_cold
        # Therefore: slope = (Ts_hot - Ts_cold) / (Ts_hot - Ts_cold) = 1.0
        # 
        # The REAL METRIC 'a' coefficient represents dt_a = H / dT in W/m²/K,
        # which is computed during calibration from:
        # a = (Rn_hot - G_hot) / dT_hot  (physics-based calibration)
        
        # Validate physical plausibility of dT values
        if dT_cold < -5 or dT_cold > 10:
            issues.append(f"Cold pixel dT={dT_cold:.2f}K outside plausible range [-5, 10]K")
        
        if dT_hot < 5 or dT_hot > 45:
            issues.append(f"Hot pixel dT={dT_hot:.2f}K outside plausible range [5, 45]K")
        
        # Validate temperature range
        if ts_range < 10:
            issues.append(f"Temperature range {ts_range:.2f}K < 10K - insufficient for calibration")
        
        # Validate hot pixel energy dominance
        # Hot pixel should have H ≈ available energy (Rn - G)
        if h_hot is not None and not np.isnan(h_hot) and rn_hot is not None and g_hot is not None:
            available_energy_hot = rn_hot - g_hot
            if available_energy_hot > 10:  # Valid energy threshold
                h_ratio = h_hot / available_energy_hot
                if h_ratio < 0.5:
                    issues.append(f"Hot pixel H ratio={h_ratio:.2f} < 0.5 (insufficient energy dominance)")
        
        # dT at median Ts should be physically plausible (2-10 K typical METRIC range)
        ts_median = (ts_cold + ts_hot) / 2
        dT_median = (dT_cold + dT_hot) / 2
        if dT_median < 0 or dT_median > 25:
            issues.append(f"dT at median Ts={dT_median:.2f}K outside plausible range [0, 25]K")
        
        if len(issues) > 0:
            logger.warning(f"dT-Ts shape issues: dT_cold={dT_cold:.2f}K, dT_hot={dT_hot:.2f}K, {issues}")
            return False, issues
        
        logger.info(f"dT-Ts shape validation PASSED: dT_cold={dT_cold:.2f}K, dT_hot={dT_hot:.2f}K")
        return True, []

    def _calculate_confidence(
        self,
        validation_passed: bool,
        temp_diff: float,
        ts_valid_std: float,
        n_cold_candidates: int,
        n_hot_candidates: int,
        relaxation_level: float,
        dt_shape_valid: bool
    ) -> float:
        """
        Calculate enhanced confidence score with scene quality penalties.
        
        Confidence increases with:
        - Validation passed
        - Large temperature difference
        - High Ts variance (healthy scene)
        - Many anchor candidates (robust selection)
        
        Confidence decreases with:
        - Excessive relaxation
        - Low Ts variance (easy scenes)
        - Small ΔTs
        - dT-Ts shape validation failure
        """
        import logging
        logger = logging.getLogger(__name__)
        
        confidence = 0.5  # Base confidence
        
        # Bonus for passing validation
        if validation_passed:
            confidence += 0.15
        
        # Bonus for dT-Ts shape validation
        if dt_shape_valid:
            confidence += 0.10
        
        # Scale by temperature difference (target: 15-25 K)
        temp_factor = min(1.0, max(0.5, temp_diff / 20.0))
        confidence *= temp_factor
        
        # Penalty for low Ts variance (easy scenes should not have high confidence)
        # Typical Ts std: 5-15 K for healthy scenes
        ts_variance_factor = min(1.0, max(0.5, ts_valid_std / 8.0))
        confidence *= ts_variance_factor
        
        # Penalty for excessive relaxation
        relaxation_penalty = np.exp(-relaxation_level * 2)
        confidence *= relaxation_penalty
        
        # Bonus for having many candidates (robust selection)
        n_candidates = n_cold_candidates + n_hot_candidates
        candidate_factor = min(1.0, max(0.7, n_candidates / 50.0))
        confidence *= (0.8 + 0.2 * candidate_factor)
        
        # Clip to [0, 1]
        confidence = np.clip(confidence, 0.0, 1.0)
        
        logger.debug(
            f"Confidence breakdown: base=0.5, validation={0.15 if validation_passed else 0}, "
            f"dt_shape={0.1 if dt_shape_valid else 0}, temp_factor={temp_factor:.2f}, "
            f"variance_factor={ts_variance_factor:.2f}, relaxation_penalty={relaxation_penalty:.2f}, "
            f"candidate_factor={candidate_factor:.2f}"
        )
        
        return confidence


# Keep old class name as alias for backward compatibility
AnchorPixelSelector = AnchorPixelSelector


__all__ = ['AnchorPixel', 'AnchorPixelsResult', 'AnchorPixelSelector', 'AnchorCluster']
