"""
Enhanced anchor pixel selection module for METRIC calibration.

This module implements a robust, physically-constrained, fully automated algorithm
to select Cold and Hot anchor pixels for METRIC internal calibration, suitable for
heterogeneous and semi-urban landscapes.

Algorithm Overview:
1. Physical Pre-Filtering (Hard Constraints)
2. Temperature Distribution Filtering  
3. Energy-Based Final Selection (Rn-G optimization)
4. Quality Control (ET-based validation)
5. Fallback Strategy (Weighted scoring)
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
import logging

from ..core.datacube import DataCube
from .anchor_pixels import AnchorPixel, AnchorPixelsResult


@dataclass
class PhysicalConstraints:
    """Physical constraints for anchor pixel selection."""
    # Cold pixel constraints
    cold_ndvi_min = 0.5
    cold_lai_min = 1.0
    cold_albedo_min = 0.15
    cold_albedo_max = 0.3
    cold_rn_minus_g_min: float = 0.0
    
    # Hot pixel constraints
    hot_ndvi_max: float = 0.20
    hot_lai_max: float = 0.50
    hot_albedo_min: float = 0.30
    
    # Temperature percentile ranges
    cold_temp_percentile_min: float = 5.0
    cold_temp_percentile_max: float = 10.0
    hot_temp_percentile_min: float = 90.0
    hot_temp_percentile_max: float = 95.0
    
    # Border exclusion
    border_pixels: int = 5
    
    # Fallback parameters
    fallback_ndvi_relax: float = 0.05
    fallback_percentile_expansion: float = 5.0
    min_candidate_threshold: int = 10


@dataclass
class SelectionDiagnostics:
    """Diagnostics for anchor pixel selection process."""
    # Cold pixel diagnostics
    cold_candidates_count: int = 0
    cold_temp_percentile_used: float = 0.0
    cold_rn_minus_g_value: float = 0.0
    cold_constraints_satisfied: int = 0
    cold_total_constraints: int = 0
    
    # Hot pixel diagnostics  
    hot_candidates_count: int = 0
    hot_temp_percentile_used: float = 0.0
    hot_rn_minus_g_value: float = 0.0
    hot_constraints_satisfied: int = 0
    hot_total_constraints: int = 0
    
    # Quality control results
    cold_etinst_vs_et0: float = 0.0
    cold_dt_k: float = 0.0
    cold_h_minimal: bool = False
    hot_le_near_zero: bool = False
    hot_dt_k: float = 0.0
    
    # Fallback activation
    fallback_used: bool = False
    fallback_reason: str = ""
    
    # Energy balance validation
    energy_balance_closure: float = 0.0


class EnhancedAnchorPixelSelector:
    """
    Enhanced anchor pixel selector implementing physically-constrained algorithm.
    
    This class provides a robust, automated method for selecting hot and cold
    anchor pixels using physical constraints, energy balance considerations,
    and quality control validation.
    """

    def __init__(
        self,
        constraints: Optional[PhysicalConstraints] = None,
        enable_quality_control: bool = True,
        enable_fallback: bool = True
    ):
        """
        Initialize enhanced anchor pixel selector.
        
        Args:
            constraints: Physical constraints for selection
            enable_quality_control: Whether to perform ET-based quality control
            enable_fallback: Whether to use fallback strategy if needed
        """
        self.constraints = constraints or PhysicalConstraints()
        self.enable_quality_control = enable_quality_control
        self.enable_fallback = enable_fallback
        self.logger = logging.getLogger(__name__)
        
    def select_anchor_pixels(
        self,
        cube: DataCube,
        rn: Optional[np.ndarray] = None,
        g_flux: Optional[np.ndarray] = None,
        air_temperature: Optional[float] = None,
        et0_inst: Optional[float] = None
    ) -> AnchorPixelsResult:
        """
        Select anchor pixels using physically-constrained algorithm.
        
        Args:
            cube: DataCube containing required bands
            rn: Net radiation array (W/m²) - uses cube.get('R_n') if not provided
            g_flux: Soil heat flux array (W/m²) - computed if not provided
            air_temperature: Air temperature (K) - uses cube metadata if not provided
            et0_inst: Instantaneous reference ET (mm/hr) - computed if not provided
            
        Returns:
            AnchorPixelsResult with enhanced diagnostics
        """
        self.logger.info("Starting enhanced physically-constrained anchor pixel selection")
        
        # Extract required data from cube
        ts = self._get_surface_temperature(cube)
        ndvi = self._get_ndvi(cube)
        lai = self._get_lai(cube)
        albedo = self._get_albedo(cube)
        
        # Get energy balance data
        if rn is None:
            rn = self._get_net_radiation(cube)
        if g_flux is None:
            g_flux = self._estimate_soil_heat_flux(cube, rn, ndvi, ts)
            
        # Calculate available energy (Rn - G)
        available_energy = rn - g_flux
        
        # Get QA mask for cloud/shadow exclusion
        qa_mask = self._get_qa_mask(cube)
        
        # Apply border exclusion
        border_mask = self._create_border_mask(ts.shape)
        
        # Combine masks
        valid_mask = qa_mask & border_mask
        
        # Step 1: Physical Pre-Filtering
        cold_mask, hot_mask = self._apply_physical_constraints(
            ts, ndvi, lai, albedo, available_energy, valid_mask
        )
        
        # Step 2: Temperature Distribution Filtering
        cold_candidates, hot_candidates, diagnostics = self._apply_temperature_filtering(
            ts, cold_mask, hot_mask, diagnostics=SelectionDiagnostics()
        )
        
        # Step 3: Energy-Based Final Selection
        cold_pixel, hot_pixel = self._energy_based_selection(
            ts, ndvi, lai, albedo, available_energy,
            cold_candidates, hot_candidates
        )
        
        # Step 4: Quality Control
        if self.enable_quality_control:
            cold_pixel, hot_pixel, diagnostics = self._quality_control(
                cube, cold_pixel, hot_pixel, rn, g_flux,
                air_temperature, et0_inst, diagnostics
            )
        
        # Step 5: Fallback Strategy
        if self.enable_fallback and (cold_pixel is None or hot_pixel is None):
            cold_pixel, hot_pixel, diagnostics = self._fallback_strategy(
                ts, ndvi, lai, albedo, available_energy,
                cold_mask, hot_mask, diagnostics
            )
        
        # Final validation
        if cold_pixel is None or hot_pixel is None:
            raise ValueError("Failed to select valid anchor pixels")
            
        # Calculate confidence
        confidence = self._calculate_confidence(cold_pixel, hot_pixel)
        
        self.logger.info(
            f"Enhanced anchor selection completed: "
            f"Cold ({cold_pixel.x}, {cold_pixel.y}) T={cold_pixel.temperature:.2f}K, "
            f"Hot ({hot_pixel.x}, {hot_pixel.y}) T={hot_pixel.temperature:.2f}K"
        )
        
        return AnchorPixelsResult(
            cold_pixel=cold_pixel,
            hot_pixel=hot_pixel,
            method="enhanced_physical",
            confidence=confidence
        )

    def _get_surface_temperature(self, cube: DataCube) -> np.ndarray:
        """Extract surface temperature from cube."""
        ts = cube.get('lst')
        if ts is None:
            raise ValueError("Surface temperature (lst) not found in cube")
        return ts.values if hasattr(ts, 'values') else ts

    def _get_ndvi(self, cube: DataCube) -> Optional[np.ndarray]:
        """Extract NDVI from cube."""
        ndvi = cube.get('ndvi')
        return ndvi.values if ndvi is not None and hasattr(ndvi, 'values') else ndvi

    def _get_lai(self, cube: DataCube) -> Optional[np.ndarray]:
        """Extract LAI from cube."""
        lai = cube.get('lai')
        return lai.values if lai is not None and hasattr(lai, 'values') else lai

    def _get_albedo(self, cube: DataCube) -> Optional[np.ndarray]:
        """Extract albedo from cube."""
        albedo = cube.get('albedo')
        return albedo.values if albedo is not None and hasattr(albedo, 'values') else albedo

    def _get_net_radiation(self, cube: DataCube) -> np.ndarray:
        """Extract net radiation from cube."""
        rn = cube.get('R_n')
        if rn is None:
            raise ValueError("Net radiation (R_n) not found in cube")
        return rn.values if hasattr(rn, 'values') else rn

    def _estimate_soil_heat_flux(self, cube: DataCube, rn: np.ndarray, 
                                ndvi: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """Estimate soil heat flux if not available."""
        # Simple estimation based on NDVI and net radiation
        # G/Rn = 0.3 for bare soil (NDVI < 0.2), 0.1 for dense vegetation (NDVI > 0.7)
        g_ratio = np.where(
            ndvi < 0.2, 0.3,
            np.where(ndvi > 0.7, 0.1, 0.2)
        )
        return g_ratio * rn

    def _get_qa_mask(self, cube: DataCube) -> np.ndarray:
        """Create QA mask to exclude clouds and shadows."""
        qa_pixel = cube.get('qa_pixel')
        if qa_pixel is None:
            # If no QA data, assume all pixels are valid
            sample_band = next(iter(cube.data.values()))
            return ~np.isnan(sample_band.values)
        
        qa_values = qa_pixel.values if hasattr(qa_pixel, 'values') else qa_pixel
        
        # Create cloud/shadow mask (True = clear, False = cloudy/shadow)
        valid = np.ones(qa_values.shape, dtype=bool)
        
        # Handle NaN values (already masked)
        valid &= ~np.isnan(qa_values)
        
        # Check for cloud/shadow bits in QA pixel
        if qa_values.dtype.kind in ['u', 'i']:  # Integer types
            cloud_bit = ((qa_values >> 2) & 1).astype(bool)  # Cloud bit
            shadow_bit = ((qa_values >> 3) & 1).astype(bool)  # Cloud shadow bit
            dilated_bit = ((qa_values >> 6) & 1).astype(bool)  # Dilated cloud bit
            
            valid &= ~(cloud_bit | shadow_bit | dilated_bit)
        
        return valid

    def _create_border_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create border mask to exclude image borders."""
        rows, cols = shape
        border = self.constraints.border_pixels
        
        mask = np.ones(shape, dtype=bool)
        mask[:border, :] = False  # Top border
        mask[-border:, :] = False  # Bottom border
        mask[:, :border] = False  # Left border
        mask[:, -border:] = False  # Right border
        
        return mask

    def _apply_physical_constraints(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        lai: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        available_energy: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply physical constraints for cold and hot pixel selection.
        
        Returns:
            Tuple of (cold_mask, hot_mask) boolean arrays
        """
        self.logger.info("Step 1: Applying physical constraints")
        
        # Start with valid mask
        cold_mask = valid_mask.copy()
        hot_mask = valid_mask.copy()
        
        # Cold pixel constraints
        if ndvi is not None:
            cold_mask &= (ndvi > self.constraints.cold_ndvi_min)
        if lai is not None:
            cold_mask &= (lai > self.constraints.cold_lai_min)
        if albedo is not None:
            cold_mask &= (albedo > self.constraints.cold_albedo_min)
            cold_mask &= (albedo < self.constraints.cold_albedo_max)
        cold_mask &= (available_energy > self.constraints.cold_rn_minus_g_min)
        
        # Hot pixel constraints
        if ndvi is not None:
            hot_mask &= (ndvi < self.constraints.hot_ndvi_max)
        if lai is not None:
            hot_mask &= (lai < self.constraints.hot_lai_max)
        if albedo is not None:
            hot_mask &= (albedo > self.constraints.hot_albedo_min)
        
        cold_count = np.sum(cold_mask)
        hot_count = np.sum(hot_mask)
        
        self.logger.info(
            f"Physical constraints applied: {cold_count} cold candidates, "
            f"{hot_count} hot candidates"
        )
        
        return cold_mask, hot_mask

    def _apply_temperature_filtering(
        self,
        ts: np.ndarray,
        cold_mask: np.ndarray,
        hot_mask: np.ndarray,
        diagnostics: SelectionDiagnostics
    ) -> Tuple[np.ndarray, np.ndarray, SelectionDiagnostics]:
        """
        Apply temperature distribution filtering using percentiles.
        
        Returns:
            Tuple of (cold_candidates, hot_candidates, updated_diagnostics)
        """
        self.logger.info("Step 2: Applying temperature distribution filtering")
        
        # Cold pixel temperature threshold
        cold_ts_values = ts[cold_mask]
        if len(cold_ts_values) > 0:
            cold_percentile = self.constraints.cold_temp_percentile_max
            cold_threshold = np.percentile(cold_ts_values, cold_percentile)
            cold_candidates = cold_mask & (ts <= cold_threshold)
            diagnostics.cold_temp_percentile_used = cold_percentile
        else:
            cold_candidates = cold_mask
            diagnostics.cold_temp_percentile_used = 0.0
        
        # Hot pixel temperature threshold
        hot_ts_values = ts[hot_mask]
        if len(hot_ts_values) > 0:
            hot_percentile = self.constraints.hot_temp_percentile_min
            hot_threshold = np.percentile(hot_ts_values, hot_percentile)
            hot_candidates = hot_mask & (ts >= hot_threshold)
            diagnostics.hot_temp_percentile_used = hot_percentile
        else:
            hot_candidates = hot_mask
            diagnostics.hot_temp_percentile_used = 100.0
        
        diagnostics.cold_candidates_count = np.sum(cold_candidates)
        diagnostics.hot_candidates_count = np.sum(hot_candidates)
        
        self.logger.info(
            f"Temperature filtering applied: {diagnostics.cold_candidates_count} cold candidates, "
            f"{diagnostics.hot_candidates_count} hot candidates"
        )
        
        return cold_candidates, hot_candidates, diagnostics

    def _energy_based_selection(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        lai: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        available_energy: np.ndarray,
        cold_candidates: np.ndarray,
        hot_candidates: np.ndarray
    ) -> Tuple[Optional[AnchorPixel], Optional[AnchorPixel]]:
        """
        Energy-based final selection optimizing Rn-G values.
        
        Returns:
            Tuple of (cold_pixel, hot_pixel)
        """
        self.logger.info("Step 3: Energy-based final selection")
        
        # Cold pixel: maximize available energy (Rn - G)
        cold_pixel = None
        if np.any(cold_candidates):
            cold_available_energy = available_energy[cold_candidates]
            max_energy_idx = np.argmax(cold_available_energy)
            cold_coords = np.where(cold_candidates)
            cold_row, cold_col = cold_coords[0][max_energy_idx], cold_coords[1][max_energy_idx]
            
            cold_pixel = AnchorPixel(
                x=cold_col,
                y=cold_row,
                temperature=ts[cold_row, cold_col],
                ts=ts[cold_row, cold_col],
                ndvi=ndvi[cold_row, cold_col] if ndvi is not None else None,
                albedo=albedo[cold_row, cold_col] if albedo is not None else None
            )
            
            self.logger.info(
                f"Cold pixel selected: ({cold_col}, {cold_row}), "
                f"T={cold_pixel.temperature:.2f}K, Rn-G={cold_available_energy[max_energy_idx]:.2f} W/m²"
            )
        
        # Hot pixel: minimize available energy (Rn - G)
        hot_pixel = None
        if np.any(hot_candidates):
            hot_available_energy = available_energy[hot_candidates]
            min_energy_idx = np.argmin(hot_available_energy)
            hot_coords = np.where(hot_candidates)
            hot_row, hot_col = hot_coords[0][min_energy_idx], hot_coords[1][min_energy_idx]
            
            hot_pixel = AnchorPixel(
                x=hot_col,
                y=hot_row,
                temperature=ts[hot_row, hot_col],
                ts=ts[hot_row, hot_col],
                ndvi=ndvi[hot_row, hot_col] if ndvi is not None else None,
                albedo=albedo[hot_row, hot_col] if albedo is not None else None
            )
            
            self.logger.info(
                f"Hot pixel selected: ({hot_col}, {hot_row}), "
                f"T={hot_pixel.temperature:.2f}K, Rn-G={hot_available_energy[min_energy_idx]:.2f} W/m²"
            )
        
        return cold_pixel, hot_pixel

    def _quality_control(
        self,
        cube: DataCube,
        cold_pixel: Optional[AnchorPixel],
        hot_pixel: Optional[AnchorPixel],
        rn: np.ndarray,
        g_flux: np.ndarray,
        air_temperature: Optional[float],
        et0_inst: Optional[float],
        diagnostics: SelectionDiagnostics
    ) -> Tuple[Optional[AnchorPixel], Optional[AnchorPixel], SelectionDiagnostics]:
        """
        Quality control validation using ET calculations.
        
        Returns:
            Tuple of (validated_cold_pixel, validated_hot_pixel, updated_diagnostics)
        """
        self.logger.info("Step 4: Quality control validation")
        
        # Calculate energy balance at selected pixels
        if cold_pixel is not None:
            cold_rn = rn[cold_pixel.y, cold_pixel.x]
            cold_g = g_flux[cold_pixel.y, cold_pixel.x]
            cold_available_energy = cold_rn - cold_g
            
            # Cold pixel validation: ET ≈ ET0 (±10%)
            if et0_inst is not None and cold_available_energy > 0:
                # Estimate ET_inst from available energy (assuming EF ≈ 1 for cold pixel)
                estimated_et_inst = cold_available_energy / 2.45  # Convert W/m² to mm/hr
                et_ratio = estimated_et_inst / et0_inst if et0_inst > 0 else 0
                diagnostics.cold_etinst_vs_et0 = et_ratio
                
                # Check if ET is within ±10% of ET0
                if 0.9 <= et_ratio <= 1.1:
                    self.logger.info(f"Cold pixel ET validation PASS: ET/ET0 = {et_ratio:.3f}")
                else:
                    self.logger.warning(f"Cold pixel ET validation FAIL: ET/ET0 = {et_ratio:.3f} (expected 0.9-1.1)")
            
            # Cold pixel validation: dT ≈ 0-0.5K
            if air_temperature is not None:
                cold_dt = cold_pixel.temperature - air_temperature
                diagnostics.cold_dt_k = cold_dt
                
                if 0.0 <= cold_dt <= 0.5:
                    self.logger.info(f"Cold pixel dT validation PASS: dT = {cold_dt:.2f}K")
                else:
                    self.logger.warning(f"Cold pixel dT validation FAIL: dT = {cold_dt:.2f}K (expected 0-0.5K)")
        
        if hot_pixel is not None:
            hot_rn = rn[hot_pixel.y, hot_pixel.x]
            hot_g  = g_flux[hot_pixel.y, hot_pixel.x]
            hot_available_energy = hot_rn - hot_g

            # 1. Energy sanity
            if hot_available_energy <= 0:
                self.logger.warning("Hot pixel available energy ≤ 0 — invalid hot pixel")
                diagnostics.hot_valid = False
                hot_pixel = None

            # 2. NDVI sanity
            elif hot_pixel.ndvi is not None and hot_pixel.ndvi > 0.25:
                self.logger.warning(
                    f"Hot pixel NDVI {hot_pixel.ndvi:.3f} too high — likely not dry soil"
                )
                diagnostics.hot_valid = False
                hot_pixel = None

            # 3. Temperature sanity (relative check)
            elif cold_pixel is not None and hot_pixel.ts <= cold_pixel.ts:
                self.logger.warning("Hot pixel Ts ≤ cold pixel Ts — invalid ordering")
                diagnostics.hot_valid = False
                hot_pixel = None

            else:
                diagnostics.hot_valid = True
                self.logger.info(
                    f"Hot pixel accepted: Rn-G={hot_available_energy:.1f} W/m², "
                    f"Ts={hot_pixel.ts:.2f} K, NDVI={hot_pixel.ndvi:.3f}"
                )

            
            # Hot pixel validation: dT typically 10-20K
            if air_temperature is not None:
                hot_dt = hot_pixel.temperature - air_temperature
                diagnostics.hot_dt_k = hot_dt
                
                if 10.0 <= hot_dt <= 20.0:
                    self.logger.info(f"Hot pixel dT validation PASS: dT = {hot_dt:.2f}K")
                else:
                    self.logger.warning(f"Hot pixel dT validation FAIL: dT = {hot_dt:.2f}K (expected 10-20K)")
        
        # Energy balance closure check
        if cold_pixel is not None and hot_pixel is not None:
            # Calculate energy balance closure at both pixels
            cold_rn = rn[cold_pixel.y, cold_pixel.x]
            cold_g = g_flux[cold_pixel.y, cold_pixel.x]
            hot_rn = rn[hot_pixel.y, hot_pixel.x]
            hot_g = g_flux[hot_pixel.y, hot_pixel.x]
            
            cold_closure = cold_rn - cold_g  # Available energy
            hot_closure = hot_rn - hot_g     # Available energy
            
            diagnostics.energy_balance_closure = min(cold_closure, hot_closure)
            
            if diagnostics.energy_balance_closure > 50:  # W/m²
                self.logger.info(f"Energy balance closure adequate: {diagnostics.energy_balance_closure:.1f} W/m²")
            else:
                self.logger.warning(f"Energy balance closure low: {diagnostics.energy_balance_closure:.1f} W/m²")
        
        return cold_pixel, hot_pixel, diagnostics

    def _fallback_strategy(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        lai: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        available_energy: np.ndarray,
        cold_mask: np.ndarray,
        hot_mask: np.ndarray,
        diagnostics: SelectionDiagnostics
    ) -> Tuple[Optional[AnchorPixel], Optional[AnchorPixel], SelectionDiagnostics]:
        """
        Fallback strategy using weighted scoring when constraints fail.
        
        Returns:
            Tuple of (cold_pixel, hot_pixel, updated_diagnostics)
        """
        self.logger.info("Step 5: Fallback strategy with weighted scoring")
        
        # Check if we have any valid pixels to work with
        valid_pixels = ~np.isnan(ts)
        if not np.any(valid_pixels):
            diagnostics.fallback_used = True
            diagnostics.fallback_reason = "No valid pixels available"
            return None, None, diagnostics
        
        # Relax NDVI constraints and expand percentiles
        cold_relaxed_mask = cold_mask.copy()
        hot_relaxed_mask = hot_mask.copy()
        
        if ndvi is not None:
            # Relax NDVI limits by ±0.05
            cold_relaxed_mask &= (ndvi > (self.constraints.cold_ndvi_min - self.constraints.fallback_ndvi_relax))
            hot_relaxed_mask &= (ndvi < (self.constraints.hot_ndvi_max + self.constraints.fallback_ndvi_relax))
        
        # Expand temperature percentiles
        cold_ts_valid = ts[cold_relaxed_mask]
        hot_ts_valid = ts[hot_relaxed_mask]
        
        cold_pixel = None
        hot_pixel = None
        
        # Cold pixel selection using weighted scoring
        if np.any(cold_relaxed_mask) and len(cold_ts_valid) > 0:
            cold_percentile = min(15.0, self.constraints.cold_temp_percentile_max + self.constraints.fallback_percentile_expansion)
            cold_threshold = np.percentile(cold_ts_valid, cold_percentile)
            cold_candidates = cold_relaxed_mask & (ts <= cold_threshold)
            
            if np.any(cold_candidates):
                # Calculate weighted scores for cold pixels
                cold_scores = self._calculate_cold_pixel_scores(
                    ts, ndvi, lai, albedo, available_energy, cold_candidates
                )
                
                # Select pixel with highest score
                max_score_idx = np.argmax(cold_scores)
                cold_coords = np.where(cold_candidates)
                cold_row, cold_col = cold_coords[0][max_score_idx], cold_coords[1][max_score_idx]
                
                cold_pixel = AnchorPixel(
                    x=cold_col,
                    y=cold_row,
                    temperature=ts[cold_row, cold_col],
                    ts=ts[cold_row, cold_col],
                    ndvi=ndvi[cold_row, cold_col] if ndvi is not None else None,
                    albedo=albedo[cold_row, cold_col] if albedo is not None else None
                )
                
                self.logger.info(
                    f"Fallback cold pixel selected: ({cold_col}, {cold_row}), "
                    f"T={cold_pixel.temperature:.2f}K, score={cold_scores[max_score_idx]:.3f}"
                )
        
        # Hot pixel selection using weighted scoring
        if np.any(hot_relaxed_mask) and len(hot_ts_valid) > 0:
            hot_percentile = max(85.0, self.constraints.hot_temp_percentile_min - self.constraints.fallback_percentile_expansion)
            hot_threshold = np.percentile(hot_ts_valid, hot_percentile)
            hot_candidates = hot_relaxed_mask & (ts >= hot_threshold)
            
            if np.any(hot_candidates):
                # Calculate weighted scores for hot pixels
                hot_scores = self._calculate_hot_pixel_scores(
                    ts, ndvi, lai, albedo, available_energy, hot_candidates
                )
                
                # Select pixel with highest score
                max_score_idx = np.argmax(hot_scores)
                hot_coords = np.where(hot_candidates)
                hot_row, hot_col = hot_coords[0][max_score_idx], hot_coords[1][max_score_idx]
                
                hot_pixel = AnchorPixel(
                    x=hot_col,
                    y=hot_row,
                    temperature=ts[hot_row, hot_col],
                    ts=ts[hot_row, hot_col],
                    ndvi=ndvi[hot_row, hot_col] if ndvi is not None else None,
                    albedo=albedo[hot_row, hot_col] if albedo is not None else None
                )
                
                self.logger.info(
                    f"Fallback hot pixel selected: ({hot_col}, {hot_row}), "
                    f"T={hot_pixel.temperature:.2f}K, score={hot_scores[max_score_idx]:.3f}"
                )
        
        diagnostics.fallback_used = True
        diagnostics.fallback_reason = "Relaxed constraints applied"
        
        return cold_pixel, hot_pixel, diagnostics
    
    def _calculate_cold_pixel_scores(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        lai: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        available_energy: np.ndarray,
        candidates: np.ndarray
    ) -> np.ndarray:
        """
        Calculate weighted scores for cold pixel candidates.
        
        Higher scores indicate better cold pixel candidates.
        
        Returns:
            Array of scores for each candidate pixel
        """
        scores = np.zeros(np.sum(candidates), dtype=float)
        
        candidate_coords = np.where(candidates)
        candidate_indices = range(len(scores))
        
        for i, (row, col) in enumerate(zip(candidate_coords[0], candidate_coords[1])):
            score = 0.0
            
            # NDVI component (higher NDVI = higher score)
            if ndvi is not None:
                score += ndvi[row, col] * 2.0
            
            # LAI component (higher LAI = higher score)
            if lai is not None:
                score += lai[row, col] * 0.5
            
            # Temperature component (lower temperature = higher score)
            ts_normalized = 1.0 - (ts[row, col] - np.nanmin(ts)) / (np.nanmax(ts) - np.nanmin(ts))
            score += ts_normalized * 3.0
            
            # Available energy component (higher Rn-G = higher score)
            ae_normalized = (available_energy[row, col] - np.nanmin(available_energy[candidates])) / \
                           (np.nanmax(available_energy[candidates]) - np.nanmin(available_energy[candidates]))
            score += ae_normalized * 2.0
            
            # Albedo component (moderate albedo preferred)
            if albedo is not None:
                albedo_val = albedo[row, col]
                if 0.18 <= albedo_val <= 0.25:
                    score += 1.0
                else:
                    score += 0.5
            
            scores[i] = score
        
        return scores
    
    def _calculate_hot_pixel_scores(
        self,
        ts: np.ndarray,
        ndvi: Optional[np.ndarray],
        lai: Optional[np.ndarray],
        albedo: Optional[np.ndarray],
        available_energy: np.ndarray,
        candidates: np.ndarray
    ) -> np.ndarray:
        """
        Calculate weighted scores for hot pixel candidates.
        
        Higher scores indicate better hot pixel candidates.
        
        Returns:
            Array of scores for each candidate pixel
        """
        scores = np.zeros(np.sum(candidates), dtype=float)
        
        candidate_coords = np.where(candidates)
        candidate_indices = range(len(scores))
        
        for i, (row, col) in enumerate(zip(candidate_coords[0], candidate_coords[1])):
            score = 0.0
            
            # NDVI component (lower NDVI = higher score)
            if ndvi is not None:
                ndvi_normalized = 1.0 - ndvi[row, col]
                score += ndvi_normalized * 2.0
            
            # LAI component (lower LAI = higher score)
            if lai is not None:
                lai_normalized = 1.0 - (lai[row, col] / 6.0)  # Normalize to [0,1]
                score += lai_normalized * 0.5
            
            # Temperature component (higher temperature = higher score)
            ts_normalized = (ts[row, col] - np.nanmin(ts)) / (np.nanmax(ts) - np.nanmin(ts))
            score += ts_normalized * 3.0
            
            # Available energy component (lower Rn-G = higher score)
            ae_normalized = 1.0 - (available_energy[row, col] - np.nanmin(available_energy[candidates])) / \
                           (np.nanmax(available_energy[candidates]) - np.nanmin(available_energy[candidates]))
            score += ae_normalized * 2.0
            
            # Albedo component (higher albedo preferred for hot pixels)
            if albedo is not None:
                albedo_val = albedo[row, col]
                if albedo_val > 0.30:
                    score += 1.0
                else:
                    score += 0.5
            
            scores[i] = score
        
        return scores

    def _calculate_confidence(self, cold_pixel: AnchorPixel, hot_pixel: AnchorPixel) -> float:
        """Calculate selection confidence based on temperature difference."""
        temp_diff = hot_pixel.temperature - cold_pixel.temperature
        return min(1.0, temp_diff / 20.0)  # Normalize to 0-1 range


# Factory function for easy instantiation
def create_enhanced_selector(
    method: str = "enhanced_physical",
    constraints: Optional[PhysicalConstraints] = None,
    **kwargs
) -> EnhancedAnchorPixelSelector:
    """
    Factory function to create enhanced anchor pixel selector.
    
    Args:
        method: Selection method (currently only "enhanced_physical")
        constraints: Physical constraints configuration
        **kwargs: Additional arguments for selector initialization
        
    Returns:
        Configured EnhancedAnchorPixelSelector instance
    """
    if method != "enhanced_physical":
        raise ValueError(f"Unknown method: {method}. Use 'enhanced_physical'")
    
    return EnhancedAnchorPixelSelector(
        constraints=constraints,
        **kwargs
    )


__all__ = [
    'EnhancedAnchorPixelSelector',
    'PhysicalConstraints', 
    'SelectionDiagnostics',
    'create_enhanced_selector'
]
