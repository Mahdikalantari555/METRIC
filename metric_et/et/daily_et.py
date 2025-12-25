"""
Daily Evapotranspiration (ET) Calculation for METRIC ETa Model.

This module scales instantaneous ET measurements to daily ET totals
using the reference ET fraction (ETrF) approach.

Formulas:
    ET_daily = ETrF × ETr_daily
    
Where:
    - ETrF = reference ET fraction (dimensionless)
    - ETr_daily = daily reference ET for alfalfa (mm/day)

Alternative Approaches:
    1. Simple scaling: ET_daily = ETrF × ETr_daily
    2. With daylight fraction: ET_daily = ETrF × ETr_daily × f_day
    3. From instantaneous: ET_daily = ET_inst × (ETr_daily / ETr_inst)

Time Factors:
    - Landsat overpass: ~10:30 local time
    - Instantaneous ET represents ~1-2 hours around overpass
    - ETr_daily can be estimated from ETr_inst × 24 (simplified)

Physical Constraints:
    - ET_min = 0 mm/day (no negative ET)
    - ET_max = 12 mm/day (upper physical limit, tropical rainforest)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from metric_et.core.constants import (
    HOURS_PER_DAY,
    SECONDS_PER_HOUR
)


@dataclass
class DailyETConfig:
    """Configuration for daily ET calculation."""
    
    # Minimum daily ET (mm/day)
    min_et_daily: float = 0.0
    
    # Maximum daily ET (mm/day) - upper physical limit
    max_et_daily: float = 12.0
    
    # Time factor for ETr extrapolation (default = 24)
    time_factor: float = 24.0
    
    # Fraction of daylight hours (optional)
    daylight_fraction: float = 1.0
    
    # Minimum valid ETrF
    min_etrf: float = 0.0
    
    # Maximum valid ETrF
    max_etrf: float = 1.3


class DailyET:
    """
    Scale instantaneous ET to daily ET totals.
    
    This class implements the conversion from instantaneous ET measurements
    at satellite overpass time to daily ET totals using the reference ET
    fraction (ETrF) approach.
    
    Attributes:
        config: Configuration parameters for daily ET calculation
    
    Example:
        >>> daily_et = DailyET()
        >>> etrf = np.array([0.8, 0.6, 0.9])
        >>> etr_daily = 5.0  # mm/day
        >>> et_daily = daily_et.calculate(etrf, etr_daily)
    """
    
    def __init__(
        self,
        config: Optional[DailyETConfig] = None
    ):
        """
        Initialize DailyET calculator.

        Args:
            config: Optional configuration parameters. Uses defaults if not provided.
        """
        self.config = config or DailyETConfig()

    def _to_numpy(self, arr):
        """Convert array to numpy if it's xarray DataArray."""
        if hasattr(arr, 'values'):
            return np.asarray(arr.values, dtype=np.float64)
        else:
            return np.asarray(arr, dtype=np.float64)
    
    def calculate_daily_et(
        self,
        etrf: np.ndarray,
        etr_daily: np.ndarray
    ) -> np.ndarray:
        """
        Calculate daily ET from reference ET fraction.
        
        ET_daily = ETrF × ETr_daily
        
        Args:
            etrf: Reference ET fraction (dimensionless)
            etr_daily: Daily reference ET for alfalfa (mm/day)
            
        Returns:
            Daily actual ET (mm/day)
        """
        etrf = self._to_numpy(etrf)
        etr_daily = self._to_numpy(etr_daily)
        
        # Calculate daily ET
        et_daily = etrf * etr_daily
        
        # Apply physical bounds
        et_daily = np.clip(
            et_daily,
            self.config.min_et_daily,
            self.config.max_et_daily
        )
        
        return et_daily
    
    def calculate_with_daylight_fraction(
        self,
        etrf: np.ndarray,
        etr_daily: np.ndarray,
        daylight_fraction: float
    ) -> np.ndarray:
        """
        Calculate daily ET with daylight fraction adjustment.
        
        ET_daily = ETrF × ETr_daily × f_day
        
        This method accounts for the fact that ET primarily occurs
        during daylight hours.
        
        Args:
            etrf: Reference ET fraction (dimensionless)
            etr_daily: Daily reference ET for alfalfa (mm/day)
            daylight_fraction: Fraction of daylight hours (0-1)
            
        Returns:
            Daily actual ET (mm/day)
        """
        etrf = self._to_numpy(etrf)
        etr_daily = self._to_numpy(etr_daily)
        
        # Apply daylight fraction
        et_daily = etrf * etr_daily * daylight_fraction
        
        # Apply physical bounds
        et_daily = np.clip(
            et_daily,
            self.config.min_et_daily,
            self.config.max_et_daily
        )
        
        return et_daily
    
    def estimate_etr_daily_from_inst(
        self,
        etr_inst: np.ndarray
    ) -> np.ndarray:
        """
        Estimate daily ETr from instantaneous ETr.
        
        ETr_daily = ETr_inst × K_T
        
        Where K_T is the time factor, typically 24 for constant conditions.
        
        Args:
            etr_inst: Instantaneous reference ET (mm/hr)
            
        Returns:
            Estimated daily reference ET (mm/day)
        """
        etr_inst = self._to_numpy(etr_inst)
        return etr_inst * self.config.time_factor
    
    def scale_instantaneous_to_daily(
        self,
        et_inst: np.ndarray,
        etr_inst: np.ndarray,
        etr_daily: np.ndarray
    ) -> np.ndarray:
        """
        Scale instantaneous ET to daily ET using reference ET ratio.
        
        ET_daily = ET_inst × (ETr_daily / ETr_inst)
        
        This method uses the ratio of daily to instantaneous reference ET
        to scale the instantaneous actual ET.
        
        Args:
            et_inst: Instantaneous ET rate (mm/hr)
            etr_inst: Instantaneous reference ET (mm/hr)
            etr_daily: Daily reference ET (mm/day)
            
        Returns:
            Daily actual ET (mm/day)
        """
        et_inst = self._to_numpy(et_inst)
        etr_inst = self._to_numpy(etr_inst)
        etr_daily = self._to_numpy(etr_daily)
        
        # Calculate scaling factor
        scaling_factor = np.where(
            etr_inst > 0.01,
            etr_daily / etr_inst,
            0.0
        )
        
        # Calculate daily ET
        et_daily = et_inst * scaling_factor
        
        # Apply physical bounds
        et_daily = np.clip(
            et_daily,
            self.config.min_et_daily,
            self.config.max_et_daily
        )
        
        return et_daily
    
    def calculate(
        self,
        etrf: np.ndarray,
        etr_daily: np.ndarray,
        use_daylight_fraction: bool = False,
        daylight_fraction: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate daily ET from ETrF and daily reference ET.

        Args:
            etrf: Reference ET fraction (dimensionless)
            etr_daily: Daily reference ET for alfalfa (mm/day)
            use_daylight_fraction: Whether to apply daylight fraction adjustment
            daylight_fraction: Optional daylight fraction (0-1)

        Returns:
            Dictionary containing:
                - 'ET_daily': Daily actual ET (mm/day)
                - 'ETrF': Input reference ET fraction
                - 'ETr_daily': Input daily reference ET
        """
        import logging
        logger = logging.getLogger(__name__)

        etrf = self._to_numpy(etrf)
        etr_daily = self._to_numpy(etr_daily)

        # DEBUG: Log spatial variation statistics
        logger.info("ET_daily calculate - Input analysis:")
        logger.info(f"  ETrF shape: {etrf.shape}")
        logger.info(f"  ETr_daily shape: {etr_daily.shape}")

        # ETrF statistics
        valid_etrf = etrf[~np.isnan(etrf)]
        if len(valid_etrf) > 0:
            logger.info(f"  ETrF stats - Min: {np.min(valid_etrf):.6f}, Max: {np.max(valid_etrf):.6f}, "
                       f"Mean: {np.mean(valid_etrf):.6f}, Std: {np.std(valid_etrf):.6f}")
            logger.info(f"  ETrF unique values: {len(np.unique(valid_etrf))}")
            logger.info(f"  ETrF clipped to max (1.05): {np.sum(valid_etrf >= 1.05)} pixels")
            logger.info(f"  ETrF clipped to min (0.0): {np.sum(valid_etrf <= 0.0)} pixels")

        # ETr_daily statistics
        if np.isscalar(etr_daily) or etr_daily.ndim == 0:
            logger.info(f"  ETr_daily is scalar: {float(etr_daily):.6f}")
        else:
            valid_etr_daily = etr_daily[~np.isnan(etr_daily)]
            if len(valid_etr_daily) > 0:
                logger.info(f"  ETr_daily stats - Min: {np.min(valid_etr_daily):.6f}, Max: {np.max(valid_etr_daily):.6f}, "
                           f"Mean: {np.mean(valid_etr_daily):.6f}, Std: {np.std(valid_etr_daily):.6f}")
                logger.info(f"  ETr_daily unique values: {len(np.unique(valid_etr_daily))}")

        if use_daylight_fraction and daylight_fraction is not None:
            et_daily = self.calculate_with_daylight_fraction(
                etrf, etr_daily, daylight_fraction
            )
        else:
            et_daily = self.calculate_daily_et(etrf, etr_daily)

        # ET_daily statistics
        valid_et_daily = et_daily[~np.isnan(et_daily)]
        if len(valid_et_daily) > 0:
            logger.info(f"  ET_daily result stats - Min: {np.min(valid_et_daily):.6f}, Max: {np.max(valid_et_daily):.6f}, "
                       f"Mean: {np.mean(valid_et_daily):.6f}, Std: {np.std(valid_et_daily):.6f}")
            logger.info(f"  ET_daily unique values: {len(np.unique(valid_et_daily))}")
            logger.info(f"  ET_daily clipped to max (12.0): {np.sum(valid_et_daily >= 12.0)} pixels")
            logger.info(f"  ET_daily clipped to min (0.0): {np.sum(valid_et_daily <= 0.0)} pixels")

            # Check for spatial variation loss
            max_mean_diff = abs(np.max(valid_et_daily) - np.mean(valid_et_daily))
            if max_mean_diff < 0.01:
                logger.warning(f"  CRITICAL: ET_daily has minimal spatial variation (Max-Mean = {max_mean_diff:.6f})")
                logger.warning("  This indicates ETrF has no spatial variation or calculation error")
            else:
                logger.info(f"  OK: ET_daily has spatial variation (Max-Mean = {max_mean_diff:.6f})")

        return {
            'ET_daily': et_daily,
            'ETrF': etrf,
            'ETr_daily': etr_daily
        }
    
    def calculate_full(
        self,
        et_inst: np.ndarray,
        etr_inst: np.ndarray,
        etr_daily: np.ndarray,
        et0_daily: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate daily ET with all available information.
        
        Args:
            et_inst: Instantaneous ET rate (mm/hr)
            etr_inst: Instantaneous reference ET for alfalfa (mm/hr)
            etr_daily: Daily reference ET for alfalfa (mm/day)
            et0_daily: Optional daily reference ET for grass (mm/day)
            
        Returns:
            Dictionary containing all ET-related parameters
        """
        et_inst = self._to_numpy(et_inst)
        etr_inst = self._to_numpy(etr_inst)
        etr_daily = self._to_numpy(etr_daily)
        
        # Calculate ETrF from instantaneous values
        etrf = np.where(
            etr_inst > 0.01,
            et_inst / etr_inst,
            0.0
        )
        etrf = np.clip(etrf, self.config.min_etrf, self.config.max_etrf)
        
        # Calculate daily ET using ETrF method
        et_daily = self.calculate_daily_et(etrf, etr_daily)
        
        result = {
            'ET_inst': et_inst,
            'ET_daily': et_daily,
            'ETrF': etrf,
            'ETr_inst': etr_inst,
            'ETr_daily': etr_daily
        }
        
        # Add grass reference ET if available
        if et0_daily is not None:
            et0_daily = self._to_numpy(et0_daily)
            result['ETo_daily'] = et0_daily
            result['EToF'] = np.where(
                et0_daily > 0.01,
                et_daily / et0_daily,
                0.0
            )
        
        return result
    
    def calculate_et_from_le(
        self,
        le: np.ndarray,
        etr_daily: np.ndarray,
        lambda_vaporization: float = 2.45e6
    ) -> Dict[str, np.ndarray]:
        """
        Calculate daily ET directly from latent heat flux.
        
        This method combines instantaneous ET calculation and daily scaling.
        
        Args:
            le: Latent heat flux (W/m²)
            etr_daily: Daily reference ET for alfalfa (mm/day)
            lambda_vaporization: Latent heat of vaporization (J/kg)
            
        Returns:
            Dictionary with ET calculations
        """
        le = self._to_numpy(le)
        etr_daily = self._to_numpy(etr_daily)
        
        # Convert LE to instantaneous ET (mm/hr)
        et_inst = le * SECONDS_PER_HOUR / lambda_vaporization
        
        # Estimate instantaneous ETr from daily ETr
        etr_inst = etr_daily / self.config.time_factor
        
        # Calculate ETrF
        etrf = np.where(
            etr_inst > 0.01,
            et_inst / etr_inst,
            0.0
        )
        etrf = np.clip(etrf, self.config.min_etrf, self.config.max_etrf)
        
        # Calculate daily ET
        et_daily = self.calculate_daily_et(etrf, etr_daily)
        
        return {
            'LE': le,
            'ET_inst': et_inst,
            'ET_daily': et_daily,
            'ETrF': etrf,
            'ETr_inst': etr_inst,
            'ETr_daily': etr_daily
        }
    
    def monthly_total(
        self,
        et_daily: np.ndarray,
        days_in_month: int
    ) -> np.ndarray:
        """
        Calculate monthly ET total from daily values.
        
        Args:
            et_daily: Daily ET array (mm/day)
            days_in_month: Number of days in the month
            
        Returns:
            Monthly ET total (mm/month)
        """
        et_daily = self._to_numpy(et_daily)
        return et_daily * days_in_month
    
    def seasonal_total(
        self,
        et_daily: np.ndarray,
        days_in_season: int
    ) -> np.ndarray:
        """
        Calculate seasonal ET total from daily values.
        
        Args:
            et_daily: Daily ET array (mm/day)
            days_in_season: Number of days in the season
            
        Returns:
            Seasonal ET total (mm/season)
        """
        et_daily = self._to_numpy(et_daily)
        return et_daily * days_in_season
    
    def compute(self, cube):
        """
        Compute daily ET and add to DataCube.

        Args:
            cube: DataCube with ETrF and ETr_daily

        Returns:
            DataCube with added ET_daily
        """
        import logging
        from ..core.datacube import DataCube

        logger = logging.getLogger(__name__)

        # Get required inputs
        etrf = cube.get("ETrF")
        etr_daily = cube.get("ETr_daily")

        if etrf is None:
            raise ValueError("ETrF not found in DataCube")
        if etr_daily is None:
            raise ValueError("ETr_daily not found in DataCube")

        # Helper function to get values, handling scalars
        def get_values(data):
            if hasattr(data, 'values'):
                return data.values
            else:
                # Scalar: create array of same shape as etrf
                return np.full_like(etrf.values, data, dtype=np.float64)

        etrf_values = get_values(etrf)
        etr_daily_values = get_values(etr_daily)

        # DEBUG: Log spatial variation statistics
        logger.info("ET_daily calculation - Input analysis:")
        logger.info(f"  ETrF shape: {etrf_values.shape}")
        logger.info(f"  ETr_daily shape: {etr_daily_values.shape}")

        # ETrF statistics
        valid_etrf = etrf_values[~np.isnan(etrf_values)]
        if len(valid_etrf) > 0:
            logger.info(f"  ETrF stats - Min: {np.min(valid_etrf):.6f}, Max: {np.max(valid_etrf):.6f}, "
                       f"Mean: {np.mean(valid_etrf):.6f}, Std: {np.std(valid_etrf):.6f}")
            logger.info(f"  ETrF unique values: {len(np.unique(valid_etrf))}")
            logger.info(f"  ETrF clipped to max (1.05): {np.sum(valid_etrf >= 1.05)} pixels")
            logger.info(f"  ETrF clipped to min (0.0): {np.sum(valid_etrf <= 0.0)} pixels")

        # ETr_daily statistics
        if np.isscalar(etr_daily_values) or etr_daily_values.ndim == 0:
            logger.info(f"  ETr_daily is scalar: {float(etr_daily_values):.6f}")
        else:
            valid_etr_daily = etr_daily_values[~np.isnan(etr_daily_values)]
            if len(valid_etr_daily) > 0:
                logger.info(f"  ETr_daily stats - Min: {np.min(valid_etr_daily):.6f}, Max: {np.max(valid_etr_daily):.6f}, "
                           f"Mean: {np.mean(valid_etr_daily):.6f}, Std: {np.std(valid_etr_daily):.6f}")
                logger.info(f"  ETr_daily unique values: {len(np.unique(valid_etr_daily))}")

        # Calculate daily ET
        result = self.calculate(etrf_values, etr_daily_values)

        # ET_daily statistics
        et_daily_values = result["ET_daily"]
        valid_et_daily = et_daily_values[~np.isnan(et_daily_values)]
        if len(valid_et_daily) > 0:
            logger.info(f"  ET_daily result stats - Min: {np.min(valid_et_daily):.6f}, Max: {np.max(valid_et_daily):.6f}, "
                       f"Mean: {np.mean(valid_et_daily):.6f}, Std: {np.std(valid_et_daily):.6f}")
            logger.info(f"  ET_daily unique values: {len(np.unique(valid_et_daily))}")
            logger.info(f"  ET_daily clipped to max (12.0): {np.sum(valid_et_daily >= 12.0)} pixels")
            logger.info(f"  ET_daily clipped to min (0.0): {np.sum(valid_et_daily <= 0.0)} pixels")

            # Check for spatial variation loss
            max_mean_diff = abs(np.max(valid_et_daily) - np.mean(valid_et_daily))
            if max_mean_diff < 0.01:
                logger.warning(f"  CRITICAL: ET_daily has minimal spatial variation (Max-Mean = {max_mean_diff:.6f})")
                logger.warning("  This indicates ETrF has no spatial variation or calculation error")
            else:
                logger.info(f"  OK: ET_daily has spatial variation (Max-Mean = {max_mean_diff:.6f})")

        # Add to cube
        cube.add("ET_daily", result["ET_daily"])

        return cube

    def __call__(
        self,
        etrf: np.ndarray,
        etr_daily: np.ndarray,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Convenience method to calculate daily ET.
        """
        return self.calculate(etrf, etr_daily, **kwargs)


def create_daily_et(
    min_et_daily: float = 0.0,
    max_et_daily: float = 12.0,
    time_factor: float = 24.0,
    **kwargs
) -> DailyET:
    """
    Factory function to create DailyET instance.
    
    Args:
        min_et_daily: Minimum daily ET (mm/day)
        max_et_daily: Maximum daily ET (mm/day)
        time_factor: Time factor for ETr extrapolation
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured DailyET instance
    """
    config = DailyETConfig(
        min_et_daily=min_et_daily,
        max_et_daily=max_et_daily,
        time_factor=time_factor,
        **kwargs
    )
    return DailyET(config)
