"""
ET Quality Assessment Module for METRIC ETa Model.

This module provides quality flags and validation for ET calculations,
including physical bounds checking, spatial consistency, and quality classes.

METRIC Quality Classes:
    | Class       | ETrF Range   | Description                     |
    |-------------|--------------|---------------------------------|
    | Excellent   | 0.8 - 1.2    | Well-watered vegetation         |
    | Good        | 0.6 - 3.0    | Normal conditions               |
    | Acceptable  | 0.3 - 3.5    | Some water stress               |
    | Poor        | 0.0 - 0.3    | Extreme stress or bare soil     |
    | Uncertain   | <0 or >1.5   | Requires review                 |

Physical Constraints:
    - ET_min = 0 mm/day (no negative ET)
    - ET_max = 12 mm/day (upper physical limit)
    - ETrF_min = 0.0
    - ETrF_max = 1.3

Spatial Consistency:
    - Outlier threshold: 50% deviation from neighbors
    - Uses 8 surrounding pixels for neighborhood mean
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum


class ETQualityClass(IntEnum):
    """
    METRIC ET Quality Classes.
    
    Higher values indicate better quality.
    """
    UNCERTAIN = 0      # ETrF < 0 or > 1.5 - Requires review
    POOR = 1           # ETrF 0.0-0.3 - Extreme stress or bare soil
    ACCEPTABLE = 2     # ETrF 0.3-1.4 - Some water stress
    GOOD = 3           # ETrF 0.6-1.3 - Normal conditions
    EXCELLENT = 4      # ETrF 0.8-1.2 - Well-watered vegetation


@dataclass
class ETQualityConfig:
    """Configuration for ET quality assessment."""
    
    # Minimum valid ET (mm/day)
    min_et: float = 0.0
    
    # Maximum valid ET (mm/day)
    max_et: float = 30.0
    
    # Minimum valid ETrF
    min_etrf: float = 0.0
    
    # Maximum valid ETrF
    max_etrf: float = 2.0
    
    # Outlier deviation threshold (fraction)
    outlier_threshold: float = 0.5
    
    # Quality class thresholds
    excellent_min: float = 0.8
    excellent_max: float = 1.2
    good_min: float = 0.6
    good_max: float = 2.0
    acceptable_min: float = 0.3
    acceptable_max: float = 1.4
    
    # Minimum ET for good quality (mm/day)
    min_et_for_good: float = 0.1
    
    # Maximum ET for good quality (mm/day)
    max_et_for_good: float = 10.0


class ETQuality:
    """
    Assess quality of ET calculations.
    
    This class provides methods for:
    - Physical bounds validation
    - Spatial consistency checking
    - Quality class assignment
    - Outlier detection
    
    Attributes:
        config: Configuration parameters for quality assessment
    
    Example:
        >>> quality = ETQuality()
        >>> et_daily = np.array([5.0, 6.0, 7.0, 0.5, 15.0])
        >>> etrf = np.array([0.9, 0.95, 0.85, 0.1, 1.5])
        >>> result = quality.assess(et_daily, etrf)
    """
    
    def __init__(
        self,
        config: Optional[ETQualityConfig] = None
    ):
        """
        Initialize ETQuality assessor.
        
        Args:
            config: Optional configuration parameters. Uses defaults if not provided.
        """
        self.config = config or ETQualityConfig()
    
    def check_physical_bounds(
        self,
        et_daily: np.ndarray,
        etrf: Optional[np.ndarray] = None,
        et_inst: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Check if ET values are within physical bounds.
        
        Args:
            et_daily: Daily ET array (mm/day)
            etrf: Optional ETrF array
            et_inst: Optional instantaneous ET array (mm/hr)
            
        Returns:
            Dictionary with validation results
        """
        et_daily = np.asarray(et_daily, dtype=np.float64)
        
        # Check daily ET bounds
        valid_et = (et_daily >= self.config.min_et) & (et_daily <= self.config.max_et)
        
        result = {
            'et_daily': et_daily,
            'et_valid': valid_et,
            'et_below_min': et_daily < self.config.min_et,
            'et_above_max': et_daily > self.config.max_et,
            'et_violation_count': np.sum(~valid_et)
        }
        
        # Check ETrF bounds if provided
        if etrf is not None:
            etrf = np.asarray(etrf, dtype=np.float64)
            valid_etrf = (etrf >= self.config.min_etrf) & (etrf <= self.config.max_etrf)
            result['etrf'] = etrf
            result['etrf_valid'] = valid_etrf
            result['etrf_below_min'] = etrf < self.config.min_etrf
            result['etrf_above_max'] = etrf > self.config.max_etrf
            result['etrf_violation_count'] = np.sum(~valid_etrf)
        
        # Check instantaneous ET bounds if provided
        if et_inst is not None:
            et_inst = np.asarray(et_inst, dtype=np.float64)
            inst_max = self.config.max_et / 24.0  # Convert daily max to hourly
            valid_et_inst = (et_inst >= 0) & (et_inst <= inst_max)
            result['et_inst'] = et_inst
            result['et_inst_valid'] = valid_et_inst
        
        return result
    
    def assign_quality_class(
        self,
        etrf: np.ndarray,
        et_daily: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Assign quality class based on ETrF values.
        
        Args:
            etrf: Reference ET fraction array
            et_daily: Optional daily ET for additional validation
            
        Returns:
            Quality class array (0-4 scale)
        """
        etrf = np.asarray(etrf, dtype=np.float64)
        
        # Initialize quality class array
        quality = np.full(etrf.shape, ETQualityClass.UNCERTAIN, dtype=np.int32)
        
        # Excellent quality: 0.8-1.2
        excellent = (etrf >= self.config.excellent_min) & (etrf <= self.config.excellent_max)
        if et_daily is not None:
            et_daily = np.asarray(et_daily, dtype=np.float64)
            excellent &= (et_daily >= self.config.min_et_for_good)
            excellent &= (et_daily <= self.config.max_et_for_good)
        quality = np.where(excellent, ETQualityClass.EXCELLENT, quality)
        
        # Good quality: 0.6-2.0 (excluding excellent)
        good = (etrf >= self.config.good_min) & (etrf <= self.config.good_max)
        good &= ~excellent
        if et_daily is not None:
            good &= (et_daily >= self.config.min_et_for_good * 0.5)
            good &= (et_daily <= self.config.max_et_for_good * 1.2)
        quality = np.where(good, ETQualityClass.GOOD, quality)
        
        # Acceptable quality: 0.3-1.4 (excluding excellent and good)
        acceptable = (etrf >= self.config.acceptable_min) & (etrf <= self.config.acceptable_max)
        acceptable &= ~excellent & ~good
        quality = np.where(acceptable, ETQualityClass.ACCEPTABLE, quality)
        
        # Poor quality: 0.0-0.3 (excluding acceptable)
        poor = (etrf >= self.config.min_etrf) & (etrf < self.config.acceptable_min)
        poor &= ~acceptable
        quality = np.where(poor, ETQualityClass.POOR, quality)
        
        return quality
    
    def check_spatial_consistency(
        self,
        et_daily: np.ndarray,
        window_size: int = 3
    ) -> Dict[str, np.ndarray]:
        """
        Check spatial consistency using neighborhood statistics.
        
        Pixels deviating significantly from their neighbors are flagged as outliers.
        
        Args:
            et_daily: Daily ET array (mm/day)
            window_size: Window size for neighborhood (odd number)
            
        Returns:
            Dictionary with spatial consistency results
        """
        et_daily = np.asarray(et_daily, dtype=np.float64)
        
        # Calculate neighborhood mean using convolution
        kernel = np.ones((window_size, window_size)) / (window_size * window_size)
        
        # Handle NaN values in input
        et_filled = np.nan_to_num(et_daily, nan=0.0)
        
        # Calculate neighborhood mean
        from scipy import ndimage
        neighbor_mean = ndimage.convolve(et_filled, kernel, mode='reflect')
        
        # Count valid neighbors
        valid_mask = np.ones_like(et_daily, dtype=np.float64)
        valid_count = ndimage.convolve(valid_mask, kernel, mode='reflect')
        
        # Calculate mean only where there are valid neighbors
        neighbor_mean = np.where(valid_count > 0, neighbor_mean / valid_count, 0.0)
        
        # Calculate deviation from neighbors
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation = np.where(
                neighbor_mean > 0,
                np.abs(et_daily - neighbor_mean) / neighbor_mean,
                0.0
            )
        
        # Flag outliers
        is_outlier = deviation > self.config.outlier_threshold
        
        return {
            'et_daily': et_daily,
            'neighbor_mean': neighbor_mean,
            'deviation': deviation,
            'is_outlier': is_outlier,
            'outlier_count': np.sum(is_outlier),
            'outlier_fraction': np.mean(is_outlier)
        }
    
    def assess(
        self,
        et_daily: np.ndarray,
        etrf: np.ndarray,
        et_inst: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform comprehensive quality assessment.
        
        Args:
            et_daily: Daily ET array (mm/day)
            etrf: Reference ET fraction array
            et_inst: Optional instantaneous ET array (mm/hr)
            
        Returns:
            Dictionary with complete quality assessment results
        """
        et_daily = np.asarray(et_daily, dtype=np.float64)
        etrf = np.asarray(etrf, dtype=np.float64)
        
        # Physical bounds check
        bounds = self.check_physical_bounds(et_daily, etrf, et_inst)
        
        # Quality class assignment
        quality_class = self.assign_quality_class(etrf, et_daily)
        
        # Spatial consistency check
        spatial = self.check_spatial_consistency(et_daily)
        
        # Combine results
        result = {
            'et_daily': et_daily,
            'etrf': etrf,
            'quality_class': quality_class,
            'physical_bounds_valid': bounds['et_valid'],
            'etrf_valid': bounds.get('etrf_valid', None),
            'spatial_outliers': spatial['is_outlier'],
            'deviation': spatial['deviation'],
            'overall_valid': bounds['et_valid'] & ~spatial['is_outlier']
        }
        
        # Add statistics
        result['et_mean'] = np.nanmean(et_daily)
        result['et_std'] = np.nanstd(et_daily)
        result['et_min'] = np.nanmin(et_daily)
        result['et_max'] = np.nanmax(et_daily)
        
        result['etrf_mean'] = np.nanmean(etrf)
        result['etrf_std'] = np.nanstd(etrf)
        
        result['quality_distribution'] = {
            'excellent': np.sum(quality_class == ETQualityClass.EXCELLENT),
            'good': np.sum(quality_class == ETQualityClass.GOOD),
            'acceptable': np.sum(quality_class == ETQualityClass.ACCEPTABLE),
            'poor': np.sum(quality_class == ETQualityClass.POOR),
            'uncertain': np.sum(quality_class == ETQualityClass.UNCERTAIN)
        }
        
        result['total_pixels'] = et_daily.size
        result['valid_pixels'] = np.sum(result['overall_valid'])
        result['valid_fraction'] = np.mean(result['overall_valid'])
        
        return result
    
    def get_quality_description(self, quality_class: int) -> str:
        """
        Get description for quality class.
        
        Args:
            quality_class: Quality class value (0-4)
            
        Returns:
            Description string
        """
        descriptions = {
            ETQualityClass.EXCELLENT: "Well-watered vegetation",
            ETQualityClass.GOOD: "Normal conditions",
            ETQualityClass.ACCEPTABLE: "Some water stress",
            ETQualityClass.POOR: "Extreme stress or bare soil",
            ETQualityClass.UNCERTAIN: "Requires review"
        }
        return descriptions.get(quality_class, "Unknown")
    
    def flag_low_quality_pixels(
        self,
        etrf: np.ndarray,
        et_daily: np.ndarray,
        quality_threshold: ETQualityClass = ETQualityClass.GOOD
    ) -> np.ndarray:
        """
        Flag pixels below a quality threshold.
        
        Args:
            etrf: Reference ET fraction array
            et_daily: Daily ET array
            quality_threshold: Minimum acceptable quality class
            
        Returns:
            Boolean array where True indicates low quality
        """
        quality_class = self.assign_quality_class(etrf, et_daily)
        return quality_class < quality_threshold
    
    def apply_quality_filter(
        self,
        et_daily: np.ndarray,
        etrf: np.ndarray,
        quality_threshold: ETQualityClass = ETQualityClass.GOOD
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter ET values to keep only high-quality pixels.
        
        Args:
            et_daily: Daily ET array
            etrf: Reference ET fraction array
            quality_threshold: Minimum acceptable quality class
            
        Returns:
            Tuple of (filtered_et_daily, filtered_etrf)
        """
        quality_class = self.assign_quality_class(etrf, et_daily)
        valid_mask = quality_class >= quality_threshold
        return et_daily * valid_mask, etrf * valid_mask
    
    def __call__(
        self,
        et_daily: np.ndarray,
        etrf: np.ndarray,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Convenience method to assess ET quality.
        """
        return self.assess(et_daily, etrf, **kwargs)


def create_et_quality(
    min_et: float = 0.0,
    max_et: float = 30.0,
    min_etrf: float = 0.0,
    max_etrf: float = 2.0,
    outlier_threshold: float = 0.5,
    **kwargs
) -> ETQuality:
    """
    Factory function to create ETQuality instance.
    
    Args:
        min_et: Minimum valid ET (mm/day)
        max_et: Maximum valid ET (mm/day)
        min_etrf: Minimum valid ETrF
        max_etrf: Maximum valid ETrF
        outlier_threshold: Outlier deviation threshold
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured ETQuality instance
    """
    config = ETQualityConfig(
        min_et=min_et,
        max_et=max_et,
        min_etrf=min_etrf,
        max_etrf=max_etrf,
        outlier_threshold=outlier_threshold,
        **kwargs
    )
    return ETQuality(config)
