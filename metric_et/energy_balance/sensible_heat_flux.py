"""
Sensible Heat Flux (H) Calculation for METRIC ETa Model.

This module implements the sensible heat flux component of the energy balance equation.
H represents the energy transferred between surface and air by convection/conduction.

Basic Equation:
    H = ρ * cp * (Ts - Ta) / rah

Where:
    ρ = air density (kg/m³)
    cp = specific heat of air (1013 J/kg/K)
    Ts = surface temperature (K)
    Ta = air temperature (K)
    rah = aerodynamic resistance (s/m)

METRIC Calibration Approach:
    When calibrated, H = a * dT, where dT = Ts - Ta, and a is derived from hot pixel.

    When not calibrated, H = ρ * cp * dT / rah, with dT calibrated or empirical.

Aerodynamic Resistance (rah):
    rah = ln((z_m - d) / z0m) * ln((z_h - d) / z0h) / (k² * u)

Stability Corrections (Monin-Obukhov):
    For unstable conditions (daytime):
        ψ_m = 2 * ln((1 + x)/2) + ln((1 + x²)/2) - 2*atan(x) + π/2
        ψ_h = 2 * ln((1 + y)/2) + ln((1 + y²)/2)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from metric_et.core.constants import (
    VON_KARMAN, AIR_DENSITY, AIR_SPECIFIC_HEAT,
    GRAVITATIONAL_ACCELERATION, FREEZING_POINT
)


@dataclass
class SensibleHeatFluxConfig:
    """Configuration for sensible heat flux calculation."""
    # Measurement heights
    z_wind: float = 10.0  # Wind measurement height (m)
    z_temp: float = 2.0   # Temperature measurement height (m)
    
    # Stability correction parameters
    use_stability_correction: bool = True
    stability_iterations: int = 5
    
    # Anchor pixel calibration (dT coefficients)
    # dT = a * (Ts - Ta) + b
    dt_a: Optional[float] = None
    dt_b: Optional[float] = None
    
    # Minimum/maximum constraints
    min_rah: float = 10.0  # s/m
    max_rah: float = 500.0  # s/m
    min_h: float = -500.0  # W/m²
    max_h: float = 3000.0  # W/m²
    
    # Roughness length ratio (z0h/z0m)
    roughness_ratio: float = 0.1


class SensibleHeatFlux:
    """
    Calculate sensible heat flux (H) for METRIC energy balance.
    
    The sensible heat flux represents the energy transferred between
    the surface and the atmosphere as heat. This is the most complex
    component of the METRIC energy balance due to the need for
    stability corrections and anchor pixel calibration.
    
    Attributes:
        config: Configuration parameters for H calculation
    """
    
    def __init__(self, config: Optional[SensibleHeatFluxConfig] = None):
        """
        Initialize SensibleHeatFlux calculator.
        
        Args:
            config: Optional configuration parameters. Uses defaults if not provided.
        """
        self.config = config or SensibleHeatFluxConfig()
    
    def calculate_air_density(
        self,
        ta_kelvin: np.ndarray,
        pressure_pa: np.ndarray
    ) -> np.ndarray:
        """
        Calculate air density using ideal gas law.
        
        ρ = P / (R_specific * T)
        
        Args:
            ta_kelvin: Air temperature in Kelvin
            pressure_pa: Atmospheric pressure in Pa
            
        Returns:
            Air density in kg/m³
        """
        # Specific gas constant for dry air (J/kg/K)
        R_specific = 287.058
        
        # Handle potential division by zero
        ta_safe = np.maximum(ta_kelvin, 100.0)
        
        air_density = pressure_pa / (R_specific * ta_safe)
        
        # Physical bounds
        air_density = np.clip(air_density, 0.5, 2.0)
        
        return air_density
    
    def calculate_displacement_height(
        self,
        z0m: np.ndarray,
        lai: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate displacement height (d) from roughness length.
        
        d ≈ 2/3 * z0m * h / k (simplified)
        For vegetation: d ≈ 0.67 * z0m * LAI^0.5 (empirical)
        
        Args:
            z0m: Roughness length for momentum (m)
            lai: Leaf Area Index (optional)
            
        Returns:
            Displacement height (m)
        """
        if lai is not None:
            # Empirical relationship for vegetated surfaces
            d = 0.67 * z0m * np.sqrt(np.maximum(lai, 0))
        else:
            # Simplified: d ≈ 2/3 * z0m * 100 (empirical factor)
            d = 2.0/3.0 * z0m * 100.0
        
        # Physical bounds - ensure d doesn't exceed reasonable limits
        d = np.clip(d, 0.0, np.maximum(z0m * 10.0, 0.1))  # More conservative upper bound
        
        return d
    
    def calculate_neutral_rah(
        self,
        z0m: np.ndarray,
        z0h: np.ndarray,
        d: np.ndarray,
        u: np.ndarray,
        z_wind: float,
        z_temp: float
    ) -> np.ndarray:
        """
        Calculate aerodynamic resistance under neutral conditions.
        
        rah = ln((z_m - d) / z0m) * ln((z_h - d) / z0h) / (k² * u)
        
        Args:
            z0m: Roughness length for momentum (m)
            z0h: Roughness length for heat (m)
            d: Displacement height (m)
            u: Wind speed (m/s)
            z_wind: Wind measurement height (m)
            z_temp: Temperature measurement height (m)
            
        Returns:
            Aerodynamic resistance (s/m) under neutral conditions
        """
        k2 = VON_KARMAN ** 2
        
        # Calculate log profiles
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Calculating rah: z_wind={z_wind}, z_temp={z_temp}")
        logger.info(f"Calculating rah: d range min={np.nanmin(d):.3f}, max={np.nanmax(d):.3f}")
        logger.info(f"Calculating rah: z0m range min={np.nanmin(z0m):.3f}, max={np.nanmax(z0m):.3f}")
        logger.info(f"Calculating rah: z0h range min={np.nanmin(z0h):.3f}, max={np.nanmax(z0h):.3f}")
        logger.info(f"Calculating rah: u range min={np.nanmin(u):.3f}, max={np.nanmax(u):.3f}")
        
        # Ensure valid arguments for log functions
        z0m_safe = np.maximum(z0m, 0.0001)  # Minimum roughness length
        z0h_safe = np.maximum(z0h, 0.00001)  # Minimum heat roughness length
        
        # Ensure displacement height doesn't exceed measurement heights
        d_safe = np.minimum(d, z_wind - z0m_safe - 0.1)  # Leave margin
        
        ln_zm_z0m = np.log(np.maximum((z_wind - d_safe) / z0m_safe, 1.001))
        ln_zh_z0h = np.log(np.maximum((z_temp - d_safe) / z0h_safe, 1.001))
        
        logger.info(f"Calculating rah: ln_zm_z0m range min={np.nanmin(ln_zm_z0m):.3f}, max={np.nanmax(ln_zm_z0m):.3f}")
        logger.info(f"Calculating rah: ln_zh_z0h range min={np.nanmin(ln_zh_z0h):.3f}, max={np.nanmax(ln_zh_z0h):.3f}")
        
        # Calculate rah
        rah_neutral = (ln_zm_z0m * ln_zh_z0h) / (k2 * np.maximum(u, 0.1))
        
        logger.info(f"Calculating rah: rah_neutral range min={np.nanmin(rah_neutral):.3f}, max={np.nanmax(rah_neutral):.3f}")
        
        # Apply physical bounds
        rah_neutral = np.clip(
            rah_neutral,
            self.config.min_rah,
            self.config.max_rah
        )
        
        return rah_neutral
    
    def calculate_stability_parameter(
        self,
        H: np.ndarray,
        rah: np.ndarray,
        rho: np.ndarray,
        ta_kelvin: np.ndarray,
        z_measure: float = 2.0
    ) -> np.ndarray:
        """
        Calculate stability parameter (z/L).
        
        z/L = g * z * H / (ρ * cp * T * u*³)
        
        Simplified for iterative solution:
        z/L ≈ g * z * rah * H / (ρ * cp * T * k² * u²)
        
        Args:
            H: Sensible heat flux (W/m²)
            rah: Aerodynamic resistance (s/m)
            rho: Air density (kg/m³)
            ta_kelvin: Air temperature (K)
            z_measure: Measurement height (m)
            
        Returns:
            Stability parameter (z/L)
        """
        # Richardson number approximation
        richardson = (
            GRAVITATIONAL_ACCELERATION * z_measure * H /
            (ta_kelvin * AIR_SPECIFIC_HEAT * np.maximum(rah, 1.0))
        )
        
        # Convert to z/L (approximate relationship)
        z_L = richardson * 10  # Empirical scaling
        
        return z_L
    
    def calculate_psi_functions(
        self,
        z_L: np.ndarray,
        z: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate stability correction functions (ψ_m, ψ_h).
        
        For unstable conditions (z/L < 0):
            ψ_m = 2 * ln((1 + x)/2) + ln((1 + x²)/2) - 2*atan(x) + π/2
            ψ_h = 2 * ln((1 + y)/2) + ln((1 + y²)/2)
            
            where x = (1 - 16*z/L)^0.25, y = (1 - 16*z/L)^0.5
        
        For stable conditions (z/L > 0):
            ψ_m = ψ_h = -β * z/L
            
        Args:
            z_L: Stability parameter (z/L)
            z: Reference height (m)
            
        Returns:
            Tuple of (ψ_m, ψ_h) arrays
        """
        psi_m = np.zeros_like(z_L, dtype=np.float64)
        psi_h = np.zeros_like(z_L, dtype=np.float64)
        
        # Unstable conditions (daytime, convective)
        unstable_mask = z_L < 0
        
        if np.any(unstable_mask):
            z_L_unstable = np.abs(z_L[unstable_mask])

            # Calculate x and y, ensuring argument is non-negative
            x_arg = np.maximum(0.0, 1.0 - 16.0 * z_L_unstable / z)
            y_arg = np.maximum(0.0, 1.0 - 16.0 * z_L_unstable / z)

            x = x_arg ** 0.25
            y = y_arg ** 0.5

            # Clip values to avoid numerical issues
            x = np.clip(x, 0.0, 10.0)
            y = np.clip(y, 0.0, 10.0)
            
            # Stability corrections for momentum
            psi_m_unstable = (
                2.0 * np.log((1.0 + x) / 2.0) +
                np.log((1.0 + x**2) / 2.0) -
                2.0 * np.arctan(x) +
                np.pi / 2.0
            )
            
            # Stability corrections for heat
            psi_h_unstable = (
                2.0 * np.log((1.0 + y) / 2.0) +
                np.log((1.0 + y**2) / 2.0)
            )
            
            psi_m[unstable_mask] = psi_m_unstable
            psi_h[unstable_mask] = psi_h_unstable
        
        # Stable conditions (nighttime, stable)
        stable_mask = z_L > 0
        
        if np.any(stable_mask):
            z_L_stable = z_L[stable_mask]
            
            # Simple linear correction for stable conditions
            beta = 5.0  # Empirical coefficient
            psi_m[stable_mask] = -beta * z_L_stable
            psi_h[stable_mask] = -beta * z_L_stable
        
        return psi_m, psi_h
    
    def calculate_stable_rah(
        self,
        z0m: np.ndarray,
        z0h: np.ndarray,
        d: np.ndarray,
        u: np.ndarray,
        psi_m: np.ndarray,
        psi_h: np.ndarray,
        z_wind: float,
        z_temp: float
    ) -> np.ndarray:
        """
        Calculate aerodynamic resistance with stability corrections.
        
        rah = [ln((z_m - d)/z0m) - ψ_m] * [ln((z_h - d)/z0h) - ψ_h] / (k² * u)
        
        Args:
            z0m: Roughness length for momentum (m)
            z0h: Roughness length for heat (m)
            d: Displacement height (m)
            u: Wind speed (m/s)
            psi_m: Stability correction for momentum
            psi_h: Stability correction for heat
            z_wind: Wind measurement height (m)
            z_temp: Temperature measurement height (m)
            
        Returns:
            Aerodynamic resistance with stability corrections (s/m)
        """
        k2 = VON_KARMAN ** 2
        
        # Ensure valid arguments for log functions with stability corrections
        z0m_safe = np.maximum(z0m, 0.0001)  # Minimum roughness length
        z0h_safe = np.maximum(z0h, 0.00001)  # Minimum heat roughness length
        
        # Ensure displacement height doesn't exceed measurement heights
        d_safe = np.minimum(d, z_wind - z0m_safe - 0.1)  # Leave margin
        
        # Calculate log profiles with stability corrections
        ln_zm_z0m = np.log(np.maximum((z_wind - d_safe) / z0m_safe, 1.001)) - psi_m
        ln_zh_z0h = np.log(np.maximum((z_temp - d_safe) / z0h_safe, 1.001)) - psi_h
        
        # Calculate rah
        rah = (ln_zm_z0m * ln_zh_z0h) / (k2 * np.maximum(u, 0.1))
        
        # Apply physical bounds
        rah = np.clip(rah, self.config.min_rah, self.config.max_rah)
        
        return rah
    
    def calculate_dt_linear(
        self,
        ts_kelvin: np.ndarray,
        ta_kelvin: np.ndarray,
        a: Optional[float] = None,
        b: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate dT using linear calibration from anchor pixels.

        dT = a * (Ts - Ta) + b

        Args:
            ts_kelvin: Surface temperature in Kelvin
            ta_kelvin: Air temperature in Kelvin
            a: Slope coefficient (default from config)
            b: Intercept coefficient (default from config)

        Returns:
            dT array (K)
        """
        a = a if a is not None else self.config.dt_a
        b = b if b is not None else self.config.dt_b

        if a is None or b is None:
            # Default coefficients if not provided
            # These should be calibrated using anchor pixels
            a = 0.0
            b = 0.0

        return a * (ts_kelvin - ta_kelvin) + b
    
    def calculate_dt_empirical(
        self,
        ts_kelvin: np.ndarray,
        ta_kelvin: np.ndarray,
        ndvi: np.ndarray
    ) -> np.ndarray:
        """
        Calculate empirical dT based on surface characteristics.
        
        This is a fallback when anchor pixel calibration is not available.
        
        Args:
            ts_kelvin: Surface temperature (K)
            ta_kelvin: Air temperature (K)
            ndvi: Normalized Difference Vegetation Index
            
        Returns:
            dT array (K)
        """
        # Base dT from temperature difference
        dt_base = ts_kelvin - ta_kelvin
        
        # Modify based on vegetation (vegetation reduces dT)
        dt_vegetation = -2.0 * ndvi
        
        # Combine
        dT = dt_base + dt_vegetation
        
        return dT
    
    def calculate_rah_iterative(
        self,
        z0m: np.ndarray,
        z0h: np.ndarray,
        d: np.ndarray,
        u: np.ndarray,
        H: np.ndarray,
        rho: np.ndarray,
        ta_kelvin: np.ndarray,
        z_wind: float,
        z_temp: float
    ) -> np.ndarray:
        """
        Iteratively calculate aerodynamic resistance with stability corrections.
        
        Args:
            z0m: Roughness length for momentum (m)
            z0h: Roughness length for heat (m)
            d: Displacement height (m)
            u: Wind speed (m/s)
            H: Sensible heat flux (W/m²)
            rho: Air density (kg/m³)
            ta_kelvin: Air temperature (K)
            z_wind: Wind measurement height (m)
            z_temp: Temperature measurement height (m)
            
        Returns:
            Aerodynamic resistance with stability corrections (s/m)
        """
        # Start with neutral rah
        rah = self.calculate_neutral_rah(
            z0m, z0h, d, u, z_wind, z_temp
        )
        
        # Iterative stability correction
        for _ in range(self.config.stability_iterations):
            # Calculate stability parameter
            z_L = self.calculate_stability_parameter(
                H, rah, rho, ta_kelvin, z_temp
            )
            
            # Calculate stability corrections
            psi_m, psi_h = self.calculate_psi_functions(z_L, z_temp)
            
            # Update rah with corrections
            rah_new = self.calculate_stable_rah(
                z0m, z0h, d, u, psi_m, psi_h, z_wind, z_temp
            )
            
            # Check convergence
            rah_diff = np.abs(rah_new - rah)
            rah = rah_new
            
            if np.max(rah_diff) < 0.1:
                break
        
        return rah
    
    def calculate(
        self,
        rn: np.ndarray,
        ts_kelvin: np.ndarray,
        ta_kelvin: np.ndarray,
        u: np.ndarray,
        z0m: np.ndarray,
        pressure_pa: Optional[np.ndarray] = None,
        ndvi: Optional[np.ndarray] = None,
        lai: Optional[np.ndarray] = None,
        dt_a: Optional[float] = None,
        dt_b: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate sensible heat flux and related parameters.
        
        Args:
            rn: Net radiation array (W/m²)
            ts_kelvin: Surface temperature in Kelvin
            ta_kelvin: Air temperature in Kelvin
            u: Wind speed at 2m (m/s)
            z0m: Roughness length for momentum (m)
            pressure_pa: Atmospheric pressure in Pa (optional)
            ndvi: Normalized Difference Vegetation Index (optional)
            lai: Leaf Area Index (optional)
            dt_a: dT slope coefficient (optional)
            dt_b: dT intercept coefficient (optional)
            
        Returns:
            Dictionary containing:
                - 'H': Sensible heat flux (W/m²)
                - 'rah': Aerodynamic resistance (s/m)
                - 'dT': Temperature difference parameter (K)
                - 'rho': Air density (kg/m³)
        """
        # Ensure arrays have same dtype
        def to_numpy(arr):
            if hasattr(arr, 'values'):
                return np.asarray(arr.values, dtype=np.float64)
            else:
                return np.asarray(arr, dtype=np.float64)

        rn = to_numpy(rn)
        ts_kelvin = to_numpy(ts_kelvin)
        ta_kelvin = to_numpy(ta_kelvin)
        u = to_numpy(u)
        z0m = to_numpy(z0m)
        
        if pressure_pa is not None:
            pressure_pa = to_numpy(pressure_pa)
        else:
            # Standard atmospheric pressure (101325 Pa)
            pressure_pa = np.full_like(ts_kelvin, 101325.0, dtype=np.float64)

        if ndvi is not None:
            ndvi = to_numpy(ndvi)
        if lai is not None:
            lai = to_numpy(lai)
        
        # Calculate air density
        rho = self.calculate_air_density(ta_kelvin, pressure_pa)
        
        # Calculate roughness lengths
        z0h = z0m * self.config.roughness_ratio
        
        # Calculate displacement height
        d = self.calculate_displacement_height(z0m, lai)
        
        # Calculate dT (temperature difference at reference height)
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Calculating dT: dt_a={dt_a}, dt_b={dt_b}")
        
        if dt_a is not None:
            logger.info(f"Using METRIC calibration: H = {dt_a} * dT, dT = Ts - Ta")
            dT = ts_kelvin - ta_kelvin
            
            # Calculate proper aerodynamic resistance instead of hardcoded value
            # First calculate basic parameters needed for RAH calculation
            z0h = z0m * self.config.roughness_ratio
            d = self.calculate_displacement_height(z0m, lai)
            
            # Calculate air density if not already calculated
            if 'rho' not in locals():
                rho = self.calculate_air_density(ta_kelvin, pressure_pa)
            
            # Calculate initial H for stability iterations
            H_initial = dt_a * dT
            
            # Calculate aerodynamic resistance with proper physics
            if self.config.use_stability_correction:
                # Use iterative method with stability corrections
                rah = self.calculate_rah_iterative(
                    z0m, z0h, d, u, H_initial, rho, ta_kelvin,
                    self.config.z_wind, self.config.z_temp
                )
            else:
                # Use neutral conditions
                rah = self.calculate_neutral_rah(
                    z0m, z0h, d, u,
                    self.config.z_wind,
                    self.config.z_temp
                )
            
            # Calculate final H using proper aerodynamic resistance
            # FIXED: Use proper physics formula H = rho * cp * dT / rah
            # The dt_a coefficient should already account for rah in calibration
            
            # CORRECTED - Enhanced fallback: Check for calibration failure indicators
            physics_fallback = False
            
            if dt_a is not None:
                if dt_a < 5.0 or dt_a > 100.0:
                    # dt_a outside physical range (10-50 W/m²/K), use physics formula
                    logger.warning(f"Calibration dt_a={dt_a:.2f} outside physical range (10-50 W/m²/K), using physics-based H")
                    physics_fallback = True
                elif abs(dt_a - 1.0) < 0.1:
                    # dt_a approximately 1.0 indicates calibration formula error
                    logger.warning(f"Calibration dt_a={dt_a:.2f} ≈ 1.0 indicates formula error, using physics-based H")
                    physics_fallback = True
                else:
                    logger.info(f"Using calibrated H: dt_a={dt_a:.2f} W/m²/K, dT = Ts - Ta")
            
            if physics_fallback:
                # Calculate rah from Monin-Obukhov similarity if available
                if rah is not None and np.nanmean(rah) > 0:
                    rah_avg = np.nanmean(rah)
                    H = rho * AIR_SPECIFIC_HEAT * dT / rah_avg
                    logger.info(f"Physics-based H using rah={rah_avg:.2f} s/m")
                else:
                    # Estimate rah from typical values (50-100 s/m for neutral conditions)
                    rah_estimated = 70.0  # Typical aerodynamic resistance
                    H = rho * AIR_SPECIFIC_HEAT * dT / rah_estimated
                    logger.info(f"Using estimated rah={rah_estimated} s/m for physics-based H calculation")
            else:
                # Use calibrated H calculation: H = dt_a * dT
                H = dt_a * dT
                logger.info(f"Calibrated H: dt_a={dt_a:.2f} W/m²/K, dT range=[{np.nanmin(dT):.2f}, {np.nanmax(dT):.2f}] K")
            
            # Apply physical constraints
            H = np.clip(H, self.config.min_h, self.config.max_h)
            
            # Energy balance constraint: H cannot exceed available energy (Rn - G)
            # NOTE: G is not available in calculate() method, estimate as 5% of Rn
            # This is a reasonable approximation for typical conditions
            g_estimated = rn * 0.05  # G ≈ 5% of Rn for typical conditions
            available_energy = np.maximum(rn - g_estimated, 1.0)  # Prevent division by zero
            
            # CORRECTED - Physics-based bounds instead of artificial constraints:
            # Upper bound: H cannot exceed 95% of available energy
            H = np.minimum(H, available_energy * 0.95)
            
            # Note: Lower bound constraint removed to allow physically accurate negative H values
            # For cold pixels (well-watered vegetation at night), H can legitimately be negative
            # when Ts < Ta (heat flows from air to surface)

            
            # Remove the minimum_h_baseline constraint as it masks calibration issues
            # The old constraint: min_h_baseline = available_energy * 0.05 was too restrictive
            
            logger.info(f"Calculated dT range: min={np.nanmin(dT):.3f}, max={np.nanmax(dT):.3f}")
            logger.info(f"Calculated H range: min={np.nanmin(H):.3f}, max={np.nanmax(H):.3f}")
            logger.info(f"Calculated RAH range: min={np.nanmin(rah):.3f}, max={np.nanmax(rah):.3f}")
            logger.info(f"METRIC DEBUG: calibration_a={dt_a}, H_mean={np.nanmean(H):.2f}, RAH_mean={np.nanmean(rah):.2f}")
            
            # Check for calibration failure indicators
            if dt_a == 0.0:
                logger.warning("WARNING: calibration_a is zero! This will cause H=0 everywhere.")
                logger.warning("This indicates calibration failure - check anchor pixel selection and dT_hot validation.")
            elif np.nanmean(H) < 5.0:
                logger.warning(f"WARNING: Mean H is very low ({np.nanmean(H):.2f} W/m²). EF may exceed realistic range.")
                logger.warning("This can occur with high vegetation cover or calibration issues.")
            elif np.nanmean(H) > 200:
                logger.warning(f"WARNING: Mean H is very high ({np.nanmean(H):.2f} W/m²). Check calibration coefficients.")
        else:
            # Calculate dT
            if dt_a is not None and dt_b is not None:
                logger.info(f"Using linear calibration: dT = {dt_a} * (Ts - Ta) + {dt_b}")
                dT = self.calculate_dt_linear(ts_kelvin, ta_kelvin, dt_a, dt_b)
            elif ndvi is not None:
                logger.info("Using empirical dT calculation based on NDVI")
                dT = self.calculate_dt_empirical(ts_kelvin, ta_kelvin, ndvi)
            else:
                logger.info("Using default dT = Ts - Ta (no calibration or NDVI available)")
                dT = ts_kelvin - ta_kelvin
                # Add empirical adjustment based on typical ranges
                dT = np.clip(dT, -10.0, 50.0)  # Reasonable bounds for dT

            logger.info(f"Calculated dT range: min={np.nanmin(dT):.3f}, max={np.nanmax(dT):.3f}")

            # Calculate aerodynamic resistance
            if self.config.use_stability_correction:
                # Need H to calculate stability, but H depends on rah
                # Start with neutral conditions, then iterate
                rah_neutral = self.calculate_neutral_rah(
                    z0m, z0h, d, u,
                    self.config.z_wind,
                    self.config.z_temp
                )

                # Initial H calculation
                H_initial = rho * AIR_SPECIFIC_HEAT * dT / rah_neutral

                # Iterative stability correction
                rah = self.calculate_rah_iterative(
                    z0m, z0h, d, u, H_initial, rho, ta_kelvin,
                    self.config.z_wind, self.config.z_temp
                )
            else:
                rah = self.calculate_neutral_rah(
                    z0m, z0h, d, u,
                    self.config.z_wind,
                    self.config.z_temp
                )

            # Calculate H
            logger.info(f"Calculating H: rho range min={np.nanmin(rho):.3f}, max={np.nanmax(rho):.3f}")
            logger.info(f"Calculating H: dT range min={np.nanmin(dT):.3f}, max={np.nanmax(dT):.3f}")
            logger.info(f"Calculating H: rah range min={np.nanmin(rah):.3f}, max={np.nanmax(rah):.3f}")

            H = rho * AIR_SPECIFIC_HEAT * dT / rah

            logger.info(f"Calculated H range: min={np.nanmin(H):.3f}, max={np.nanmax(H):.3f}")

            # Apply physical constraints
            H = np.clip(H, self.config.min_h, self.config.max_h)

            # Energy balance constraint: H cannot exceed available energy (Rn - G)
            # NOTE: G is not available in uncalibrated mode, estimate as 5% of Rn
            # This is a reasonable approximation for typical conditions
            g_estimated = rn * 0.05  # G ≈ 5% of Rn for typical conditions
            available_energy = np.maximum(rn - g_estimated, 1.0)  # Prevent division by zero
            
            # CORRECTED - Physics-based bounds:
            # Upper bound: H cannot exceed 95% of available energy
            H = np.minimum(H, available_energy * 0.95)
            
            # Note: Lower bound constraint removed to allow physically accurate negative H values
            # For cold pixels (well-watered vegetation at night), H can legitimately be negative
            # when Ts < Ta (heat flows from air to surface)
        
        return {
            'H': H,
            'rah': rah,
            'dT': dT,
            'rho': rho
        }
    
    def compute(self, cube, calibration):
        """
        Compute sensible heat flux and add to DataCube.

        Args:
            cube: DataCube with required bands
            calibration: CalibrationResult from DTCalibration

        Returns:
            DataCube with added H, rah, dT, rho
        """
        from ..core.datacube import DataCube

        # Get required inputs
        rn = cube.get("R_n")
        ts_kelvin = cube.get("lwir11")
        ta_kelvin = cube.get("temperature_2m")
        u = cube.get("u")
        z0m = cube.get("z0m")  # Roughness length
        pressure_pa = cube.get("P")
        ndvi = cube.get("ndvi")
        lai = cube.get("lai") if cube.get("lai") is not None else None

        if rn is None:
            raise ValueError("Net radiation (R_n) not found in DataCube")
        if ts_kelvin is None:
            raise ValueError("Surface temperature (lwir11) not found in DataCube")
        if ta_kelvin is None:
            raise ValueError("Air temperature (temperature_2m) not found in DataCube")
        if u is None:
            raise ValueError("Wind speed (u) not found in DataCube")
        if z0m is None:
            raise ValueError("Roughness length (z0m) not found in DataCube")

        # Helper function to get values, handling scalars
        def get_values(data):
            if hasattr(data, 'values'):
                return data.values
            else:
                # Scalar: create array of same shape as rn
                return np.full_like(rn.values, data, dtype=np.float64)

        # Calculate sensible heat flux
        result = self.calculate(
            rn=get_values(rn),
            ts_kelvin=get_values(ts_kelvin),
            ta_kelvin=get_values(ta_kelvin),
            u=get_values(u),
            z0m=get_values(z0m),
            pressure_pa=get_values(pressure_pa) if pressure_pa is not None else None,
            ndvi=get_values(ndvi) if ndvi is not None else None,
            lai=get_values(lai) if lai is not None else None,
            dt_a=calibration.a_coefficient,
            dt_b=calibration.b_coefficient
        )

        # Add to cube
        cube.add("H", result["H"])
        cube.add("rah", result["rah"])
        cube.add("dT", result["dT"])
        cube.add("rho", result["rho"])

        return cube

    def __call__(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Convenience method to calculate sensible heat flux.
        """
        return self.calculate(*args, **kwargs)


def create_sensible_heat_flux(
    z_wind: float = 10.0,
    z_temp: float = 2.0,
    use_stability: bool = True,
    **kwargs
) -> SensibleHeatFlux:
    """
    Factory function to create SensibleHeatFlux instance.
    
    Args:
        z_wind: Wind measurement height (m)
        z_temp: Temperature measurement height (m)
        use_stability: Whether to apply stability corrections
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured SensibleHeatFlux instance
    """
    config = SensibleHeatFluxConfig(
        z_wind=z_wind,
        z_temp=z_temp,
        use_stability_correction=use_stability,
        **kwargs
    )
    return SensibleHeatFlux(config)
