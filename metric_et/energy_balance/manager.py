"""
Energy Balance Manager for METRIC ETa Model.

This module provides a unified interface for calculating all energy balance
components (G, H, LE) and integrating them with the DataCube.

Energy Balance Equation:
    Rn - G - H - LE = 0

Example usage:
    manager = EnergyBalanceManager()
    results = manager.calculate(cube)
    
    # Access results
    G = results['G']
    H = results['H']
    LE = results['LE']
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from metric_et.core.datacube import DataCube
from metric_et.energy_balance.soil_heat_flux import (
    SoilHeatFlux, SoilHeatFluxConfig
)
from metric_et.energy_balance.sensible_heat_flux import (
    SensibleHeatFlux, SensibleHeatFluxConfig
)
from metric_et.energy_balance.latent_heat_flux import (
    LatentHeatFlux, LatentHeatFluxConfig
)


@dataclass
class EnergyBalanceConfig:
    """Configuration for the energy balance manager."""
    # G configuration
    g_method: str = 'automatic'
    g_ndvi_threshold: float = 0.2
    
    # H configuration
    h_z_wind: float = 10.0
    h_z_temp: float = 2.0
    h_use_stability: bool = True
    
    # LE configuration
    le_allow_negative: bool = False
    le_min_ef: float = 0.0
    le_max_ef: float = 1.2
    
    # Anchor pixel calibration (to be set during METRIC calibration)
    dt_a: Optional[float] = None
    dt_b: Optional[float] = None
    
    # Output keys
    output_keys: tuple = ('G', 'H', 'LE', 'rah', 'EF', 'ET_inst')


class EnergyBalanceManager:
    """
    Unified manager for calculating energy balance components.
    
    This class provides a simple interface for calculating all energy
    balance components (G, H, LE) and integrates with the DataCube.
    
    Example:
        >>> manager = EnergyBalanceManager()
        >>> results = manager.calculate(cube)
        >>> print(f"G: {results['G'].mean():.2f} W/m²")
        >>> print(f"H: {results['H'].mean():.2f} W/m²")
        >>> print(f"LE: {results['LE'].mean():.2f} W/m²")
    """
    
    def __init__(self, config: Optional[EnergyBalanceConfig] = None):
        """
        Initialize EnergyBalanceManager.
        
        Args:
            config: Optional configuration parameters. Uses defaults if not provided.
        """
        self.config = config or EnergyBalanceConfig()
        
        # Initialize component calculators
        self._init_components()
    
    def _init_components(self):
        """Initialize the energy balance component calculators."""
        # Soil heat flux
        g_config = SoilHeatFluxConfig(
            method=self.config.g_method,
            ndvi_threshold=self.config.g_ndvi_threshold
        )
        self.g_calculator = SoilHeatFlux(g_config)
        
        # Sensible heat flux
        h_config = SensibleHeatFluxConfig(
            z_wind=self.config.h_z_wind,
            z_temp=self.config.h_z_temp,
            use_stability_correction=self.config.h_use_stability,
            dt_a=self.config.dt_a,
            dt_b=self.config.dt_b
        )
        self.h_calculator = SensibleHeatFlux(h_config)
        
        # Latent heat flux
        le_config = LatentHeatFluxConfig(
            allow_negative_le=self.config.le_allow_negative,
            min_ef=self.config.le_min_ef,
            max_ef=self.config.le_max_ef
        )
        self.le_calculator = LatentHeatFlux(le_config)
    
    def calculate_from_arrays(
        self,
        rn: np.ndarray,
        ts_kelvin: np.ndarray,
        ta_kelvin: np.ndarray,
        u: np.ndarray,
        z0m: np.ndarray,
        ndvi: np.ndarray,
        albedo: Optional[np.ndarray] = None,
        pressure_pa: Optional[np.ndarray] = None,
        lai: Optional[np.ndarray] = None,
        g_flux: Optional[np.ndarray] = None  # Optional pre-computed G
    ) -> Dict[str, np.ndarray]:
        """
        Calculate all energy balance components from arrays.
        
        Args:
            rn: Net radiation (W/m²)
            ts_kelvin: Surface temperature (K)
            ta_kelvin: Air temperature (K)
            u: Wind speed at 2m (m/s)
            z0m: Roughness length for momentum (m)
            ndvi: Normalized Difference Vegetation Index
            albedo: Surface albedo (optional)
            pressure_pa: Atmospheric pressure (Pa, optional)
            lai: Leaf Area Index (optional)
            
        Returns:
            Dictionary containing:
                - 'G': Soil heat flux (W/m²)
                - 'H': Sensible heat flux (W/m²)
                - 'LE': Latent heat flux (W/m²)
                - 'rah': Aerodynamic resistance (s/m)
                - 'EF': Evaporative fraction
                - 'ET_inst': Instantaneous ET rate (mm/hr)
        """
        # Calculate soil heat flux (G) - either use pre-computed or calculate
        if g_flux is not None:
            # Use pre-computed G from pipeline
            G = g_flux
            # Create a minimal result dict for compatibility
            g_result = {'G': G, 'G_Rn_ratio': G / np.maximum(rn, 1.0)}
        else:
            # Calculate G as part of energy balance
            g_result = self.g_calculator.calculate(rn, ndvi, ts_kelvin, ta_kelvin)
            G = g_result['G']
        
        # Calculate sensible heat flux (H)
        h_result = self.h_calculator.calculate(
            rn=rn,
            ts_kelvin=ts_kelvin,
            ta_kelvin=ta_kelvin,
            u=u,
            z0m=z0m,
            pressure_pa=pressure_pa,
            ndvi=ndvi,
            lai=lai,
            dt_a=self.config.dt_a,
            dt_b=self.config.dt_b
        )
        H = h_result['H']
        rah = h_result['rah']
        
        # Calculate latent heat flux (LE)
        le_result = self.le_calculator.calculate(rn, G, H, ts_kelvin)
        LE = le_result['LE']
        EF = le_result['EF']
        ET_inst = le_result['ET_inst']
        
        return {
            'G': G,
            'H': H,
            'LE': LE,
            'rah': rah,
            'EF': EF,
            'ET_inst': ET_inst,
            'dT': h_result.get('dT'),
            'rho': h_result.get('rho'),
            'G_Rn_ratio': g_result.get('G_Rn_ratio'),
            'available_energy': le_result.get('available_energy')
        }
    
    def calculate(self, cube: DataCube) -> Dict[str, np.ndarray]:
        """
        Calculate all energy balance components from DataCube.
        
        Args:
            cube: DataCube containing required data arrays
            
        Required cube arrays:
            - 'R_n': Net radiation (W/m²)
            - 'lwir11': Surface temperature (K)
            - 'temperature_2m': Air temperature (K)
            - 'wind_speed_10m': Wind speed at 10m (m/s)
            - 'z0m': Roughness length for momentum (m)
            - 'ndvi': Normalized Difference Vegetation Index
            
        Optional cube arrays:
            - 'surface_pressure': Atmospheric pressure (Pa)
            - 'lai': Leaf Area Index
            - 'albedo': Surface albedo
            
        Returns:
            Dictionary with energy balance arrays. Also adds arrays to cube.
        """
        # Get required arrays from cube
        try:
            rn = cube.get('R_n')
            ts_kelvin = cube.get('lwir11')
            ta_kelvin = cube.get('temperature_2m')
            u = cube.get('wind_speed_10m')
            z0m = cube.get('z0m')
            ndvi = cube.get('ndvi')
        except KeyError as e:
            raise ValueError(f"Missing required data in cube: {e}")
        
        # Get optional arrays
        pressure_pa = cube.get('surface_pressure') if 'surface_pressure' in cube.bands() else None
        lai = cube.get('lai') if 'lai' in cube.bands() else None
        albedo = cube.get('albedo') if 'albedo' in cube.bands() else None
        
        # Check if G is already computed (for METRIC pipeline reordering)
        g_flux = cube.get('G')
        if g_flux is not None:
            # Use existing G calculation from pipeline
            results = self.calculate_from_arrays(
                rn=rn,
                ts_kelvin=ts_kelvin,
                ta_kelvin=ta_kelvin,
                u=u,
                z0m=z0m,
                ndvi=ndvi,
                albedo=albedo,
                pressure_pa=pressure_pa,
                lai=lai,
                g_flux=g_flux  # Pass existing G
            )
        else:
            # Calculate G as part of energy balance
            results = self.calculate_from_arrays(
                rn=rn,
                ts_kelvin=ts_kelvin,
                ta_kelvin=ta_kelvin,
                u=u,
                z0m=z0m,
                ndvi=ndvi,
                albedo=albedo,
                pressure_pa=pressure_pa,
                lai=lai
            )
        
        # Add results to cube
        for key in self.config.output_keys:
            if key in results:
                cube.add(key, results[key])
        
        return results
    
    def calculate_quality_report(
        self,
        rn: np.ndarray,
        G: np.ndarray,
        H: np.ndarray,
        LE: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Generate quality report for energy balance.
        
        Args:
            rn: Net radiation (W/m²)
            G: Soil heat flux (W/m²)
            H: Sensible heat flux (W/m²)
            LE: Latent heat flux (W/m²)
            
        Returns:
            Dictionary with quality metrics
        """
        available_energy = rn - G
        residual = rn - G - H - LE
        rel_residual = np.where(
            np.abs(available_energy) > 1.0,
            residual / np.abs(available_energy),
            0.0
        )
        
        # Energy balance closure
        closure = np.where(
            np.abs(available_energy) > 1.0,
            (H + LE) / np.abs(available_energy),
            1.0
        )
        
        return {
            'residual': residual,
            'relative_residual': rel_residual,
            'energy_balance_closure': closure,
            'available_energy': available_energy
        }
    
    def set_anchor_pixel_calibration(self, a: float, b: float):
        """
        Set anchor pixel calibration coefficients for dT.

        dT = a * (Ts - Ta) + b

        These coefficients are determined during METRIC calibration
        using hot and cold anchor pixels.

        Args:
            a: Slope coefficient
            b: Intercept coefficient
        """
        self.config.dt_a = a
        self.config.dt_b = b
        
        # Update H calculator
        self.h_calculator.config.dt_a = a
        self.h_calculator.config.dt_b = b
    
    def __repr__(self) -> str:
        """String representation of EnergyBalanceManager."""
        return (
            f"EnergyBalanceManager("
            f"g_method={self.config.g_method}, "
            f"h_use_stability={self.config.h_use_stability}, "
            f"dt_a={self.config.dt_a}, "
            f"dt_b={self.config.dt_b})"
        )


def create_energy_balance_manager(
    g_method: str = 'automatic',
    h_use_stability: bool = True,
    dt_a: Optional[float] = None,
    dt_b: Optional[float] = None
) -> EnergyBalanceManager:
    """
    Factory function to create EnergyBalanceManager.
    
    Args:
        g_method: Soil heat flux calculation method
        h_use_stability: Whether to use stability corrections
        dt_a: dT slope coefficient (anchor pixel calibration)
        dt_b: dT intercept coefficient (anchor pixel calibration)
        
    Returns:
        Configured EnergyBalanceManager instance
    """
    config = EnergyBalanceConfig(
        g_method=g_method,
        h_use_stability=h_use_stability,
        dt_a=dt_a,
        dt_b=dt_b
    )
    return EnergyBalanceManager(config)
