"""
Energy Balance Components for METRIC ETa Model.

This package contains the core energy balance calculations:
- SoilHeatFlux (G): Energy used to heat the soil
- SensibleHeatFlux (H): Energy transferred between surface and air
- LatentHeatFlux (LE): Energy used for evapotranspiration
- EnergyBalanceManager: Unified interface for all components

Energy Balance Equation:
    Rn - G - H - LE = 0

Example usage:
    from metric_et.energy_balance import EnergyBalanceManager
    
    manager = EnergyBalanceManager()
    results = manager.calculate(cube)
    
    G = results['G']
    H = results['H']
    LE = results['LE']
"""

from metric_et.energy_balance.soil_heat_flux import (
    SoilHeatFlux,
    SoilHeatFluxConfig,
    create_soil_heat_flux
)

from metric_et.energy_balance.sensible_heat_flux import (
    SensibleHeatFlux,
    SensibleHeatFluxConfig,
    create_sensible_heat_flux
)

from metric_et.energy_balance.latent_heat_flux import (
    LatentHeatFlux,
    LatentHeatFluxConfig,
    create_latent_heat_flux
)

from metric_et.energy_balance.manager import (
    EnergyBalanceManager,
    EnergyBalanceConfig,
    create_energy_balance_manager
)

__all__ = [
    # Soil Heat Flux
    'SoilHeatFlux',
    'SoilHeatFluxConfig',
    'create_soil_heat_flux',
    
    # Sensible Heat Flux
    'SensibleHeatFlux',
    'SensibleHeatFluxConfig',
    'create_sensible_heat_flux',
    
    # Latent Heat Flux
    'LatentHeatFlux',
    'LatentHeatFluxConfig',
    'create_latent_heat_flux',
    
    # Manager
    'EnergyBalanceManager',
    'EnergyBalanceConfig',
    'create_energy_balance_manager',
]
