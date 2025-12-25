"""Net radiation calculations for METRIC ETa model."""

import numpy as np
import xarray as xr
from typing import Optional

from ..core.datacube import DataCube
from ..core.constants import STEFAN_BOLTZMANN


class NetRadiation:
    """
    Compute net radiation for energy balance calculations.
    
    Net radiation (R_n) is the balance between incoming and outgoing
    shortwave and longwave radiation. It represents the available energy
    at the surface for partitioning into turbulent fluxes (H, LE) and
    ground heat flux (G).
    
    Energy balance equation:
        R_n = R_ns - R_nl
        R_n = (1 - α) * Rs↓ - (ε * σ * Ts⁴ - ε_clear * σ * Ta⁴)
    
    Where:
        R_ns = Net shortwave radiation = (1 - α) * Rs↓
        R_nl = Net longwave radiation = R_l↓ - R_l↑
    
    For ET calculations, only positive (daytime) net radiation is used
    since ET does not occur at night in most conditions.
    
    Attributes:
        clip_negative: Clip negative Rn values to 0 (for daytime ET)
    """
    
    def __init__(self, clip_negative: bool = True):
        """
        Initialize NetRadiation calculator.
        
        Args:
            clip_negative: Clip negative Rn values to 0 for daytime ET
        """
        self.clip_negative = clip_negative
    
    def compute(self, cube: DataCube) -> DataCube:
        """
        Compute net radiation and add to DataCube.
        
        This method combines net shortwave and net longwave radiation
        to calculate the total net radiation balance.
        
        Required inputs:
            - R_ns: Net shortwave radiation (W/m²)
            - R_nl: Net longwave radiation (W/m²)
            
        Or alternatively:
            - albedo: Surface albedo
            - Rs_down: Incoming shortwave radiation
            - R_l_up: Outgoing longwave radiation
            - R_l_down: Incoming longwave radiation
        
        Outputs:
            - R_n: Net radiation (W/m²)
            - R_n_daytime: Daytime net radiation (Rn ≥ 0, W/m²)
        
        Args:
            cube: Input DataCube with radiation components
            
        Returns:
            DataCube with added net radiation
            
        Raises:
            ValueError: If required inputs are missing
        """
        # Check if shortwave and longwave are already computed
        if "R_ns" in cube.bands() and "R_nl" in cube.bands():
            R_ns = cube.get("R_ns")
            R_nl = cube.get("R_nl")
            R_n = self.compute_net_radiation_from_components(R_ns, R_nl)
        else:
            # Need to compute from basic components
            R_n = self.compute_net_radiation(cube)
        
        # Clip negative values for daytime ET
        if self.clip_negative:
            R_n_daytime = xr.where(R_n < 0, 0.0, R_n)
            R_n_daytime.name = "R_n_daytime"
            R_n_daytime.attrs = {
                'long_name': 'Daytime Net Radiation (Rn ≥ 0)',
                'units': 'W/m²',
                'method': 'Clipped net radiation for ET calculations'
            }
            cube.add("R_n_daytime", R_n_daytime)
        
        # Add to DataCube
        cube.add("R_n", R_n)
        
        return cube
    
    def compute_net_radiation(self, cube: DataCube) -> xr.DataArray:
        """
        Compute net radiation from basic components.
        
        Computes R_n = R_ns - R_nl or equivalently:
            R_n = (1 - α) * Rs↓ - ε * σ * Ts⁴ + ε_clear * σ * Ta⁴
        
        Args:
            cube: Input DataCube with albedo, Rs_down, R_l_up, R_l_down
            
        Returns:
            Net radiation (W/m²)
        """
        # Get required components
        albedo = cube.get("albedo")
        Rs_down = cube.get("Rs_down")
        R_l_up = cube.get("R_l_up")
        R_l_down = cube.get("R_l_down")
        
        # Validate inputs
        if any(x is None for x in [albedo, Rs_down, R_l_up, R_l_down]):
            raise ValueError(
                "Missing required components for net radiation. "
                "Need: albedo, Rs_down, R_l_up, R_l_down"
            )
        
        # Compute net shortwave
        R_ns = (1.0 - albedo) * Rs_down
        
        # Compute net longwave
        R_nl = R_l_down - R_l_up
        
        # Compute net radiation
        R_n = R_ns - R_nl
        
        R_n.name = "R_n"
        R_n.attrs = {
            'long_name': 'Net Radiation',
            'units': 'W/m²',
            'method': 'R_ns - R_nl',
            'clip_negative': self.clip_negative
        }
        
        return R_n
    
    def compute_net_radiation_from_components(
        self, 
        R_ns: xr.DataArray, 
        R_nl: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute net radiation from net shortwave and net longwave.
        
        R_n = R_ns - R_nl
        
        Args:
            R_ns: Net shortwave radiation (W/m²)
            R_nl: Net longwave radiation (W/m²)
            
        Returns:
            Net radiation (W/m²)
        """
        R_n = R_ns - R_nl
        
        R_n.name = "R_n"
        R_n.attrs = {
            'long_name': 'Net Radiation',
            'units': 'W/m²',
            'method': 'R_ns - R_nl'
        }
        
        return R_n
    
    def compute_net_radiation_alternative(
        self,
        cube: DataCube
    ) -> xr.DataArray:
        """
        Compute net radiation using alternative formulation.
        
        R_n = (1 - α) * Rs↓ - ε * σ * Ts⁴ + ε_clear * σ * Ta⁴
        
        This formulation explicitly shows the temperature dependence
        of the longwave components.
        
        Args:
            cube: Input DataCube with required components
            
        Returns:
            Net radiation (W/m²)
        """
        # Get components
        albedo = cube.get("albedo")
        Rs_down = cube.get("Rs_down")
        emissivity = cube.get("emissivity")
        Ts = cube.get("lwir11")
        
        # Get air temperature and clear-sky emissivity
        Ta = cube.get("air_temperature")
        if Ta is None:
            Ta = cube.metadata.get("temperature_2m", 25.0) + 273.15
        
        e_a = cube.get("vapor_pressure")
        if e_a is None:
            e_a = cube.metadata.get("e_a", 1.5)
        
        # Clear-sky emissivity (Brunt equation)
        epsilon_clear = 0.70 + 0.005 * np.sqrt(e_a)
        
        # Ts (lwir11) is already in Kelvin
        
        # Convert Ta to array if scalar
        if isinstance(Ta, (int, float)):
            Ta_val = Ta
            Ta = Rs_down.copy()
            Ta.values[:] = Ta_val
        
        # Compute net radiation
        R_n = (
            (1.0 - albedo) * Rs_down
            - emissivity * STEFAN_BOLTZMANN * (Ts ** 4)
            + epsilon_clear * STEFAN_BOLTZMANN * (Ta ** 4)
        )
        
        R_n.name = "R_n_alt"
        R_n.attrs = {
            'long_name': 'Net Radiation (Alternative Method)',
            'units': 'W/m²',
            'method': '(1-α)Rs↓ - εσTs⁴ + ε_clearσTa⁴'
        }
        
        return R_n
    
    def compute_radiation_balance_components(
        self, 
        cube: DataCube
    ) -> dict:
        """
        Compute all radiation balance components for analysis.
        
        Returns a dictionary with all radiation components and their
        contributions to the net radiation balance.
        
        Args:
            cube: Input DataCube with all radiation data
            
        Returns:
            Dictionary of radiation components
        """
        components = {}
        
        # Shortwave components
        Rs_down = cube.get("Rs_down")
        albedo = cube.get("albedo")
        R_s_up = cube.get("R_s_up")
        R_ns = cube.get("R_ns")
        
        if Rs_down is not None:
            components['Rs_down'] = Rs_down
        if albedo is not None:
            components['albedo'] = albedo
        if R_s_up is not None:
            components['R_s_up'] = R_s_up
        if R_ns is not None:
            components['R_ns'] = R_ns
        
        # Longwave components
        R_l_up = cube.get("R_l_up")
        R_l_down = cube.get("R_l_down")
        R_nl = cube.get("R_nl")
        
        if R_l_up is not None:
            components['R_l_up'] = R_l_up
        if R_l_down is not None:
            components['R_l_down'] = R_l_down
        if R_nl is not None:
            components['R_nl'] = R_nl
        
        # Net radiation
        R_n = cube.get("R_n")
        if R_n is not None:
            components['R_n'] = R_n
            components['R_n_daytime'] = cube.get("R_n_daytime")
        
        return components
    
    def compute_albedo_effect(
        self, 
        Rs_down: xr.DataArray, 
        albedo: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute the effect of albedo on net shortwave radiation.
        
        Shows how much radiation is reflected due to surface albedo.
        
        Args:
            Rs_down: Incoming shortwave radiation (W/m²)
            albedo: Surface albedo
            
        Returns:
            Reflected shortwave radiation (W/m²)
        """
        R_reflected = albedo * Rs_down
        R_absorbed = (1.0 - albedo) * Rs_down
        
        effect = xr.Dataset({
            'reflected': R_reflected,
            'absorbed': R_absorbed,
            'albedo_ratio': albedo
        })
        
        effect['reflected'].attrs = {
            'long_name': 'Reflected Shortwave Radiation',
            'units': 'W/m²'
        }
        effect['absorbed'].attrs = {
            'long_name': 'Absorbed Shortwave Radiation',
            'units': 'W/m²'
        }
        effect['albedo_ratio'].attrs = {
            'long_name': 'Albedo Ratio',
            'units': 'dimensionless'
        }
        
        return effect
    
    def compute_greenhouse_effect(
        self, 
        R_l_up: xr.DataArray,
        R_l_down: xr.DataArray,
        emissivity: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute the greenhouse effect from longwave radiation.
        
        The greenhouse effect is the difference between outgoing
        longwave radiation and what would be emitted by a blackbody
        at the same temperature.
        
        Args:
            R_l_up: Actual outgoing longwave radiation
            R_l_down: Incoming longwave radiation
            emissivity: Surface emissivity
            
        Returns:
            Greenhouse effect dataset
        """
        # Blackbody emission at surface temperature
        # (Assuming Ts from R_l_up = εσTs⁴)
        Ts_4 = R_l_up / (emissivity * STEFAN_BOLTZMANN)
        R_l_up_blackbody = STEFAN_BOLTZMANN * Ts_4
        
        # Greenhouse effect
        ghe = R_l_up_blackbody - R_l_up
        
        effect = xr.Dataset({
            'R_l_up_actual': R_l_up,
            'R_l_up_blackbody': R_l_up_blackbody,
            'greenhouse_effect': ghe,
            'R_l_down': R_l_down
        })
        
        effect['greenhouse_effect'].attrs = {
            'long_name': 'Greenhouse Effect',
            'units': 'W/m²',
            'description': 'Reduction in outgoing LW due to non-blackbody surface'
        }
        
        return effect


# Alias for backward compatibility
NetRadiationCalculator = NetRadiation


__all__ = ['NetRadiation', 'NetRadiationCalculator']
