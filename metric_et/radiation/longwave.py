"""Longwave radiation calculations for METRIC ETa model."""

import numpy as np
import xarray as xr
from typing import Optional

from ..core.datacube import DataCube
from ..core.constants import STEFAN_BOLTZMANN, FREEZING_POINT


class LongwaveRadiation:
    """
    Compute longwave radiation components for energy balance.
    
    This class calculates incoming longwave radiation (R_l↓) from the 
    atmosphere and outgoing longwave radiation (R_l↑) emitted by the 
    surface. The net longwave radiation (R_nl) is the difference.
    
    The Stefan-Boltzmann law is used for blackbody radiation:
        L = ε * σ * T⁴
    
    Where:
        ε = emissivity
        σ = 5.670374419e-8 W/m²/K⁴
        T = temperature in Kelvin
    
    Attributes:
        use_cloud_correction: Apply cloud cover correction to R_l↓
        use_simplified_brunt: Use simplified Brunt equation for R_l↓
    """
    
    def __init__(
        self, 
        use_cloud_correction: bool = True,
        use_simplified_brunt: bool = True
    ):
        """
        Initialize LongwaveRadiation calculator.
        
        Args:
            use_cloud_correction: Apply cloud cover correction
            use_simplified_brunt: Use simplified Brunt equation
        """
        self.use_cloud_correction = use_cloud_correction
        self.use_simplified_brunt = use_simplified_brunt
    
    def compute(self, cube: DataCube) -> DataCube:
        """
        Compute all longwave radiation components and add to DataCube.
        
        Required inputs:
            - emissivity: Surface emissivity (from emissivity module)
            - Ts (lwir11): Surface temperature in Kelvin (from thermal band)
            - Ta: Air temperature in Kelvin (from weather data)
            - e_a: Actual vapor pressure in kPa (from weather data)
            - C: Cloud cover fraction 0-1 (optional, from weather data)
        
        Outputs:
            - R_l_up: Outgoing longwave radiation (W/m²)
            - R_l_down: Incoming longwave radiation (W/m²)
            - R_nl: Net longwave radiation (W/m²)
        
        Args:
            cube: Input DataCube with required bands
            
        Returns:
            DataCube with added longwave radiation components
            
        Raises:
            ValueError: If required inputs are missing
        """
        # Validate required inputs
        self._validate_inputs(cube)
        
        # Get required data
        emissivity = cube.get("emissivity")
        Ts = cube.get("lwir11")  # Surface temperature in Kelvin (already converted)
        
        # Compute outgoing longwave (surface emission)
        R_l_up = self.compute_outgoing_longwave(Ts, emissivity)
        
        # Compute incoming longwave (atmospheric emission)
        R_l_down = self.compute_incoming_longwave(cube)
        
        # Compute net longwave
        R_nl = self.compute_net_longwave(R_l_down, R_l_up)
        
        # Add to DataCube
        cube.add("R_l_up", R_l_up)
        cube.add("R_l_down", R_l_down)
        cube.add("R_nl", R_nl)
        
        return cube
    
    def _validate_inputs(self, cube: DataCube) -> None:
        """
        Validate that required inputs are present in DataCube.
        
        Args:
            cube: Input DataCube
            
        Raises:
            ValueError: If required inputs are missing
        """
        required = ["emissivity", "lwir11"]
        missing = [r for r in required if r not in cube.bands()]
        
        # Check for air temperature and vapor pressure
        if "air_temperature" not in cube.bands() and "air_temperature" not in cube.scalars():
            # Check metadata for temperature
            if "temperature_2m" not in cube.scalars() and "temperature_2m" not in cube.metadata:
                # Assume Ta is available
                pass
        
        if "vapor_pressure" not in cube.bands() and "vapor_pressure" not in cube.scalars():
            # Assume e_a is available
            pass
        
        if missing:
            raise ValueError(
                f"Missing required inputs for longwave radiation: {missing}. "
                f"Available: {cube.bands() + cube.scalars()}"
            )
    
    def compute_outgoing_longwave(
        self,
        Ts: xr.DataArray,
        emissivity: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute outgoing longwave radiation emitted by the surface.

        Uses the Stefan-Boltzmann law:
            R_l↑ = ε * σ * Ts⁴

        Args:
            Ts: Surface temperature in Kelvin
            emissivity: Surface emissivity (dimensionless)

        Returns:
            Outgoing longwave radiation (W/m²)
        """
        import logging
        logger = logging.getLogger(__name__)

        # Log temperature statistics
        logger.info(f"Outgoing LW: Ts min={Ts.min().values:.2f}, max={Ts.max().values:.2f}, mean={Ts.mean().values:.2f}")
        logger.info(f"Outgoing LW: emissivity min={emissivity.min().values:.3f}, max={emissivity.max().values:.3f}, mean={emissivity.mean().values:.3f}")

        # Stefan-Boltzmann law
        R_l_up = emissivity * STEFAN_BOLTZMANN * (Ts ** 4)

        logger.info(f"Outgoing LW: R_l_up min={R_l_up.min().values:.2f}, max={R_l_up.max().values:.2f}, mean={R_l_up.mean().values:.2f}")

        R_l_up.name = "R_l_up"
        R_l_up.attrs = {
            'long_name': 'Outgoing Longwave Radiation (Surface Emission)',
            'units': 'W/m²',
            'method': 'ε * σ * T⁴',
            'stefan_boltzmann_constant': STEFAN_BOLTZMANN
        }

        return R_l_up
    
    def compute_incoming_longwave(self, cube: DataCube) -> xr.DataArray:
        """
        Compute incoming longwave radiation from the atmosphere.

        Uses the Brunt equation for clear-sky emissivity:
            ε_clear = 0.70 + 0.005 * sqrt(e_a)
            R_l↓ = ε_clear * σ * Ta⁴

        With cloud correction:
            ε_clear = 0.23 + 0.433 * sqrt(e_a) * (1 - 0.14 * C) - 0.1

        Args:
            cube: Input DataCube with temperature and vapor pressure

        Returns:
            Incoming longwave radiation (W/m²)
        """
        import logging
        logger = logging.getLogger(__name__)

        # Get air temperature
        Ta = self._get_air_temperature(cube)

        # Get actual vapor pressure
        e_a = self._get_vapor_pressure(cube)

        logger.info(f"Incoming LW: Ta min={Ta.min().values:.2f}, max={Ta.max().values:.2f}, mean={Ta.mean().values:.2f}")
        logger.info(f"Incoming LW: e_a min={e_a.min().values:.2f}, max={e_a.max().values:.2f}, mean={e_a.mean().values:.2f}")

        # Get cloud cover if available and correction enabled
        C = 0.0
        if self.use_cloud_correction:
            cloud_cover = cube.get("cloud_cover")
            if cloud_cover is not None:
                C = cloud_cover
            else:
                # Try metadata
                C = cube.metadata.get("cloud_cover", 0.0)

        # Compute clear-sky emissivity
        if self.use_simplified_brunt:
            epsilon_clear = self._brunt_equation_simplified(e_a)
        else:
            epsilon_clear = self._brunt_equation_full(e_a, C)

        logger.info(f"Incoming LW: epsilon_clear={epsilon_clear.mean().values:.3f}")

        # Compute incoming longwave
        R_l_down = epsilon_clear * STEFAN_BOLTZMANN * (Ta ** 4)

        logger.info(f"Incoming LW: R_l_down min={R_l_down.min().values:.2f}, max={R_l_down.max().values:.2f}, mean={R_l_down.mean().values:.2f}")

        R_l_down.name = "R_l_down"
        R_l_down.attrs = {
            'long_name': 'Incoming Longwave Radiation (Atmospheric Emission)',
            'units': 'W/m²',
            'method': 'Brunt equation',
            'cloud_cover_fraction': float(C) if isinstance(C, (int, float)) else 'variable'
        }

        return R_l_down
    
    def _get_air_temperature(self, cube: DataCube) -> xr.DataArray:
        """
        Get air temperature in Kelvin from DataCube.

        Args:
            cube: Input DataCube

        Returns:
            Air temperature in Kelvin
        """
        # Try different keys
        Ta = cube.get("air_temperature")
        if Ta is None:
            Ta = cube.get("Ta")
        if Ta is None:
            Ta = cube.get("temperature_2m")
        if Ta is None:
            # Try from metadata
            temp_celsius = cube.metadata.get("temperature_2m", 25.0)
            Ta = xr.DataArray(
                np.full((cube.y_dim, cube.x_dim), temp_celsius + FREEZING_POINT),
                dims=['y', 'x']
            )

        # Convert to Kelvin if in Celsius
        if Ta.max() < 200:
            Ta = Ta + FREEZING_POINT

        return Ta
    
    def _get_vapor_pressure(self, cube: DataCube) -> xr.DataArray:
        """
        Get actual vapor pressure in kPa from DataCube.
        
        Args:
            cube: Input DataCube
            
        Returns:
            Actual vapor pressure in kPa
        """
        # Try direct vapor pressure
        e_a = cube.get("vapor_pressure")
        if e_a is None:
            e_a = cube.get("e_a")
        if e_a is None:
            e_a = cube.get("actual_vapor_pressure")
        
        if e_a is None:
            # Try to compute from dewpoint temperature
            Td = cube.get("dewpoint_temperature")
            if Td is None:
                Td = cube.metadata.get("Td")
            
            if Td is not None:
                if isinstance(Td, (int, float)):
                    # Scalar dewpoint - create array
                    e_a = self._compute_vapor_pressure(float(Td))
                    e_a = xr.DataArray(
                        np.full((cube.y_dim, cube.x_dim), e_a),
                        dims=['y', 'x']
                    )
                else:
                    # Array dewpoint
                    e_a = self._compute_vapor_pressure_array(Td)
            else:
                # Default value
                e_a = xr.DataArray(
                    np.full((cube.y_dim, cube.x_dim), 1.5),
                    dims=['y', 'x']
                )
        
        return e_a
    
    def _brunt_equation_simplified(self, e_a: xr.DataArray) -> xr.DataArray:
        """
        Compute clear-sky emissivity using simplified Brunt equation.
        
        ε_clear = 0.70 + 0.005 * sqrt(e_a)
        
        Args:
            e_a: Actual vapor pressure in kPa
            
        Returns:
            Clear-sky emissivity
        """
        epsilon_clear = 0.70 + 0.005 * np.sqrt(e_a)
        return epsilon_clear.clip(0.0, 1.0)
    
    def _brunt_equation_full(
        self, 
        e_a: xr.DataArray, 
        C: float = 0.0
    ) -> xr.DataArray:
        """
        Compute clear-sky emissivity using full Brunt equation.
        
        ε_clear = 0.23 + 0.433 * sqrt(e_a) * (1 - 0.14 * C) - 0.1
        
        Args:
            e_a: Actual vapor pressure in kPa
            C: Cloud cover fraction (0-1)
            
        Returns:
            Clear-sky emissivity
        """
        epsilon_clear = (
            0.23 + 
            0.433 * np.sqrt(e_a) * (1.0 - 0.14 * C) - 
            0.1
        )
        return epsilon_clear.clip(0.0, 1.0)
    
    def _compute_vapor_pressure(self, Td_celsius: float) -> float:
        """
        Compute actual vapor pressure from dewpoint temperature.
        
        Uses the Magnus formula:
            e_a = 0.6108 * exp(17.27 * Td / (Td + 237.3))
        
        Args:
            Td_celsius: Dewpoint temperature in °C
            
        Returns:
            Actual vapor pressure in kPa
        """
        return 0.6108 * np.exp(17.27 * Td_celsius / (Td_celsius + 237.3))
    
    def _compute_vapor_pressure_array(self, Td: xr.DataArray) -> xr.DataArray:
        """
        Compute actual vapor pressure from dewpoint temperature (array version).
        
        Args:
            Td: Dewpoint temperature in °C
            
        Returns:
            Actual vapor pressure in kPa
        """
        e_a = 0.6108 * np.exp(17.27 * Td / (Td + 237.3))
        return e_a
    
    def compute_net_longwave(
        self,
        R_l_down: xr.DataArray,
        R_l_up: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute net longwave radiation.

        R_nl = R_l↓ - R_l↑

        Args:
            R_l_down: Incoming longwave radiation (W/m²)
            R_l_up: Outgoing longwave radiation (W/m²)

        Returns:
            Net longwave radiation (W/m²)
        """
        import logging
        logger = logging.getLogger(__name__)

        R_nl = R_l_down - R_l_up

        logger.info(f"Net LW: R_nl min={R_nl.min().values:.2f}, max={R_nl.max().values:.2f}, mean={R_nl.mean().values:.2f}")

        R_nl.name = "R_nl"
        R_nl.attrs = {
            'long_name': 'Net Longwave Radiation',
            'units': 'W/m²',
            'method': 'R_l_down - R_l_up'
        }

        return R_nl
    
    def compute_atmospheric_emission_correction(
        self, 
        cube: DataCube
    ) -> xr.DataArray:
        """
        Compute correction factor for atmospheric downward emission.
        
        Accounts for atmospheric transmittance and view factor effects
        in the longwave radiation balance.
        
        Args:
            cube: Input DataCube
            
        Returns:
            Atmospheric emission correction factor
        """
        # Get inputs
        e_a = self._get_vapor_pressure(cube)
        C = cube.metadata.get("cloud_cover", 0.0)
        
        # Clear-sky emissivity
        epsilon_clear = self._brunt_equation_full(e_a, C)
        
        # Atmospheric transmittance
        tau_atm = 1.0 - epsilon_clear
        
        corr = tau_atm
        
        corr.name = "atm_emission_corr"
        corr.attrs = {
            'long_name': 'Atmospheric Emission Correction',
            'units': 'dimensionless'
        }
        
        return corr


# Alias for backward compatibility
Longwave = LongwaveRadiation


__all__ = ['LongwaveRadiation', 'Longwave']
