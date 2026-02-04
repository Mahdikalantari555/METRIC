"""Shortwave radiation calculations for METRIC ETa model."""

import numpy as np
import xarray as xr
from typing import Optional

from ..core.datacube import DataCube
from ..core.constants import STEFAN_BOLTZMANN


class ShortwaveRadiation:
    """
    Compute shortwave radiation components for energy balance.
    
    This class calculates incoming shortwave radiation (Rs↓), outgoing 
    shortwave reflection (R_s↑), and net shortwave radiation (R_ns).
    
    The net shortwave radiation is the difference between incoming and 
    reflected shortwave radiation, representing the solar energy absorbed
    by the surface.
    
    Attributes:
        apply_sun_angle_correction: Apply solar zenith angle correction
    """
    
    def __init__(self, apply_sun_angle_correction: bool = True):
        """
        Initialize ShortwaveRadiation calculator.
        
        Args:
            apply_sun_angle_correction: Apply solar zenith angle correction
        """
        self.apply_sun_angle_correction = apply_sun_angle_correction
    
    def compute(self, cube: DataCube) -> DataCube:
        """
        Compute all shortwave radiation components and add to DataCube.
        
        Required inputs:
            - shortwave_radiation: Incoming shortwave from weather data (W/m²)
            - albedo: Surface broadband albedo (from albedo module)
            - sun_elevation: Sun elevation angle in degrees (from MTL)
        
        Outputs:
            - Rs_down: Incoming shortwave radiation (W/m²)
            - R_s_up: Outgoing/reflected shortwave radiation (W/m²)
            - R_ns: Net shortwave radiation (W/m²)
        
        Args:
            cube: Input DataCube with required bands
            
        Returns:
            DataCube with added shortwave radiation components
            
        Raises:
            ValueError: If required inputs are missing
        """
        # Validate required inputs
        self._validate_inputs(cube)
        
        # Get required data
        shortwave_rad = cube.get("shortwave_radiation")
        albedo = cube.get("albedo")
        
        # Apply sun angle correction if enabled
        if self.apply_sun_angle_correction:
            shortwave_rad = self._apply_sun_angle_correction(cube, shortwave_rad)
        
        # Compute outgoing shortwave (reflected)
        R_s_up = self.compute_outgoing_shortwave(shortwave_rad, albedo)
        
        # Compute net shortwave
        R_ns = self.compute_net_shortwave(shortwave_rad, albedo)
        
        # Add to DataCube
        cube.add("Rs_down", shortwave_rad)
        cube.add("R_s_up", R_s_up)
        cube.add("R_ns", R_ns)
        
        return cube
    
    def _validate_inputs(self, cube: DataCube) -> None:
        """
        Validate that required inputs are present in DataCube.
        
        Args:
            cube: Input DataCube
            
        Raises:
            ValueError: If required inputs are missing
        """
        required = ["shortwave_radiation", "albedo"]
        missing = [r for r in required if r not in cube.bands() and r not in cube.scalars()]
        
        if missing:
            # Check for sun_elevation if sun angle correction is enabled
            if self.apply_sun_angle_correction:
                if "sun_elevation" not in cube.bands() and "sun_elevation" not in cube.scalars():
                    missing.append("sun_elevation (required for sun angle correction)")
            
            raise ValueError(
                f"Missing required inputs for shortwave radiation: {missing}. "
                f"Available: {cube.bands() + cube.scalars()}"
            )
    
    def _apply_sun_angle_correction(
        self, 
        cube: DataCube, 
        shortwave_rad: xr.DataArray
    ) -> xr.DataArray:
        """
        Apply solar zenith angle correction to shortwave radiation.
        
        Uses the cosine of the solar zenith angle to adjust the incoming
        radiation based on sun position:
            cos(θz) = sin(sun_elevation)
            Rs↓_corrected = Rs↓ * cos(θz)
        
        Args:
            cube: Input DataCube containing sun_elevation
            shortwave_rad: Original shortwave radiation
            
        Returns:
            Sun angle corrected shortwave radiation
        """
        sun_elevation = cube.get("sun_elevation")
        
        if sun_elevation is None:
            # Fallback: try from metadata
            sun_elevation = cube.metadata.get("sun_elevation", 45.0)
        
        # Convert elevation to zenith angle and compute cosine
        sun_elevation_rad = np.deg2rad(sun_elevation)
        cos_zenith = np.sin(sun_elevation_rad)
        
        # Ensure cos_zenith is positive (sun above horizon)
        cos_zenith = np.maximum(cos_zenith, 0.0)
        
        # Apply correction
        corrected_rad = shortwave_rad * cos_zenith
        
        corrected_rad.name = "Rs_down_corrected"
        corrected_rad.attrs = {
            'long_name': 'Incoming Shortwave Radiation (Sun Angle Corrected)',
            'units': 'W/m²',
            'sun_elevation_deg': sun_elevation if isinstance(sun_elevation, (int, float)) else 'variable',
            'cos_zenith_factor': 'applied'
        }
        
        return corrected_rad
    
    def compute_outgoing_shortwave(
        self, 
        Rs_down: xr.DataArray, 
        albedo: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute outgoing (reflected) shortwave radiation.
        
        Uses the surface albedo to calculate reflected shortwave:
            R_s↑ = α * Rs↓
        
        Args:
            Rs_down: Incoming shortwave radiation (W/m²)
            albedo: Surface broadband albedo (dimensionless)
            
        Returns:
            Outgoing shortwave radiation (W/m²)
        """
        R_s_up = albedo * Rs_down
        
        R_s_up.name = "R_s_up"
        R_s_up.attrs = {
            'long_name': 'Outgoing Shortwave Radiation (Reflected)',
            'units': 'W/m²',
            'method': 'albedo * Rs_down'
        }
        
        return R_s_up
    
    def compute_net_shortwave(
        self, 
        Rs_down: xr.DataArray, 
        albedo: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute net shortwave radiation absorbed by the surface.
        
        Net shortwave is the incoming radiation minus the reflected portion:
            R_ns = (1 - α) * Rs↓
        
        Args:
            Rs_down: Incoming shortwave radiation (W/m²)
            albedo: Surface broadband albedo (dimensionless)
            
        Returns:
            Net shortwave radiation (W/m²)
        """
        R_ns = (1.0 - albedo) * Rs_down
        
        R_ns.name = "R_ns"
        R_ns.attrs = {
            'long_name': 'Net Shortwave Radiation',
            'units': 'W/m²',
            'method': '(1 - albedo) * Rs_down'
        }
        
        return R_ns
    
    def compute_direct_diffuse_split(
        self, 
        Rs_down: xr.DataArray,
        sun_elevation: float,
        atmospheric_transmittance: float = 0.75
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Split total shortwave into direct and diffuse components.
        
        Uses a simple model based on solar elevation and atmospheric
        transmittance to estimate the direct and diffuse fractions.
        
        Args:
            Rs_down: Total incoming shortwave (W/m²)
            sun_elevation: Sun elevation angle in degrees
            atmospheric_transmittance: Clear-sky transmittance (default: 0.75)
            
        Returns:
            Tuple of (direct_component, diffuse_component) DataArrays
        """
        # Calculate solar zenith angle
        zenith_angle = 90.0 - sun_elevation
        
        # Cosine of zenith angle
        mu0 = np.cos(np.deg2rad(zenith_angle))
        mu0 = np.maximum(mu0, 0.05)  # Avoid very small values
        
        # Direct radiation follows Beer-Lambert law
        # Diffuse is the remaining portion
        tau = atmospheric_transmittance
        
        # Fraction of direct radiation
        f_direct = tau ** (1.0 / mu0)
        f_direct = np.clip(f_direct, 0.0, 1.0)
        
        # Direct and diffuse components
        F_direct = f_direct * Rs_down
        F_diffuse = (1.0 - f_direct) * Rs_down
        
        F_direct.name = "Rs_down_direct"
        F_diffuse.name = "Rs_down_diffuse"
        
        F_direct.attrs = {'long_name': 'Direct Shortwave Radiation', 'units': 'W/m²'}
        F_diffuse.attrs = {'long_name': 'Diffuse Shortwave Radiation', 'units': 'W/m²'}
        
        return F_direct, F_diffuse


# Alias for backward compatibility
Shortwave = ShortwaveRadiation


__all__ = ['ShortwaveRadiation', 'Shortwave']
