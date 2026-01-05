"""Land Surface Temperature (LST) calculations for METRIC ETa model."""

import numpy as np
import xarray as xr
from typing import Optional

from ..core.datacube import DataCube


class LSTCalculator:
    """
    Compute Land Surface Temperature (LST) for METRIC ETa model.

    LST is calculated from brightness temperature and surface emissivity
    using the METRIC-compatible formulation.

    Formula: Ts = Tb / (1 + (λ * Tb / ρ) * ln(ε))

    Where:
        Ts = Land Surface Temperature (K)
        Tb = Brightness Temperature (K) from thermal band
        ε = Surface emissivity (dimensionless)
        λ = Effective wavelength of thermal band (m)
        ρ = hc/σ = 1.438 × 10^-2 m·K

    For Landsat 8/9 Band 10 (TIRS):
        λ = 10.895 × 10^-6 m
        ρ = 1.438 × 10^-2 m·K
    """

    # Landsat 8/9 Band 10 constants (as provided in user specification)
    LAMBDA = 10.895e-6  # m (effective wavelength)
    RHO = 1.438e-2     # m·K (hc/σ)

    def __init__(self):
        """Initialize LSTCalculator."""
        pass

    def compute(self, cube: DataCube) -> DataCube:
        """
        Compute LST and add to DataCube.

        Args:
            cube: Input DataCube containing brightness temperature and emissivity

        Returns:
            DataCube with added LST band

        Raises:
            ValueError: If required inputs are missing
        """
        lst = self.compute_lst(cube)
        cube.add("lst", lst)

        return cube

    def compute_lst(
        self,
        cube: DataCube,
        brightness_temp: Optional[xr.DataArray] = None,
        emissivity: Optional[xr.DataArray] = None
    ) -> xr.DataArray:
        """
        Compute Land Surface Temperature using METRIC formulation.

        Args:
            cube: DataCube containing thermal data
            brightness_temp: Pre-computed brightness temperature (uses cube.get("lwir11") if not provided)
            emissivity: Pre-computed emissivity (uses cube.get("emissivity") if not provided)

        Returns:
            LST as xarray.DataArray in Kelvin

        Raises:
            ValueError: If required inputs are missing
        """
        if brightness_temp is None:
            if "lwir11" not in cube.bands():
                raise ValueError("Brightness temperature (lwir11) not found. Required for LST calculation.")
            brightness_temp = cube.get("lwir11")

        if emissivity is None:
            if "emissivity" not in cube.bands():
                raise ValueError("Emissivity not found. Compute emissivity first.")
            emissivity = cube.get("emissivity")

        # METRIC LST formula: Ts = Tb / (1 + (λ * Tb / ρ) * ln(ε))
        # Avoid division by zero and log of invalid values
        with np.errstate(invalid='ignore', divide='ignore'):
            # Calculate the correction term
            ln_emissivity = np.log(emissivity)
            correction_term = (self.LAMBDA * brightness_temp / self.RHO) * ln_emissivity

            # Apply LST formula
            lst = brightness_temp / (1.0 + correction_term)

        # Apply valid range mask (LST should be close to brightness temperature)
        # Brightness temperature range for Landsat: ~250-350 K
        valid_mask = (brightness_temp >= 200) & (brightness_temp <= 400) & \
                    (emissivity >= 0.8) & (emissivity <= 1.0)

        lst = lst.where(valid_mask, np.nan)

        # Clamp to reasonable LST range
        lst = lst.clip(250.0, 350.0)

        lst.name = "lst"
        lst.attrs = {
            'long_name': 'Land Surface Temperature',
            'units': 'K',
            'range': '[250.0, 350.0]',
            'method': 'METRIC LST from brightness temperature and emissivity',
            'thermal_band': 'lwir11',
            'wavelength': f'{self.LAMBDA*1e6:.3f} μm',
            'formula': 'Ts = Tb / (1 + (λ*Tb/ρ)*ln(ε))'
        }

        return lst


# Alias for backward compatibility
LandSurfaceTemperature = LSTCalculator