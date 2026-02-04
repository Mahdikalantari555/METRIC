"""
Evapotranspiration (ET) Computation Module for METRIC Model.

This module computes Actual Evapotranspiration (ETa) from latent heat flux,
including instantaneous ET, daily ET, and quality assessment.

Classes:
    - InstantaneousET: Convert latent heat flux to instantaneous ET
    - DailyET: Scale instantaneous ET to daily ET
    - ETQuality: ET quality flags and validation
    - ETQualityClass: Quality class enumerations

Formulas:
    1. ET_inst = LE / λ (mm/s)
    2. ETrF = ET_inst / ETr_inst (dimensionless)
    3. ET_daily = ETrF × ETr_daily (mm/day)

Quality Classes:
    | Class       | ETrF Range   | Description                     |
    |-------------|--------------|---------------------------------|
    | Excellent   | 0.8 - 1.2    | Well-watered vegetation         |
    | Good        | 0.6 - 2.0    | Normal conditions               |
    | Acceptable  | 0.3 - 2.5    | Some water stress               |
    | Poor        | 0.0 - 0.3    | Extreme stress or bare soil     |
    | Uncertain   | <0 or >1.5   | Requires review                 |
"""

from .instantaneous_et import InstantaneousET, create_instantaneous_et
from .daily_et import DailyET, create_daily_et
from .quality import ETQuality, ETQualityClass, create_et_quality

__all__ = [
    'InstantaneousET',
    'create_instantaneous_et',
    'DailyET',
    'create_daily_et',
    'ETQuality',
    'ETQualityClass',
    'create_et_quality'
]
