"""
FETCH Range Calculation Package

A modular refactoring of the FETCH Range Calculation notebook for
oceanographic acoustic range measurement and deformation analysis.
"""

from . import config
from . import data_loading
from . import velocity
from . import salinity
from . import pressure
from . import range_calculation
from . import triangle
from . import figures
from . import pipeline

__version__ = "1.0.0"
__all__ = [
    "config",
    "data_loading",
    "velocity",
    "salinity",
    "pressure",
    "range_calculation",
    "triangle",
    "figures",
    "pipeline",
]
