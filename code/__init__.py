"""
FETCH StreamLine Analysis Package

A clean, modular implementation of the FETCH StreamLine oceanographic
data processing and sound velocity analysis tools.

This package provides:
- Data processing utilities for oceanographic CSV files
- Sound velocity calculations using multiple methods (TEOS-10, basic, Chen-Millero)
- Positioning and tilt correction functions
- Optimization and fitting tools
- Statistical utilities and data analysis functions

Modules:
    data_processing: Load and process oceanographic data files
    sound_velocity: Calculate sound velocity in seawater
    positioning: Handle coordinate transformations and tilt corrections
    optimization: Parameter estimation and curve fitting
    utils: Statistical calculations and data utilities
"""

__version__ = "1.0.0"
__author__ = "FETCH StreamLine Analysis Team"

# Import main functions for convenience
from .data_processing import (
    process_data,
    create_nested_dictionary,
    ensure_datetime,
    set_datetime_index,
    remove_outliers
)

from .sound_velocity import (
    velocity_timeseries_TEOS10,
    velocity_timeseries_basic,
    velocity_timeseries_chen_millero,
    velocity_from_teos10,
    calculate_sensitivities_TEOS10,
    calculate_sensitivities_basic
)

from .positioning import (
    local_xy,
    to_enu,
    apply_tilt_correction,
    geodetic_to_enu,
    calculate_range_from_coordinates,
    build_baseline_perturbations
)

from .optimization import (
    fit_and_extrapolate,
    find_best_temperature_offset,
    optimize_parameters,
    fit_polynomial
)

from .utils import (
    calculate_harmonic_mean,
    compute_harmonic_mean,
    calculate_rms_error,
    calculate_statistics,
    normalize_data
)

from .range_salinity import (
    SalinityCalibrationConfig,
    GapMeanShiftConfig,
    prepare_salinity_series,
    bottle_residuals
)

from .tidal_correction import (
    load_tidal_predictions,
    optimize_tidal_correction,
    apply_tidal_correction
)

from .velocity_interpolation import interpolate_velocity

from .data_persistence import (
    save_dataframe_as_pickle,
    load_dataframe_from_pickle,
    save_processed_results,
    save_baseline_data,
    save_instrument_data,
    load_existing_pickles
)

__all__ = [
    # Data processing
    'process_data',
    'create_nested_dictionary',
    'ensure_datetime',
    'set_datetime_index',
    'remove_outliers',
    
    # Sound velocity
    'velocity_timeseries_TEOS10',
    'velocity_timeseries_basic', 
    'velocity_timeseries_chen_millero',
    'velocity_from_teos10',
    'calculate_sensitivities_TEOS10',
    'calculate_sensitivities_basic',
    
    # Positioning
    'local_xy',
    'to_enu',
    'apply_tilt_correction',
    'geodetic_to_enu',
    'calculate_range_from_coordinates',
    'build_baseline_perturbations',
    
    # Optimization
    'fit_and_extrapolate',
    'find_best_temperature_offset',
    'optimize_parameters',
    'fit_polynomial',
    
    # Utils
    'calculate_harmonic_mean',
    'compute_harmonic_mean',
    'calculate_rms_error',
    'calculate_statistics',
    'normalize_data',

    # Range calculation helpers
    'SalinityCalibrationConfig',
    'GapMeanShiftConfig',
    'prepare_salinity_series',
    'bottle_residuals',
    'load_tidal_predictions',
    'optimize_tidal_correction',
    'apply_tidal_correction',
    'interpolate_velocity',
    
    # Data persistence
    'save_dataframe_as_pickle',
    'load_dataframe_from_pickle',
    'save_processed_results',
    'save_baseline_data',
    'save_instrument_data',
    'load_existing_pickles'
]
