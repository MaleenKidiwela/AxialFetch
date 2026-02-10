"""
Sound-speed / velocity computation: interpolation from SSP records,
and TEOS-10 calculation using gsw.
"""
import numpy as np
import pandas as pd
import itertools
import gsw
from scipy.interpolate import interp1d
from .config import LONGITUDE, LATITUDE


def iterate_or_constant(value):
    """Iterate over a timeseries or repeat a constant."""
    if isinstance(value, (list, np.ndarray)):
        return value
    else:
        return itertools.repeat(value)


def velocity_timeseries_TEOS10(T_series, P_series_kPa, S_series):
    """Compute sound speed using TEOS-10 conversions via gsw."""
    velocity_series = []
    for T, P_kPa, S in zip(
        iterate_or_constant(T_series),
        iterate_or_constant(P_series_kPa),
        iterate_or_constant(S_series),
    ):
        P_dbar = P_kPa * 0.1
        SA = gsw.SA_from_SP(S, P_dbar, LONGITUDE, LATITUDE)
        CT = gsw.CT_from_t(SA, T, P_dbar)
        velocity_series.append(gsw.sound_speed(SA, CT, P_dbar))
    return velocity_series


def interpolate_velocity(time_series, velocity_df):
    """Interpolate SSP SoundSpeed onto a given time_series."""
    velocity_df = velocity_df.copy()
    velocity_df['Record Time'] = pd.to_datetime(velocity_df['Record Time'])
    time_numeric = velocity_df['Record Time'].astype('int64')

    interpolator = interp1d(
        time_numeric,
        velocity_df['SoundSpeed'],
        fill_value='extrapolate',
        bounds_error=False,
    )

    return interpolator(pd.to_datetime(time_series).astype('int64'))


def add_velocities(combined_df, result_dfs, identifier):
    """
    Add interpolated SSP velocity (interp_v) and TEOS-10 velocity to a
    combined station DataFrame. Returns the modified DataFrame.
    """
    combined_df = combined_df.copy()

    # Interpolated SSP velocity
    combined_df['Record Time'] = pd.to_datetime(combined_df['Record Time'])
    combined_df['interp_v'] = interpolate_velocity(
        combined_df['Record Time'], result_dfs[identifier]
    )

    # TEOS-10 velocity
    T_series = np.array(combined_df['Temperature Deg C'])
    P_series_kPa = np.array(combined_df['Pressure (kPa)'])
    S_constant = np.array(combined_df['Salinity'])

    teos10_vel = velocity_timeseries_TEOS10(T_series, P_series_kPa, S_constant)
    combined_df['velocity'] = teos10_vel
    combined_df.reset_index(inplace=True)

    return combined_df
