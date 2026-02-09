"""
Sound Velocity Calculation Module for FETCH Range Calculation

Provides TEOS-10 based sound velocity calculations using the GSW library,
along with sensitivity analysis and interpolation functions.
"""

from typing import List, Union, Tuple, Any
import numpy as np
import pandas as pd
import itertools
import gsw
from scipy.interpolate import interp1d

from .config import DEFAULT_LONGITUDE, DEFAULT_LATITUDE


def iterate_or_constant(value: Union[List, np.ndarray, float]) -> Any:
    """
    Return an iterator over timeseries or a constant value repeater.

    Args:
        value: Either a list/array of values or a single constant value

    Returns:
        Iterator over the values or repeating constant
    """
    if isinstance(value, (list, np.ndarray, pd.Series)):
        return value
    else:
        return itertools.repeat(value)


def velocity_from_teos10(
    temp: float,
    pressure_kPa: float,
    salinity: float,
    longitude: float = DEFAULT_LONGITUDE,
    latitude: float = DEFAULT_LATITUDE,
) -> float:
    """
    Calculate sound velocity from temperature, salinity, and pressure using TEOS-10.

    Args:
        temp: Temperature in degrees Celsius
        pressure_kPa: Pressure in kPa
        salinity: Practical salinity
        longitude: Longitude in decimal degrees
        latitude: Latitude in decimal degrees

    Returns:
        Sound velocity in m/s
    """
    # Convert pressure from kPa to dbar (1 dbar = 10 kPa)
    pressure_dbar = pressure_kPa * 0.1

    # Calculate Absolute Salinity (SA) and Conservative Temperature (CT)
    SA = gsw.SA_from_SP(salinity, pressure_dbar, longitude, latitude)
    CT = gsw.CT_from_t(SA, temp, pressure_dbar)

    # Calculate sound speed
    return gsw.sound_speed(SA, CT, pressure_dbar)


def velocity_timeseries_teos10(
    T_series: Union[List, np.ndarray, float],
    P_series_kPa: Union[List, np.ndarray, float],
    S_series: Union[List, np.ndarray, float],
    longitude: float = DEFAULT_LONGITUDE,
    latitude: float = DEFAULT_LATITUDE,
) -> List[float]:
    """
    Calculate sound velocity time series using GSW with TEOS-10 conversions.

    Uses Absolute Salinity and Conservative Temperature for more accurate
    oceanographic calculations.

    Args:
        T_series: Temperature values in degrees Celsius
        P_series_kPa: Pressure values in kPa
        S_series: Salinity values (practical salinity)
        longitude: Longitude in decimal degrees
        latitude: Latitude in decimal degrees

    Returns:
        List of sound velocity values in m/s
    """
    velocity_series = []

    for T, P_kPa, S in zip(
        iterate_or_constant(T_series),
        iterate_or_constant(P_series_kPa),
        iterate_or_constant(S_series),
    ):
        P_dbar = P_kPa * 0.1  # Convert kPa to dbar
        SA = gsw.SA_from_SP(S, P_dbar, longitude, latitude)
        CT = gsw.CT_from_t(SA, T, P_dbar)
        velocity_series.append(gsw.sound_speed(SA, CT, P_dbar))

    return velocity_series


def velocity_timeseries_basic(
    T_series: Union[List, np.ndarray, float],
    P_series_kPa: Union[List, np.ndarray, float],
    S_series: Union[List, np.ndarray, float],
) -> List[float]:
    """
    Calculate sound velocity using basic T, P, S parameters with GSW.

    Args:
        T_series: Temperature values in degrees Celsius
        P_series_kPa: Pressure values in kPa
        S_series: Salinity values (practical salinity)

    Returns:
        List of sound velocity values in m/s
    """
    velocity_series = []

    for T, P_kPa, S in zip(
        iterate_or_constant(T_series),
        iterate_or_constant(P_series_kPa),
        iterate_or_constant(S_series),
    ):
        P_dbar = P_kPa * 0.1  # Convert kPa to dbar
        velocity_series.append(gsw.sound_speed(S, T, P_dbar))

    return velocity_series


def sound_speed_chen_millero(S: float, T: float, P_bar: float) -> float:
    """
    Calculate sound speed using the Chen and Millero (1977) equation.

    Args:
        S: Salinity (practical salinity units)
        T: Temperature in degrees Celsius
        P_bar: Pressure in bar

    Returns:
        Sound speed in m/s
    """
    # Chen and Millero (1977) coefficients
    C00, C01, C02, C03, C04, C05 = 1402.388, 5.03830, -5.81090e-2, 3.3432e-4, -1.47797e-6, 3.1419e-9
    C10, C11, C12, C13, C14 = 0.153563, 6.8999e-4, -8.1829e-6, 1.3632e-7, -6.1260e-10
    C20, C21, C22, C23, C24 = 3.1260e-5, -1.7111e-6, 2.5986e-8, -2.5353e-10, 1.0415e-12
    C30, C31, C32 = -9.7729e-9, 3.8513e-10, -2.3654e-12

    A00, A01, A02, A03, A04 = 1.389, -1.262e-2, 7.166e-5, 2.008e-6, -3.21e-8
    A10, A11, A12, A13, A14 = 9.4742e-5, -1.2583e-5, -6.4928e-8, 1.0515e-8, -2.0142e-10
    A20, A21, A22, A23 = -3.9064e-7, 9.1061e-9, -1.6009e-10, 7.994e-12
    A30, A31, A32 = 1.100e-10, 6.651e-12, -3.391e-13

    B00, B01, B10, B11 = -1.922e-2, -4.42e-5, 7.3637e-5, 1.7950e-7
    D00, D10 = 1.727e-3, -7.9836e-6

    # Calculate Cw (sound speed in pure water)
    Cw = (
        C00 + C01 * T + C02 * T**2 + C03 * T**3 + C04 * T**4 + C05 * T**5
        + (C10 + C11 * T + C12 * T**2 + C13 * T**3 + C14 * T**4) * P_bar
        + (C20 + C21 * T + C22 * T**2 + C23 * T**3 + C24 * T**4) * P_bar**2
        + (C30 + C31 * T + C32 * T**2) * P_bar**3
    )

    # Calculate A (salinity effect)
    A = (
        A00 + A01 * T + A02 * T**2 + A03 * T**3 + A04 * T**4
        + (A10 + A11 * T + A12 * T**2 + A13 * T**3 + A14 * T**4) * P_bar
        + (A20 + A21 * T + A22 * T**2 + A23 * T**3) * P_bar**2
        + (A30 + A31 * T + A32 * T**2) * P_bar**3
    ) * S

    # Calculate B and D (higher order salinity effects)
    B = (B00 + B01 * T + (B10 + B11 * T) * P_bar) * S ** (3 / 2)
    D = (D00 + D10 * P_bar) * S**2

    return Cw + A + B + D


def velocity_timeseries_chen_millero(
    T_series: Union[List, np.ndarray, float],
    P_series_kPa: Union[List, np.ndarray, float],
    S_series: Union[List, np.ndarray, float],
) -> List[float]:
    """
    Calculate sound velocity time series using the Chen and Millero equation.

    Args:
        T_series: Temperature values in degrees Celsius
        P_series_kPa: Pressure values in kPa
        S_series: Salinity values (practical salinity)

    Returns:
        List of sound velocity values in m/s
    """
    velocity_series = []

    for T, P_kPa, S in zip(
        iterate_or_constant(T_series),
        iterate_or_constant(P_series_kPa),
        iterate_or_constant(S_series),
    ):
        P_bar = P_kPa * 0.01  # Convert kPa to bar
        velocity_series.append(sound_speed_chen_millero(S, T, P_bar))

    return velocity_series


def interpolate_velocity(
    time_series: Any,
    velocity_df: pd.DataFrame,
    time_col: str = "Record Time",
    velocity_col: str = "SoundSpeed",
) -> np.ndarray:
    """
    Interpolate velocity values for given time series based on velocity DataFrame.

    Args:
        time_series: Time values for interpolation
        velocity_df: DataFrame containing time and velocity columns
        time_col: Name of the time column
        velocity_col: Name of the velocity column

    Returns:
        Array of interpolated velocity values
    """
    df = velocity_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    time_numeric = df[time_col].astype("int64")

    interpolator = interp1d(
        time_numeric,
        df[velocity_col],
        fill_value="extrapolate",
        bounds_error=False,
    )

    return interpolator(pd.to_datetime(time_series).astype("int64"))


def calculate_sensitivities(
    salinity: float,
    temperature: float,
    pressure_kPa: float,
    longitude: float = DEFAULT_LONGITUDE,
    latitude: float = DEFAULT_LATITUDE,
    delta: float = 0.1,
) -> Tuple[float, float, float, float]:
    """
    Calculate sensitivity coefficients for sound velocity with respect to
    salinity, temperature, and pressure using TEOS-10.

    Args:
        salinity: Practical salinity
        temperature: Temperature in degrees Celsius
        pressure_kPa: Pressure in kPa
        longitude: Longitude in decimal degrees
        latitude: Latitude in decimal degrees
        delta: Small change for sensitivity calculation

    Returns:
        Tuple of (baseline_velocity, sensitivity_S, sensitivity_T, sensitivity_P)
    """
    pressure_dbar = pressure_kPa * 0.1

    SA_ref = gsw.SA_from_SP(salinity, pressure_dbar, longitude, latitude)
    CT_ref = gsw.CT_from_t(SA_ref, temperature, pressure_dbar)
    baseline_velocity = gsw.sound_speed(SA_ref, CT_ref, pressure_dbar)

    # Sensitivity to salinity
    SA_new = gsw.SA_from_SP(salinity + delta, pressure_dbar, longitude, latitude)
    sensitivity_S = (
        gsw.sound_speed(SA_new, CT_ref, pressure_dbar) - baseline_velocity
    ) / delta

    # Sensitivity to temperature
    CT_new = gsw.CT_from_t(SA_ref, temperature + delta, pressure_dbar)
    sensitivity_T = (
        gsw.sound_speed(SA_ref, CT_new, pressure_dbar) - baseline_velocity
    ) / delta

    # Sensitivity to pressure
    sensitivity_P = (
        gsw.sound_speed(SA_ref, CT_ref, pressure_dbar + delta) - baseline_velocity
    ) / delta

    return baseline_velocity, sensitivity_S, sensitivity_T, sensitivity_P


def add_velocity_to_combined_df(
    combined_df: pd.DataFrame,
    longitude: float = DEFAULT_LONGITUDE,
    latitude: float = DEFAULT_LATITUDE,
) -> pd.DataFrame:
    """
    Add TEOS-10 calculated velocity column to a combined DataFrame.

    Args:
        combined_df: DataFrame with Temperature Deg C, Pressure (kPa), and Salinity columns
        longitude: Longitude for TEOS-10 calculation
        latitude: Latitude for TEOS-10 calculation

    Returns:
        DataFrame with added 'velocity' column
    """
    df = combined_df.copy()

    T_series = np.array(df["Temperature Deg C"])
    P_series = np.array(df["Pressure (kPa)"])
    S_series = np.array(df["Salinity"])

    velocities = velocity_timeseries_teos10(
        T_series, P_series, S_series, longitude, latitude
    )
    df["velocity"] = velocities

    return df
