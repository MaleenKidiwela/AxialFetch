"""
Sound Velocity Calculation Module for FETCH StreamLine Analysis

This module contains functions for calculating sound velocity in seawater
using various methods including TEOS-10, basic calculations, and Chen-Millero.
"""

from typing import List, Union, Any
import numpy as np
import itertools
import gsw
from scipy.interpolate import interp1d
import pandas as pd


# Default coordinates for FETCH project
DEFAULT_LONGITUDE = -130.01149966
DEFAULT_LATITUDE = 45.95882833


def iterate_or_constant(value: Union[List, np.ndarray, float]) -> Any:
    """
    Return an iterator over timeseries or a constant value repeater.
    
    Args:
        value: Either a list/array of values or a single constant value
        
    Returns:
        Iterator over the values or repeating constant
    """
    if isinstance(value, (list, np.ndarray)):
        return value
    else:
        return itertools.repeat(value)


def velocity_timeseries_TEOS10(
    T_series: Union[List, np.ndarray, float], 
    P_series_kPa: Union[List, np.ndarray, float], 
    S_series: Union[List, np.ndarray, float],
    longitude: float = DEFAULT_LONGITUDE,
    latitude: float = DEFAULT_LATITUDE
) -> List[float]:
    """
    Calculate sound velocity using GSW with TEOS-10 conversions.
    
    Uses Absolute Salinity and Conservative Temperature for more accurate
    oceanographic calculations.
    
    Args:
        T_series: Temperature values in degrees Celsius
        P_series_kPa: Pressure values in kPa
        S_series: Salinity values (practical salinity)
        longitude: Longitude in decimal degrees (default: FETCH project location)
        latitude: Latitude in decimal degrees (default: FETCH project location)
        
    Returns:
        List of sound velocity values in m/s
    """
    velocity_series = []
    
    for T, P_kPa, S in zip(
        iterate_or_constant(T_series), 
        iterate_or_constant(P_series_kPa), 
        iterate_or_constant(S_series)
    ):
        P_dbar = P_kPa * 0.1  # Convert kPa to dbar
        SA = gsw.SA_from_SP(S, P_dbar, longitude, latitude)
        CT = gsw.CT_from_t(SA, T, P_dbar)
        velocity_series.append(gsw.sound_speed(SA, CT, P_dbar))
        
    return velocity_series


def velocity_timeseries_basic(
    T_series: Union[List, np.ndarray, float], 
    P_series_kPa: Union[List, np.ndarray, float], 
    S_series: Union[List, np.ndarray, float]
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
        iterate_or_constant(S_series)
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
    C00 = 1402.388
    C01 = 5.03830
    C02 = -5.81090e-2
    C03 = 3.3432e-4
    C04 = -1.47797e-6
    C05 = 3.1419e-9
    
    C10 = 0.153563
    C11 = 6.8999e-4
    C12 = -8.1829e-6
    C13 = 1.3632e-7
    C14 = -6.1260e-10
    
    C20 = 3.1260e-5
    C21 = -1.7111e-6
    C22 = 2.5986e-8
    C23 = -2.5353e-10
    C24 = 1.0415e-12
    
    C30 = -9.7729e-9
    C31 = 3.8513e-10
    C32 = -2.3654e-12
    
    A00 = 1.389
    A01 = -1.262e-2
    A02 = 7.166e-5
    A03 = 2.008e-6
    A04 = -3.21e-8
    
    A10 = 9.4742e-5
    A11 = -1.2583e-5
    A12 = -6.4928e-8
    A13 = 1.0515e-8
    A14 = -2.0142e-10
    
    A20 = -3.9064e-7
    A21 = 9.1061e-9
    A22 = -1.6009e-10
    A23 = 7.994e-12
    
    A30 = 1.100e-10
    A31 = 6.651e-12
    A32 = -3.391e-13
    
    B00 = -1.922e-2
    B01 = -4.42e-5
    B10 = 7.3637e-5
    B11 = 1.7950e-7
    
    D00 = 1.727e-3
    D10 = -7.9836e-6
    
    # Calculate Cw (sound speed in pure water)
    Cw = (C00 + C01*T + C02*T**2 + C03*T**3 + C04*T**4 + C05*T**5 +
          (C10 + C11*T + C12*T**2 + C13*T**3 + C14*T**4)*P_bar +
          (C20 + C21*T + C22*T**2 + C23*T**3 + C24*T**4)*P_bar**2 +
          (C30 + C31*T + C32*T**2)*P_bar**3)
    
    # Calculate A (salinity effect)
    A = (A00 + A01*T + A02*T**2 + A03*T**3 + A04*T**4 +
         (A10 + A11*T + A12*T**2 + A13*T**3 + A14*T**4)*P_bar +
         (A20 + A21*T + A22*T**2 + A23*T**3)*P_bar**2 +
         (A30 + A31*T + A32*T**2)*P_bar**3)*S
    
    # Calculate B and D (higher order salinity effects)
    B = (B00 + B01*T + (B10 + B11*T)*P_bar)*S**(3/2)
    D = (D00 + D10*P_bar)*S**2
    
    # Calculate final sound speed
    sound_speed = Cw + A + B + D
    
    return sound_speed


def velocity_timeseries_chen_millero(
    T_series: Union[List, np.ndarray, float], 
    P_series_kPa: Union[List, np.ndarray, float], 
    S_series: Union[List, np.ndarray, float]
) -> List[float]:
    """
    Calculate sound velocity using the Chen and Millero equation.
    
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
        iterate_or_constant(S_series)
    ):
        P_bar = P_kPa * 0.01  # Convert kPa to bar
        velocity_series.append(sound_speed_chen_millero(S, T, P_bar))
        
    return velocity_series


def velocity_from_teos10(
    temp: float, 
    salinity: float, 
    pressure: float,
    longitude: float = DEFAULT_LONGITUDE,
    latitude: float = DEFAULT_LATITUDE
) -> float:
    """
    Calculate velocity from temperature, salinity, and pressure using TEOS-10.
    
    Args:
        temp: Temperature in degrees Celsius
        salinity: Salinity (practical salinity units)
        pressure: Pressure in kPa
        longitude: Longitude in decimal degrees
        latitude: Latitude in decimal degrees
        
    Returns:
        Sound velocity in m/s
    """
    # Convert pressure from kPa to dbar (1 dbar = 10 kPa)
    pressure_dbar = pressure * 0.1
    
    # Calculate Absolute Salinity (SA) and Conservative Temperature (CT)
    SA = gsw.SA_from_SP(salinity, pressure_dbar, longitude, latitude)
    CT = gsw.CT_from_t(SA, temp, pressure_dbar)
    
    # Calculate sound speed
    return gsw.sound_speed(SA, CT, pressure_dbar)


def interpolate_velocity(time_series: Any, velocity_df: pd.DataFrame) -> np.ndarray:
    """
    Interpolate velocity values for given time series based on velocity DataFrame.
    
    Args:
        time_series: Time values for interpolation
        velocity_df: DataFrame containing 'Record Time' and 'SoundSpeed' columns
        
    Returns:
        Array of interpolated velocity values
    """
    # Convert 'Record Time' to numeric for interpolation
    velocity_df['Record Time'] = pd.to_datetime(velocity_df['Record Time'])
    time_numeric = velocity_df['Record Time'].astype('int64')
    
    # Interpolator function
    interpolator = interp1d(
        time_numeric, 
        velocity_df['SoundSpeed'], 
        fill_value='extrapolate', 
        bounds_error=False
    )
    
    # Apply interpolation
    return interpolator(pd.to_datetime(time_series).astype('int64'))


def calculate_sensitivities_TEOS10(
    salinity: float, 
    temperature: float, 
    pressure_dbar: float, 
    longitude: float, 
    latitude: float,
    delta: float = 0.1
) -> tuple:
    """
    Calculate sensitivity coefficients for sound velocity with respect to 
    salinity, temperature, and pressure using TEOS-10.
    
    Args:
        salinity: Practical salinity
        temperature: Temperature in degrees Celsius
        pressure_dbar: Pressure in dbar
        longitude: Longitude in decimal degrees
        latitude: Latitude in decimal degrees
        delta: Small change for sensitivity calculation
        
    Returns:
        Tuple of (baseline_velocity, sensitivity_S, sensitivity_T, sensitivity_P)
    """
    SA_ref = gsw.SA_from_SP(salinity, pressure_dbar, longitude, latitude)
    CT_ref = gsw.CT_from_t(SA_ref, temperature, pressure_dbar)
    baseline_velocity = gsw.sound_speed(SA_ref, CT_ref, pressure_dbar)

    # Sensitivity to salinity
    SA_new = gsw.SA_from_SP(salinity + delta, pressure_dbar, longitude, latitude)
    sensitivity_S = (gsw.sound_speed(SA_new, CT_ref, pressure_dbar) - baseline_velocity) / delta

    # Sensitivity to temperature
    CT_new = gsw.CT_from_t(SA_ref, temperature + delta, pressure_dbar)
    sensitivity_T = (gsw.sound_speed(SA_ref, CT_new, pressure_dbar) - baseline_velocity) / delta

    # Sensitivity to pressure
    sensitivity_P = (gsw.sound_speed(SA_ref, CT_ref, pressure_dbar + delta) - baseline_velocity) / delta

    return baseline_velocity, sensitivity_S, sensitivity_T, sensitivity_P


def calculate_sensitivities_basic(
    salinity: float, 
    temperature: float, 
    pressure_dbar: float,
    delta: float = 0.1
) -> tuple:
    """
    Calculate sensitivity coefficients for sound velocity using basic (non-TEOS-10) approach.
    
    Args:
        salinity: Practical salinity
        temperature: Temperature in degrees Celsius
        pressure_dbar: Pressure in dbar
        delta: Small change for sensitivity calculation
        
    Returns:
        Tuple of (baseline_velocity, sensitivity_S, sensitivity_T, sensitivity_P)
    """
    baseline_velocity = gsw.sound_speed(salinity, temperature, pressure_dbar)

    # Sensitivity to salinity
    sensitivity_S = (gsw.sound_speed(salinity + delta, temperature, pressure_dbar) - baseline_velocity) / delta

    # Sensitivity to temperature
    sensitivity_T = (gsw.sound_speed(salinity, temperature + delta, pressure_dbar) - baseline_velocity) / delta

    # Sensitivity to pressure
    sensitivity_P = (gsw.sound_speed(salinity, temperature, pressure_dbar + delta) - baseline_velocity) / delta

    return baseline_velocity, sensitivity_S, sensitivity_T, sensitivity_P