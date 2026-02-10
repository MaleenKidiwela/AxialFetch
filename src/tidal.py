"""
Tidal pressure correction: load tidal predictions, optimise amplitude and density
parameters, and remove tidal influence from pressure records.
"""
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from .config import TIDAL_FILES


def parse_to_dataframe(data):
    """Parse a tidal prediction text file into a DataFrame."""
    lines = data.split('\n')
    parsed_data = [line.split() for line in lines if line]
    df = pd.DataFrame(
        parsed_data,
        columns=['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Value'],
    )
    df['DateTime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
    df['Value'] = pd.to_numeric(df['Value'])
    df.drop(['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'], axis=1, inplace=True)
    return df


def load_tidal_df():
    """Load and concatenate all tidal prediction files."""
    dfs = []
    for fpath in TIDAL_FILES:
        with open(fpath, 'r') as f:
            dfs.append(parse_to_dataframe(f.read()))
    tidal_df = pd.concat(dfs, ignore_index=True)
    tidal_df.set_index('DateTime', inplace=True)
    return tidal_df


def _optimize_tidal_influence(params, tidal_df, pressure_df, adjusted_col):
    """Objective function for least_squares: minimise pressure variance."""
    amplitude, rho = params
    adjusted_tidal = amplitude * tidal_df['Value']
    tidal_df[adjusted_col] = adjusted_tidal
    interpolated_tidal = tidal_df.reindex(pressure_df.index, method='nearest')[adjusted_col]
    corrected_pressure = (
        pressure_df['Pressure (kPa)']
        - (rho * 9.81 * interpolated_tidal) * 0.001
    )
    return np.var(corrected_pressure)


def apply_tidal_correction(combined_df, tidal_df, adjusted_col='Adjusted Value'):
    """
    Apply tidal pressure correction to a combined station DataFrame.
    Adds 'Tidal Influence (kPa)' and 'Corrected Pressure (kPa)' columns.
    Returns the modified DataFrame.
    """
    combined_df = combined_df.copy()
    combined_df['DateTime'] = pd.to_datetime(combined_df['Record Time'])
    combined_df.set_index('DateTime', inplace=True)

    def optimize_tidal_influence(params, tidal_df, pressure_df):
        amplitude, rho = params
        adjusted_tidal = amplitude * tidal_df['Value']
        tidal_df[adjusted_col] = adjusted_tidal
        interpolated_tidal = tidal_df.reindex(pressure_df.index, method='nearest')[adjusted_col]
        corrected_pressure = (
            pressure_df['Pressure (kPa)']
            - (rho * 9.81 * interpolated_tidal) * 0.001
        )
        return np.var(corrected_pressure)

    initial_amplitude = 1
    initial_rho = 1025

    result = least_squares(
        optimize_tidal_influence,
        x0=[initial_amplitude, initial_rho],
        args=(tidal_df, combined_df),
    )

    optimized_amplitude, optimized_rho = result.x

    tidal_df[adjusted_col] = optimized_amplitude * tidal_df['Value']
    combined_df['Tidal Influence (kPa)'] = (
        optimized_rho
        * 9.81
        * tidal_df.reindex(combined_df.index, method='nearest')[adjusted_col]
    ) * 0.001
    combined_df['Corrected Pressure (kPa)'] = (
        combined_df['Pressure (kPa)'] - combined_df['Tidal Influence (kPa)']
    )

    combined_df.reset_index(inplace=True)
    return combined_df
