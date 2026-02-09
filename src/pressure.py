"""
Pressure and Tidal Correction Module for FETCH Range Calculation

Handles tidal signal removal, pressure anomaly calculation,
and node pressure data loading.
"""

from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import glob

from .config import PSI_TO_KPA, GRAVITY


def parse_tidal_file(data: str) -> pd.DataFrame:
    """
    Parse tidal prediction text data into a DataFrame.

    Args:
        data: String containing space-separated tidal prediction data
              with columns: Year Month Day Hour Minute Second Value

    Returns:
        DataFrame with DateTime index and Value column
    """
    lines = data.split("\n")
    parsed_data = [line.split() for line in lines if line]
    df = pd.DataFrame(
        parsed_data,
        columns=["Year", "Month", "Day", "Hour", "Minute", "Second", "Value"],
    )
    df["DateTime"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour", "Minute", "Second"]]
    )
    df["Value"] = pd.to_numeric(df["Value"])
    df.drop(["Year", "Month", "Day", "Hour", "Minute", "Second"], axis=1, inplace=True)
    return df


def load_tidal_predictions(filepaths: List[str]) -> pd.DataFrame:
    """
    Load and combine tidal prediction files.

    Args:
        filepaths: List of paths to tidal prediction text files

    Returns:
        Combined DataFrame with DateTime index and Value column
    """
    dfs = []
    for filepath in filepaths:
        with open(filepath, "r") as f:
            data = f.read()
        dfs.append(parse_tidal_file(data))

    tidal_df = pd.concat(dfs, ignore_index=True)
    tidal_df.set_index("DateTime", inplace=True)
    return tidal_df


def optimize_tidal_influence(
    params: Tuple[float, float],
    tidal_df: pd.DataFrame,
    pressure_df: pd.DataFrame,
    pressure_col: str = "Pressure (kPa)",
) -> float:
    """
    Objective function for optimizing tidal amplitude and density.

    Minimizes variance of tide-corrected pressure.

    Args:
        params: Tuple of (amplitude_scale, density)
        tidal_df: DataFrame with tidal predictions
        pressure_df: DataFrame with pressure measurements
        pressure_col: Name of pressure column

    Returns:
        Variance of corrected pressure
    """
    amplitude, rho = params
    adjusted_tidal = amplitude * tidal_df["Value"]
    tidal_df = tidal_df.copy()
    tidal_df["Adjusted Value"] = adjusted_tidal
    interpolated_tidal = tidal_df.reindex(pressure_df.index, method="nearest")[
        "Adjusted Value"
    ]
    corrected_pressure = pressure_df[pressure_col] - (
        rho * GRAVITY * interpolated_tidal
    ) * 0.001
    return np.var(corrected_pressure)


def remove_tidal_signal(
    combined_df: pd.DataFrame,
    tidal_df: pd.DataFrame,
    pressure_col: str = "Pressure (kPa)",
    initial_amplitude: float = 1.0,
    initial_rho: float = 1025.0,
) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    """
    Remove tidal signal from pressure data using optimization.

    Args:
        combined_df: DataFrame with pressure data and DateTime index
        tidal_df: DataFrame with tidal predictions
        pressure_col: Name of pressure column
        initial_amplitude: Initial guess for tidal amplitude scaling
        initial_rho: Initial guess for seawater density (kg/m^3)

    Returns:
        Tuple of (corrected_df, (optimized_amplitude, optimized_rho))
    """
    df = combined_df.copy()

    # Ensure datetime index
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df.set_index("DateTime", inplace=True)
    elif "Record Time" in df.columns:
        df["DateTime"] = pd.to_datetime(df["Record Time"])
        df.set_index("DateTime", inplace=True)

    # Perform optimization
    result = least_squares(
        lambda p: optimize_tidal_influence(p, tidal_df.copy(), df, pressure_col),
        x0=[initial_amplitude, initial_rho],
    )

    optimized_amplitude, optimized_rho = result.x

    # Apply optimized parameters
    tidal_df_adj = tidal_df.copy()
    tidal_df_adj["Adjusted Value"] = optimized_amplitude * tidal_df_adj["Value"]
    df["Tidal Influence (kPa)"] = (
        optimized_rho
        * GRAVITY
        * tidal_df_adj.reindex(df.index, method="nearest")["Adjusted Value"]
    ) * 0.001
    df["Corrected Pressure (kPa)"] = df[pressure_col] - df["Tidal Influence (kPa)"]

    # Reset index
    df.reset_index(inplace=True)

    return df, (optimized_amplitude, optimized_rho)


def calculate_moving_average(
    df: pd.DataFrame,
    value_col: str = "Corrected Pressure (kPa)",
    window: str = "15D",
    time_col: str = "DateTime",
) -> pd.DataFrame:
    """
    Compute moving average of pressure anomalies.

    Args:
        df: DataFrame with pressure data
        value_col: Name of value column
        window: Rolling window specification (e.g., '15D', '30D')
        time_col: Name of time column

    Returns:
        DataFrame with added 'pressure_demeaned' and 'ma' columns
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df.sort_values(time_col, inplace=True)

    mu = df[value_col].mean(skipna=True)
    df["pressure_demeaned"] = df[value_col] - mu
    df["ma"] = (
        df.set_index(time_col)["pressure_demeaned"]
        .rolling(window, min_periods=1)
        .mean()
        .values
    )

    return df


def load_node_pressure(
    pattern: str,
    node: str = "F",
) -> pd.DataFrame:
    """
    Load OOI node pressure data from Data files.

    Args:
        pattern: File path pattern with {node} placeholder
        node: Node identifier (B, E, or F)

    Returns:
        DataFrame with time and pressure data
    """
    file_pattern = pattern.format(node=node)
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    df_list = []
    for path in file_paths:
        temp_df = pd.read_csv(
            path, header=None, names=["time", "rawpsi", "depth", "temp"]
        )
        temp_df["time"] = pd.to_datetime(temp_df["time"])
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True).sort_values("time")
    df["time"] = pd.to_datetime(df["time"])
    df["p"] = df["depth"] / 0.67  # Convert to PSI

    return df.reset_index(drop=True)


def load_all_node_pressures(
    pattern: str,
) -> Dict[str, pd.DataFrame]:
    """
    Load pressure data from all OOI nodes.

    Args:
        pattern: File path pattern with {node} placeholder

    Returns:
        Dictionary with node identifiers as keys and DataFrames as values
    """
    nodes = {"B": "West", "E": "East", "F": "Central"}
    result = {}

    for node_id, node_name in nodes.items():
        try:
            result[node_id] = load_node_pressure(pattern, node_id)
            result[node_id]["node_name"] = node_name
        except FileNotFoundError:
            print(f"Warning: No data found for node {node_id} ({node_name})")

    return result


def compute_pressure_ma_series(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    window: str = "15D",
    unit: str = "kpa",
) -> pd.Series:
    """
    Compute 15-day (or custom) moving average of pressure as a Series.

    Args:
        df: DataFrame with time and value columns
        time_col: Name of time column
        value_col: Name of value column
        window: Rolling window specification
        unit: Unit of input ('kpa' or 'psi')

    Returns:
        Series with DatetimeIndex and MA values
    """
    x = df[[time_col, value_col]].copy()
    x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
    x = x.dropna(subset=[time_col]).sort_values(time_col)

    v = x[value_col].astype(float)
    if unit.lower() == "psi":
        v = v * PSI_TO_KPA

    s = pd.Series(v.values, index=x[time_col])
    # Use trailing window instead of center for datetime index compatibility
    return s.rolling(window, min_periods=1).mean()


def rebase_to_window(
    ma: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """
    Re-baseline a moving average series to have zero mean within a window.

    Args:
        ma: Moving average Series with DatetimeIndex
        start: Start of reference window
        end: End of reference window

    Returns:
        Re-baselined Series
    """
    w = ma.loc[start:end]
    return ma - w.mean()
