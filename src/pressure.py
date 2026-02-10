"""
Pressure post-processing: demeaning and 15-day moving averages.
"""
import pandas as pd
import glob
from .config import NANO_MJ03F_PATTERN, NANO_MJ03E_PATTERN, NANO_MJ03B_PATTERN

COL = 'Corrected Pressure (kPa)'


def prep_ma15d(df):
    """Add demeaned pressure and trailing 15-day moving average columns."""
    df = df.copy()
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df.sort_values('DateTime', inplace=True)
    mu = df[COL].mean(skipna=True)
    df['pressure_demeaned'] = df[COL] - mu
    df['ma15d'] = (
        df.set_index('DateTime')['pressure_demeaned']
          .rolling('15D', min_periods=1)
          .mean()
          .values
    )
    return df


def ensure_demeaned(df):
    """Ensure demeaned and 15-day MA columns exist (idempotent)."""
    df = df.copy()
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df.sort_values('DateTime', inplace=True)
    if 'pressure_demeaned' not in df.columns:
        mu = df[COL].mean(skipna=True)
        df['pressure_demeaned'] = df[COL] - mu
    df['ma15d'] = (
        df.set_index('DateTime')['pressure_demeaned']
          .rolling('15D', min_periods=1)
          .mean()
          .values
    )
    return df


def load_nano_pressure(pattern):
    """Load and concatenate NANO .Data files matching a glob pattern."""
    file_paths = glob.glob(pattern)
    df_list = []
    for path in file_paths:
        temp_df = pd.read_csv(
            path, header=None, names=["time", "rawpsi", "depth", "temp"]
        )
        temp_df['time'] = pd.to_datetime(temp_df['time'])
        df_list.append(temp_df)

    combined = pd.concat(df_list, ignore_index=True).sort_values('time')
    combined['time'] = pd.to_datetime(combined['time'])
    combined['p'] = combined['depth'] / 0.67
    return combined


def apply_pressure_postprocessing(combined_df4, combined_df3, combined_df2):
    """
    Apply demeaning and 15-day MA to all three station DataFrames.
    Returns the updated DataFrames.
    """
    combined_df4 = prep_ma15d(combined_df4)
    combined_df2 = prep_ma15d(combined_df2)
    combined_df3 = prep_ma15d(combined_df3)

    # Load external pressure data
    cp = load_nano_pressure(NANO_MJ03F_PATTERN)
    ep = load_nano_pressure(NANO_MJ03E_PATTERN)
    wp = load_nano_pressure(NANO_MJ03B_PATTERN)

    # Second pass: ensure_demeaned (overwrites ma15d with recomputed values)
    combined_df4 = ensure_demeaned(combined_df4)
    combined_df2 = ensure_demeaned(combined_df2)
    combined_df3 = ensure_demeaned(combined_df3)

    return combined_df4, combined_df3, combined_df2
