"""
BSL (baseline/range) data extraction, outlier filtering, and distance computation.

Extracts acoustic range measurements from raw FETCH data, applies station-specific
filtering, interpolates harmonic-mean velocities, and computes one-way distances.
"""
import numpy as np
import pandas as pd
from .tilt import apply_tilt_correction


def extract_bsl_data(df_dict, station, target_address):
    """
    Extract and type-convert BSL data from a station targeting a specific address.

    Parameters
    ----------
    df_dict : dict
        Nested dictionary from load_all_data(), keyed by station then code.
    station : str
        Source station identifier (e.g. '2504').
    target_address : str
        Target station identifier (e.g. '2502').

    Returns
    -------
    DataFrame with typed 'RangeAddress', 'Range(ms)', 'TAT(ms)', 'Record Time'.
    """
    bsl_df = df_dict[station]['BSL'].copy()
    bsl_df['RangeAddress'] = bsl_df['RangeAddress'].astype(int).astype(str)
    bsl_df['Range(ms)'] = pd.to_numeric(bsl_df['Range(ms)'], errors='coerce')
    bsl_df['TAT(ms)'] = pd.to_numeric(bsl_df['TAT(ms)'], errors='coerce')
    filtered = bsl_df[bsl_df['RangeAddress'] == target_address].copy()
    filtered['Record Time'] = pd.to_datetime(filtered['Record Time'])
    return filtered


def filter_bsl_iqr(df, column='Range(ms)', q_low=0.25, q_high=0.75):
    """
    Filter DataFrame using IQR method with configurable quantile bounds.

    Parameters
    ----------
    df : DataFrame
    column : str
    q_low, q_high : float
        Quantile bounds for IQR computation.

    Returns
    -------
    Filtered DataFrame (rows within [Q1 - 1.5*IQR, Q3 + 1.5*IQR]).
    """
    Q1 = df[column].quantile(q_low)
    Q3 = df[column].quantile(q_high)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]


def filter_bsl_range(df, column='Range(ms)', min_val=None, max_val=None):
    """Filter DataFrame by hard min/max bounds on a column."""
    mask = pd.Series(True, index=df.index)
    if min_val is not None:
        mask &= df[column] >= min_val
    if max_val is not None:
        mask &= df[column] <= max_val
    return df[mask]


def compute_distance(bsl_df, harmonic_df, harmonic_mean_df,
                     baseline_perturb, perturb_tag,
                     compute_harmonic_dist=True):
    """
    Compute acoustic distances for a single directed baseline.

    Steps:
      1. Interpolate HMean from harmonic_df onto BSL timestamps
      2. Calculated Distance (m) = Interpolated_Sound_Speed * (Range - TAT) / 2000
      3. Optionally: Interpolate Harmonic Mean, compute Harmonic Distance
      4. Apply tilt correction

    Parameters
    ----------
    bsl_df : DataFrame
        Filtered BSL data with 'Record Time', 'Range(ms)', 'TAT(ms)'.
    harmonic_df : DataFrame
        From harmonic_df_dict[key], with 'Record Time' and 'HMean'.
    harmonic_mean_df : DataFrame or None
        From harmonic_mean_dfs[key], with 'Record Time' and 'Harmonic Mean'.
    baseline_perturb : DataFrame
        Datetime-indexed perturbation table.
    perturb_tag : str
        E.g. '2504-2502' for tilt correction column lookup.
    compute_harmonic_dist : bool
        Whether to also compute Harmonic Distance (not done for 2502 baselines).

    Returns
    -------
    DataFrame with Calculated Distance (m), optionally Harmonic Distance (m),
    and Distance_tiltcorr(m).
    """
    df = bsl_df.copy()
    df['Record Time'] = pd.to_datetime(df['Record Time'])

    hdf = harmonic_df.copy()
    hdf['Record Time'] = pd.to_datetime(hdf['Record Time'])

    # Interpolate HMean velocity onto BSL timestamps
    df['Interpolated Sound Speed'] = np.interp(
        df['Record Time'].astype(np.int64),
        hdf['Record Time'].values.astype(np.int64),
        hdf['HMean'],
    )

    # Calculate distance: velocity * one-way travel time
    df['Calculated Distance (m)'] = (
        df['Interpolated Sound Speed']
        * (df['Range(ms)'] - df['TAT(ms)']) / 2000
    )

    # Harmonic distance (for 2504 and 2503 baselines)
    if compute_harmonic_dist and harmonic_mean_df is not None:
        hmdf = harmonic_mean_df.copy()
        hmdf['Record Time'] = pd.to_datetime(hmdf['Record Time'])
        df['Interpolated Harmonic Velocity'] = np.interp(
            df['Record Time'].astype(np.int64),
            hmdf['Record Time'].values.astype(np.int64),
            hmdf['Harmonic Mean'],
        )
        df['Harmonic Distance (m)'] = (
            df['Interpolated Harmonic Velocity']
            * (df['Range(ms)'] - df['TAT(ms)']) / 2000
        )

    # Apply tilt correction
    df = apply_tilt_correction(df, perturb_tag, baseline_perturb)

    return df


def build_all_baselines(df_dict, harmonic_df_dict, harmonic_mean_dfs,
                        baseline_perturb):
    """
    Build all 6 directed baseline DataFrames, each filtered and distance-computed.

    Returns
    -------
    dict mapping baseline name (e.g. 'R2504_2502') to its DataFrame.
    """
    baselines = {}

    # --- Station 2504 baselines ---

    # 2504 -> 2502: IQR with quantiles (0.15, 0.85)
    bsl = extract_bsl_data(df_dict, '2504', '2502')
    bsl = filter_bsl_iqr(bsl, q_low=0.15, q_high=0.85)
    baselines['R2504_2502'] = compute_distance(
        bsl,
        harmonic_df_dict['2502_2504'],
        harmonic_mean_dfs['2502_2504'],
        baseline_perturb, '2504-2502',
        compute_harmonic_dist=True,
    )

    # 2504 -> 2503: hard filter Range(ms) >= 2618
    bsl = extract_bsl_data(df_dict, '2504', '2503')
    bsl = filter_bsl_range(bsl, min_val=2618)
    baselines['R2504_2503'] = compute_distance(
        bsl,
        harmonic_df_dict['2503_2504'],
        harmonic_mean_dfs['2503_2504'],
        baseline_perturb, '2504-2503',
        compute_harmonic_dist=True,
    )

    # --- Station 2503 baselines ---

    # 2503 -> 2502: standard IQR (0.25, 0.75)
    bsl = extract_bsl_data(df_dict, '2503', '2502')
    bsl = filter_bsl_iqr(bsl, q_low=0.25, q_high=0.75)
    baselines['R2503_2502'] = compute_distance(
        bsl,
        harmonic_df_dict['2502_2503'],
        harmonic_mean_dfs['2502_2504'],   # uses 2502_2504 (2503 SSP is NaN)
        baseline_perturb, '2502-2503',
        compute_harmonic_dist=True,
    )

    # 2503 -> 2504: standard IQR (0.25, 0.75)
    bsl = extract_bsl_data(df_dict, '2503', '2504')
    bsl = filter_bsl_iqr(bsl, q_low=0.25, q_high=0.75)
    baselines['R2503_2504'] = compute_distance(
        bsl,
        harmonic_df_dict['2503_2504'],
        harmonic_mean_dfs['2503_2504'],
        baseline_perturb, '2504-2503',    # undirected: reuses 2504-2503
        compute_harmonic_dist=True,
    )

    # --- Station 2502 baselines (no Harmonic Distance) ---

    # 2502 -> 2503: hard filter Range(ms) in [4636.2, 4638]
    bsl = extract_bsl_data(df_dict, '2502', '2503')
    bsl = filter_bsl_range(bsl, min_val=4636.2, max_val=4638)
    baselines['R2502_2503'] = compute_distance(
        bsl,
        harmonic_df_dict['2502_2503'],
        None,
        baseline_perturb, '2502-2503',
        compute_harmonic_dist=False,
    )

    # 2502 -> 2504: hard filter Range(ms) >= 2400
    bsl = extract_bsl_data(df_dict, '2502', '2504')
    bsl = filter_bsl_range(bsl, min_val=2400)
    baselines['R2502_2504'] = compute_distance(
        bsl,
        harmonic_df_dict['2502_2504'],
        None,
        baseline_perturb, '2504-2502',
        compute_harmonic_dist=False,
    )

    return baselines
