"""
Salinity Correction Module for FETCH Range Calculation

Handles CTD salinity drift correction, bottle calibration,
gap handling, and data quality control.
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress


def load_ctd_salinity(
    filepath: str,
    sal_bounds: Tuple[float, float] = (34.48, 100.0),
) -> pd.DataFrame:
    """
    Load OOI CTD salinity data from CSV file.

    Args:
        filepath: Path to the CTD CSV file
        sal_bounds: Min and max salinity bounds for filtering

    Returns:
        DataFrame with 'time' and 'sal_raw' columns
    """
    df = pd.read_csv(filepath, usecols=[0, 1]).drop(0).reset_index(drop=True)
    df.columns = ["time", "sal_raw"]
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df["sal_raw"] = pd.to_numeric(df["sal_raw"], errors="coerce")
    df = df[df["sal_raw"].between(sal_bounds[0], sal_bounds[1])].copy()

    return df


def apply_gap_mean_shift(
    sdf: pd.DataFrame,
    gap_start: pd.Timestamp,
    gap_end: pd.Timestamp,
    pre_hours: int = 24,
    post_hours: int = 24,
    col: str = "corrected_salinity",
    overwrite: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply a hard-gap correction by removing gap rows and shifting post-gap data.

    Drops rows in [gap_start, gap_end) and shifts all rows with time >= gap_end
    by (mean_before - mean_after), where means are computed over windows before
    and after the gap.

    Args:
        sdf: DataFrame with 'time' and salinity columns
        gap_start: Start of the gap to remove
        gap_end: End of the gap
        pre_hours: Hours before gap for mean calculation
        post_hours: Hours after gap for mean calculation
        col: Column name containing salinity values
        overwrite: If True, overwrite the column; if False, create new column
        verbose: Print diagnostic information

    Returns:
        DataFrame with gap removed and post-gap data shifted
    """
    sdf = sdf.copy()

    # Drop gap rows
    in_gap = (sdf["time"] >= gap_start) & (sdf["time"] < gap_end)
    sdf = sdf[~in_gap].reset_index(drop=True)

    # Compute means in pre/post windows
    pre_mask = (sdf["time"] >= (gap_start - pd.Timedelta(hours=pre_hours))) & (
        sdf["time"] < gap_start
    )
    post_mask = (sdf["time"] >= gap_end) & (
        sdf["time"] < (gap_end + pd.Timedelta(hours=post_hours))
    )

    pre_vals = sdf.loc[pre_mask, col].dropna()
    post_vals = sdf.loc[post_mask, col].dropna()

    if len(pre_vals) == 0 or len(post_vals) == 0:
        raise ValueError(
            "Not enough data in pre/post windows to compute means. "
            "Increase pre_hours/post_hours or adjust gap bounds."
        )

    pre_mean = float(pre_vals.mean())
    post_mean = float(post_vals.mean())
    shift_amt = pre_mean - post_mean

    # Apply constant shift to all samples at/after gap_end
    after = sdf["time"] >= gap_end
    if overwrite:
        sdf[col + "_before_gapfix"] = sdf[col].copy()
        sdf.loc[after, col] = sdf.loc[after, col] + shift_amt
    else:
        sdf[col + "_gapfix"] = sdf[col]
        sdf.loc[after, col + "_gapfix"] = sdf.loc[after, col + "_gapfix"] + shift_amt

    if verbose:
        print(
            f"[Gap fix] {gap_start} → {gap_end} | pre_mean={pre_mean:.5f} PSU, "
            f"post_mean={post_mean:.5f} PSU, applied shift={shift_amt:+.5f} PSU, "
            f"rows_dropped={int(in_gap.sum())}"
        )

    return sdf


def apply_bottle_calibration(
    df: pd.DataFrame,
    bottle_times: List[pd.Timestamp],
    bottle_values: List[float],
    sal_col: str = "sal_raw",
    jump_start: Optional[pd.Timestamp] = None,
    jump_end: Optional[pd.Timestamp] = None,
    smooth_len: int = 250,
) -> pd.DataFrame:
    """
    Apply bottle calibration to salinity data using discrete samples.

    Args:
        df: DataFrame with 'time' and salinity column
        bottle_times: List of bottle sample times
        bottle_values: List of bottle salinity values
        sal_col: Name of the raw salinity column
        jump_start: Start of gap period (for stitching)
        jump_end: End of gap period
        smooth_len: Smoothing window length for post-gap data

    Returns:
        DataFrame with 'corrected_salinity' column added
    """
    df = df.copy()

    # Handle gap stitching if specified
    if jump_start is not None and jump_end is not None:
        gap_mask = (df["time"] >= jump_start) & (df["time"] < jump_end)
        df = df[~gap_mask].reset_index(drop=True)

        pre_gap_val = df.loc[df["time"] < jump_start, sal_col].iloc[-1]
        post_gap_val = df.loc[df["time"] >= jump_end, sal_col].iloc[0]
        gap_shift = pre_gap_val - post_gap_val
        df.loc[df["time"] >= jump_end, sal_col] += gap_shift

        # Smooth post-gap data
        post_idx = df.index[df["time"] >= jump_end]
        if len(post_idx):
            df.loc[post_idx, sal_col] = uniform_filter1d(
                df.loc[post_idx, sal_col], size=smooth_len
            )

    # Find nearest values to bottles
    def nearest_val(t):
        return df.iloc[(df["time"] - t).abs().argmin()][sal_col]

    t1, t2, t3 = bottle_times[:3]
    raw_b1, raw_b2 = map(nearest_val, bottle_times[:2])

    d1 = bottle_values[0] - raw_b1  # Bias at t1
    d2 = bottle_values[1] - raw_b2  # Bias at t2

    # Offset from Bottle-1 → Bottle-2 (linear)
    df["offset_A"] = 0.0
    mask_A = (df["time"] >= t1) & (df["time"] <= t2)
    w = (df.loc[mask_A, "time"] - t1) / (t2 - t1)
    df.loc[mask_A, "offset_A"] = d1 + w * (d2 - d1)
    df.loc[df["time"] > t2, "offset_A"] = d2  # Flat after t2

    # Fit line through segment B (t2 → jump_start) if exists
    if jump_start is not None:
        mask_B = (df["time"] > t2) & (df["time"] < jump_start)
        if mask_B.any():
            slope, intercept, *_ = linregress(
                df.loc[mask_B, "time"].astype("int64"),
                df.loc[mask_B, sal_col] + d2,
            )

    # Shift segment C (post-gap) so Bottle-3 is accurate
    raw_b3 = nearest_val(t3)
    shift_C = bottle_values[2] - (raw_b3 + d2)

    smooth_after = jump_end if jump_end is not None else t3
    df["offset_C"] = 0.0
    df.loc[df["time"] >= smooth_after, "offset_C"] = shift_C

    # Ramp shift into segment B for continuity
    df["offset_P"] = 0.0
    if jump_end is not None:
        ramp_mask = (df["time"] > t2) & (df["time"] < jump_end)
        if ramp_mask.any():
            rr = (df.loc[ramp_mask, "time"] - t2) / (jump_end - t2)
            df.loc[ramp_mask, "offset_P"] = rr * shift_C

    # Total correction
    df["total_offset"] = df["offset_A"] + df["offset_P"] + df["offset_C"]
    df["corrected_salinity"] = df[sal_col] + df["total_offset"]

    return df


def correct_salinity_drift(
    filepath: str,
    bottle_times: List[str],
    bottle_values: List[float],
    gap_start: str = "2023-09-07 00:00:00",
    gap_end: str = "2023-09-14 00:00:00",
    start_time: str = "2022-08-14 05:49:53",
    end_time: str = "2025-09-05 23:59:00",
    smooth_len: int = 250,
    additional_gaps: Optional[List[Tuple[str, str]]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full salinity correction pipeline.

    Loads CTD data, applies bottle calibration, handles gaps, and returns
    corrected salinity time series.

    Args:
        filepath: Path to CTD salinity CSV
        bottle_times: List of bottle sample times as strings
        bottle_values: List of bottle salinity values
        gap_start: Start of main gap period
        gap_end: End of main gap period
        start_time: Analysis start time
        end_time: Analysis end time
        smooth_len: Smoothing window length
        additional_gaps: List of (start, end) tuples for additional gap corrections
        verbose: Print diagnostic information

    Returns:
        DataFrame with corrected salinity time series
    """
    # Load raw data
    df = load_ctd_salinity(filepath)

    # Convert times
    bottle_ts = pd.to_datetime(bottle_times)
    jump_start = pd.to_datetime(gap_start)
    jump_end = pd.to_datetime(gap_end)
    t_start = pd.to_datetime(start_time)
    t_end = pd.to_datetime(end_time)

    # Apply bottle calibration with main gap handling
    df = apply_bottle_calibration(
        df,
        bottle_times=list(bottle_ts),
        bottle_values=bottle_values,
        jump_start=jump_start,
        jump_end=jump_end,
        smooth_len=smooth_len,
    )

    # Apply time window filter
    salinity_df = df[(df["time"] >= t_start) & (df["time"] <= t_end)].copy()

    # Print residuals at bottle times
    if verbose:
        for bt, true_S in zip(bottle_ts, bottle_values):
            est = salinity_df.iloc[(salinity_df["time"] - bt).abs().argmin()]
            print(f"Residual @ {bt} : {(est['corrected_salinity']-true_S):+6.2e} PSU")

    # Apply additional gap corrections if specified
    if additional_gaps:
        for gap_s, gap_e in additional_gaps:
            salinity_df = apply_gap_mean_shift(
                salinity_df,
                gap_start=pd.to_datetime(gap_s),
                gap_end=pd.to_datetime(gap_e),
                pre_hours=24,
                post_hours=24,
                col="corrected_salinity",
                overwrite=True,
                verbose=verbose,
            )

    return salinity_df


def get_default_additional_gaps() -> List[Tuple[str, str]]:
    """
    Get the default list of additional gap corrections for 2024-2025.

    Returns:
        List of (start, end) time string tuples
    """
    return [
        ("2024-08-31 21:43:00", "2024-09-01 20:13:00"),
        ("2025-06-27 02:21:00", "2025-06-27 03:21:00"),
        ("2025-07-07 07:28:00", "2025-07-07 16:17:00"),
        ("2025-07-15 07:34:00", "2025-07-15 08:34:00"),
        ("2025-08-16 22:36:00", "2025-08-16 23:36:00"),
        ("2025-08-25 11:19:00", "2025-08-25 12:19:00"),
    ]
