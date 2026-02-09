"""
Salinity correction workflow for FETCH Range Calculation analysis.

This module mirrors the salinity correction steps from the
FETCH Range Calculation notebook, including gap stitching,
post-gap smoothing, bottle calibration, and optional gap mean shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress


@dataclass(frozen=True)
class SalinityCalibrationConfig:
    file_path: str
    bottle_times: Iterable[pd.Timestamp]
    bottle_sal: Iterable[float]
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    jump_start: pd.Timestamp
    jump_end: pd.Timestamp
    smooth_after: pd.Timestamp
    smooth_len: int = 250


@dataclass(frozen=True)
class GapMeanShiftConfig:
    gap_start: pd.Timestamp
    gap_end: pd.Timestamp
    pre_hours: int = 24
    post_hours: int = 24
    column: str = "corrected_salinity"
    overwrite_column: str | None = None


def load_salinity_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, usecols=[0, 1]).drop(0).reset_index(drop=True)
    df.columns = ["time", "sal_raw"]
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df["sal_raw"] = pd.to_numeric(df["sal_raw"], errors="coerce")
    return df[df["sal_raw"].between(34.48, 100)].copy()


def remove_gap_and_stitch(
    df: pd.DataFrame, jump_start: pd.Timestamp, jump_end: pd.Timestamp
) -> pd.DataFrame:
    gap_mask = (df["time"] >= jump_start) & (df["time"] < jump_end)
    df = df[~gap_mask].reset_index(drop=True)

    pre_gap_val = df.loc[df["time"] < jump_start, "sal_raw"].iloc[-1]
    post_gap_val = df.loc[df["time"] >= jump_end, "sal_raw"].iloc[0]
    gap_shift = pre_gap_val - post_gap_val
    df.loc[df["time"] >= jump_end, "sal_raw"] += gap_shift
    return df


def smooth_post_gap(
    df: pd.DataFrame, smooth_after: pd.Timestamp, smooth_len: int
) -> pd.DataFrame:
    post_idx = df.index[df["time"] >= smooth_after]
    if len(post_idx):
        df.loc[post_idx, "sal_raw"] = uniform_filter1d(
            df.loc[post_idx, "sal_raw"], size=smooth_len
        )
    return df


def calibrate_bottle_salinity(
    df: pd.DataFrame,
    bottle_times: Iterable[pd.Timestamp],
    bottle_sal: Iterable[float],
    jump_start: pd.Timestamp,
    jump_end: pd.Timestamp,
    smooth_after: pd.Timestamp,
) -> pd.DataFrame:
    bottle_times = pd.to_datetime(list(bottle_times))
    bottle_sal = np.array(list(bottle_sal), dtype=float)

    def nearest_val(t: pd.Timestamp) -> float:
        return df.iloc[(df["time"] - t).abs().argmin()]["sal_raw"]

    raw_b1, raw_b2 = map(nearest_val, bottle_times[:2])
    d1 = bottle_sal[0] - raw_b1
    d2 = bottle_sal[1] - raw_b2

    t1, t2, t3 = bottle_times

    df["offset_A"] = 0.0
    mask_A = (df["time"] >= t1) & (df["time"] <= t2)
    w = (df.loc[mask_A, "time"] - t1) / (t2 - t1)
    df.loc[mask_A, "offset_A"] = d1 + w * (d2 - d1)
    df.loc[df["time"] > t2, "offset_A"] = d2

    mask_B = (df["time"] > t2) & (df["time"] < jump_start)
    if mask_B.any():
        linregress(
            df.loc[mask_B, "time"].astype("int64"),
            df.loc[mask_B, "sal_raw"] + d2,
        )

    raw_b3 = nearest_val(t3)
    shift_C = bottle_sal[2] - (raw_b3 + d2)
    df["offset_C"] = 0.0
    df.loc[df["time"] >= smooth_after, "offset_C"] = shift_C

    df["offset_P"] = 0.0
    ramp_mask = (df["time"] > t2) & (df["time"] < jump_end)
    if ramp_mask.any():
        rr = (df.loc[ramp_mask, "time"] - t2) / (jump_end - t2)
        df.loc[ramp_mask, "offset_P"] = rr * shift_C

    df["total_offset"] = df["offset_A"] + df["offset_P"] + df["offset_C"]
    df["corrected_salinity"] = df["sal_raw"] + df["total_offset"]
    return df


def apply_gap_mean_shift(
    sdf: pd.DataFrame,
    gap_start: pd.Timestamp,
    gap_end: pd.Timestamp,
    pre_hours: int = 24,
    post_hours: int = 24,
    column: str = "corrected_salinity",
    overwrite_column: str | None = None,
) -> pd.DataFrame:
    df = sdf.copy()
    target_col = overwrite_column or column

    pre_window = df[(df["time"] >= gap_start - pd.Timedelta(hours=pre_hours))
                    & (df["time"] < gap_start)]
    post_window = df[(df["time"] >= gap_end)
                     & (df["time"] < gap_end + pd.Timedelta(hours=post_hours))]

    if pre_window.empty or post_window.empty:
        df[target_col] = df[column]
        return df

    pre_mean = pre_window[column].mean()
    post_mean = post_window[column].mean()
    shift = pre_mean - post_mean

    df[target_col] = df[column]
    mask = df["time"] >= gap_end
    df.loc[mask, target_col] = df.loc[mask, target_col] + shift

    gap_mask = (df["time"] >= gap_start) & (df["time"] < gap_end)
    df = df[~gap_mask].reset_index(drop=True)
    return df


def prepare_salinity_series(
    config: SalinityCalibrationConfig,
    gap_shift: GapMeanShiftConfig | None = None,
) -> pd.DataFrame:
    df = load_salinity_data(config.file_path)
    df = remove_gap_and_stitch(df, config.jump_start, config.jump_end)
    df = smooth_post_gap(df, config.smooth_after, config.smooth_len)
    df = calibrate_bottle_salinity(
        df,
        config.bottle_times,
        config.bottle_sal,
        config.jump_start,
        config.jump_end,
        config.smooth_after,
    )

    if gap_shift is not None:
        df = apply_gap_mean_shift(
            df,
            gap_shift.gap_start,
            gap_shift.gap_end,
            pre_hours=gap_shift.pre_hours,
            post_hours=gap_shift.post_hours,
            column=gap_shift.column,
            overwrite_column=gap_shift.overwrite_column,
        )

    mask = (df["time"] >= config.start_time) & (df["time"] <= config.end_time)
    return df.loc[mask].copy()


def bottle_residuals(
    salinity_df: pd.DataFrame,
    bottle_times: Iterable[pd.Timestamp],
    bottle_sal: Iterable[float],
    column: str = "corrected_salinity",
) -> Tuple[Tuple[pd.Timestamp, float], ...]:
    residuals = []
    for bt, true_s in zip(pd.to_datetime(list(bottle_times)), bottle_sal):
        est = salinity_df.iloc[(salinity_df["time"] - bt).abs().argmin()]
        residuals.append((bt, float(est[column] - true_s)))
    return tuple(residuals)
