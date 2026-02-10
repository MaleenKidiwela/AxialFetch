"""
Salinity correction pipeline: load CTD data, remove gaps, smooth, bottle-calibrate,
and apply mean-shift corrections for instrument steps.
"""
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress
from .config import SALINITY_CSV


def apply_gap_mean_shift(sdf, gap_start, gap_end,
                         pre_hours=24, post_hours=24,
                         col='corrected_salinity', overwrite=True):
    """
    Drop rows in [gap_start, gap_end) and shift all rows with time >= gap_end
    by (mean_before - mean_after).
    """
    sdf = sdf.copy()
    in_gap = (sdf['time'] >= gap_start) & (sdf['time'] < gap_end)
    sdf = sdf[~in_gap].reset_index(drop=True)

    pre_mask = ((sdf['time'] >= (gap_start - pd.Timedelta(hours=pre_hours)))
                & (sdf['time'] < gap_start))
    post_mask = ((sdf['time'] >= gap_end)
                 & (sdf['time'] < (gap_end + pd.Timedelta(hours=post_hours))))

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

    after = sdf['time'] >= gap_end
    if overwrite:
        sdf[col + '_before_gapfix'] = sdf[col].copy()
        sdf.loc[after, col] = sdf.loc[after, col] + shift_amt
    else:
        sdf[col + '_gapfix'] = sdf[col]
        sdf.loc[after, col + '_gapfix'] = sdf.loc[after, col + '_gapfix'] + shift_amt

    print(f"[Gap fix] {gap_start} -> {gap_end} | pre_mean={pre_mean:.5f} PSU, "
          f"post_mean={post_mean:.5f} PSU, applied shift={shift_amt:+.5f} PSU, "
          f"rows_dropped={int(in_gap.sum())}")

    return sdf


def compute_corrected_salinity():
    """
    Full salinity correction pipeline. Returns the final salinity_df with
    columns: time, sal_raw, corrected_salinity (plus offset columns).
    """
    # --- Configuration ---
    file_path = SALINITY_CSV

    bottle_times = pd.to_datetime([
        '2022-08-14 05:49:53',   # t1
        '2022-08-30 19:37:13',   # t2
        '2023-09-17 03:35:09',   # t3
    ])
    bottle_sal = np.array([34.52543602, 34.52720340, 34.52700000])

    # 2023 calibration gap
    jump_start = pd.to_datetime('2023-09-07 00:00:00')
    jump_end = pd.to_datetime('2023-09-14 00:00:00')
    smooth_after = jump_end
    smooth_len = 250

    # Analysis window
    start_time = pd.to_datetime('2022-08-14 05:49:53')
    end_time = pd.to_datetime('2025-09-05 23:59:00')

    # 2024 gap
    gap_start_24 = pd.to_datetime('2024-08-31 21:43:00')
    gap_end_24 = pd.to_datetime('2024-09-01 20:13:00')
    pre_hours = 24
    post_hours = 24

    # --- 1) Load & basic filter ---
    df = pd.read_csv(file_path, usecols=[0, 1]).drop(0).reset_index(drop=True)
    df.columns = ['time', 'sal_raw']
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    df['sal_raw'] = pd.to_numeric(df['sal_raw'], errors='coerce')
    df = df[df['sal_raw'].between(34.48, 100)].copy()

    # --- 2) Remove 2023 gap & stitch ---
    gap_mask = (df['time'] >= jump_start) & (df['time'] < jump_end)
    df = df[~gap_mask].reset_index(drop=True)

    pre_gap_val = df.loc[df['time'] < jump_start, 'sal_raw'].iloc[-1]
    post_gap_val = df.loc[df['time'] >= jump_end, 'sal_raw'].iloc[0]
    gap_shift = pre_gap_val - post_gap_val
    df.loc[df['time'] >= jump_end, 'sal_raw'] += gap_shift

    # Smooth post-gap
    post_idx = df.index[df['time'] >= smooth_after]
    if len(post_idx):
        df.loc[post_idx, 'sal_raw'] = uniform_filter1d(
            df.loc[post_idx, 'sal_raw'], size=smooth_len
        )

    # --- 3) Bottle calibration ---
    def nearest_val(t):
        return df.iloc[(df['time'] - t).abs().argmin()]['sal_raw']

    raw_b1, raw_b2 = map(nearest_val, bottle_times[:2])
    d1 = bottle_sal[0] - raw_b1
    d2 = bottle_sal[1] - raw_b2

    t1, t2, t3 = bottle_times

    # Offset A: linear ramp from Bottle-1 to Bottle-2
    df['offset_A'] = 0.0
    mask_A = (df['time'] >= t1) & (df['time'] <= t2)
    w = (df.loc[mask_A, 'time'] - t1) / (t2 - t1)
    df.loc[mask_A, 'offset_A'] = d1 + w * (d2 - d1)
    df.loc[df['time'] > t2, 'offset_A'] = d2

    # Fit line through Segment-B (t2 -> jump_start)
    mask_B = (df['time'] > t2) & (df['time'] < jump_start)
    if mask_B.any():
        slope, intercept, *_ = linregress(
            df.loc[mask_B, 'time'].astype('int64'),
            df.loc[mask_B, 'sal_raw'] + d2,
        )

    # Shift Segment-C so Bottle-3 is perfect
    raw_b3 = nearest_val(t3)
    shift_C = bottle_sal[2] - (raw_b3 + d2)
    df['offset_C'] = 0.0
    df.loc[df['time'] >= smooth_after, 'offset_C'] = shift_C

    # Ramp into Segment-B
    df['offset_P'] = 0.0
    ramp_mask = (df['time'] > t2) & (df['time'] < jump_end)
    if ramp_mask.any():
        rr = (df.loc[ramp_mask, 'time'] - t2) / (jump_end - t2)
        df.loc[ramp_mask, 'offset_P'] = rr * shift_C

    # Total correction
    df['total_offset'] = df['offset_A'] + df['offset_P'] + df['offset_C']
    df['corrected_salinity'] = df['sal_raw'] + df['total_offset']

    # Apply analysis window
    salinity_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)].copy()

    for bt, true_S in zip(bottle_times, bottle_sal):
        est = salinity_df.iloc[(salinity_df['time'] - bt).abs().argmin()]
        print(f"Residual @ {bt} : {(est['corrected_salinity']-true_S):+6.2e} PSU")

    # --- 4) 2024 gap + mean-shift ---
    salinity_df = apply_gap_mean_shift(
        salinity_df,
        gap_start=gap_start_24,
        gap_end=gap_end_24,
        pre_hours=pre_hours,
        post_hours=post_hours,
        col='corrected_salinity',
        overwrite=True,
    )

    # --- 5) 2025 gaps + mean-shifts ---
    gaps_2025 = [
        (pd.Timestamp('2025-06-27 02:21:00'), pd.Timestamp('2025-06-27 03:21:00')),
        (pd.Timestamp('2025-07-07 07:28:00'), pd.Timestamp('2025-07-07 16:17:00')),
        (pd.Timestamp('2025-07-15 07:34:00'), pd.Timestamp('2025-07-15 08:34:00')),
        (pd.Timestamp('2025-08-16 22:36:00'), pd.Timestamp('2025-08-16 23:36:00')),
        (pd.Timestamp('2025-08-25 11:19:00'), pd.Timestamp('2025-08-25 12:19:00')),
    ]

    for s, e in gaps_2025:
        salinity_df = apply_gap_mean_shift(
            salinity_df,
            gap_start=s,
            gap_end=e,
            pre_hours=24,
            post_hours=24,
            col='corrected_salinity',
            overwrite=True,
        )

    return salinity_df
