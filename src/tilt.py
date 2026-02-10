"""
Tilt correction: compute baseline perturbations from inclinometer data.

Projects per-station tilt offsets (Pitch/Roll) through instrument headings
onto each baseline direction to produce a one-way path-length change (dL).
"""
import numpy as np
import pandas as pd
from .config import STATION_META, R_EARTH, H_TXR


# Baseline pairs (tx -> rx), matching notebook Cell 11
BASELINE_PAIRS = [
    ("2504", "2502"),
    ("2504", "2503"),
    ("2502", "2503"),
]


def _interp_unique(df, t):
    """Interpolate Pitch/Roll onto timeline *t*, dropping duplicate stamps."""
    tidy = (df.set_index("Record Time")
              .sort_index()
              .loc[lambda x: ~x.index.duplicated(keep="first")])
    return (tidy.reindex(t)
                .interpolate(limit_direction="both")
                .astype(float))


def _local_xy(df, h=H_TXR):
    pitch, roll = np.radians(df["Pitch"]), np.radians(df["Roll"])
    dx = h * np.sin(roll)       # starboard
    dy = h * np.sin(pitch)      # forward
    return dx.values, dy.values


def _to_enu(dx, dy, heading_deg):
    h = np.radians(heading_deg)
    east = dx * np.sin(h) + dy * np.cos(h)
    north = dx * np.cos(h) - dy * np.sin(h)
    return east, north


def _unit(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    mlat = np.radians((lat1 + lat2) / 2)
    de, dn = dlon * np.cos(mlat) * R_EARTH, dlat * R_EARTH
    L = np.hypot(de, dn)
    return de / L, dn / L       # EN components of unit vector


def compute_baseline_perturb(INC_df4, INC_df3, INC_df2):
    """
    Compute the baseline perturbation table from inclinometer data.

    Returns a DataFrame indexed by datetime with columns:
        '2504-2502_dL', '2504-2503_dL', '2502-2503_dL'
    each giving the one-way path-length change in metres.
    """
    stations = {
        "2502": dict(inc=INC_df2, **STATION_META["2502"]),
        "2503": dict(inc=INC_df3, **STATION_META["2503"]),
        "2504": dict(inc=INC_df4, **STATION_META["2504"]),
    }

    series_list = []

    for tx, rx in BASELINE_PAIRS:
        S_tx, S_rx = stations[tx], stations[rx]

        # Common timeline (union of both inclinometer traces)
        timeline = pd.Index(sorted(
            set(S_tx["inc"]["Record Time"]) | set(S_rx["inc"]["Record Time"])
        ))

        # Interpolate, convert to ENU
        inc_tx = _interp_unique(S_tx["inc"], timeline)
        inc_rx = _interp_unique(S_rx["inc"], timeline)
        e_tx, n_tx = _to_enu(*_local_xy(inc_tx), S_tx["heading"])
        e_rx, n_rx = _to_enu(*_local_xy(inc_rx), S_rx["heading"])

        # Project offsets onto baseline direction
        ue, un = _unit(S_tx["lat"], S_tx["lon"], S_rx["lat"], S_rx["lon"])
        dL_oneway = (e_tx - e_rx) * ue + (n_tx - n_rx) * un  # metres

        series_list.append(
            pd.DataFrame({f"{tx}-{rx}_dL": dL_oneway}, index=timeline)
        )

    baseline_perturb = pd.concat(series_list, axis=1).sort_index()
    return baseline_perturb


def apply_tilt_correction(df_range, baseline_tag, perturb_df,
                          dist_col="Calculated Distance (m)"):
    """
    Merge tilt-perturbation onto a range DataFrame and compute corrected distance.

    Parameters
    ----------
    df_range : DataFrame
        Must contain 'Record Time' and *dist_col*.
    baseline_tag : str
        E.g. '2504-2502'; the perturbation column is '{baseline_tag}_dL'.
    perturb_df : DataFrame
        Datetime-indexed with columns like '2504-2502_dL'.
    dist_col : str
        Column name of the distance to correct.

    Returns
    -------
    DataFrame with new column 'Distance_tiltcorr(m)'.
    """
    col = f"{baseline_tag}_dL"
    merged = df_range.merge(
        perturb_df[[col]],
        left_on="Record Time",
        right_index=True,
        how="left",
    )
    merged["Distance_tiltcorr(m)"] = merged[dist_col] - merged[col]
    return merged
