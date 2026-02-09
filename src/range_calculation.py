"""
Range/Distance Calculation Module for FETCH Range Calculation

Core acoustic range calculation from travel times, harmonic velocity
computation, and tilt correction for baseline measurements.
"""

from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
from scipy.stats import hmean

from .config import R_EARTH, H_TXR, STATIONS, StationMetadata


# Baseline-specific filtering configurations from the original notebook
# Each key is (source_station, target_station)
# Values are dictionaries with filter parameters
BASELINE_FILTERS = {
    # 2504 source (Central/North station)
    ("2504", "2502"): {
        "filter_type": "iqr",
        "quantiles": (0.15, 0.85),
        "min_range": None,
        "max_range": None,
    },
    ("2504", "2503"): {
        "filter_type": "hard",
        "quantiles": None,
        "min_range": 2618,
        "max_range": None,
    },
    # 2503 source (West station)
    ("2503", "2502"): {
        "filter_type": "iqr",
        "quantiles": (0.25, 0.75),
        "min_range": None,
        "max_range": None,
    },
    ("2503", "2504"): {
        "filter_type": "iqr",
        "quantiles": (0.25, 0.75),
        "min_range": None,
        "max_range": None,
    },
    # 2502 source (East station)
    ("2502", "2503"): {
        "filter_type": "hard",
        "quantiles": None,
        "min_range": 4636.2,
        "max_range": 4638,
    },
    ("2502", "2504"): {
        "filter_type": "hard",
        "quantiles": None,
        "min_range": 2400,
        "max_range": None,
    },
}


def calculate_distance(
    travel_time_ms: float,
    velocity: float,
    tat_ms: float = 0.0,
) -> float:
    """
    Convert acoustic travel time to distance.

    Args:
        travel_time_ms: Two-way travel time in milliseconds
        velocity: Sound velocity in m/s
        tat_ms: Turn-around time in milliseconds

    Returns:
        One-way distance in meters
    """
    # Subtract TAT and convert to one-way time in seconds
    one_way_time_s = (travel_time_ms - tat_ms) / 2000.0
    return velocity * one_way_time_s


def compute_harmonic_mean(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    column: str = "velocity",
) -> Tuple[pd.Index, np.ndarray]:
    """
    Compute harmonic mean for a specific column between two DataFrames.

    Implementation:
    1. Remove duplicate indices from each DataFrame
    2. Drop NaN values from the column in each DataFrame
    3. Find intersection of the remaining valid indices
    4. Compute harmonic mean on the aligned valid values

    This ensures velocities are only paired at timestamps where BOTH
    stations have valid data, which is mathematically correct.

    Args:
        df1: First DataFrame with DatetimeIndex
        df2: Second DataFrame with DatetimeIndex
        column: Name of the column to compute harmonic mean for

    Returns:
        Tuple of (common_index, harmonic_mean_values)
    """
    # Remove duplicate indices (keep first occurrence)
    df1_clean = df1[~df1.index.duplicated(keep='first')]
    df2_clean = df2[~df2.index.duplicated(keep='first')]

    # Drop NaN values from the column FIRST
    df1_valid = df1_clean[df1_clean[column].notna()]
    df2_valid = df2_clean[df2_clean[column].notna()]

    # Find intersection of valid indices
    common_index = df1_valid.index.intersection(df2_valid.index)

    # Select using common index
    aligned_df1 = df1_valid.loc[common_index, column]
    aligned_df2 = df2_valid.loc[common_index, column]

    # Ensure no zero or negative values as they are invalid for harmonic mean
    aligned_df1 = aligned_df1.clip(lower=0.0001)
    aligned_df2 = aligned_df2.clip(lower=0.0001)

    # Compute the harmonic mean
    harmonic_mean_values = hmean(np.vstack([aligned_df1.to_numpy(), aligned_df2.to_numpy()]), axis=0)

    return common_index, harmonic_mean_values


def compute_harmonic_velocity(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute harmonic mean velocity between two stations.

    Args:
        df1: DataFrame for station 1 with 'velocity' column and datetime index
        df2: DataFrame for station 2 with 'velocity' column and datetime index

    Returns:
        DataFrame with Record Time and HMean columns
    """
    index, values = compute_harmonic_mean(df1, df2, column="velocity")
    return pd.DataFrame({"Record Time": index, "HMean": values})


def compute_all_harmonic_velocities(
    combined_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Compute harmonic mean velocities for all station pairs.

    Matches the notebook's exact argument order for compute_harmonic_mean:
    - Key '2502_2503': compute_harmonic_mean(combined_df3, combined_df2) -> (2503, 2502)
    - Key '2502_2504': compute_harmonic_mean(combined_df4, combined_df2) -> (2504, 2502)
    - Key '2503_2504': compute_harmonic_mean(combined_df4, combined_df3) -> (2504, 2503)

    Args:
        combined_dfs: Dictionary of station identifier to combined DataFrame

    Returns:
        Dictionary with pair keys (e.g., "2502_2503") and harmonic DataFrames
    """
    # Define pairs with (key, first_arg_station, second_arg_station)
    # to match notebook's exact argument order
    pairs = [
        ("2502_2503", "2503", "2502"),  # compute_harmonic_mean(combined_df3, combined_df2)
        ("2502_2504", "2504", "2502"),  # compute_harmonic_mean(combined_df4, combined_df2)
        ("2503_2504", "2504", "2503"),  # compute_harmonic_mean(combined_df4, combined_df3)
    ]

    harmonic_dfs = {}
    for key, s1, s2 in pairs:
        if s1 in combined_dfs and s2 in combined_dfs:
            df1 = combined_dfs[s1].copy()
            df2 = combined_dfs[s2].copy()

            # Ensure datetime index
            for df in [df1, df2]:
                if "Record Time" in df.columns:
                    df["Record Time"] = pd.to_datetime(df["Record Time"])
                    df.set_index("Record Time", inplace=True)

            harmonic_dfs[key] = compute_harmonic_velocity(df1, df2)

    return harmonic_dfs


# ---- Tilt Correction Functions ----


def _interp_unique(df: pd.DataFrame, t: pd.Index) -> pd.DataFrame:
    """
    Interpolate Pitch/Roll onto timeline t, after dropping duplicate stamps.

    Args:
        df: DataFrame with Record Time, Pitch, Roll columns
        t: Target timeline

    Returns:
        DataFrame with Pitch and Roll interpolated to target times
    """
    tidy = (
        df.set_index("Record Time")
        .sort_index()
        .loc[lambda x: ~x.index.duplicated(keep="first")]
    )
    return tidy.reindex(t).interpolate(limit_direction="both").astype(float)


def _local_xy(df: pd.DataFrame, h: float = H_TXR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert pitch and roll to local x,y offsets.

    Args:
        df: DataFrame with Pitch and Roll columns (in degrees)
        h: Lever arm length (transducer to tilt sensor)

    Returns:
        Tuple of (dx, dy) arrays in meters
    """
    pitch = np.radians(df["Pitch"])
    roll = np.radians(df["Roll"])
    dx = h * np.sin(roll)  # Starboard
    dy = h * np.sin(pitch)  # Forward
    return dx.values, dy.values


def _to_enu(
    dx: np.ndarray, dy: np.ndarray, heading_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert local frame offsets to East-North-Up coordinates.

    Args:
        dx: Starboard offset
        dy: Forward offset
        heading_deg: Station heading in degrees

    Returns:
        Tuple of (east, north) arrays
    """
    h = np.radians(heading_deg)
    east = dx * np.sin(h) + dy * np.cos(h)
    north = dx * np.cos(h) - dy * np.sin(h)
    return east, north


def _unit_vector(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> Tuple[float, float]:
    """
    Compute unit vector in ENU coordinates from station 1 to station 2.

    Args:
        lat1, lon1: Station 1 coordinates
        lat2, lon2: Station 2 coordinates

    Returns:
        Tuple of (unit_east, unit_north)
    """
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    mlat = np.radians((lat1 + lat2) / 2)
    de = dlon * np.cos(mlat) * R_EARTH
    dn = dlat * R_EARTH
    L = np.hypot(de, dn)
    return de / L, dn / L


def build_baseline_perturbation(
    inc_dfs: Dict[str, pd.DataFrame],
    stations: Dict[str, StationMetadata],
    pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    Build table of baseline perturbations due to tilt.

    Args:
        inc_dfs: Dictionary of station ID to inclinometer DataFrame
        stations: Dictionary of station ID to StationMetadata
        pairs: List of (tx, rx) station pairs

    Returns:
        DataFrame with columns "{tx}-{rx}_dL" for each pair
    """
    series_list = []

    for tx, rx in pairs:
        S_tx = stations[tx]
        S_rx = stations[rx]
        inc_tx_df = inc_dfs[tx]
        inc_rx_df = inc_dfs[rx]

        # Common timeline (union of both inclinometer traces)
        timeline = pd.Index(
            sorted(
                set(pd.to_datetime(inc_tx_df["Record Time"]))
                | set(pd.to_datetime(inc_rx_df["Record Time"]))
            )
        )

        # Interpolate, convert to ENU
        inc_tx = _interp_unique(inc_tx_df, timeline)
        inc_rx = _interp_unique(inc_rx_df, timeline)
        e_tx, n_tx = _to_enu(*_local_xy(inc_tx), S_tx.heading)
        e_rx, n_rx = _to_enu(*_local_xy(inc_rx), S_rx.heading)

        # Project offsets onto baseline
        ue, un = _unit_vector(S_tx.lat, S_tx.lon, S_rx.lat, S_rx.lon)
        dL_oneway = (e_tx - e_rx) * ue + (n_tx - n_rx) * un  # Meters

        series_list.append(pd.DataFrame({f"{tx}-{rx}_dL": dL_oneway}, index=timeline))

    baseline_perturb = pd.concat(series_list, axis=1).sort_index()
    return baseline_perturb


def apply_tilt_correction(
    df_range: pd.DataFrame,
    baseline_tag: str,
    perturb_df: pd.DataFrame,
    dist_col: str = "Calculated Distance (m)",
    tolerance: str = "1H",
) -> pd.DataFrame:
    """
    Apply tilt correction to range measurements.

    Uses merge_asof with time tolerance to match perturbation values
    to distance measurements (matching notebook lines 10287-10300).

    Args:
        df_range: DataFrame with range measurements
        baseline_tag: Baseline identifier (e.g., "2504-2502")
        perturb_df: DataFrame from build_baseline_perturbation()
        dist_col: Name of distance column to correct
        tolerance: Time tolerance for merge_asof (default "1H")

    Returns:
        DataFrame with added 'Distance_tiltcorr(m)' column
    """
    col = f"{baseline_tag}_dL"
    if col not in perturb_df.columns:
        print(f"Warning: {col} not found in perturbation table")
        df_range["Distance_tiltcorr(m)"] = df_range[dist_col]
        return df_range

    # Prepare perturbation data for merge_asof
    perturb_reset = perturb_df[[col]].reset_index()
    perturb_reset.columns = ["Record Time", "perturbation"]
    perturb_reset = perturb_reset.sort_values("Record Time")

    # Prepare distance data
    df_sorted = df_range.copy()
    df_sorted["Record Time"] = pd.to_datetime(df_sorted["Record Time"])
    df_sorted = df_sorted.sort_values("Record Time")

    # Use merge_asof with time tolerance (notebook line 10291-10294)
    merged = pd.merge_asof(
        df_sorted,
        perturb_reset,
        on="Record Time",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )

    # Fill NaN with 0 (notebook line 10297)
    merged["perturbation"] = merged["perturbation"].fillna(0)

    # Apply correction: Distance_corrected = Distance + dL
    # (notebook line 10300 uses ADDITION)
    merged["Distance_tiltcorr(m)"] = merged[dist_col] + merged["perturbation"]

    # Rename perturbation column back to original name
    merged.rename(columns={"perturbation": col}, inplace=True)

    return merged


def calculate_baseline_distances(
    bsl_df: pd.DataFrame,
    harmonic_df: pd.DataFrame,
    source_station: str,
    target_station: str,
    outlier_quantiles: Optional[Tuple[float, float]] = None,
    min_range: Optional[float] = None,
    max_range: Optional[float] = None,
    use_baseline_config: bool = True,
) -> pd.DataFrame:
    """
    Calculate baseline distances from travel time data.

    Uses baseline-specific filtering configurations from BASELINE_FILTERS
    when use_baseline_config=True.

    Args:
        bsl_df: Baseline DataFrame with Range(ms), TAT(ms), RangeAddress columns
        harmonic_df: Harmonic velocity DataFrame with Record Time and HMean columns
        source_station: Transmitting station
        target_station: Receiving station
        outlier_quantiles: Quantiles for IQR-based outlier removal (overrides config)
        min_range: Optional minimum range threshold (overrides config)
        max_range: Optional maximum range threshold (overrides config)
        use_baseline_config: Whether to use BASELINE_FILTERS configuration

    Returns:
        DataFrame with calculated distances
    """
    # Get baseline-specific configuration
    baseline_key = (source_station, target_station)

    if use_baseline_config and baseline_key in BASELINE_FILTERS:
        config = BASELINE_FILTERS[baseline_key]
        filter_type = config["filter_type"]
        if outlier_quantiles is None:
            outlier_quantiles = config["quantiles"]
        if min_range is None:
            min_range = config["min_range"]
        if max_range is None:
            max_range = config["max_range"]
    else:
        filter_type = "iqr"  # Default to IQR filtering
        if outlier_quantiles is None:
            outlier_quantiles = (0.15, 0.85)

    # Filter for target station
    bsl_df = bsl_df.copy()
    bsl_df["RangeAddress"] = bsl_df["RangeAddress"].astype(int).astype(str)
    bsl_df["Range(ms)"] = pd.to_numeric(bsl_df["Range(ms)"], errors="coerce")
    bsl_df["TAT(ms)"] = pd.to_numeric(bsl_df["TAT(ms)"], errors="coerce")

    filtered_df = bsl_df[bsl_df["RangeAddress"] == target_station].copy()

    if filtered_df.empty:
        raise ValueError(f"No data for baseline {source_station} -> {target_station}")

    # Apply filtering based on type
    if filter_type == "iqr" and outlier_quantiles is not None:
        # IQR-based outlier removal
        Q1 = filtered_df["Range(ms)"].quantile(outlier_quantiles[0])
        Q3 = filtered_df["Range(ms)"].quantile(outlier_quantiles[1])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        filtered_df = filtered_df[
            (filtered_df["Range(ms)"] >= lower_bound)
            & (filtered_df["Range(ms)"] <= upper_bound)
        ]
    elif filter_type == "hard":
        # Hard threshold filtering
        if min_range is not None:
            filtered_df = filtered_df[filtered_df["Range(ms)"] >= min_range]
        if max_range is not None:
            filtered_df = filtered_df[filtered_df["Range(ms)"] <= max_range]
    else:
        # Fallback: apply min/max range if specified
        if min_range is not None:
            filtered_df = filtered_df[filtered_df["Range(ms)"] >= min_range]
        if max_range is not None:
            filtered_df = filtered_df[filtered_df["Range(ms)"] <= max_range]

    if filtered_df.empty:
        raise ValueError(f"No data after filtering for baseline {source_station} -> {target_station}")

    # Ensure datetime
    filtered_df["Record Time"] = pd.to_datetime(filtered_df["Record Time"])
    harmonic_df["Record Time"] = pd.to_datetime(harmonic_df["Record Time"])

    # Interpolate sound speed onto baseline timestamps
    filtered_df["Interpolated Sound Speed"] = np.interp(
        filtered_df["Record Time"].view(np.int64),
        harmonic_df["Record Time"].view(np.int64),
        harmonic_df["HMean"],
    )

    # Calculate distance
    filtered_df["Calculated Distance (m)"] = filtered_df[
        "Interpolated Sound Speed"
    ] * ((filtered_df["Range(ms)"] - filtered_df["TAT(ms)"]) / 2000)

    return filtered_df


def add_moving_average(
    df: pd.DataFrame,
    window: str = "30D",
    dist_col: str = "Calculated Distance (m)",
    scale: float = 100.0,
) -> pd.DataFrame:
    """
    Add moving average column to distance DataFrame.

    Args:
        df: DataFrame with distance measurements
        window: Rolling window specification
        dist_col: Name of distance column
        scale: Scale factor (e.g., 100 for cm)

    Returns:
        DataFrame with added 'Moving Average' column
    """
    df = df.copy()
    df["Record Time"] = pd.to_datetime(df["Record Time"])
    df = df.sort_values(by="Record Time")
    df = df.set_index("Record Time")
    df["Moving Average"] = df[dist_col].rolling(window).mean() * scale
    df = df.reset_index()
    return df


def save_distance_csv(
    df: pd.DataFrame,
    output_path: str,
    dist_col: str = "Calculated Distance (m)",
    scale: float = 100.0,
    use_tilt_corrected: bool = True,
    exclude_first_days: int = 0,
) -> None:
    """
    Save distance time series to CSV file.

    Args:
        df: DataFrame with distance measurements
        output_path: Path for output CSV file
        dist_col: Name of distance column (used if tilt-corrected not available)
        scale: Scale factor for distance (e.g., 100 for cm)
        use_tilt_corrected: If True, use 'Distance_tiltcorr(m)' column when available
        exclude_first_days: Number of days to exclude from the start of data
    """
    # Use tilt-corrected distance if available and requested
    actual_dist_col = dist_col
    if use_tilt_corrected and "Distance_tiltcorr(m)" in df.columns:
        # Only use if we have valid values
        if df["Distance_tiltcorr(m)"].notna().any():
            actual_dist_col = "Distance_tiltcorr(m)"

    df = add_moving_average(df, dist_col=actual_dist_col, scale=scale)

    # Exclude first N days if requested
    if exclude_first_days > 0:
        df["Record Time"] = pd.to_datetime(df["Record Time"])
        start_date = df["Record Time"].min() + pd.Timedelta(days=exclude_first_days)
        df = df[df["Record Time"] >= start_date]

    output_df = pd.DataFrame(
        {
            "Record Time": df["Record Time"],
            "Distance (cm)": df[actual_dist_col] * scale,
            "Moving Average (cm)": df["Moving Average"],
        }
    )

    output_df.to_csv(output_path, index=False)
    print(f"Distance data saved to '{output_path}'")
