"""
Triangle Area and Areal Strain Module for FETCH Range Calculation

Geometric calculations for deformation monitoring using acoustic
baseline measurements between three stations.
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats


def calculate_triangle_area(
    side1: float,
    side2: float,
    side3: float,
) -> float:
    """
    Calculate triangle area using Heron's formula.

    Args:
        side1: Length of first side (any unit)
        side2: Length of second side
        side3: Length of third side

    Returns:
        Area of triangle (in squared units of input)
    """
    s = (side1 + side2 + side3) / 2  # Semi-perimeter
    area_squared = s * (s - side1) * (s - side2) * (s - side3)

    if area_squared < 0:
        return np.nan  # Invalid triangle

    return np.sqrt(area_squared)


def merge_baseline_distances(
    wc_df: pd.DataFrame,
    we_df: pd.DataFrame,
    ce_df: pd.DataFrame,
    time_col: str = "Record Time",
    dist_col: str = "Distance (cm)",
    tolerance: str = "1H",
) -> pd.DataFrame:
    """
    Merge three baseline distance time series for triangle calculation.

    Aligns West-Central, West-East, and Central-East baselines temporally.

    Args:
        wc_df: West-Central distance DataFrame
        we_df: West-East distance DataFrame
        ce_df: Central-East distance DataFrame
        time_col: Name of time column
        dist_col: Name of distance column
        tolerance: Time tolerance for merge

    Returns:
        Merged DataFrame with WC, WE, CE distance columns
    """
    # Prepare DataFrames - handle both DataFrame and None
    if wc_df is None or we_df is None or ce_df is None:
        raise ValueError("All three baseline DataFrames must be provided")

    wc = wc_df[[time_col, dist_col]].copy()
    we = we_df[[time_col, dist_col]].copy()
    ce = ce_df[[time_col, dist_col]].copy()

    wc.columns = [time_col, "WC"]
    we.columns = [time_col, "WE"]
    ce.columns = [time_col, "CE"]

    # Convert to datetime and sort
    wc[time_col] = pd.to_datetime(wc[time_col])
    we[time_col] = pd.to_datetime(we[time_col])
    ce[time_col] = pd.to_datetime(ce[time_col])

    wc = wc.sort_values(time_col).reset_index(drop=True)
    we = we.sort_values(time_col).reset_index(drop=True)
    ce = ce.sort_values(time_col).reset_index(drop=True)

    # Merge with tolerance using merge_asof
    merged = pd.merge_asof(
        wc, we, on=time_col, tolerance=pd.Timedelta(tolerance), direction="nearest"
    )
    merged = pd.merge_asof(
        merged, ce, on=time_col, tolerance=pd.Timedelta(tolerance), direction="nearest"
    )

    # Drop rows with any missing values
    merged = merged.dropna(subset=["WC", "WE", "CE"])

    return merged


def compute_area_timeseries(
    merged_df: pd.DataFrame,
    wc_col: str = "WC",
    we_col: str = "WE",
    ce_col: str = "CE",
    input_unit: str = "cm",
) -> pd.DataFrame:
    """
    Compute triangle area time series from merged baseline distances.

    Args:
        merged_df: DataFrame with WC, WE, CE columns
        wc_col: Name of West-Central distance column
        we_col: Name of West-East distance column
        ce_col: Name of Central-East distance column
        input_unit: Unit of input distances ('cm' or 'm')

    Returns:
        DataFrame with added area columns
    """
    df = merged_df.copy()

    # Convert distances to meters for area calculation
    if input_unit == "cm":
        df["WC_m"] = df[wc_col] / 100
        df["WE_m"] = df[we_col] / 100
        df["CE_m"] = df[ce_col] / 100
    else:
        df["WC_m"] = df[wc_col]
        df["WE_m"] = df[we_col]
        df["CE_m"] = df[ce_col]

    # Calculate triangle area using Heron's formula
    df["s"] = (df["WC_m"] + df["WE_m"] + df["CE_m"]) / 2  # Semi-perimeter
    df["Area (m^2)"] = np.sqrt(
        df["s"] * (df["s"] - df["WC_m"]) * (df["s"] - df["WE_m"]) * (df["s"] - df["CE_m"])
    )

    return df


def add_area_moving_average(
    df: pd.DataFrame,
    time_col: str = "Record Time",
    area_col: str = "Area (m^2)",
    window: str = "30D",
) -> pd.DataFrame:
    """
    Add moving average of area to DataFrame.

    Args:
        df: DataFrame with area column
        time_col: Name of time column
        area_col: Name of area column
        window: Rolling window specification

    Returns:
        DataFrame with added 'Area MA (m^2)' column
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    df["Area MA (m^2)"] = df[area_col].rolling(window).mean()
    df = df.reset_index()
    return df


def calculate_areal_strain(
    area_series: pd.Series,
    reference_area: Optional[float] = None,
) -> pd.Series:
    """
    Calculate areal strain from area time series.

    Areal strain = (A - A_ref) / A_ref

    Args:
        area_series: Time series of area values
        reference_area: Reference area for strain calculation
                       If None, uses mean of series

    Returns:
        Series of areal strain values (dimensionless)
    """
    if reference_area is None:
        reference_area = area_series.mean()

    return (area_series - reference_area) / reference_area


def compute_area_trend(
    df: pd.DataFrame,
    time_col: str = "Record Time",
    area_col: str = "Area MA (m^2)",
    exclude_first_days: int = 30,
) -> Tuple[float, float, float, float]:
    """
    Compute linear trend in area time series.

    Args:
        df: DataFrame with time and area columns
        time_col: Name of time column
        area_col: Name of area column
        exclude_first_days: Number of days to exclude from start

    Returns:
        Tuple of (slope_per_year, intercept, r_squared, p_value)
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).set_index(time_col)

    # Remove first N days
    start_date = df.index[0] + pd.Timedelta(days=exclude_first_days)
    df_filtered = df[df.index >= start_date]

    ma = df_filtered[area_col].dropna()

    if len(ma) < 2:
        return np.nan, np.nan, np.nan, np.nan

    # Convert time to days
    days = (ma.index - ma.index[0]).total_seconds() / 86400

    slope, intercept, r_value, p_value, std_err = stats.linregress(days, ma.values)

    # Convert slope to per year
    slope_per_year = slope * 365.25

    return slope_per_year, intercept, r_value**2, p_value


def compute_triangle_metrics(
    wc_df: pd.DataFrame,
    we_df: pd.DataFrame,
    ce_df: pd.DataFrame,
    time_col: str = "Record Time",
    dist_col: str = "Distance (cm)",
    exclude_first_days: int = 30,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute complete triangle area metrics from baseline distances.

    Args:
        wc_df: West-Central distance DataFrame
        we_df: West-East distance DataFrame
        ce_df: Central-East distance DataFrame
        time_col: Name of time column
        dist_col: Name of distance column
        exclude_first_days: Days to exclude from trend calculation
        output_path: Optional path to save CSV output

    Returns:
        DataFrame with triangle area time series
    """
    # Merge baselines
    merged = merge_baseline_distances(wc_df, we_df, ce_df, time_col, dist_col)

    # Compute area
    merged = compute_area_timeseries(merged)

    # Add moving average
    merged = add_area_moving_average(merged, time_col)

    # Compute trend
    slope, intercept, r_squared, p_value = compute_area_trend(
        merged, time_col, exclude_first_days=exclude_first_days
    )

    print(f"Triangle Area Analysis:")
    print(f"  Area range: {merged['Area (m^2)'].min():.2f} to {merged['Area (m^2)'].max():.2f} m²")
    print(f"  Linear trend: {slope:.1f} m²/yr (R² = {r_squared:.3f})")

    # Save to CSV if path provided
    if output_path:
        output_df = merged[[time_col, "Area (m^2)", "Area MA (m^2)"]]
        output_df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")

    return merged


def calculate_full_2d_strain(
    wc_df: pd.DataFrame,
    we_df: pd.DataFrame,
    ce_df: pd.DataFrame,
    reference_lengths: Optional[Tuple[float, float, float]] = None,
    time_col: str = "Record Time",
    dist_col: str = "Distance (cm)",
) -> pd.DataFrame:
    """
    Calculate full 2D strain tensor components from three baseline measurements.

    This uses the three baseline lengths to compute components of the
    2D strain tensor.

    Args:
        wc_df: West-Central distance DataFrame
        we_df: West-East distance DataFrame
        ce_df: Central-East distance DataFrame
        reference_lengths: Optional (WC_ref, WE_ref, CE_ref) in cm
                          If None, uses mean of each series
        time_col: Name of time column
        dist_col: Name of distance column

    Returns:
        DataFrame with strain tensor components
    """
    # Merge baselines
    merged = merge_baseline_distances(wc_df, we_df, ce_df, time_col, dist_col)

    # Set reference lengths
    if reference_lengths is None:
        wc_ref = merged["WC"].mean()
        we_ref = merged["WE"].mean()
        ce_ref = merged["CE"].mean()
    else:
        wc_ref, we_ref, ce_ref = reference_lengths

    # Calculate relative length changes
    merged["dL_WC"] = (merged["WC"] - wc_ref) / wc_ref
    merged["dL_WE"] = (merged["WE"] - we_ref) / we_ref
    merged["dL_CE"] = (merged["CE"] - ce_ref) / ce_ref

    # Calculate areal strain (simple average for small strains)
    merged["areal_strain"] = (
        merged["dL_WC"] + merged["dL_WE"] + merged["dL_CE"]
    ) / 3

    return merged
