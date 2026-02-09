"""
Publication Figures Module for FETCH Range Calculation

Generate all visualization outputs including velocity comparisons,
distance time series, pressure plots, and triangle area analysis.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.signal import find_peaks
from pathlib import Path

# Configure matplotlib for publication quality
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# Y-axis limits for distance plots (in cm) from notebook
# Key format: "source_target" e.g., "2504_2502" for Central->East
DISTANCE_YLIM = {
    "2504_2502": (164200, 164236),  # Central -> East (CE) - single plot
    "2504_2503": (176550, 176570),  # Central -> West (CW) - single plot
    "2503_2502": (326010, 326055),  # West -> East (WE) - single plot
    "2503_2504": (176550, 176570),  # West -> Central (WC) - single plot
    "2502_2503": (326000, 326060),  # East -> West (EW) - single plot
    "2502_2504": (164200, 164236),  # East -> Central (EC) - single plot
}

# Y-axis limits for bidirectional comparison plots
BIDIRECTIONAL_YLIM = {
    ("2504_2503", "2503_2504"): (176555, 176575),  # CW/WC
    ("2504_2502", "2502_2504"): (164200, 164240),  # CE/EC
    ("2503_2502", "2502_2503"): (326000, 326060),  # WE/EW
}

# Default time limits for distance plots (from notebook)
DEFAULT_XLIM = (pd.Timestamp("2022-09-01"), pd.Timestamp("2025-10-01"))

# Station pair display names (matching notebook titles)
STATION_PAIR_NAMES = {
    "2504_2502": "Central -> East",
    "2504_2503": "Central -> West",
    "2503_2502": "West -> East",
    "2503_2504": "West -> Central",
    "2502_2503": "East -> West",
    "2502_2504": "East -> Central",
}

# Title format for distance plots (matching notebook)
DISTANCE_TITLES = {
    "2504_2502": "Central -> East Distance",
    "2504_2503": "Central -> West Distance",
    "2503_2502": "West -> East Distance",
    "2503_2504": "West -> Central Distance",
    "2502_2503": "East -> West Distance",
    "2502_2504": "East -> Central Distance",
}

# Baseline lengths in km (for normalized precision calculations)
BASELINE_LENGTHS_KM = {
    "2504_2502": 1.642,  # Central -> East (CE)
    "2502_2504": 1.642,  # East -> Central (EC)
    "2504_2503": 1.765,  # Central -> West (CW)
    "2503_2504": 1.765,  # West -> Central (WC)
    "2503_2502": 3.26,   # West -> East (WE)
    "2502_2503": 3.26,   # East -> West (EW)
}


def set_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 300,
    })


def plot_velocity_comparison(
    combined_dfs: Dict[str, pd.DataFrame],
    result_dfs: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot comparison of velocimeter vs TEOS-10 calculated velocity.

    Args:
        combined_dfs: Dictionary of station ID to combined DataFrame with 'velocity' column
        result_dfs: Dictionary of station ID to sound speed DataFrame with 'SoundSpeed' column
        output_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    set_publication_style()

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )

    # Plot velocities
    if "2504" in combined_dfs:
        df4 = combined_dfs["2504"]
        ax1.plot(df4["Record Time"], df4.get("interp_v", df4["velocity"]),
                 label="NT Recorded Velocity")
        ax1.plot(df4["Record Time"], df4["velocity"],
                 label="TEOS-10 Velocity NT")

    if "2503" in combined_dfs:
        df3 = combined_dfs["2503"]
        ax1.plot(df3["Record Time"], df3["velocity"],
                 label="TEOS-10 Velocity WT", linewidth=1)

    if "2502" in combined_dfs:
        df2 = combined_dfs["2502"]
        ax1.plot(df2["Record Time"], df2.get("interp_v", df2["velocity"]),
                 label="ET Recorded Velocity")
        ax1.plot(df2["Record Time"], df2["velocity"],
                 label="TEOS-10 Velocity ET", color="black", linewidth=1)

    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title("(a) Comparison of Calculated Velocity")
    ax1.legend(loc="upper left", fontsize=11)

    # Plot differences
    ax2.axhline(0, linewidth=1)

    if "2504" in combined_dfs and "interp_v" in combined_dfs["2504"].columns:
        df4 = combined_dfs["2504"]
        nt_diff = df4["interp_v"] - df4["velocity"]
        ax2.plot(df4["Record Time"], nt_diff, label="NT (Recorded - TEOS-10)")

    if "2502" in combined_dfs and "interp_v" in combined_dfs["2502"].columns:
        df2 = combined_dfs["2502"]
        et_diff = df2["interp_v"] - df2["velocity"]
        ax2.plot(df2["Record Time"], et_diff, label="ET (Recorded - TEOS-10)")

    ax2.set_ylabel("Δv (m/s)")
    ax2.set_xlabel("Time")
    ax2.set_title("(b) Difference: Recorded minus TEOS-10")
    ax2.legend(loc="upper left", fontsize=11)

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_distance_timeseries(
    distance_df: pd.DataFrame,
    station_pair: str,
    output_path: Optional[str] = None,
    dist_col: str = "Calculated Distance (m)",
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> plt.Figure:
    """
    Plot distance time series with moving average.

    Args:
        distance_df: DataFrame with Record Time and distance columns
        station_pair: Station pair label (e.g., "Central -> East")
        output_path: Optional path to save figure
        dist_col: Name of distance column
        ylim: Optional y-axis limits in cm
        xlim: Optional x-axis limits (timestamps)

    Returns:
        Matplotlib Figure object
    """
    set_publication_style()

    df = distance_df.copy()
    df["Record Time"] = pd.to_datetime(df["Record Time"])
    df = df.sort_values(by="Record Time")
    df = df.set_index("Record Time")
    df["Moving Average"] = df[dist_col].rolling("30D").mean() * 100
    df = df.reset_index()

    # Use notebook figure size (14, 5)
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        df["Record Time"],
        df[dist_col] * 100,
        ".",
        label="TEOS10 Distance",
        color="salmon",
    )
    ax.plot(
        df["Record Time"],
        df["Moving Average"],
        "r.",
        label="30-day Moving Average",
    )

    ax.set_xlabel("Record Time")
    ax.set_ylabel("Distance (cm)")
    ax.legend()
    ax.set_title(f"{station_pair}")
    ax.grid(True)

    if ylim:
        ax.set_ylim(ylim)

    # Apply x-axis limits (default from notebook: 2022-09-01 to 2025-10-01)
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(DEFAULT_XLIM)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_pressure_comparison(
    fetch_dfs: Dict[str, pd.DataFrame],
    node_dfs: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> plt.Figure:
    """
    Plot pressure comparison between FETCH stations and OOI nodes.

    Args:
        fetch_dfs: Dictionary of FETCH station DataFrames
        node_dfs: Dictionary of OOI node DataFrames
        output_path: Optional path to save figure
        start: Start time for plot
        end: End time for plot

    Returns:
        Matplotlib Figure object
    """
    from .pressure import compute_pressure_ma_series, rebase_to_window

    set_publication_style()

    if start is None:
        start = pd.Timestamp("2022-09-01")
    if end is None:
        end = pd.Timestamp("2025-09-01")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot FETCH stations
    station_names = {"2504": "Central", "2502": "East", "2503": "West"}
    for sid, name in station_names.items():
        if sid in fetch_dfs:
            df = fetch_dfs[sid]
            ma = compute_pressure_ma_series(
                df, "DateTime", "Corrected Pressure (kPa)", "15D", "kpa"
            )
            ma = rebase_to_window(ma, start, end)
            sw = ma.loc[start:end]
            if not sw.empty:
                ax.plot(sw.index, sw.values, linewidth=2, label=f"{name} (15-day MA)")

    # Plot OOI nodes
    node_names = {"F": "Node-F", "E": "Node-E", "B": "Node-B"}
    for nid, name in node_names.items():
        if nid in node_dfs:
            df = node_dfs[nid]
            ma = compute_pressure_ma_series(df, "time", "p", "15D", "psi")
            ma = rebase_to_window(ma, start, end)
            sw = ma.loc[start:end]
            if not sw.empty:
                ax.plot(
                    sw.index, sw.values, linewidth=2, linestyle="--", label=f"{name} (15-day MA)"
                )

    ax.axhline(0, lw=1, ls="--", alpha=0.5)
    ax.set_xlim(start, end)
    ax.set_xlabel("Time")
    ax.set_ylabel("15-day mean pressure anomaly (kPa)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_triangle_area(
    area_df: pd.DataFrame,
    output_path: Optional[str] = None,
    exclude_first_days: int = 30,
) -> plt.Figure:
    """
    Plot triangle area time series with trend analysis.

    Args:
        area_df: DataFrame with Record Time and Area MA (m^2) columns
        output_path: Optional path to save figure
        exclude_first_days: Days to exclude from start

    Returns:
        Matplotlib Figure object
    """
    set_publication_style()

    df = area_df.copy()
    df["Record Time"] = pd.to_datetime(df["Record Time"])
    df = df.sort_values("Record Time").set_index("Record Time")

    # Remove first N days
    start_date = df.index[0] + pd.Timedelta(days=exclude_first_days)
    df_filtered = df[df.index >= start_date]
    ma = df_filtered["Area MA (m^2)"].dropna()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(ma.index, ma.values, color="darkred", linewidth=2, label="30-day Moving Average")

    # Add linear trend
    days = (ma.index - ma.index[0]).total_seconds() / 86400
    slope, intercept, r_value, p_value, std_err = stats.linregress(days, ma.values)
    trend_line = intercept + slope * days

    ax.plot(
        ma.index,
        trend_line,
        "k--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Linear Trend: {slope*365.25:.1f} m²/yr (R²={r_value**2:.3f})",
    )

    ax.fill_between(
        ma.index, ma.values, trend_line,
        where=(ma.values > trend_line), color="red", alpha=0.2, label="Above trend"
    )
    ax.fill_between(
        ma.index, ma.values, trend_line,
        where=(ma.values < trend_line), color="blue", alpha=0.2, label="Below trend"
    )

    ax.set_ylabel("Area (m²)")
    ax.set_title("Triangle Area (30-day MA) - Axial Seamount")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {output_path}")

    return fig


def plot_triangle_area_full(
    area_df: pd.DataFrame,
    output_path: Optional[str] = None,
    exclude_first_days: int = 30,
) -> plt.Figure:
    """
    Create comprehensive visualization of triangle area analysis.

    Includes: trend plot, detrended anomaly, and residuals.

    Args:
        area_df: DataFrame with Record Time and Area MA (m^2) columns
        output_path: Optional path to save figure
        exclude_first_days: Days to exclude from start

    Returns:
        Matplotlib Figure object
    """
    set_publication_style()

    df = area_df.copy()
    df["Record Time"] = pd.to_datetime(df["Record Time"])
    df = df.sort_values("Record Time").set_index("Record Time")

    # Remove first N days
    start_date = df.index[0] + pd.Timedelta(days=exclude_first_days)
    df_filtered = df[df.index >= start_date]
    ma = df_filtered["Area MA (m^2)"].dropna()

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # Plot 1: 30-day MA with trend line
    ax1 = axes[0]
    ax1.plot(ma.index, ma.values, color="darkred", linewidth=2, label="30-day Moving Average")

    days = (ma.index - ma.index[0]).total_seconds() / 86400
    slope, intercept, r_value, p_value, std_err = stats.linregress(days, ma.values)
    trend_line = intercept + slope * days

    ax1.plot(
        ma.index, trend_line, "k--", linewidth=1.5, alpha=0.7,
        label=f"Linear Trend: {slope*365.25:.1f} m²/yr (R²={r_value**2:.3f})",
    )

    ax1.fill_between(
        ma.index, ma.values, trend_line,
        where=(ma.values > trend_line), color="red", alpha=0.2, label="Above trend"
    )
    ax1.fill_between(
        ma.index, ma.values, trend_line,
        where=(ma.values < trend_line), color="blue", alpha=0.2, label="Below trend"
    )

    ax1.set_ylabel("Area (m²)")
    ax1.set_title("Triangle Area (30-day MA) - Axial Seamount")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Detrended (relative to mean)
    ax2 = axes[1]
    ma_anomaly = ma - ma.mean()
    ax2.plot(ma.index, ma_anomaly, color="darkblue", linewidth=2)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax2.axhline(
        y=ma_anomaly.std(), color="gray", linestyle=":", alpha=0.5,
        label=f"±1σ ({ma_anomaly.std():.1f} m²)"
    )
    ax2.axhline(y=-ma_anomaly.std(), color="gray", linestyle=":", alpha=0.5)
    ax2.fill_between(ma.index, 0, ma_anomaly, where=(ma_anomaly > 0), color="red", alpha=0.3)
    ax2.fill_between(ma.index, 0, ma_anomaly, where=(ma_anomaly < 0), color="blue", alpha=0.3)

    # Mark peaks and troughs
    peaks_idx, _ = find_peaks(ma.values, prominence=100)
    troughs_idx, _ = find_peaks(-ma.values, prominence=100)
    if len(peaks_idx) > 0:
        ax2.plot(ma.index[peaks_idx], ma_anomaly.iloc[peaks_idx], "r^", markersize=10)
    if len(troughs_idx) > 0:
        ax2.plot(ma.index[troughs_idx], ma_anomaly.iloc[troughs_idx], "bv", markersize=10)

    ax2.set_ylabel("Area Anomaly (m²)")
    ax2.set_title("Detrended Triangle Area")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals from trend
    ax3 = axes[2]
    residuals = ma.values - trend_line
    ax3.plot(ma.index, residuals, color="green", linewidth=1)
    ax3.axhline(0, color="k", linestyle="--")
    ax3.set_ylabel("Residual (m²)")
    ax3.set_xlabel("Time")
    ax3.set_title("Residuals from Linear Trend")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {output_path}")

    return fig


def plot_baseline_precision(
    distance_dfs: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None,
    max_lag_days: int = 90,
) -> plt.Figure:
    """
    Plot baseline precision analysis (scatter vs time separation).

    Args:
        distance_dfs: Dictionary of baseline name to distance DataFrame
        output_path: Optional path to save figure
        max_lag_days: Maximum lag in days for analysis

    Returns:
        Matplotlib Figure object
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(11, 5))

    for name, df in distance_dfs.items():
        res = _scatter_vs_lag(df, max_lag_days=max_lag_days)
        ax.plot(res["lag_days"], res["rms_diff_cm"], ".", label=f"{name} RMS")

    ax.set_xlabel("Time separation Δt (days)")
    ax.set_ylabel("Scatter (cm)")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def _scatter_vs_lag(
    df: pd.DataFrame,
    time_col: str = "Record Time",
    dist_col: str = "Calculated Distance (m)",
    max_lag_days: int = 60,
    min_pairs: int = 200,
) -> pd.DataFrame:
    """
    Compute scatter as a function of time separation.

    Internal helper for precision analysis.
    """
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    x_cm = pd.to_numeric(df[dist_col], errors="coerce") * 100.0
    s = pd.Series(x_cm.values, index=t).dropna().sort_index()

    # Regularize to 4H
    x4 = s.resample("4H").median()

    samples_per_day = 6
    max_k = int(max_lag_days * samples_per_day)

    rows = []
    for k in range(0, max_k + 1):
        a = x4
        b = x4.shift(-k)
        valid = a.notna() & b.notna()
        diffs = (b[valid] - a[valid]).values
        n = diffs.size

        if n < min_pairs:
            rows.append([k, k / samples_per_day, n, np.nan])
            continue

        rms = float(np.sqrt(np.mean(diffs**2)))
        rows.append([k, k / samples_per_day, n, rms])

    return pd.DataFrame(
        rows, columns=["lag_samples", "lag_days", "n_pairs", "rms_diff_cm"]
    )


def _compute_moving_average(df: pd.DataFrame, dist_col: str, window: str = "30D") -> pd.Series:
    """Compute 30-day moving average for distance column."""
    df_sorted = df.copy()
    df_sorted["Record Time"] = pd.to_datetime(df_sorted["Record Time"])
    df_sorted = df_sorted.sort_values("Record Time").set_index("Record Time")
    ma = df_sorted[dist_col].rolling(window).mean()
    return ma.values


def plot_bidirectional_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label1: str,
    label2: str,
    title: str,
    ylim: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None,
    xlim: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> plt.Figure:
    """
    Plot bidirectional baseline comparison (e.g., CE vs EC).

    Args:
        df1: First baseline DataFrame with distance data
        df2: Second baseline DataFrame
        label1: Label for first baseline (e.g., "Central → East")
        label2: Label for second baseline (e.g., "East → Central")
        title: Plot title
        ylim: Y-axis limits as (min, max)
        output_path: Optional path to save figure
        xlim: Optional x-axis limits (timestamps)

    Returns:
        Figure object
    """
    # Use notebook figure size (14, 6) for bidirectional plots
    fig, ax = plt.subplots(figsize=(14, 6))

    # Make copies and prepare data
    df1 = df1.copy()
    df2 = df2.copy()

    # Ensure datetime
    df1["Record Time"] = pd.to_datetime(df1["Record Time"])
    df2["Record Time"] = pd.to_datetime(df2["Record Time"])

    # Use tilt-corrected distance if available, otherwise use calculated
    if "Distance_tiltcorr(m)" in df1.columns:
        df1["Distance (cm)"] = df1["Distance_tiltcorr(m)"] * 100
    elif "Calculated Distance (m)" in df1.columns:
        df1["Distance (cm)"] = df1["Calculated Distance (m)"] * 100

    if "Distance_tiltcorr(m)" in df2.columns:
        df2["Distance (cm)"] = df2["Distance_tiltcorr(m)"] * 100
    elif "Calculated Distance (m)" in df2.columns:
        df2["Distance (cm)"] = df2["Calculated Distance (m)"] * 100

    # Compute moving average if not present
    if "Moving Average (cm)" not in df1.columns:
        df1["Moving Average (cm)"] = _compute_moving_average(df1, "Distance (cm)")
    if "Moving Average (cm)" not in df2.columns:
        df2["Moving Average (cm)"] = _compute_moving_average(df2, "Distance (cm)")

    # Colors
    color1_light = "lightcoral"
    color1_dark = "darkred"
    color2_light = "lightblue"
    color2_dark = "darkblue"

    # Scatter plots
    ax.plot(df1["Record Time"], df1["Distance (cm)"], ".",
            color=color1_light, label=f"{label1} Distance", alpha=0.5, markersize=3)
    ax.plot(df2["Record Time"], df2["Distance (cm)"], ".",
            color=color2_light, label=f"{label2} Distance", alpha=0.5, markersize=3)

    # Moving average lines
    ax.plot(df1["Record Time"], df1["Moving Average (cm)"],
            color=color1_dark, linewidth=2, label=f"{label1} 30-day MA")
    ax.plot(df2["Record Time"], df2["Moving Average (cm)"],
            color=color2_dark, linewidth=2, label=f"{label2} 30-day MA")

    ax.set_xlabel("Record Time")
    ax.set_ylabel("Distance (cm)")
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    # Apply x-axis limits (default from notebook: 2022-09-01 to 2025-10-01)
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(DEFAULT_XLIM)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_30day_binned_errorbar(
    df: pd.DataFrame,
    baseline_name: str,
    exclude_first_days: int = 30,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 30-day binned distance with error bars (mean ± std).

    Args:
        df: DataFrame with Record Time and distance data
        baseline_name: Name for plot title
        exclude_first_days: Days to exclude from start
        output_path: Optional path to save figure

    Returns:
        Figure object
    """
    df = df.copy()
    df["Record Time"] = pd.to_datetime(df["Record Time"])

    # Get distance in meters
    if "Distance_tiltcorr(m)" in df.columns:
        df["distance_m"] = df["Distance_tiltcorr(m)"]
    elif "Calculated Distance (m)" in df.columns:
        df["distance_m"] = df["Calculated Distance (m)"]
    else:
        raise ValueError("No distance column found")

    # Exclude first N days
    start_date = df["Record Time"].min() + pd.Timedelta(days=exclude_first_days)
    df = df[df["Record Time"] >= start_date].copy()

    # Remove NaN
    df = df.dropna(subset=["distance_m"])

    # Group into 30-day bins
    binned = df.resample("30D", on="Record Time").agg({
        "distance_m": ["mean", "std", "count"]
    })
    binned.columns = ["Mean Distance", "Std Dev", "Count"]
    binned = binned.reset_index()

    # Convert to cm
    binned["Mean Distance"] *= 100
    binned["Std Dev"] *= 100

    # Filter bins with enough points
    binned = binned[binned["Count"] >= 10]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.errorbar(
        binned["Record Time"],
        binned["Mean Distance"],
        yerr=binned["Std Dev"],
        fmt="o-", color="blue", label="30-day Binned Mean (±1σ)", capsize=5
    )

    ax.set_title(f"{baseline_name} Distance (30-Day Binned) with Error Bars")
    ax.set_xlabel("Record Time")
    ax.set_ylabel("Calculated Distance (cm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_rolling_std_normalized(
    distance_dfs: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None,
    window_days: int = 30,
    min_points: int = 10,
    exclude_first_days: int = 30,
    use_residuals: bool = True,
) -> plt.Figure:
    """
    Plot rolling standard deviation normalized by baseline length (cm/km).

    Args:
        distance_dfs: Dictionary of baseline DataFrames
        output_path: Optional path to save figure
        window_days: Rolling window size in days
        min_points: Minimum points required in window
        exclude_first_days: Days to exclude from start
        use_residuals: If True, use scatter from MA; else use raw distance

    Returns:
        Figure object
    """
    # Prepare data for each baseline
    dfs = {}
    tmins, tmaxs = [], []

    # Map our keys to the notebook's naming
    key_map = {
        "2504_2502": "CE",
        "2502_2504": "EC",
        "2504_2503": "CW",
        "2503_2504": "WC",
        "2503_2502": "WE",
        "2502_2503": "EW",
    }

    for key, df in distance_dfs.items():
        df = df.copy()
        df["Record Time"] = pd.to_datetime(df["Record Time"])

        # Get distance in cm
        if "Distance_tiltcorr(m)" in df.columns:
            df["Distance (cm)"] = df["Distance_tiltcorr(m)"] * 100
        elif "Calculated Distance (m)" in df.columns:
            df["Distance (cm)"] = df["Calculated Distance (m)"] * 100

        # Compute moving average if needed
        df_sorted = df.sort_values("Record Time").set_index("Record Time")
        df["Moving Average (cm)"] = df_sorted["Distance (cm)"].rolling("30D").mean().values

        # Compute scatter (residual from MA)
        if use_residuals:
            df["scatter"] = df["Distance (cm)"] - df["Moving Average (cm)"]
        else:
            df["scatter"] = df["Distance (cm)"]

        # Exclude first N days
        start_date = df["Record Time"].min() + pd.Timedelta(days=exclude_first_days)
        df = df[df["Record Time"] >= start_date].copy()

        name = key_map.get(key, key)
        dfs[name] = df
        tmins.append(df["Record Time"].min())
        tmaxs.append(df["Record Time"].max())

    if not dfs:
        return None

    global_start = min(tmins)
    global_end = max(tmaxs)

    def rolling_std_cm_per_km(df, L_km):
        s = df.set_index("Record Time")["scatter"].dropna().sort_index()
        s = s.groupby(level=0).mean()  # Handle duplicates
        std_cm = s.rolling(f"{window_days}D", min_periods=min_points).std()
        return std_cm / L_km

    # Compute std series for each baseline
    std_series = {}
    length_km = {
        "CE": 1.642, "EC": 1.642,
        "CW": 1.765, "WC": 1.765,
        "WE": 3.26, "EW": 3.26,
    }

    for name, df in dfs.items():
        if name in length_km:
            std_series[name] = rolling_std_cm_per_km(df, length_km[name])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Define pairs and colors
    pairs = [
        ("CW", "WC", "Central-West", "lightcoral", "darkred"),
        ("CE", "EC", "Central-East", "lightblue", "darkblue"),
        ("WE", "EW", "West-East", "lightgreen", "darkgreen"),
    ]

    for a, b, pair_label, light_c, dark_c in pairs:
        if a in std_series:
            ax.plot(std_series[a].index, std_series[a].values,
                    linewidth=2, color=dark_c, label=f"{a} ({pair_label})")
        if b in std_series:
            ax.plot(std_series[b].index, std_series[b].values,
                    linewidth=2, color=light_c, label=f"{b} ({pair_label})")

    ax.set_ylabel(f"{window_days}-day Rolling Std Dev (cm/km)")
    ax.set_xlabel("Time")
    ax.set_title(f"{window_days}-day Rolling Standard Deviation (Normalized by Baseline Length)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    ax.set_xlim(global_start, global_end)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_precision_summary(
    distance_dfs: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None,
    exclude_first_days: int = 30,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Compute and plot 30-day precision metrics (cm/km) for each baseline.

    Args:
        distance_dfs: Dictionary of baseline DataFrames
        output_path: Optional path to save figure
        exclude_first_days: Days to exclude from start

    Returns:
        Tuple of (Figure, summary DataFrame)
    """
    results = []
    key_map = {
        "2504_2502": "CE", "2502_2504": "EC",
        "2504_2503": "CW", "2503_2504": "WC",
        "2503_2502": "WE", "2502_2503": "EW",
    }
    length_km = {
        "CE": 1.642, "EC": 1.642,
        "CW": 1.765, "WC": 1.765,
        "WE": 3.26, "EW": 3.26,
    }

    for key, df in distance_dfs.items():
        name = key_map.get(key, key)
        if name not in length_km:
            continue

        df = df.copy()
        df["Record Time"] = pd.to_datetime(df["Record Time"])

        if "Distance_tiltcorr(m)" in df.columns:
            df["distance_m"] = df["Distance_tiltcorr(m)"]
        elif "Calculated Distance (m)" in df.columns:
            df["distance_m"] = df["Calculated Distance (m)"]
        else:
            continue

        # Exclude first N days
        start_date = df["Record Time"].min() + pd.Timedelta(days=exclude_first_days)
        df = df[df["Record Time"] >= start_date].copy()
        df = df.dropna(subset=["distance_m"])

        # 30-day binned stats
        binned = (
            df.groupby(pd.Grouper(key="Record Time", freq="30D"))
            .agg(
                distance_mean_m=("distance_m", "mean"),
                distance_std_m=("distance_m", "std"),
                count=("distance_m", "size")
            )
            .reset_index()
        )
        binned = binned[binned["count"] >= 10].copy()

        if len(binned) == 0:
            continue

        # Precision metrics
        binned["precision_cm_per_km"] = 1e5 * (binned["distance_std_m"] / binned["distance_mean_m"])

        median_prec = binned["precision_cm_per_km"].median()
        q25 = binned["precision_cm_per_km"].quantile(0.25)
        q75 = binned["precision_cm_per_km"].quantile(0.75)

        results.append({
            "Baseline": name,
            "Length (km)": length_km[name],
            "Median Precision (cm/km)": median_prec,
            "IQR Low": q25,
            "IQR High": q75,
            "N Windows": len(binned),
        })

    summary_df = pd.DataFrame(results)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(summary_df))
    ax.bar(x, summary_df["Median Precision (cm/km)"], color="steelblue", alpha=0.8)
    ax.errorbar(
        x, summary_df["Median Precision (cm/km)"],
        yerr=[
            summary_df["Median Precision (cm/km)"] - summary_df["IQR Low"],
            summary_df["IQR High"] - summary_df["Median Precision (cm/km)"]
        ],
        fmt="none", color="black", capsize=5
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Baseline"])
    ax.set_ylabel("30-day Precision (cm/km)")
    ax.set_title("Baseline Precision Summary (Median ± IQR)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig, summary_df


def create_all_figures(
    results: Dict,
    output_dir: str,
) -> List[str]:
    """
    Generate all publication figures from pipeline results.

    Args:
        results: Dictionary containing pipeline output DataFrames
        output_dir: Directory to save figures

    Returns:
        List of paths to generated figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated = []

    # Velocity comparison
    if "combined_dfs" in results and "result_dfs" in results:
        path = str(output_path / "velocity_comparison.pdf")
        plot_velocity_comparison(results["combined_dfs"], results["result_dfs"], path)
        generated.append(path)

    # Distance time series
    if "distance_dfs" in results:
        for name, df in results["distance_dfs"].items():
            path = str(output_path / f"distance_timeseries_{name}.png")
            # Use DISTANCE_TITLES for the full title (matching notebook format)
            display_name = DISTANCE_TITLES.get(name, f"{STATION_PAIR_NAMES.get(name, name)} Distance")
            ylim = DISTANCE_YLIM.get(name, None)
            plot_distance_timeseries(df, display_name, path, ylim=ylim)
            generated.append(path)

        # Bidirectional comparison plots
        bidirectional_pairs = [
            ("2504_2503", "2503_2504", "Central → West", "West → Central",
             "Central-West vs West-Central", (176555, 176575)),
            ("2504_2502", "2502_2504", "Central → East", "East → Central",
             "East-Central vs Central-East", (164200, 164240)),
            ("2503_2502", "2502_2503", "West → East", "East → West",
             "West-East vs East-West", (326000, 326060)),
        ]

        for key1, key2, label1, label2, title, ylim in bidirectional_pairs:
            if key1 in results["distance_dfs"] and key2 in results["distance_dfs"]:
                df1 = results["distance_dfs"][key1]
                df2 = results["distance_dfs"][key2]
                filename = f"bidirectional_{key1}_{key2}.png"
                path = str(output_path / filename)
                plot_bidirectional_comparison(df1, df2, label1, label2, title, ylim, path)
                generated.append(path)

    # Triangle area
    if "triangle_df" in results:
        path = str(output_path / "triangle_area_analysis.png")
        plot_triangle_area(results["triangle_df"], path)
        generated.append(path)

        path_full = str(output_path / "triangle_area_full_analysis.png")
        plot_triangle_area_full(results["triangle_df"], path_full)
        generated.append(path_full)

    # Pressure comparison
    if "combined_dfs" in results and "node_dfs" in results:
        path = str(output_path / "pressure_comparison.png")
        plot_pressure_comparison(results["combined_dfs"], results["node_dfs"], path)
        generated.append(path)

    # 30-day binned error bar plots for each baseline
    if "distance_dfs" in results:
        for name, df in results["distance_dfs"].items():
            display_name = STATION_PAIR_NAMES.get(name, name)
            path = str(output_path / f"binned_errorbar_{name}.png")
            try:
                plot_30day_binned_errorbar(df, display_name, exclude_first_days=30, output_path=path)
                generated.append(path)
            except Exception as e:
                print(f"  Warning: Could not generate binned plot for {name}: {e}")

        # Rolling std normalized plot
        path = str(output_path / "rolling_std_normalized.png")
        try:
            plot_rolling_std_normalized(results["distance_dfs"], output_path=path)
            generated.append(path)
        except Exception as e:
            print(f"  Warning: Could not generate rolling std plot: {e}")

        # Precision summary
        path = str(output_path / "precision_summary.png")
        try:
            fig, summary_df = plot_precision_summary(results["distance_dfs"], output_path=path)
            # Print summary
            print("\n30-day Precision Summary:")
            for _, row in summary_df.iterrows():
                print(f"  {row['Baseline']}: {row['Median Precision (cm/km)']:.2f} cm/km "
                      f"(IQR: {row['IQR Low']:.2f}-{row['IQR High']:.2f})")
            generated.append(path)
        except Exception as e:
            print(f"  Warning: Could not generate precision summary: {e}")

    print(f"\nGenerated {len(generated)} figures in {output_dir}")
    return generated
