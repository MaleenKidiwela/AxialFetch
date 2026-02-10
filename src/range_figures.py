"""
Figure generation for the Calculate_Range pipeline.

All plots are saved as PNG files to the specified output directory.
Reproduces the figures from Calculate_Range.ipynb.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import BASELINE_LENGTH_KM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_corrected(corrected_dir, names=None):
    """Load corrected CSVs into a dict keyed by short label (NW, WN, etc.)."""
    if names is None:
        names = ["NW", "WN", "NE", "EN", "WE", "EW"]
    dfs = {}
    for name in names:
        fp = os.path.join(corrected_dir, f"{name}_distance_corrected.csv")
        df = pd.read_csv(fp)
        df["Record Time"] = pd.to_datetime(df["Record Time"], utc=True,
                                           errors="coerce")
        df = df.dropna(subset=["Record Time"]).sort_values("Record Time")
        dfs[name] = df
    return dfs


def _global_time_range(dfs):
    """Return (global_start, global_end) floored/ceiled to day."""
    tmins = [df["Record Time"].min() for df in dfs.values()]
    tmaxs = [df["Record Time"].max() for df in dfs.values()]
    return min(tmins).floor("D"), max(tmaxs).ceil("D")


# ---------------------------------------------------------------------------
# 1. Distance timeseries: scatter + 30-day MA
# ---------------------------------------------------------------------------

def plot_distance_timeseries(baseline_df, title, ylim, save_path,
                             figsize=(14, 5)):
    """Scatter of Calculated Distance (cm) + 30D rolling MA."""
    plt.rcParams.update({'font.size': 16})
    df = baseline_df.copy()
    df['Record Time'] = pd.to_datetime(df['Record Time'])
    df = df.sort_values('Record Time').set_index('Record Time')
    df['Moving Average'] = df['Calculated Distance (m)'].rolling('30D').mean() * 100
    df = df.reset_index()

    plt.figure(figsize=figsize)
    plt.plot(df['Record Time'],
             df['Calculated Distance (m)'] * 100, '.',
             label="Distance", color='salmon')
    plt.plot(df['Record Time'],
             df['Moving Average'], 'r.',
             label="30-day Moving Average")
    plt.xlabel("Record Time")
    plt.ylabel("Distance (cm)")
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 2. Binned error-bar plot (30-day bins)
# ---------------------------------------------------------------------------

def plot_binned_errorbar(baseline_df, title, save_path, bin_size='30D',
                         figsize=(14, 5)):
    """30-day binned mean +/- 1 std error bars."""
    plt.rcParams.update({'font.size': 14})
    df = baseline_df.copy()
    df['Record Time'] = pd.to_datetime(df['Record Time'])
    df['Distance (cm)'] = df['Calculated Distance (m)'] * 100
    df = df.set_index('Record Time')

    binned_mean = df['Distance (cm)'].resample(bin_size).mean()
    binned_std = df['Distance (cm)'].resample(bin_size).std()

    plt.figure(figsize=figsize)
    plt.errorbar(binned_mean.index, binned_mean.values, yerr=binned_std.values,
                 fmt='o', capsize=3, label='Mean +/- 1 std')
    plt.title(title)
    plt.xlabel("Record Time")
    plt.ylabel("Calculated Distance (cm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 3. Bidirectional comparison (3 panels)
# ---------------------------------------------------------------------------

def plot_bidirectional_comparison(corrected_dir, save_path, exclude_days=30):
    """3-panel bidirectional baseline comparison with scatter + 30D MA."""
    plt.rcParams.update({'font.size': 14})

    baseline_pairs = [
        ('NW_distance_corrected.csv', 'WN_distance_corrected.csv',
         'North \u2192 West', 'West \u2192 North',
         'North-West vs West-North', (176555, 176575)),
        ('EN_distance_corrected.csv', 'NE_distance_corrected.csv',
         'East \u2192 North', 'North \u2192 East',
         'East-North vs North-East', (164200, 164240)),
        ('WE_distance_corrected.csv', 'EW_distance_corrected.csv',
         'West \u2192 East', 'East \u2192 West',
         'West-East vs East-West', (326000, 326060)),
    ]

    for file1, file2, label1, label2, title, ylim in baseline_pairs:
        df1 = pd.read_csv(os.path.join(corrected_dir, file1))
        df1['Record Time'] = pd.to_datetime(df1['Record Time'], errors='coerce')
        df2 = pd.read_csv(os.path.join(corrected_dir, file2))
        df2['Record Time'] = pd.to_datetime(df2['Record Time'], errors='coerce')

        t0 = min(df1['Record Time'].min(), df2['Record Time'].min())
        cutoff = t0 + pd.Timedelta(days=exclude_days)

        df1p = df1[df1['Record Time'] >= cutoff].copy()
        df2p = df2[df2['Record Time'] >= cutoff].copy()

        plt.figure(figsize=(14, 6))
        plt.plot(df1p['Record Time'], df1p['Distance (cm)'], '.',
                 color='salmon', alpha=0.5, label=label1)
        plt.plot(df2p['Record Time'], df2p['Distance (cm)'], '.',
                 color='lightblue', alpha=0.5, label=label2)

        plt.plot(df1p['Record Time'], df1p['Moving Average (cm)'],
                 color='red', linewidth=2, label=f'{label1} 30D MA')
        plt.plot(df2p['Record Time'], df2p['Moving Average (cm)'],
                 color='blue', linewidth=2, label=f'{label2} 30D MA')

        plt.xlabel("Record Time")
        plt.ylabel("Distance (cm)")
        plt.title(f"{title} (excluding first {exclude_days} days; "
                  f"cutoff = {cutoff.date()})")
        plt.ylim(ylim)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        fname = title.replace(' ', '_').replace('-', '_') + ".png"
        plt.savefig(os.path.join(os.path.dirname(save_path), fname), dpi=150)
        plt.close()


# ---------------------------------------------------------------------------
# 4. 10-day binned STD (all 6 baselines)
# ---------------------------------------------------------------------------

def plot_10day_std(corrected_dir, save_path):
    """10-day binned standard deviation for all 6 baselines."""
    dfs = _load_corrected(corrected_dir)
    global_start, global_end = _global_time_range(dfs)

    BIN = "10D"
    MIN_POINTS = 2

    std_series = {}
    for name, df in dfs.items():
        s = df.set_index("Record Time")["Distance (cm)"].dropna()
        g = s.resample(BIN, origin=global_start)
        std = g.std()
        cnt = g.count()
        std[cnt < MIN_POINTS] = pd.NA
        std_series[name] = std

    std_df = pd.DataFrame(std_series).sort_index()

    plt.figure(figsize=(14, 6))
    for name in std_df.columns:
        plt.plot(std_df.index, std_df[name], linewidth=2, label=name)

    plt.ylabel("10-day STD of Distance [cm]")
    plt.xlabel("Time")
    plt.title("10-day binned standard deviation (scatter) for 6 baselines")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3)
    plt.xlim(global_start, global_end)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 5. 10-day binned STD normalized by baseline length (cm/km)
# ---------------------------------------------------------------------------

def plot_10day_std_normalized(corrected_dir, save_path):
    """10-day binned STD normalized by baseline length, with paired colors."""
    length_km = {
        "NW": 1.765, "WN": 1.765,
        "NE": 1.642, "EN": 1.642,
        "WE": 3.26,  "EW": 3.26,
    }
    pairs = [
        ("NW", "WN", "North\u2013West", "lightcoral", "darkred"),
        ("NE", "EN", "North\u2013East", "lightblue", "darkblue"),
        ("WE", "EW", "West\u2013East", "lightgreen", "darkgreen"),
    ]

    dfs = _load_corrected(corrected_dir)
    global_start, global_end = _global_time_range(dfs)

    BIN = "10D"
    MIN_POINTS = 2

    def ten_day_std_per_km(df, L_km):
        s = df.set_index("Record Time")["Distance (cm)"].dropna()
        g = s.resample(BIN, origin=global_start)
        std = g.std()
        cnt = g.count()
        std[cnt < MIN_POINTS] = pd.NA
        return std / L_km

    std_series = {}
    for name in length_km:
        std_series[name] = ten_day_std_per_km(dfs[name], length_km[name])

    plt.figure(figsize=(14, 6))
    for a, b, pair_label, light_c, dark_c in pairs:
        plt.plot(std_series[a].index, std_series[a].values,
                 linewidth=2, color=dark_c, label=f"{a} ({pair_label})")
        plt.plot(std_series[b].index, std_series[b].values,
                 linewidth=2, color=light_c, label=f"{b} ({pair_label})")

    plt.ylabel("10-day STD (cm/km) of Distance")
    plt.xlabel("Time")
    plt.title("10-day binned standard deviation (normalized by baseline length)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.xlim(global_start, global_end)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 6. 30-day rolling STD normalized (cm/km)
# ---------------------------------------------------------------------------

def plot_rolling_std_normalized(corrected_dir, save_path, window_days=30,
                                min_pts=20):
    """30-day rolling STD normalized by baseline length."""
    length_km = {
        "NW": 1.765, "WN": 1.765,
        "NE": 1.642, "EN": 1.642,
        "WE": 3.26,  "EW": 3.26,
    }
    pairs = [
        ("NW", "WN", "North\u2013West", "lightcoral", "darkred"),
        ("NE", "EN", "North\u2013East", "lightblue", "darkblue"),
        ("WE", "EW", "West\u2013East", "lightgreen", "darkgreen"),
    ]

    dfs = _load_corrected(corrected_dir)
    global_start, global_end = _global_time_range(dfs)

    def rolling_std_per_km(df, L_km):
        s = df.set_index("Record Time")["Distance (cm)"].dropna().sort_index()
        s = s.groupby(level=0).mean()  # collapse duplicates
        std_cm = s.rolling(f"{window_days}D", min_periods=min_pts).std()
        return std_cm / L_km

    std_series = {}
    for name in length_km:
        std_series[name] = rolling_std_per_km(dfs[name], length_km[name])

    plt.figure(figsize=(14, 6))
    for a, b, pair_label, light_c, dark_c in pairs:
        plt.plot(std_series[a].index, std_series[a].values,
                 linewidth=2, color=dark_c, label=f"{a} ({pair_label})")
        plt.plot(std_series[b].index, std_series[b].values,
                 linewidth=2, color=light_c, label=f"{b} ({pair_label})")

    plt.ylabel("Normalized STD (cm/km)")
    plt.xlabel("Time")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.xlim(global_start, global_end)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 7. 3-baseline rolling STD (single direction per pair)
# ---------------------------------------------------------------------------

def plot_rolling_std_3baselines(corrected_dir, save_path, window_days=30,
                                min_pts=20):
    """30-day rolling STD for 3 single baselines: WN, NE, WE."""
    targets = [
        ("WN", "green", "WN (North\u2013West)"),
        ("NE", "blue",  "NE (North\u2013East)"),
        ("WE", "red",   "WE (West\u2013East)"),
    ]
    length_km = {"WN": 1.765, "NE": 1.642, "WE": 3.26}

    dfs = _load_corrected(corrected_dir, names=[t[0] for t in targets])
    global_start, global_end = _global_time_range(dfs)

    def rolling_std_per_km(df, L_km):
        s = df.set_index("Record Time")["Distance (cm)"].dropna().sort_index()
        s = s.groupby(level=0).mean()
        std_cm = s.rolling(f"{window_days}D", min_periods=min_pts).std()
        return std_cm / L_km

    plt.figure(figsize=(14, 6))
    for name, color, label in targets:
        std = rolling_std_per_km(dfs[name], length_km[name])
        plt.plot(std.index, std.values, linewidth=2, color=color, label=label)

    plt.ylabel("Normalized STD (cm/km)")
    plt.xlabel("Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(global_start, global_end)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 8. 10-day STD subplots (3x1, unnormalized)
# ---------------------------------------------------------------------------

def plot_10day_std_subplots(corrected_dir, save_path):
    """3x1 subplots of 10-day binned STD for paired baselines."""
    dfs = _load_corrected(corrected_dir)
    global_start, global_end = _global_time_range(dfs)

    BIN = "10D"
    MIN_POINTS = 2

    pairs = [
        ("NW", "WN", "North \u2194 West (NW & WN)"),
        ("NE", "EN", "North \u2194 East (NE & EN)"),
        ("WE", "EW", "West \u2194 East (WE & EW)"),
    ]

    def binned_std(df):
        s = df.set_index("Record Time")["Distance (cm)"].dropna()
        g = s.resample(BIN, origin=global_start)
        std = g.std()
        cnt = g.count()
        std[cnt < MIN_POINTS] = pd.NA
        return std

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, (a, b, title) in zip(axes, pairs):
        std_a = binned_std(dfs[a])
        std_b = binned_std(dfs[b])
        ax.plot(std_a.index, std_a.values, linewidth=2, label=a)
        ax.plot(std_b.index, std_b.values, linewidth=2, label=b)
        ax.set_ylabel("10-day STD (cm)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(global_start, global_end)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 9. 10-day STD subplots normalized by baseline length (3x1)
# ---------------------------------------------------------------------------

def plot_10day_std_normalized_subplots(corrected_dir, save_path):
    """3x1 subplots of 10-day binned STD normalized by baseline length."""
    dfs = _load_corrected(corrected_dir)
    global_start, global_end = _global_time_range(dfs)

    BIN = "10D"
    MIN_POINTS = 2

    pairs = [
        ("NW", "WN", "North \u2194 West (NW & WN)", 1.765),
        ("NE", "EN", "North \u2194 East (NE & EN)", 1.642),
        ("WE", "EW", "West \u2194 East (WE & EW)", 3.26),
    ]

    def binned_std_per_km(df, L_km):
        s = df.set_index("Record Time")["Distance (cm)"].dropna()
        g = s.resample(BIN, origin=global_start)
        std = g.std()
        cnt = g.count()
        std[cnt < MIN_POINTS] = pd.NA
        return std / L_km

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, (a, b, title, L) in zip(axes, pairs):
        std_a = binned_std_per_km(dfs[a], L)
        std_b = binned_std_per_km(dfs[b], L)
        ax.plot(std_a.index, std_a.values, linewidth=2, label=a)
        ax.plot(std_b.index, std_b.values, linewidth=2, label=b)
        ax.set_ylabel("10-day STD (cm/km)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(global_start, global_end)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Master function: generate all figures
# ---------------------------------------------------------------------------

def generate_all_figures(baselines, corrected_dir, figures_dir):
    """
    Generate all figures and save to figures_dir.

    Parameters
    ----------
    baselines : dict
        Maps baseline name (e.g. 'R2504_2502') to its DataFrame.
    corrected_dir : str
        Directory containing *_corrected.csv files.
    figures_dir : str
        Directory to save figure PNGs.
    """
    os.makedirs(figures_dir, exist_ok=True)

    # --- Individual distance timeseries ---
    timeseries_config = [
        ('R2504_2503', 'North -> West Distance',
         (176550, 176570), 'NW_distance.png'),
        ('R2504_2502', 'North -> East Distance',
         (164200, 164236), 'NE_distance.png'),
        ('R2503_2502', 'West -> East Distance',
         (326000, 326060), 'WE_distance.png'),
        ('R2503_2504', 'West -> North Distance',
         (176550, 176575), 'WN_distance.png'),
        ('R2502_2503', 'East -> West Distance',
         (326000, 326060), 'EW_distance.png'),
        ('R2502_2504', 'East -> North Distance',
         (164200, 164236), 'EN_distance.png'),
    ]

    for key, title, ylim, fname in timeseries_config:
        print(f"    {fname}")
        plot_distance_timeseries(
            baselines[key], title, ylim,
            os.path.join(figures_dir, fname),
        )

    # --- Binned error-bar plots ---
    errorbar_config = [
        ('R2504_2503', 'North-West Distance (30-Day Binned) with Error Bars',
         'NW_binned.png'),
        ('R2504_2502', 'North-East Distance (30-Day Binned) with Error Bars',
         'NE_binned.png'),
        ('R2503_2502', 'West-East Distance (30-Day Binned) with Error Bars',
         'WE_binned.png'),
        ('R2503_2504', 'West-North Distance (30-Day Binned) with Error Bars',
         'WN_binned.png'),
        ('R2502_2503', 'East-West Distance (30-Day Binned) with Error Bars',
         'EW_binned.png'),
        ('R2502_2504', 'East-North Distance (30-Day Binned) with Error Bars',
         'EN_binned.png'),
    ]

    for key, title, fname in errorbar_config:
        print(f"    {fname}")
        plot_binned_errorbar(
            baselines[key], title,
            os.path.join(figures_dir, fname),
        )

    # --- Bidirectional comparison ---
    print("    bidirectional_comparison.png")
    plot_bidirectional_comparison(
        corrected_dir,
        os.path.join(figures_dir, "bidirectional_comparison.png"),
    )

    # --- 10-day STD ---
    print("    10day_std_all.png")
    plot_10day_std(corrected_dir, os.path.join(figures_dir, "10day_std_all.png"))

    # --- 10-day STD normalized ---
    print("    10day_std_normalized.png")
    plot_10day_std_normalized(
        corrected_dir, os.path.join(figures_dir, "10day_std_normalized.png")
    )

    # --- 30-day rolling STD normalized ---
    print("    rolling_std_normalized.png")
    plot_rolling_std_normalized(
        corrected_dir, os.path.join(figures_dir, "rolling_std_normalized.png")
    )

    # --- 3-baseline rolling STD ---
    print("    rolling_std_3baselines.png")
    plot_rolling_std_3baselines(
        corrected_dir, os.path.join(figures_dir, "rolling_std_3baselines.png")
    )

    # --- 10-day STD subplots ---
    print("    10day_std_subplots.png")
    plot_10day_std_subplots(
        corrected_dir, os.path.join(figures_dir, "10day_std_subplots.png")
    )

    # --- 10-day STD normalized subplots ---
    print("    10day_std_normalized_subplots.png")
    plot_10day_std_normalized_subplots(
        corrected_dir,
        os.path.join(figures_dir, "10day_std_normalized_subplots.png"),
    )
