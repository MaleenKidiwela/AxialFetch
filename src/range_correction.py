"""
Distance CSV generation and perturbation correction.

Computes 30-day moving averages, saves baseline distance CSVs,
and applies tilt-perturbation corrections via merge_asof.
"""
import os
import pandas as pd
from .config import BASELINE_CSV_MAP, PERTURBATION_CORRECTIONS


def compute_moving_average(df, dist_col='Calculated Distance (m)',
                           window='30D'):
    """
    Compute 30-day rolling mean on a distance column and return a
    DataFrame with columns suitable for CSV export.

    Returns
    -------
    DataFrame with 'Record Time', 'Distance (cm)', 'Moving Average (cm)'.
    """
    df = df.copy()
    df['Record Time'] = pd.to_datetime(df['Record Time'])
    df = df.sort_values(by='Record Time').set_index('Record Time')
    df['Moving Average'] = df[dist_col].rolling(window).mean() * 100
    df = df.reset_index()
    return pd.DataFrame({
        'Record Time': df['Record Time'],
        'Distance (cm)': df[dist_col] * 100,
        'Moving Average (cm)': df['Moving Average'],
    })


def save_distance_csvs(baselines, output_dir):
    """
    Save all 6 baseline distance CSVs to output_dir.

    Parameters
    ----------
    baselines : dict
        Maps baseline name (e.g. 'R2504_2502') to its DataFrame.
    output_dir : str
        Directory to save CSVs.
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, (csv_name, _label) in BASELINE_CSV_MAP.items():
        scatter = compute_moving_average(baselines[name])
        filepath = os.path.join(output_dir, csv_name)
        scatter.to_csv(filepath, index=False)
        print(f"  Saved {filepath}")


def apply_perturbation_correction(baseline_perturb, input_dir, output_dir):
    """
    Apply perturbation correction to all distance CSVs.

    For each perturbation column, loads the corresponding distance CSVs,
    does merge_asof with baseline_perturb, adds perturbation*100 to Distance,
    recomputes 30D rolling mean, and saves as *_corrected.csv.
    """
    if not isinstance(baseline_perturb.index, pd.DatetimeIndex):
        baseline_perturb.index = pd.to_datetime(baseline_perturb.index)

    os.makedirs(output_dir, exist_ok=True)

    for perturb_col, file_list in PERTURBATION_CORRECTIONS.items():
        for filename in file_list:
            df = pd.read_csv(os.path.join(input_dir, filename))
            df['Record Time'] = pd.to_datetime(df['Record Time'])

            perturb_df = baseline_perturb[[perturb_col]].reset_index()
            perturb_df.columns = ['Record Time', 'perturbation']

            df_sorted = df.sort_values('Record Time')
            perturb_sorted = perturb_df.sort_values('Record Time')

            df_merged = pd.merge_asof(
                df_sorted, perturb_sorted,
                on='Record Time',
                direction='nearest',
                tolerance=pd.Timedelta('1H'),
            )

            df_merged['perturbation'] = df_merged['perturbation'].fillna(0)
            df_merged['Distance (cm)'] += df_merged['perturbation'] * 100

            df_merged = df_merged.set_index('Record Time').sort_index()
            df_merged['Moving Average (cm)'] = (
                df_merged['Distance (cm)'].rolling('30D').mean()
            )

            df_merged = df_merged[
                ['Distance (cm)', 'Moving Average (cm)']
            ].reset_index()

            out_name = filename.replace('.csv', '_corrected.csv')
            out_path = os.path.join(output_dir, out_name)
            df_merged.to_csv(out_path, index=False)
            print(f"  Corrected: {out_path}")
