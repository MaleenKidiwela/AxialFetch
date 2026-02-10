"""
Harmonic mean computation: both SSP-based (from result_dfs) and
TEOS-10 velocity-based (from combined DataFrames).
"""
import pandas as pd
import numpy as np
from scipy.stats import hmean


def calculate_harmonic_mean(series1, series2):
    """Compute element-wise harmonic mean of two Series, dropping NaN rows."""
    combined = pd.concat([series1, series2], axis=1).dropna()
    harmonic_values = hmean(combined, axis=1)
    return pd.Series(harmonic_values, index=combined.index)


def compute_ssp_harmonic_means(result_dfs):
    """
    Compute harmonic mean DataFrames from SSP SoundSpeed data.
    Returns harmonic_mean_dfs dict with keys '2502_2503', '2502_2504', '2503_2504'.
    """
    harmonic_mean_dfs = {}
    pairs = [('2502', '2503'), ('2502', '2504'), ('2503', '2504')]

    for pair in pairs:
        df1 = result_dfs[pair[0]].set_index('Record Time')
        df2 = result_dfs[pair[1]].set_index('Record Time')
        df1 = df1[~df1.index.duplicated(keep='first')]
        df2 = df2[~df2.index.duplicated(keep='first')]

        harmonic_mean = calculate_harmonic_mean(
            df1['SoundSpeed'], df2['SoundSpeed']
        )
        harmonic_mean_df = pd.DataFrame({
            'Record Time': harmonic_mean.index,
            'Harmonic Mean': harmonic_mean.values,
        })
        harmonic_mean_dfs[f'{pair[0]}_{pair[1]}'] = harmonic_mean_df

    # Overrides per notebook logic:
    # 2502_2503 uses same data as 2502_2504 (because 2503 SSP is NaN)
    harmonic_mean_dfs['2502_2503'] = harmonic_mean_dfs['2502_2504']
    # 2503_2504 uses station 2504 SSP directly
    harmonic_mean_dfs['2503_2504'] = result_dfs['2504'][
        ['Record Time', 'SoundSpeed']
    ].copy()

    return harmonic_mean_dfs


def compute_velocity_harmonic_means(combined_df4, combined_df3, combined_df2):
    """
    Compute harmonic means from TEOS-10 velocity columns in the combined DataFrames.
    Returns harmonic_df_dict with keys '2502_2503', '2502_2504', '2503_2504'.
    """
    def set_datetime_index(df):
        df = df.copy()
        if 'Record Time' in df.columns:
            df['Record Time'] = pd.to_datetime(df['Record Time'])
            df.set_index('Record Time', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

    df4 = set_datetime_index(combined_df4)
    df3 = set_datetime_index(combined_df3)
    df2 = set_datetime_index(combined_df2)

    def compute_harmonic_mean_pair(df1, df2, column='velocity'):
        common_index = df1.index.intersection(df2.index)
        aligned_df1 = df1.loc[common_index, [column]].copy()
        aligned_df2 = df2.loc[common_index, [column]].copy()
        aligned_df1.dropna(inplace=True)
        aligned_df2.dropna(inplace=True)
        min_length = min(len(aligned_df1), len(aligned_df2))
        aligned_df1 = aligned_df1.iloc[:min_length]
        aligned_df2 = aligned_df2.iloc[:min_length]
        aligned_df1[column] = aligned_df1[column].clip(lower=0.0001)
        aligned_df2[column] = aligned_df2[column].clip(lower=0.0001)
        values1 = aligned_df1[column].to_numpy()
        values2 = aligned_df2[column].to_numpy()
        harmonic_mean_values = hmean(np.vstack([values1, values2]), axis=0)
        return common_index[:min_length], harmonic_mean_values

    index_2504_2503, hm_2504_2503 = compute_harmonic_mean_pair(df4, df3)
    index_2504_2502, hm_2504_2502 = compute_harmonic_mean_pair(df4, df2)
    index_2503_2502, hm_2503_2502 = compute_harmonic_mean_pair(df3, df2)

    harmonic_means = {
        '2502_2503': (index_2503_2502, hm_2503_2502),
        '2502_2504': (index_2504_2502, hm_2504_2502),
        '2503_2504': (index_2504_2503, hm_2504_2503),
    }

    harmonic_df_dict = {}
    for key, (index, values) in harmonic_means.items():
        df = pd.DataFrame({'Record Time': index, 'HMean': values})
        harmonic_df_dict[key] = df

    return harmonic_df_dict


def build_final_harmonic_mean_dfs(result_dfs, combined_df4, combined_df3, combined_df2):
    """
    Build the final harmonic_mean_dfs dict that matches the notebook output.

    This combines:
    1. SSP-based harmonic means (from Cell 3)
    2. Rename of SoundSpeed -> Harmonic Mean for the 2503_2504 entry (Cell 23)

    Also computes velocity-based harmonic means (harmonic_df_dict) as a side product.
    """
    # SSP-based (Cell 3)
    harmonic_mean_dfs = compute_ssp_harmonic_means(result_dfs)

    # Rename SoundSpeed -> Harmonic Mean in the 2503_2504 entry (Cell 23)
    harmonic_mean_dfs['2503_2504'].rename(
        columns={'SoundSpeed': 'Harmonic Mean'}, inplace=True
    )

    # Also compute velocity-based harmonic means (harmonic_df_dict from Cell 23)
    # This is a separate dict, but not part of harmonic_mean_dfs
    harmonic_df_dict = compute_velocity_harmonic_means(
        combined_df4, combined_df3, combined_df2
    )

    return harmonic_mean_dfs, harmonic_df_dict
