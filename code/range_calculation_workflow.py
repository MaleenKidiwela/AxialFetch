"""
Structured workflow for the FETCH Range Calculation notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .data_processing import create_nested_dictionary, ensure_datetime, remove_outliers
from .utils import calculate_harmonic_mean, compute_harmonic_mean


@dataclass(frozen=True)
class FetchFileGroup:
    filepaths: Iterable[str]


def load_fetch_data(filepaths: Iterable[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    return create_nested_dictionary(list(filepaths))


def extract_sensor_data(
    df_dict: Dict[str, Dict[str, pd.DataFrame]],
    identifiers: Iterable[str],
    sensor_keys: Iterable[str] = ("TMP", "DQZ", "INC", "SSP", "BSL"),
) -> Dict[str, Dict[str, pd.DataFrame]]:
    data_extracted: Dict[str, Dict[str, pd.DataFrame]] = {}
    for identifier in identifiers:
        data_extracted[identifier] = {}
        for key in sensor_keys:
            data_extracted[identifier][key] = df_dict.get(identifier, {}).get(key)
    return data_extracted


def build_sound_speed_tables(
    data_extracted: Dict[str, Dict[str, pd.DataFrame]],
    identifiers: Iterable[str],
    column: str = "SoundSpeed (m/s)",
) -> Dict[str, pd.DataFrame]:
    result_dfs: Dict[str, pd.DataFrame] = {}
    for identifier in identifiers:
        ssp_df = data_extracted[identifier].get("SSP")
        if ssp_df is None:
            continue
        ensure_datetime(ssp_df, "Record Time")
        ssp_df[column] = pd.to_numeric(ssp_df[column], errors="coerce")
        result_dfs[identifier] = remove_outliers(ssp_df, column)
    return result_dfs


def build_harmonic_means(
    result_dfs: Dict[str, pd.DataFrame],
    pairs: Iterable[Tuple[str, str]],
    column: str = "SoundSpeed (m/s)",
) -> Dict[str, pd.DataFrame]:
    harmonic_mean_dfs: Dict[str, pd.DataFrame] = {}
    for inst_a, inst_b in pairs:
        df1 = result_dfs[inst_a].set_index("Record Time")
        df2 = result_dfs[inst_b].set_index("Record Time")
        harmonic_series = calculate_harmonic_mean(df1[column], df2[column])
        harmonic_mean_dfs[f"{inst_a}_{inst_b}"] = pd.DataFrame(
            {"Record Time": harmonic_series.index, "HMean": harmonic_series.values}
        )
    return harmonic_mean_dfs


def build_harmonic_mean_from_combined(
    combined_df4: pd.DataFrame,
    combined_df3: pd.DataFrame,
    combined_df2: pd.DataFrame,
    column: str = "velocity",
) -> Dict[str, pd.DataFrame]:
    for df in (combined_df4, combined_df3, combined_df2):
        if "Record Time" in df.columns:
            df["Record Time"] = pd.to_datetime(df["Record Time"])
            df.set_index("Record Time", inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    index_2504_2503, harmonic_mean_2504_2503 = compute_harmonic_mean(
        combined_df4, combined_df3, column=column
    )
    index_2504_2502, harmonic_mean_2504_2502 = compute_harmonic_mean(
        combined_df4, combined_df2, column=column
    )
    index_2503_2502, harmonic_mean_2503_2502 = compute_harmonic_mean(
        combined_df3, combined_df2, column=column
    )

    harmonic_means = {
        "2502_2503": (index_2503_2502, harmonic_mean_2503_2502),
        "2502_2504": (index_2504_2502, harmonic_mean_2504_2502),
        "2503_2504": (index_2504_2503, harmonic_mean_2504_2503),
    }

    harmonic_df_dict: Dict[str, pd.DataFrame] = {}
    for key, (index, values) in harmonic_means.items():
        harmonic_df_dict[key] = pd.DataFrame({"Record Time": index, "HMean": values})
    return harmonic_df_dict


def build_pressure_temperature_data(
    data_extracted: Dict[str, Dict[str, pd.DataFrame]],
    identifier: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pressure_df = pd.DataFrame(
        {
            "Record Time": pd.to_datetime(data_extracted[identifier]["DQZ"]["Record Time"]),
            "Pressure (kPa)": data_extracted[identifier]["DQZ"]["Pressure (kPa)"],
        }
    ).set_index("Record Time")

    temperature_df = pd.DataFrame(
        {
            "Record Time": pd.to_datetime(data_extracted[identifier]["TMP"]["Record Time"]),
            "Temperature Deg C": data_extracted[identifier]["TMP"]["Temperature Deg C"],
        }
    ).set_index("Record Time")

    return pressure_df, temperature_df
