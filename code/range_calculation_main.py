"""
Main entry point for the FETCH Range Calculation workflow.

This script is intended to be imported and executed from a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from .range_calculation_workflow import (
    build_harmonic_means,
    build_sound_speed_tables,
    extract_sensor_data,
    load_fetch_data,
)
from .range_salinity import (
    GapMeanShiftConfig,
    SalinityCalibrationConfig,
    bottle_residuals,
    prepare_salinity_series,
)
from .tidal_correction import apply_tidal_correction, load_tidal_predictions, optimize_tidal_correction
from .velocity_interpolation import interpolate_velocity


@dataclass(frozen=True)
class FetchRangeInputs:
    filepaths: List[str]
    identifiers: List[str]


def run_fetch_range_workflow(
    inputs: FetchRangeInputs,
    salinity_config: SalinityCalibrationConfig,
    gap_shift_config: GapMeanShiftConfig | None,
    tidal_prediction_paths: List[str],
) -> Dict[str, object]:
    df_dict = load_fetch_data(inputs.filepaths)
    data_extracted = extract_sensor_data(df_dict, inputs.identifiers)

    result_dfs = build_sound_speed_tables(data_extracted, inputs.identifiers)
    pairs = [("2502", "2503"), ("2502", "2504"), ("2503", "2504")]
    harmonic_mean_dfs = build_harmonic_means(result_dfs, pairs)

    salinity_df = prepare_salinity_series(salinity_config, gap_shift_config)
    salinity_residuals = bottle_residuals(
        salinity_df,
        salinity_config.bottle_times,
        salinity_config.bottle_sal,
    )

    tidal_df = load_tidal_predictions(tidal_prediction_paths)

    outputs = {
        "df_dict": df_dict,
        "data_extracted": data_extracted,
        "result_dfs": result_dfs,
        "harmonic_mean_dfs": harmonic_mean_dfs,
        "salinity_df": salinity_df,
        "salinity_residuals": salinity_residuals,
        "tidal_df": tidal_df,
    }

    return outputs


def apply_tidal_and_velocity(
    combined_df: pd.DataFrame,
    tidal_df: pd.DataFrame,
    result_df: pd.DataFrame,
    amplitude: float | None = None,
    rho: float | None = None,
) -> pd.DataFrame:
    if amplitude is None or rho is None:
        amplitude, rho = optimize_tidal_correction(combined_df, tidal_df)

    corrected = apply_tidal_correction(combined_df, tidal_df, amplitude, rho)
    corrected["interp_v"] = interpolate_velocity(corrected["Record Time"], result_df)
    return corrected
