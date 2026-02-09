"""
Tidal correction workflow for FETCH Range Calculation analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .data_processing import parse_to_dataframe
from .optimization import optimize_tidal_influence


@dataclass(frozen=True)
class TidalPredictionFiles:
    paths: Iterable[str]


def load_tidal_predictions(paths: Iterable[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            frames.append(parse_to_dataframe(handle.read()))

    tidal_df = pd.concat(frames, ignore_index=True)
    tidal_df.set_index("DateTime", inplace=True)
    return tidal_df


def optimize_tidal_correction(
    pressure_df: pd.DataFrame,
    tidal_df: pd.DataFrame,
    initial_amplitude: float = 1.0,
    initial_rho: float = 1025.0,
) -> Tuple[float, float]:
    result = least_squares(
        optimize_tidal_influence,
        x0=[initial_amplitude, initial_rho],
        args=(tidal_df, pressure_df),
    )
    amplitude, rho = result.x
    return float(amplitude), float(rho)


def apply_tidal_correction(
    combined_df: pd.DataFrame,
    tidal_df: pd.DataFrame,
    amplitude: float,
    rho: float,
) -> pd.DataFrame:
    df = combined_df.copy()
    df["DateTime"] = pd.to_datetime(df["Record Time"])
    df.set_index("DateTime", inplace=True)

    adjusted_tidal = amplitude * tidal_df["Value"]
    tidal_df = tidal_df.copy()
    tidal_df["Adjusted Value"] = adjusted_tidal

    interpolated = tidal_df.reindex(df.index, method="nearest")["Adjusted Value"]
    df["Tidal Influence (kPa)"] = (rho * 9.81 * interpolated) * 0.001
    df["Corrected Pressure (kPa)"] = df["Pressure (kPa)"] - df["Tidal Influence (kPa)"]

    df.reset_index(inplace=True)
    return df
