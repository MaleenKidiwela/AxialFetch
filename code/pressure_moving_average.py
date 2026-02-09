"""
Pressure moving-average utilities for FETCH Range Calculation analysis.
"""

from __future__ import annotations

import pandas as pd

PSI_TO_KPA = 6.894757
COL_KPA = "Corrected Pressure (kPa)"


def prep_ma15d(df: pd.DataFrame, column: str = COL_KPA) -> pd.DataFrame:
    df = df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df.sort_values("DateTime", inplace=True)
    mu = df[column].mean(skipna=True)
    df["pressure_demeaned"] = df[column] - mu
    df["ma15d"] = (
        df.set_index("DateTime")["pressure_demeaned"]
        .rolling("15D", min_periods=1)
        .mean()
        .values
    )
    return df


def ensure_demeaned(df: pd.DataFrame, column: str = COL_KPA) -> pd.DataFrame:
    df = df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df.sort_values("DateTime", inplace=True)
    if "pressure_demeaned" not in df.columns:
        mu = df[column].mean(skipna=True)
        df["pressure_demeaned"] = df[column] - mu
    df["ma15d"] = (
        df.set_index("DateTime")["pressure_demeaned"]
        .rolling("15D", min_periods=1)
        .mean()
        .values
    )
    return df


def to_ma15d(df: pd.DataFrame, tcol: str, vcol: str, unit: str = "kpa") -> pd.Series:
    x = df[[tcol, vcol]].copy()
    x[tcol] = pd.to_datetime(x[tcol], errors="coerce")
    x = x.dropna(subset=[tcol]).sort_values(tcol)
    v = x[vcol].astype(float)
    if unit.lower() == "psi":
        v = v * PSI_TO_KPA
    s = pd.Series(v.values, index=x[tcol])
    return s.rolling("15D", center=True, min_periods=1).mean()


def rebase_to_window(ma: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    w = ma.loc[start:end]
    return ma - w.mean()
