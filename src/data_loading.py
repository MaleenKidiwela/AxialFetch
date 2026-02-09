"""
Data Loading and Parsing Module for FETCH Range Calculation

Handles raw CSV parsing, multi-header file processing, and data extraction
for temperature, pressure, inclinometer, baseline, and sound speed data.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path


def process_data(filepath: str) -> Dict[str, pd.DataFrame]:
    """
    Load and process CSV files by separating header rows and data rows.

    The FETCH data files have a special format with 14 header rows that define
    the column structure for different data codes (TMP, DQZ, INC, BSL, SSP, etc.).

    Args:
        filepath: Path to the CSV file to process

    Returns:
        Dictionary containing DataFrames for each code found in the header
    """
    data_df = pd.read_csv(filepath)

    NUM_HEADER_ROWS = 14
    DATA_START_MARKER = "# Data"

    header_rows = data_df.iloc[:NUM_HEADER_ROWS]
    data_rows = data_df[data_df["Code"] != DATA_START_MARKER]

    dataframes_dict = {}

    for _, row in header_rows.iterrows():
        code, *headers = row.dropna().values

        rename_dict = {f"V{i+1}": header for i, header in enumerate(headers)}
        temp_df = data_rows.copy()
        temp_df.rename(columns=rename_dict, inplace=True)

        corresponding_data_rows = temp_df[temp_df["Code"] == code]

        if not corresponding_data_rows.empty:
            corresponding_data_rows.columns = corresponding_data_rows.iloc[0].values
            corresponding_data_rows = corresponding_data_rows.drop(
                corresponding_data_rows.index[0]
            )
            dataframes_dict[code] = corresponding_data_rows

    return dataframes_dict


def create_nested_dictionary(filepaths: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Process multiple files and return a dictionary containing dataframes for each file.

    Args:
        filepaths: List of file paths to process

    Returns:
        Dictionary with station identifiers as keys and processed data dictionaries as values.
        The keys are extracted from the last underscore-separated segment of the filename.
    """
    all_dfs = {}

    for filepath in filepaths:
        # Extract identifier from filename (e.g., "2502" from "...2502_.csv")
        identifier = Path(filepath).stem.split("_")[-2]
        all_dfs[identifier] = process_data(filepath)

    return all_dfs


def ensure_datetime(df: pd.DataFrame, column_name: str) -> None:
    """
    Check and convert a column to datetime format if not already.

    Args:
        df: DataFrame to modify in place
        column_name: Name of the column to convert
    """
    if not pd.api.types.is_datetime64_any_dtype(df[column_name]):
        df[column_name] = pd.to_datetime(df[column_name])


def remove_outliers(
    df: pd.DataFrame,
    column_name: str,
    quantiles: Tuple[float, float] = (0.25, 0.75),
    iqr_factor: float = 1.5,
) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame based on the Interquartile Range (IQR).

    Args:
        df: Input DataFrame
        column_name: Name of the column to analyze for outliers
        quantiles: Tuple of (lower, upper) quantiles for IQR calculation
        iqr_factor: Multiplier for IQR to define outlier bounds

    Returns:
        DataFrame with outliers removed
    """
    Q1 = df[column_name].quantile(quantiles[0])
    Q3 = df[column_name].quantile(quantiles[1])
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR

    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]


def extract_sensor_data(
    df_dict: Dict[str, Dict[str, pd.DataFrame]],
    station: str,
    sensor_type: str,
) -> Optional[pd.DataFrame]:
    """
    Extract sensor data for a specific station and sensor type.

    Args:
        df_dict: Nested dictionary from create_nested_dictionary()
        station: Station identifier (e.g., "2502", "2503", "2504")
        sensor_type: Type of sensor data ("TMP", "DQZ", "INC", "BSL", "SSP")

    Returns:
        DataFrame with the requested sensor data, or None if not found
    """
    if station not in df_dict:
        return None
    if sensor_type not in df_dict[station]:
        return None
    return df_dict[station][sensor_type].copy()


def extract_temperature_data(
    df_dict: Dict[str, Dict[str, pd.DataFrame]],
    station: str,
    apply_correction: float = 0.0,
) -> pd.DataFrame:
    """
    Extract temperature data for a station with optional correction.

    Args:
        df_dict: Nested dictionary from create_nested_dictionary()
        station: Station identifier
        apply_correction: Temperature correction to add (e.g., 0.351 for 2504)

    Returns:
        DataFrame with Record Time and Temperature Deg C columns
    """
    tmp_df = extract_sensor_data(df_dict, station, "TMP")
    if tmp_df is None:
        raise ValueError(f"No TMP data found for station {station}")

    tmp_df["Temperature Deg C"] = pd.to_numeric(
        tmp_df["Temperature Deg C"], errors="coerce"
    )
    tmp_df.dropna(subset=["Temperature Deg C"], inplace=True)
    tmp_df = remove_outliers(tmp_df, "Temperature Deg C")

    result = pd.DataFrame({
        "Record Time": pd.to_datetime(tmp_df["Record Time"]),
        "Temperature Deg C": tmp_df["Temperature Deg C"].values + apply_correction,
    })

    return result.set_index("Record Time")


def extract_pressure_data(
    df_dict: Dict[str, Dict[str, pd.DataFrame]],
    station: str,
) -> pd.DataFrame:
    """
    Extract pressure data for a station.

    Args:
        df_dict: Nested dictionary from create_nested_dictionary()
        station: Station identifier

    Returns:
        DataFrame with Record Time index and Pressure (kPa) column
    """
    dqz_df = extract_sensor_data(df_dict, station, "DQZ")
    if dqz_df is None:
        raise ValueError(f"No DQZ data found for station {station}")

    dqz_df["Pressure (kPa)"] = pd.to_numeric(
        dqz_df["Pressure (kPa)"], errors="coerce"
    )
    dqz_df.dropna(subset=["Pressure (kPa)"], inplace=True)
    dqz_df = remove_outliers(dqz_df, "Pressure (kPa)")

    result = pd.DataFrame({
        "Record Time": pd.to_datetime(dqz_df["Record Time"]),
        "Pressure (kPa)": dqz_df["Pressure (kPa)"].values,
    })

    return result.set_index("Record Time")


def extract_inclinometer_data(
    df_dict: Dict[str, Dict[str, pd.DataFrame]],
    station: str,
    apply_filters: bool = True,
) -> pd.DataFrame:
    """
    Extract inclinometer (pitch/roll) data for a station.

    Applies station-specific quality filters by default:
    - Station 2502: Keep rows where -0.35 < Roll < 0
    - Station 2503: Keep rows where -3.5 < Pitch < -3

    Args:
        df_dict: Nested dictionary from create_nested_dictionary()
        station: Station identifier
        apply_filters: Whether to apply station-specific quality filters

    Returns:
        DataFrame with Record Time, Pitch, and Roll columns
    """
    inc_df = extract_sensor_data(df_dict, station, "INC")
    if inc_df is None:
        raise ValueError(f"No INC data found for station {station}")

    inc_df["Pitch (deg)"] = pd.to_numeric(inc_df["Pitch (deg)"], errors="coerce")
    inc_df["Roll (deg)"] = pd.to_numeric(inc_df["Roll (deg)"], errors="coerce")
    inc_df.dropna(subset=["Pitch (deg)", "Roll (deg)"], inplace=True)

    result = pd.DataFrame({
        "Record Time": pd.to_datetime(inc_df["Record Time"]),
        "Pitch": inc_df["Pitch (deg)"].values,
        "Roll": inc_df["Roll (deg)"].values,
    })

    # Apply station-specific quality filters (from notebook analysis)
    if apply_filters:
        if station == "2502":
            # Filter Roll to remove outliers
            result = result[(result["Roll"] < 0) & (result["Roll"] > -0.35)]
        elif station == "2503":
            # Filter Pitch to remove outliers
            result = result[(result["Pitch"] < -3) & (result["Pitch"] > -3.5)]

    return result


def extract_baseline_data(
    df_dict: Dict[str, Dict[str, pd.DataFrame]],
    source_station: str,
    target_station: str,
) -> pd.DataFrame:
    """
    Extract baseline (acoustic range) data between two stations.

    Args:
        df_dict: Nested dictionary from create_nested_dictionary()
        source_station: Transmitting station identifier
        target_station: Receiving station identifier (RangeAddress)

    Returns:
        DataFrame with Record Time, Range(ms), TAT(ms) columns
    """
    bsl_df = extract_sensor_data(df_dict, source_station, "BSL")
    if bsl_df is None:
        raise ValueError(f"No BSL data found for station {source_station}")

    bsl_df["RangeAddress"] = bsl_df["RangeAddress"].astype(int).astype(str)
    bsl_df["Range(ms)"] = pd.to_numeric(bsl_df["Range(ms)"], errors="coerce")
    bsl_df["TAT(ms)"] = pd.to_numeric(bsl_df["TAT(ms)"], errors="coerce")

    # Filter for the target station
    filtered_df = bsl_df[bsl_df["RangeAddress"] == target_station].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No baseline data from {source_station} to {target_station}"
        )

    result = pd.DataFrame({
        "Record Time": pd.to_datetime(filtered_df["Record Time"]),
        "Range(ms)": filtered_df["Range(ms)"].values,
        "TAT(ms)": filtered_df["TAT(ms)"].values,
    })

    return result


def extract_sound_speed_data(
    df_dict: Dict[str, Dict[str, pd.DataFrame]],
    station: str,
) -> pd.DataFrame:
    """
    Extract sound speed data for a station.

    Args:
        df_dict: Nested dictionary from create_nested_dictionary()
        station: Station identifier

    Returns:
        DataFrame with Record Time and SoundSpeed columns
    """
    ssp_df = extract_sensor_data(df_dict, station, "SSP")
    if ssp_df is None:
        raise ValueError(f"No SSP data found for station {station}")

    ssp_df["SoundSpeed (m/s)"] = pd.to_numeric(
        ssp_df["SoundSpeed (m/s)"], errors="coerce"
    )
    ssp_df.dropna(subset=["SoundSpeed (m/s)"], inplace=True)
    ssp_df = remove_outliers(ssp_df, "SoundSpeed (m/s)")

    result = pd.DataFrame({
        "Record Time": pd.to_datetime(ssp_df["Record Time"]),
        "SoundSpeed": ssp_df["SoundSpeed (m/s)"].values,
    })

    return result


def set_datetime_index(df: pd.DataFrame) -> None:
    """
    Set datetime index for DataFrames, converting Record Time column to datetime index.

    Args:
        df: DataFrame to modify in place
    """
    if "Record Time" in df.columns:
        df["Record Time"] = pd.to_datetime(df["Record Time"])
        df.set_index("Record Time", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)


def build_combined_dataframe(
    df_dict: Dict[str, Dict[str, pd.DataFrame]],
    station: str,
    salinity_df: pd.DataFrame,
    temp_correction: float = 0.0,
) -> pd.DataFrame:
    """
    Build a combined DataFrame with pressure, temperature, and salinity for a station.

    Args:
        df_dict: Nested dictionary from create_nested_dictionary()
        station: Station identifier
        salinity_df: DataFrame with corrected salinity (time index, corrected_salinity column)
        temp_correction: Temperature correction to apply

    Returns:
        DataFrame with Record Time, Pressure (kPa), Temperature Deg C, and Salinity columns
    """
    # Get pressure data
    pressure_df = extract_pressure_data(df_dict, station)

    # Get temperature data
    temp_df = extract_temperature_data(df_dict, station, apply_correction=temp_correction)

    # Merge pressure and temperature
    combined_df = pressure_df.join(temp_df, how="left")

    # Prepare salinity for reindexing
    salinity_indexed = salinity_df.copy()
    if "time" in salinity_indexed.columns:
        salinity_indexed["time"] = pd.to_datetime(salinity_indexed["time"])
        salinity_indexed = salinity_indexed.set_index("time")

    # Reindex salinity to match pressure timestamps
    aligned_salinity = salinity_indexed.reindex(combined_df.index, method="nearest")

    # Get the salinity column name
    sal_col = "corrected_salinity" if "corrected_salinity" in aligned_salinity.columns else aligned_salinity.columns[0]
    combined_df["Salinity"] = aligned_salinity[sal_col]

    # Reset index
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={"index": "Record Time"}, inplace=True)

    return combined_df
