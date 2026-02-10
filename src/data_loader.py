"""
Data loading: parse FETCH instrument CSV files into nested dictionaries of DataFrames.
"""
import pandas as pd
import numpy as np
from .config import FETCH_CSV_FILES, IDENTIFIERS


def process_data(filepath):
    """
    Parse a single FETCH CSV file into a dictionary of DataFrames keyed by
    instrument code (e.g. 'TMP', 'DQZ', 'INC', 'SSP', 'BSL', ...).
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


def create_nested_dictionary(filepaths):
    """
    Process multiple FETCH CSV files and return a dictionary keyed by the
    4-digit station identifier (last 4 digits before the trailing underscore).
    """
    all_dfs = {}
    for filepath in filepaths:
        identifier = filepath.split("_")[-2]
        all_dfs[identifier] = process_data(filepath)
    return all_dfs


def remove_outliers(df, column_name):
    """Remove outliers using the IQR method."""
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]


def load_all_data():
    """
    Main entry point: loads FETCH CSVs and returns:
      - df_dict:        raw nested dictionary {identifier: {code: DataFrame}}
      - data_extracted: cleaned TMP/DQZ/INC data per station
      - result_dfs:     SSP sound-speed DataFrames per station
    """
    df_dict = create_nested_dictionary(FETCH_CSV_FILES)

    # --- Build result_dfs from SSP data ---
    result_dfs = {}
    for identifier in IDENTIFIERS:
        ssp_df = df_dict[identifier]["SSP"].copy()
        ssp_df["SoundSpeed (m/s)"] = pd.to_numeric(
            ssp_df["SoundSpeed (m/s)"], errors="coerce"
        )
        cleaned_df = remove_outliers(ssp_df, "SoundSpeed (m/s)")
        result_df = cleaned_df[["Record Time", "SoundSpeed (m/s)"]].rename(
            columns={"SoundSpeed (m/s)": "SoundSpeed"}
        )
        result_dfs[identifier] = result_df

    for identifier, df in result_dfs.items():
        df["Record Time"] = pd.to_datetime(df["Record Time"])

    # Station 2503 SSP is set to NaN (no reliable velocimeter data)
    result_dfs["2503"]["SoundSpeed"] = np.nan

    # --- Extract and clean TMP / DQZ / INC ---
    data_extracted = {}
    for identifier in IDENTIFIERS:
        data_extracted[identifier] = {
            "TMP": (
                df_dict[identifier]["TMP"]
                if "TMP" in df_dict[identifier]
                else None
            ),
            "DQZ": (
                df_dict[identifier]["DQZ"]
                if "DQZ" in df_dict[identifier]
                else None
            ),
            "INC": (
                df_dict[identifier]["INC"]
                if "INC" in df_dict[identifier]
                else None
            ),
        }

    for identifier in IDENTIFIERS:
        tmp = data_extracted[identifier]["TMP"]
        if tmp is not None:
            tmp["Temperature Deg C"] = pd.to_numeric(
                tmp["Temperature Deg C"], errors="coerce"
            )
            tmp.dropna(subset=["Temperature Deg C"], inplace=True)

        dqz = data_extracted[identifier]["DQZ"]
        if dqz is not None:
            dqz["Pressure (kPa)"] = pd.to_numeric(
                dqz["Pressure (kPa)"], errors="coerce"
            )
            dqz.dropna(subset=["Pressure (kPa)"], inplace=True)
            dqz["Temperature (Deg C)"] = pd.to_numeric(
                dqz["Temperature (Deg C)"], errors="coerce"
            )
            dqz.dropna(subset=["Temperature (Deg C)"], inplace=True)

        inc = data_extracted[identifier]["INC"]
        if inc is not None:
            inc["Pitch (deg)"] = pd.to_numeric(inc["Pitch (deg)"], errors="coerce")
            inc.dropna(subset=["Pitch (deg)"], inplace=True)
            inc["Roll (deg)"] = pd.to_numeric(inc["Roll (deg)"], errors="coerce")
            inc.dropna(subset=["Roll (deg)"], inplace=True)

    # Remove outliers from TMP and DQZ
    for identifier in IDENTIFIERS:
        tmp = data_extracted[identifier]["TMP"]
        if tmp is not None:
            data_extracted[identifier]["TMP"] = remove_outliers(
                tmp, "Temperature Deg C"
            )

        dqz = data_extracted[identifier]["DQZ"]
        if dqz is not None:
            data_extracted[identifier]["DQZ"] = remove_outliers(dqz, "Pressure (kPa)")
            data_extracted[identifier]["DQZ"] = remove_outliers(
                data_extracted[identifier]["DQZ"], "Temperature (Deg C)"
            )

    return df_dict, data_extracted, result_dfs
