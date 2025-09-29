"""
Data Processing Module for FETCH StreamLine Analysis

This module contains functions for loading, parsing, and processing
oceanographic data from CSV files.
"""

from typing import Dict, List, Any, Union
import pandas as pd
import numpy as np


def process_data(filepath: str) -> Dict[str, pd.DataFrame]:
    """
    Load and process CSV files by separating header rows and data rows.
    
    Args:
        filepath: Path to the CSV file to process
        
    Returns:
        Dictionary containing DataFrames for each code found in the header
    """
    # Load the CSV file into a DataFrame
    data_df = pd.read_csv(filepath)
    
    # Define constants for the number of header rows and the row indicating the start of data
    NUM_HEADER_ROWS = 14
    DATA_START_MARKER = "# Data"
    
    # Separate header rows and data rows
    header_rows = data_df.iloc[:NUM_HEADER_ROWS]
    data_rows = data_df[data_df["Code"] != DATA_START_MARKER]
    
    # Create a dictionary to store DataFrames for each code
    dataframes_dict = {}
    
    # Iterate through the header rows
    for _, row in header_rows.iterrows():
        # Extract the code and headers for this code
        code, *headers = row.dropna().values
        
        # Rename the columns in a temporary DataFrame based on the extracted headers
        rename_dict = {f"V{i+1}": header for i, header in enumerate(headers)}
        temp_df = data_rows.copy()
        temp_df.rename(columns=rename_dict, inplace=True)
        
        # Get the corresponding data rows based on the code
        corresponding_data_rows = temp_df[temp_df["Code"] == code]
        
        # If there are corresponding data rows:
        if not corresponding_data_rows.empty:
            # Set the first row as the header
            corresponding_data_rows.columns = corresponding_data_rows.iloc[0].values
            # Drop the first row (now header)
            corresponding_data_rows = corresponding_data_rows.drop(corresponding_data_rows.index[0])
            dataframes_dict[code] = corresponding_data_rows
            
    return dataframes_dict


def create_nested_dictionary(filepaths: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Process multiple files and return a dictionary containing dataframes for each file.
    
    Args:
        filepaths: List of file paths to process
        
    Returns:
        Dictionary with file identifiers as keys and processed data dictionaries as values.
        The keys are extracted from the last 4 digits of the csv file name.
    """
    all_dfs = {}
    
    for filepath in filepaths:
        # Extracting the identifier from the filename (using the last 4 digits)
        identifier = filepath.split("_")[-2]
        
        all_dfs[identifier] = process_data(filepath)
        
    return all_dfs


def ensure_datetime(df: pd.DataFrame, column_name: str) -> None:
    """
    Check and convert a column to datetime format if not already.
    
    Args:
        df: DataFrame to modify
        column_name: Name of the column to convert
    """
    if not pd.api.types.is_datetime64_any_dtype(df[column_name]):
        df[column_name] = pd.to_datetime(df[column_name])


def parse_to_dataframe(data: str) -> pd.DataFrame:
    """
    Parse text data into a DataFrame with DateTime and Value columns.
    
    Args:
        data: String containing space-separated data lines
        
    Returns:
        DataFrame with DateTime and Value columns
    """
    lines = data.split('\n')
    parsed_data = [line.split() for line in lines if line]
    df = pd.DataFrame(parsed_data, columns=['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Value'])
    df['DateTime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
    df['Value'] = pd.to_numeric(df['Value'])
    df.drop(['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'], axis=1, inplace=True)
    return df


def set_datetime_index(df: pd.DataFrame) -> None:
    """
    Set datetime index for DataFrames, converting Record Time column to datetime index.
    
    Args:
        df: DataFrame to modify in place
    """
    if 'Record Time' in df.columns:
        df['Record Time'] = pd.to_datetime(df['Record Time'])
        df.set_index('Record Time', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)


def remove_outliers(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame based on the Interquartile Range (IQR) for a given column.
    
    Args:
        df: Input DataFrame
        column_name: Name of the column to analyze for outliers
        
    Returns:
        DataFrame with outliers removed
    """
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Return only rows where the value is within the bounds
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]


def find_closest_time(df: pd.DataFrame, target_time: Any) -> pd.Series:
    """
    Find the row in DataFrame with the closest time to the target time.
    
    Args:
        df: DataFrame with 'time' column
        target_time: Target time to find closest match for
        
    Returns:
        Series representing the row with the closest time
    """
    absolute_difference = abs(df['time'] - target_time)
    closest_index = absolute_difference.idxmin()
    return df.loc[closest_index]