"""
Utility Functions Module for FETCH StreamLine Analysis

This module contains statistical calculations, data cleaning,
and general helper functions.
"""

from typing import Union, Tuple
import pandas as pd
import numpy as np
from scipy.stats import hmean


def calculate_harmonic_mean(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Calculate the harmonic mean of two pandas Series.
    
    Args:
        series1: First data series
        series2: Second data series
        
    Returns:
        Series containing harmonic mean values
    """
    # Drop NaN values
    combined = pd.concat([series1, series2], axis=1).dropna()
    harmonic_values = hmean(combined, axis=1)
    return pd.Series(harmonic_values, index=combined.index)


def compute_harmonic_mean(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    column: str = 'velocity'
) -> Tuple[pd.Index, np.ndarray]:
    """
    Compute harmonic mean for a specific column between two DataFrames.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        column: Name of the column to compute harmonic mean for
        
    Returns:
        Tuple of (common_index, harmonic_mean_values)
    """
    # Intersect the indices (Record Time) of both DataFrames
    common_index = df1.index.intersection(df2.index)
    
    # Select only the common timestamps
    aligned_df1 = df1.loc[common_index]
    aligned_df2 = df2.loc[common_index]
    
    # Ensure no zero or negative values as they are invalid for harmonic mean
    aligned_df1[column] = aligned_df1[column].clip(lower=0.0001)
    aligned_df2[column] = aligned_df2[column].clip(lower=0.0001)
    
    # Compute the harmonic mean
    harmonic_mean_values = hmean([aligned_df1[column], aligned_df2[column]], axis=0)
    
    return common_index, harmonic_mean_values


def calculate_rms_error(values1: Union[list, np.ndarray], values2: Union[list, np.ndarray]) -> float:
    """
    Calculate the Root Mean Square (RMS) error between two sets of values.
    
    Excludes NaN values from the calculation.
    
    Args:
        values1: First set of values
        values2: Second set of values
        
    Returns:
        RMS error value
    """
    # Ensure the arrays are numpy arrays
    values1 = np.array(values1)
    values2 = np.array(values2)

    # Calculate the difference
    differences = values1 - values2

    # Remove NaN values from the differences
    differences = differences[~np.isnan(differences)]

    # Calculate the mean squared error, excluding NaN values
    mean_squared_error = np.nanmean(differences ** 2)

    # Take the square root of the mean squared error
    rms_error = np.sqrt(mean_squared_error)
    
    return rms_error


def calculate_statistics(data: Union[pd.Series, np.ndarray]) -> dict:
    """
    Calculate basic statistical measures for a dataset.
    
    Args:
        data: Input data series or array
        
    Returns:
        Dictionary containing statistical measures
    """
    data_array = np.array(data)
    data_clean = data_array[~np.isnan(data_array)]
    
    if len(data_clean) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'count': 0
        }
    
    return {
        'mean': np.mean(data_clean),
        'median': np.median(data_clean),
        'std': np.std(data_clean),
        'min': np.min(data_clean),
        'max': np.max(data_clean),
        'count': len(data_clean)
    }


def normalize_data(data: Union[pd.Series, np.ndarray], method: str = 'z-score') -> np.ndarray:
    """
    Normalize data using specified method.
    
    Args:
        data: Input data to normalize
        method: Normalization method ('z-score', 'min-max', or 'robust')
        
    Returns:
        Normalized data array
        
    Raises:
        ValueError: If method is not supported
    """
    data_array = np.array(data)
    
    if method == 'z-score':
        mean = np.nanmean(data_array)
        std = np.nanstd(data_array)
        if std == 0:
            return np.zeros_like(data_array)
        return (data_array - mean) / std
        
    elif method == 'min-max':
        min_val = np.nanmin(data_array)
        max_val = np.nanmax(data_array)
        if max_val == min_val:
            return np.zeros_like(data_array)
        return (data_array - min_val) / (max_val - min_val)
        
    elif method == 'robust':
        median = np.nanmedian(data_array)
        mad = np.nanmedian(np.abs(data_array - median))
        if mad == 0:
            return np.zeros_like(data_array)
        return (data_array - median) / mad
        
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def detect_outliers(
    data: Union[pd.Series, np.ndarray], 
    method: str = 'iqr',
    threshold: float = 1.5
) -> np.ndarray:
    """
    Detect outliers in data using specified method.
    
    Args:
        data: Input data to analyze
        method: Detection method ('iqr' or 'z-score')
        threshold: Threshold value for outlier detection
        
    Returns:
        Boolean array indicating outliers (True = outlier)
        
    Raises:
        ValueError: If method is not supported
    """
    data_array = np.array(data)
    
    if method == 'iqr':
        Q1 = np.nanpercentile(data_array, 25)
        Q3 = np.nanpercentile(data_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data_array < lower_bound) | (data_array > upper_bound)
        
    elif method == 'z-score':
        mean = np.nanmean(data_array)
        std = np.nanstd(data_array)
        if std == 0:
            return np.zeros(len(data_array), dtype=bool)
        z_scores = np.abs((data_array - mean) / std)
        return z_scores > threshold
        
    else:
        raise ValueError(f"Unsupported outlier detection method: {method}")


def interpolate_gaps(
    data: pd.Series, 
    method: str = 'linear',
    max_gap: int = None
) -> pd.Series:
    """
    Interpolate gaps in time series data.
    
    Args:
        data: Time series data with potential gaps
        method: Interpolation method ('linear', 'cubic', 'nearest')
        max_gap: Maximum gap size to interpolate (None = no limit)
        
    Returns:
        Series with interpolated values
    """
    if max_gap is not None:
        # Only interpolate gaps smaller than max_gap
        return data.interpolate(method=method, limit=max_gap)
    else:
        return data.interpolate(method=method)


def sliding_window_stats(
    data: Union[pd.Series, np.ndarray], 
    window_size: int,
    stat: str = 'mean'
) -> np.ndarray:
    """
    Calculate sliding window statistics.
    
    Args:
        data: Input data
        window_size: Size of the sliding window
        stat: Statistic to calculate ('mean', 'median', 'std', 'min', 'max')
        
    Returns:
        Array of windowed statistics
        
    Raises:
        ValueError: If statistic is not supported
    """
    if isinstance(data, pd.Series):
        if stat == 'mean':
            return data.rolling(window=window_size, center=True).mean().values
        elif stat == 'median':
            return data.rolling(window=window_size, center=True).median().values
        elif stat == 'std':
            return data.rolling(window=window_size, center=True).std().values
        elif stat == 'min':
            return data.rolling(window=window_size, center=True).min().values
        elif stat == 'max':
            return data.rolling(window=window_size, center=True).max().values
        else:
            raise ValueError(f"Unsupported statistic: {stat}")
    else:
        # For numpy arrays, use a simple implementation
        data_array = np.array(data)
        result = np.full(len(data_array), np.nan)
        half_window = window_size // 2
        
        for i in range(half_window, len(data_array) - half_window):
            window_data = data_array[i - half_window:i + half_window + 1]
            window_data_clean = window_data[~np.isnan(window_data)]
            
            if len(window_data_clean) > 0:
                if stat == 'mean':
                    result[i] = np.mean(window_data_clean)
                elif stat == 'median':
                    result[i] = np.median(window_data_clean)
                elif stat == 'std':
                    result[i] = np.std(window_data_clean)
                elif stat == 'min':
                    result[i] = np.min(window_data_clean)
                elif stat == 'max':
                    result[i] = np.max(window_data_clean)
                else:
                    raise ValueError(f"Unsupported statistic: {stat}")
        
        return result