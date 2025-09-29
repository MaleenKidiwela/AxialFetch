"""
Optimization and Fitting Module for FETCH StreamLine Analysis

This module contains functions for optimization, curve fitting,
and parameter estimation in oceanographic data analysis.
"""

from typing import Tuple, Callable, Any, Union, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import linregress
from .sound_velocity import velocity_from_teos10
from .utils import calculate_rms_error


def fit_and_extrapolate(
    df: pd.DataFrame, 
    start_time: pd.Timestamp, 
    end_time: pd.Timestamp, 
    target_time: pd.Timestamp,
    x_column: str = 'time',
    y_column: str = 'sea_water_practical_salinity'
) -> float:
    """
    Perform linear regression and extrapolation using scipy.stats.linregress.
    
    Args:
        df: DataFrame containing the data
        start_time: Start time for regression period
        end_time: End time for regression period
        target_time: Time to extrapolate to
        x_column: Name of time column
        y_column: Name of data column to extrapolate
        
    Returns:
        Extrapolated value at target_time
    """
    subset = df[(df[x_column] >= start_time) & (df[x_column] <= end_time)]
    
    if len(subset) < 2:
        raise ValueError("Insufficient data points for linear regression")
    
    slope, intercept, _, _, _ = linregress(
        x=subset[x_column].map(pd.Timestamp.timestamp),
        y=subset[y_column]
    )
    
    return slope * target_time.timestamp() + intercept


def optimize_tidal_influence(
    params: Tuple[float, float], 
    tidal_df: pd.DataFrame, 
    pressure_df: pd.DataFrame
) -> float:
    """
    Optimization function for tidal influence on pressure measurements.
    
    Args:
        params: Tuple of (amplitude, density_factor) parameters
        tidal_df: DataFrame containing tidal data with 'Value' column
        pressure_df: DataFrame containing pressure data with 'Pressure (kPa)' column
        
    Returns:
        Variance of corrected pressure (to be minimized)
    """
    amplitude, rho = params
    
    # Apply amplitude scaling to tidal values
    adjusted_tidal = amplitude * tidal_df['Value']
    tidal_df['Adjusted Value'] = adjusted_tidal
    
    # Interpolate tidal data to pressure timestamps
    interpolated_tidal = tidal_df.reindex(pressure_df.index, method='nearest')['Adjusted Value']
    
    # Apply tidal correction to pressure
    # Pressure correction = rho * g * tidal_height * unit_conversion
    g = 9.81  # gravitational acceleration
    unit_conversion = 0.001  # convert to kPa
    corrected_pressure = pressure_df['Pressure (kPa)'] - (rho * g * interpolated_tidal) * unit_conversion
    
    # Return variance (to be minimized)
    return np.var(corrected_pressure)


def objective_function(
    delta_T: float, 
    temp_obs: Union[float, np.ndarray], 
    salinity: Union[float, np.ndarray], 
    pressure: Union[float, np.ndarray], 
    velocity_obs: Union[float, np.ndarray]
) -> float:
    """
    Objective function for temperature offset optimization based on velocity measurements.
    
    Args:
        delta_T: Temperature offset to optimize
        temp_obs: Observed temperature values
        salinity: Salinity values
        pressure: Pressure values
        velocity_obs: Observed velocity values
        
    Returns:
        Mean squared error between calculated and observed velocities
    """
    temp_true = temp_obs + delta_T
    velocity_calc = velocity_from_teos10(temp_true, salinity, pressure)
    mse = np.mean((velocity_calc - velocity_obs) ** 2)
    return mse


def find_best_temperature_offset(
    temp_obs: Union[float, np.ndarray], 
    salinity: Union[float, np.ndarray], 
    pressure: Union[float, np.ndarray], 
    velocity_obs: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Find the best temperature offset using scipy optimization.
    
    Args:
        temp_obs: Observed temperature values
        salinity: Salinity values
        pressure: Pressure values
        velocity_obs: Observed velocity values
        
    Returns:
        Optimal temperature offset value(s)
    """
    result = minimize(
        objective_function, 
        x0=0, 
        args=(temp_obs, salinity, pressure, velocity_obs),
        method='Nelder-Mead'
    )
    return result.x


def find_best_offset_iterative(
    combined_df: pd.DataFrame,
    temp_column: str = 'Temperature Deg C',
    pressure_column: str = 'Corrected Pressure (kPa)',
    salinity_column: str = 'Salinity',
    reference_velocity_column: str = 'interp_v',
    offset_range: Tuple[float, float, float] = (0.01, 0.5, 0.01)
) -> Tuple[float, float]:
    """
    Find the best temperature offset by iterating over possible offsets and minimizing RMS error.
    
    Args:
        combined_df: DataFrame containing temperature, pressure, salinity, and reference velocity data
        temp_column: Name of temperature column
        pressure_column: Name of pressure column
        salinity_column: Name of salinity column
        reference_velocity_column: Name of reference velocity column
        offset_range: Tuple of (start, stop, step) for offset search
        
    Returns:
        Tuple of (best_offset, min_rms_error)
    """
    min_rms_error = float('inf')
    best_offset = None
    
    start, stop, step = offset_range
    
    # Iterate over a range of possible offsets
    for offset in np.arange(start, stop, step):
        T_series = np.array(combined_df[temp_column])
        P_series_kPa = np.array(combined_df[pressure_column])
        S_series = np.array(combined_df[salinity_column])
        
        # Calculate velocities with offset temperature
        velocity_calc = []
        for T, P, S in zip(T_series + offset, P_series_kPa, S_series):
            velocity_calc.append(velocity_from_teos10(T, S, P))
        
        # Calculate RMS error against reference
        interp_v = combined_df[reference_velocity_column].values
        rms_error = calculate_rms_error(interp_v, velocity_calc)
        
        if rms_error < min_rms_error:
            min_rms_error = rms_error
            best_offset = offset
    
    return best_offset, min_rms_error


def optimize_parameters(
    objective_func: Callable,
    initial_params: Union[List, np.ndarray],
    bounds: List[Tuple[float, float]] = None,
    method: str = 'Nelder-Mead',
    **kwargs
) -> dict:
    """
    General parameter optimization function.
    
    Args:
        objective_func: Function to minimize
        initial_params: Initial parameter values
        bounds: Parameter bounds as list of (min, max) tuples
        method: Optimization method
        **kwargs: Additional arguments passed to scipy.optimize.minimize
        
    Returns:
        Dictionary containing optimization results
    """
    result = minimize(
        objective_func,
        x0=initial_params,
        bounds=bounds,
        method=method,
        **kwargs
    )
    
    return {
        'optimal_params': result.x,
        'optimal_value': result.fun,
        'success': result.success,
        'message': result.message,
        'iterations': result.nit if hasattr(result, 'nit') else None,
        'function_evaluations': result.nfev if hasattr(result, 'nfev') else None
    }


def fit_polynomial(
    x_data: np.ndarray, 
    y_data: np.ndarray, 
    degree: int = 1,
    weights: np.ndarray = None
) -> Tuple[np.ndarray, float]:
    """
    Fit polynomial to data and return coefficients and R-squared.
    
    Args:
        x_data: Independent variable data
        y_data: Dependent variable data
        degree: Polynomial degree
        weights: Optional weights for fitting
        
    Returns:
        Tuple of (coefficients, r_squared)
    """
    # Remove NaN values
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    
    if weights is not None:
        w_clean = weights[mask]
    else:
        w_clean = None
    
    if len(x_clean) < degree + 1:
        raise ValueError(f"Insufficient data points for degree {degree} polynomial")
    
    # Fit polynomial
    coeffs = np.polyfit(x_clean, y_clean, degree, w=w_clean)
    
    # Calculate R-squared
    y_pred = np.polyval(coeffs, x_clean)
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return coeffs, r_squared


def robust_regression(
    x_data: np.ndarray, 
    y_data: np.ndarray,
    outlier_threshold: float = 2.0,
    max_iterations: int = 10
) -> Tuple[float, float, np.ndarray]:
    """
    Perform robust linear regression with outlier rejection.
    
    Args:
        x_data: Independent variable data
        y_data: Dependent variable data
        outlier_threshold: Z-score threshold for outlier rejection
        max_iterations: Maximum number of iterations
        
    Returns:
        Tuple of (slope, intercept, inlier_mask)
    """
    # Remove initial NaN values
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_work = x_data[mask]
    y_work = y_data[mask]
    current_mask = np.ones(len(x_work), dtype=bool)
    
    for iteration in range(max_iterations):
        # Fit linear regression on current inliers
        x_current = x_work[current_mask]
        y_current = y_work[current_mask]
        
        if len(x_current) < 2:
            break
            
        slope, intercept, _, _, _ = linregress(x_current, y_current)
        
        # Calculate residuals for all points
        y_pred = slope * x_work + intercept
        residuals = y_work - y_pred
        
        # Calculate standardized residuals
        residual_std = np.std(residuals[current_mask])
        if residual_std == 0:
            break
            
        z_scores = np.abs(residuals) / residual_std
        
        # Update mask to exclude outliers
        new_mask = z_scores <= outlier_threshold
        
        # Check for convergence
        if np.array_equal(current_mask, new_mask):
            break
            
        current_mask = new_mask
    
    # Create final mask for original data
    final_mask = np.zeros(len(x_data), dtype=bool)
    final_mask[mask] = current_mask
    
    return slope, intercept, final_mask


def cross_validate_model(
    model_func: Callable,
    x_data: np.ndarray,
    y_data: np.ndarray,
    k_folds: int = 5,
    random_state: int = None
) -> dict:
    """
    Perform k-fold cross validation on a model.
    
    Args:
        model_func: Function that takes (x_train, y_train, x_test) and returns y_pred
        x_data: Independent variable data
        y_data: Dependent variable data
        k_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with cross-validation statistics
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Remove NaN values
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    
    n_samples = len(x_clean)
    fold_size = n_samples // k_folds
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    fold_scores = []
    fold_rmse = []
    
    for fold in range(k_folds):
        # Define test indices for this fold
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < k_folds - 1 else n_samples
        test_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        # Split data
        x_train, y_train = x_clean[train_indices], y_clean[train_indices]
        x_test, y_test = x_clean[test_indices], y_clean[test_indices]
        
        # Make predictions
        y_pred = model_func(x_train, y_train, x_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate R-squared
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        fold_scores.append(r_squared)
        fold_rmse.append(rmse)
    
    return {
        'mean_r_squared': np.mean(fold_scores),
        'std_r_squared': np.std(fold_scores),
        'mean_rmse': np.mean(fold_rmse),
        'std_rmse': np.std(fold_rmse),
        'fold_r_squared': fold_scores,
        'fold_rmse': fold_rmse
    }