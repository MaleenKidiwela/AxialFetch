#!/usr/bin/env python3
"""
FETCH StreamLine Analysis - Main Demonstration Script

This script demonstrates the usage of the FETCH StreamLine analysis package
for processing oceanographic data and calculating sound velocities.

Example usage:
    python main.py --data_path ../Data_230912111909_East_006870_2502_.csv
    python main.py --demo  # Run with synthetic demo data
"""

import argparse
import os
import sys
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import process_data, create_nested_dictionary, remove_outliers
from sound_velocity import (
    velocity_timeseries_TEOS10, 
    velocity_timeseries_basic,
    velocity_from_teos10,
    calculate_sensitivities_TEOS10
)
from positioning import local_xy, to_enu, apply_tilt_correction
from optimization import find_best_temperature_offset, fit_polynomial
from utils import calculate_rms_error, calculate_statistics, normalize_data
from data_persistence import (
    save_processed_results, 
    save_baseline_data, 
    save_instrument_data,
    create_analysis_summary
)


def create_demo_data() -> pd.DataFrame:
    """
    Create synthetic oceanographic data for demonstration purposes.
    
    Returns:
        DataFrame with synthetic temperature, pressure, salinity, and time data
    """
    # Create time series (24 hours of data every 10 minutes)
    n_points = 144
    time_range = pd.date_range(
        start='2023-09-12 11:00:00', 
        periods=n_points, 
        freq='10T'
    )
    
    # Create realistic oceanographic data
    np.random.seed(42)  # For reproducible results
    
    # Temperature: seasonal + daily variation + noise
    base_temp = 4.0  # Base deep water temperature
    daily_temp_var = 0.5 * np.sin(2 * np.pi * np.arange(n_points) / 144)  # Daily variation
    temp_noise = np.random.normal(0, 0.1, n_points)
    temperature = base_temp + daily_temp_var + temp_noise
    
    # Pressure: tidal variation + instrument depth
    base_pressure = 2000.0  # Base pressure at depth (kPa)
    tidal_pressure = 10.0 * np.sin(2 * np.pi * np.arange(n_points) / 72)  # Semi-diurnal tide
    pressure_noise = np.random.normal(0, 0.5, n_points)
    pressure = base_pressure + tidal_pressure + pressure_noise
    
    # Salinity: relatively stable with small variations
    base_salinity = 34.5
    salinity_drift = 0.2 * np.sin(2 * np.pi * np.arange(n_points) / n_points)  # Long-term drift
    salinity_noise = np.random.normal(0, 0.05, n_points)
    salinity = base_salinity + salinity_drift + salinity_noise
    
    # Create DataFrame
    demo_df = pd.DataFrame({
        'Record Time': time_range,
        'Temperature Deg C': temperature,
        'Pressure (kPa)': pressure,
        'Salinity': salinity,
        'Pitch': np.random.normal(0, 1, n_points),  # Platform attitude
        'Roll': np.random.normal(0, 1, n_points),
        'Heading': np.random.normal(45, 5, n_points)  # Roughly northeast heading
    })
    
    return demo_df


def analyze_sound_velocity(df: pd.DataFrame, method: str = 'TEOS10') -> Dict[str, Any]:
    """
    Analyze sound velocity using specified method.
    
    Args:
        df: DataFrame with temperature, pressure, and salinity data
        method: Method to use ('TEOS10', 'basic', or 'chen_millero')
        
    Returns:
        Dictionary containing analysis results
    """
    print(f"\n=== Sound Velocity Analysis ({method}) ===")
    
    # Extract data
    temperature = df['Temperature Deg C'].values
    pressure = df['Pressure (kPa)'].values
    salinity = df['Salinity'].values
    
    # Calculate sound velocity
    if method == 'TEOS10':
        velocity = velocity_timeseries_TEOS10(temperature, pressure, salinity)
    elif method == 'basic':
        velocity = velocity_timeseries_basic(temperature, pressure, salinity)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Add to DataFrame
    df[f'SoundSpeed_{method}'] = velocity
    
    # Calculate statistics
    stats = calculate_statistics(velocity)
    print(f"Sound Speed Statistics:")
    print(f"  Mean: {stats['mean']:.2f} m/s")
    print(f"  Std:  {stats['std']:.2f} m/s")
    print(f"  Min:  {stats['min']:.2f} m/s")
    print(f"  Max:  {stats['max']:.2f} m/s")
    
    return {
        'velocity': velocity,
        'statistics': stats,
        'method': method
    }


def analyze_sensitivities(df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze sound velocity sensitivities to temperature, salinity, and pressure.
    
    Args:
        df: DataFrame with oceanographic data
        
    Returns:
        Dictionary with sensitivity coefficients
    """
    print(f"\n=== Sensitivity Analysis ===")
    
    # Use mean values for sensitivity analysis
    mean_temp = df['Temperature Deg C'].mean()
    mean_pressure = df['Pressure (kPa)'].mean() * 0.1  # Convert to dbar
    mean_salinity = df['Salinity'].mean()
    
    # Default coordinates for FETCH project
    longitude = -130.01149966
    latitude = 45.95882833
    
    baseline_vel, sens_S, sens_T, sens_P = calculate_sensitivities_TEOS10(
        mean_salinity, mean_temp, mean_pressure, longitude, latitude
    )
    
    print(f"Baseline velocity: {baseline_vel:.2f} m/s")
    print(f"Sensitivity to salinity: {sens_S:.3f} m/s per psu")
    print(f"Sensitivity to temperature: {sens_T:.3f} m/s per °C")
    print(f"Sensitivity to pressure: {sens_P:.3f} m/s per dbar")
    
    return {
        'baseline_velocity': baseline_vel,
        'sensitivity_salinity': sens_S,
        'sensitivity_temperature': sens_T,
        'sensitivity_pressure': sens_P
    }


def analyze_positioning(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze platform positioning and tilt effects.
    
    Args:
        df: DataFrame with attitude data
        
    Returns:
        Dictionary with positioning analysis results
    """
    print(f"\n=== Positioning Analysis ===")
    
    # Calculate local displacements from tilt
    dx, dy = local_xy(df, h=1.0)  # Assuming 1m transducer height
    
    # Convert to ENU coordinates
    east, north = to_enu(dx, dy, df['Heading'].values)
    
    # Add to DataFrame
    df['East_displacement'] = east
    df['North_displacement'] = north
    
    # Calculate statistics
    east_stats = calculate_statistics(east)
    north_stats = calculate_statistics(north)
    total_displacement = np.sqrt(east**2 + north**2)
    total_stats = calculate_statistics(total_displacement)
    
    print(f"Platform displacement statistics:")
    print(f"  East RMS: {np.sqrt(np.mean(east**2)):.3f} m")
    print(f"  North RMS: {np.sqrt(np.mean(north**2)):.3f} m")
    print(f"  Total RMS: {np.sqrt(np.mean(total_displacement**2)):.3f} m")
    print(f"  Max total: {total_stats['max']:.3f} m")
    
    return {
        'east_displacement': east,
        'north_displacement': north,
        'total_displacement': total_displacement,
        'east_stats': east_stats,
        'north_stats': north_stats,
        'total_stats': total_stats
    }


def create_visualizations(df: pd.DataFrame, output_dir: str = '.') -> None:
    """
    Create visualization plots of the analysis results.
    
    Args:
        df: DataFrame with analysis results
        output_dir: Directory to save plots
    """
    print(f"\n=== Creating Visualizations ===")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('FETCH StreamLine Analysis Results', fontsize=16)
    
    # Plot 1: Temperature and Pressure time series
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(df['Record Time'], df['Temperature Deg C'], 'b-', label='Temperature')
    line2 = ax1_twin.plot(df['Record Time'], df['Pressure (kPa)'], 'r-', label='Pressure')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature (°C)', color='b')
    ax1_twin.set_ylabel('Pressure (kPa)', color='r')
    ax1.set_title('Temperature and Pressure')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # Plot 2: Sound velocity
    ax2 = axes[0, 1]
    if 'SoundSpeed_TEOS10' in df.columns:
        ax2.plot(df['Record Time'], df['SoundSpeed_TEOS10'], 'g-', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Sound Speed (m/s)')
    ax2.set_title('Sound Velocity (TEOS-10)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Platform attitude
    ax3 = axes[1, 0]
    ax3.plot(df['Record Time'], df['Pitch'], 'b-', alpha=0.7, label='Pitch')
    ax3.plot(df['Record Time'], df['Roll'], 'r-', alpha=0.7, label='Roll')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Angle (degrees)')
    ax3.set_title('Platform Attitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Platform displacement
    ax4 = axes[1, 1]
    if 'East_displacement' in df.columns:
        total_disp = np.sqrt(df['East_displacement']**2 + df['North_displacement']**2)
        ax4.plot(df['Record Time'], total_disp * 1000, 'purple', linewidth=2)  # Convert to mm
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Total Displacement (mm)')
    ax4.set_title('Platform Displacement')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'fetch_analysis_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()


def save_results_with_pickles(df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    """
    Save analysis results in the original FETCH pickle format.
    
    Args:
        df: DataFrame with analysis results
        output_dir: Directory to save files
        
    Returns:
        Dictionary of saved file paths
    """
    saved_files = {}
    
    # Save main results as pickle (like original 2504.pkl)
    main_pickle_path = os.path.join(output_dir, 'analysis_results.pkl')
    df.to_pickle(main_pickle_path)
    saved_files['main_results'] = main_pickle_path
    
    # Also save as CSV for compatibility
    csv_path = os.path.join(output_dir, 'fetch_analysis_results.csv')
    df.to_csv(csv_path, index=True)
    saved_files['csv_results'] = csv_path
    
    print(f"Results saved in FETCH pickle format: {main_pickle_path}")
    print(f"Results also saved as CSV: {csv_path}")
    
    return saved_files


def main():
    """Main function to run the FETCH StreamLine analysis."""
    parser = argparse.ArgumentParser(description='FETCH StreamLine Analysis Tool')
    parser.add_argument('--data_path', type=str, help='Path to oceanographic data CSV file')
    parser.add_argument('--demo', action='store_true', help='Run with synthetic demo data')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for results')
    parser.add_argument('--methods', nargs='+', default=['TEOS10'], 
                       choices=['TEOS10', 'basic'],
                       help='Sound velocity calculation methods to use')
    parser.add_argument('--save_pickles', action='store_true', 
                       help='Save results in original FETCH pickle format')
    
    args = parser.parse_args()
    
    print("=== FETCH StreamLine Analysis Tool ===")
    print("A clean, modular implementation for oceanographic data processing")
    print()
    
    # Load or create data
    if args.demo:
        print("Creating synthetic demonstration data...")
        df = create_demo_data()
        print(f"Created {len(df)} data points spanning {df['Record Time'].iloc[-1] - df['Record Time'].iloc[0]}")
    elif args.data_path:
        if not os.path.exists(args.data_path):
            print(f"Error: Data file not found: {args.data_path}")
            return
        print(f"Loading data from: {args.data_path}")
        # This would need to be adapted based on the actual file format
        # For now, assume it's a simple CSV
        try:
            df = pd.read_csv(args.data_path)
            df['Record Time'] = pd.to_datetime(df['Record Time'])
            print(f"Loaded {len(df)} data points")
        except Exception as e:
            print(f"Error loading data: {e}")
            return
    else:
        print("Please specify --data_path or use --demo for synthetic data")
        return
    
    # Perform analyses
    try:
        # Sound velocity analysis
        velocity_results = {}
        for method in args.methods:
            velocity_results[method] = analyze_sound_velocity(df, method)
        
        # Sensitivity analysis
        sensitivity_results = analyze_sensitivities(df)
        
        # Positioning analysis
        positioning_results = analyze_positioning(df)
        
        # Create visualizations
        create_visualizations(df, args.output_dir)
        
        # Summary
        print(f"\n=== Analysis Summary ===")
        print(f"Total data points analyzed: {len(df)}")
        print(f"Time range: {df['Record Time'].iloc[0]} to {df['Record Time'].iloc[-1]}")
        print(f"Methods used: {', '.join(args.methods)}")
        print()
        print("Analysis completed successfully!")
        
        # Save results in multiple formats
        if args.save_pickles:
            # Save in original FETCH pickle format
            pickle_files = save_results_with_pickles(df, args.output_dir)
            
            # Create analysis summary
            summary_path = create_analysis_summary({
                'main_results': df,
                'velocity_results': velocity_results,
                'sensitivity_results': sensitivity_results,
                'positioning_results': positioning_results
            }, args.output_dir)
            print(f"Analysis summary saved to: {summary_path}")
        else:
            # Default: save as CSV
            output_file = os.path.join(args.output_dir, 'fetch_analysis_results.csv')
            df.to_csv(output_file, index=False)
            print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()