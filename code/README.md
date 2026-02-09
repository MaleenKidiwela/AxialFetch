# FETCH StreamLine Analysis Package

A clean, modular implementation of oceanographic data processing and sound velocity analysis tools, extracted and refactored from the FETCH StreamLine Final notebook.

## Features

- **Data Processing**: Load and process oceanographic CSV files with proper header handling
- **Sound Velocity Calculations**: Multiple methods including TEOS-10, basic calculations, and Chen-Millero
- **Positioning Analysis**: Coordinate transformations and tilt corrections for underwater measurements
- **Optimization Tools**: Parameter estimation, curve fitting, and robust regression
- **Statistical Utilities**: Data cleaning, outlier detection, and statistical analysis

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Import the package:
```python
from data_processing import process_data
from sound_velocity import velocity_timeseries_TEOS10
from positioning import apply_tilt_correction
from optimization import find_best_temperature_offset
from utils import calculate_rms_error
```

## Quick Start

### Using Demo Data
```bash
python main.py --demo
```

### Using Your Own Data
```bash
python main.py --data_path /path/to/your/data.csv
```

### Programmatic Usage
```python
import pandas as pd
from sound_velocity import velocity_timeseries_TEOS10

# Example data
temperature = [4.0, 4.1, 4.2]  # °C
pressure = [2000, 2005, 2010]  # kPa  
salinity = [34.5, 34.6, 34.5]  # PSU

# Calculate sound velocity
velocity = velocity_timeseries_TEOS10(temperature, pressure, salinity)
print(f"Sound velocities: {velocity}")
```

## Module Overview

### `data_processing.py`
Functions for loading and processing oceanographic data files:
- `process_data()`: Parse CSV files with complex header structures
- `create_nested_dictionary()`: Process multiple files
- `remove_outliers()`: IQR-based outlier detection
- `ensure_datetime()`: Handle datetime conversions

### `sound_velocity.py`
Sound velocity calculations using various methods:
- `velocity_timeseries_TEOS10()`: TEOS-10 standard (most accurate)
- `velocity_timeseries_basic()`: Basic T,P,S calculation
- `velocity_timeseries_chen_millero()`: Chen & Millero (1977) equation
- `calculate_sensitivities_TEOS10()`: Sensitivity analysis

### `positioning.py`
Coordinate transformations and positioning:
- `local_xy()`: Convert tilt angles to local coordinates
- `to_enu()`: Transform to East-North-Up coordinates
- `apply_tilt_correction()`: Correct range measurements for platform tilt
- `geodetic_to_enu()`: Geographic coordinate conversions

### `optimization.py`
Parameter estimation and fitting:
- `find_best_temperature_offset()`: Optimize temperature corrections
- `fit_and_extrapolate()`: Linear regression with extrapolation
- `robust_regression()`: Outlier-resistant fitting
- `cross_validate_model()`: Model validation

### `utils.py`
Statistical and utility functions:
- `calculate_rms_error()`: Root mean square error
- `calculate_harmonic_mean()`: Harmonic mean for velocity averaging
- `normalize_data()`: Data normalization (z-score, min-max, robust)
- `detect_outliers()`: Multiple outlier detection methods

### Range Calculation Workflow (FETCH Range Calculation Notebook)
Structured helpers extracted from `FETCH Range Calculation.ipynb`:
- `range_salinity.py`: Salinity gap stitching, smoothing, bottle calibration
- `tidal_correction.py`: Tidal prediction parsing and pressure correction
- `velocity_interpolation.py`: Interpolate recorded velocities to data timelines
- `pressure_moving_average.py`: 15-day moving average and re-basing helpers
- `range_calculation_workflow.py`: Data extraction and harmonic mean utilities
- `range_calculation_main.py`: Notebook-oriented main workflow entry point

Use the included notebook scaffold:
```bash
jupyter notebook code/fetch_range_calculation.ipynb
```

## Example Analysis Workflow

```python
# 1. Load and process data
from data_processing import process_data, remove_outliers
data_dict = process_data('your_file.csv')
clean_data = remove_outliers(data_dict['2502'], 'Temperature Deg C')

# 2. Calculate sound velocity
from sound_velocity import velocity_timeseries_TEOS10
velocity = velocity_timeseries_TEOS10(
    clean_data['Temperature Deg C'], 
    clean_data['Pressure (kPa)'], 
    clean_data['Salinity']
)

# 3. Apply positioning corrections  
from positioning import apply_tilt_correction
corrected_ranges = apply_tilt_correction(
    range_data, 'baseline_tag', attitude_corrections
)

# 4. Optimize parameters
from optimization import find_best_temperature_offset
best_offset = find_best_temperature_offset(
    temperature_obs, salinity, pressure, velocity_obs
)

# 5. Statistical analysis
from utils import calculate_statistics, calculate_rms_error
stats = calculate_statistics(velocity)
error = calculate_rms_error(velocity_calculated, velocity_observed)
```

## Pickle File Workflow (Original FETCH Format)

The original FETCH StreamLine notebook creates several types of pickle files:

### 1. Individual Instrument Data
```
2502.pkl, 2503.pkl, 2504.pkl
```
These contain combined DataFrames with all sensor data for each instrument:
- Temperature measurements (TMP)
- Pressure measurements (DQZ) 
- Platform attitude (INC)
- Sound speed measurements (SSP)
- Calculated velocities and corrections

### 2. Baseline/Range Data
```
R2502_2503.pkl, R2502_2504.pkl, R2503_2502.pkl, etc.
```
These contain processed range measurements between instrument pairs:
- Calculated distances between instruments
- Tilt corrections applied
- Environmental corrections
- Quality flags and metadata

### 3. Using Pickle Files in Clean Code

```python
# Generate pickle files in original format
python process_fetch_data.py --data_dir /path/to/csv/files

# Or use in main analysis
python main.py --demo --save_pickles

# Load existing pickle files (with compatibility handling)
from data_persistence import load_dataframe_from_pickle

# Load instrument data
instrument_data = load_dataframe_from_pickle('2504.pkl')

# Load baseline data  
baseline_data = load_dataframe_from_pickle('R2502_2503.pkl')
```

### 4. Pickle Compatibility

The clean code handles pandas version compatibility issues:

```python
# Handle legacy pickle files
python pickle_compatibility.py --convert_dir .

# Analyze problematic pickle files
python pickle_compatibility.py --analyze 2504.pkl
```

## Key Improvements Over Original Notebook

1. **Modular Design**: Functions organized by purpose into separate modules
2. **Type Hints**: Full type annotations for better code clarity
3. **Documentation**: Comprehensive docstrings for all functions
4. **Error Handling**: Robust error checking and validation
5. **Flexibility**: Configurable parameters and multiple method options
6. **Performance**: Optimized algorithms and vectorized operations
7. **Testing**: Clean functions that are easy to unit test
8. **Reusability**: Functions can be used independently or combined
9. **Pickle Compatibility**: Handles pandas version issues with legacy files
10. **Multiple Formats**: Save results as pickle, CSV, or HDF5

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **scipy**: Scientific computing and optimization
- **gsw**: Gibbs SeaWater oceanographic toolbox
- **matplotlib**: Visualization (for main.py demo)

## Contributing

This package was extracted from the FETCH StreamLine Final notebook to create a cleaner, more maintainable codebase. The original notebook's functionality has been preserved while improving code organization and adding comprehensive documentation.

## License

This code is derived from the FETCH StreamLine project for oceanographic research.
