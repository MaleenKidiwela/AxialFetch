#!/usr/bin/env python3
"""
Pickle Compatibility Utilities for FETCH StreamLine Analysis

This module handles loading older pickle files that may have pandas version
compatibility issues, and provides utilities for creating compatible pickle files.
"""

import pickle
import pandas as pd
import numpy as np
from typing import Any, Dict, Union
import warnings
import os


def load_legacy_pickle(filepath: str, fallback_to_csv: bool = True) -> Union[pd.DataFrame, Any]:
    """
    Load pickle files with compatibility handling for different pandas versions.
    
    Args:
        filepath: Path to the pickle file
        fallback_to_csv: If True, try to find a CSV version if pickle fails
        
    Returns:
        Loaded data (usually a DataFrame)
        
    Raises:
        Exception: If loading fails with all methods
    """
    print(f"Attempting to load: {filepath}")
    
    # Method 1: Try pandas.read_pickle (modern pandas)
    try:
        data = pd.read_pickle(filepath)
        print(f"✓ Loaded with pd.read_pickle: {type(data)}")
        return data
    except Exception as e1:
        print(f"✗ pd.read_pickle failed: {e1}")
    
    # Method 2: Try regular pickle.load
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded with pickle.load: {type(data)}")
        return data
    except Exception as e2:
        print(f"✗ pickle.load failed: {e2}")
    
    # Method 3: Try with different pickle protocols
    for protocol in [None, 2, 3, 4, 5]:
        try:
            with open(filepath, 'rb') as f:
                # Try to load with specific protocol
                unpickler = pickle.Unpickler(f)
                if protocol is not None:
                    unpickler.protocol_version = protocol
                data = unpickler.load()
            print(f"✓ Loaded with pickle protocol {protocol}: {type(data)}")
            return data
        except Exception as e:
            continue
    
    # Method 4: Try to find CSV fallback
    if fallback_to_csv:
        csv_path = filepath.replace('.pkl', '.csv')
        if os.path.exists(csv_path):
            try:
                data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                print(f"✓ Loaded CSV fallback: {csv_path}")
                return data
            except Exception as e:
                print(f"✗ CSV fallback failed: {e}")
    
    raise Exception(f"Could not load {filepath} with any method")


def save_compatible_pickle(data: Union[pd.DataFrame, Any], filepath: str, protocol: int = None) -> None:
    """
    Save data as pickle with maximum compatibility.
    
    Args:
        data: Data to save (usually DataFrame)
        filepath: Output file path
        protocol: Pickle protocol version (None for default)
    """
    # Use protocol 4 for good compatibility (Python 3.4+)
    if protocol is None:
        protocol = 4
        
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=protocol)
        print(f"✓ Saved compatible pickle: {filepath} (protocol {protocol})")
        
        # Also save as CSV for maximum compatibility
        if isinstance(data, pd.DataFrame):
            csv_path = filepath.replace('.pkl', '.csv')
            data.to_csv(csv_path)
            print(f"✓ Saved CSV backup: {csv_path}")
            
    except Exception as e:
        print(f"✗ Failed to save {filepath}: {e}")
        raise


def analyze_pickle_file(filepath: str) -> Dict[str, Any]:
    """
    Analyze a pickle file without fully loading it (for debugging).
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Dictionary with file analysis information
    """
    info = {
        'filepath': filepath,
        'size_bytes': os.path.getsize(filepath),
        'size_mb': os.path.getsize(filepath) / 1024**2,
        'loadable': False,
        'data_type': None,
        'pandas_version_issue': False
    }
    
    # Try to peek at the pickle content
    try:
        with open(filepath, 'rb') as f:
            # Read just the first few bytes to get protocol info
            header = f.read(10)
            f.seek(0)
            
            # Try to load
            data = pickle.load(f)
            info['loadable'] = True
            info['data_type'] = str(type(data))
            
            if isinstance(data, pd.DataFrame):
                info['shape'] = data.shape
                info['columns'] = list(data.columns)[:10]  # First 10 columns
                info['total_columns'] = len(data.columns)
                info['index_type'] = str(type(data.index))
                
    except Exception as e:
        info['error'] = str(e)
        if '_unpickle_block' in str(e) or 'pandas._libs' in str(e):
            info['pandas_version_issue'] = True
    
    return info


def convert_legacy_pickles(input_dir: str = '.', output_dir: str = None) -> Dict[str, str]:
    """
    Convert legacy pickle files to compatible format.
    
    Args:
        input_dir: Directory containing legacy pickle files
        output_dir: Directory to save converted files (None = same as input)
        
    Returns:
        Dictionary mapping original files to converted files
    """
    if output_dir is None:
        output_dir = input_dir
        
    converted_files = {}
    pkl_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    
    print(f"Found {len(pkl_files)} pickle files to analyze")
    
    for pkl_file in pkl_files:
        input_path = os.path.join(input_dir, pkl_file)
        output_path = os.path.join(output_dir, pkl_file.replace('.pkl', '_compatible.pkl'))
        
        print(f"\nProcessing: {pkl_file}")
        
        # Analyze the file
        info = analyze_pickle_file(input_path)
        print(f"File info: {info['size_mb']:.2f} MB, Type: {info.get('data_type', 'Unknown')}")
        
        if info['pandas_version_issue']:
            print("⚠ Detected pandas version compatibility issue")
            
            # Try alternative loading methods
            try:
                data = load_legacy_pickle(input_path)
                save_compatible_pickle(data, output_path)
                converted_files[pkl_file] = output_path
                
            except Exception as e:
                print(f"✗ Could not convert {pkl_file}: {e}")
                
        elif info['loadable']:
            print("✓ File loads correctly with current pandas version")
            # Still create a compatible version
            try:
                data = load_legacy_pickle(input_path)
                save_compatible_pickle(data, output_path)
                converted_files[pkl_file] = output_path
            except Exception as e:
                print(f"✗ Unexpected error: {e}")
        else:
            print(f"✗ Cannot load file: {info.get('error', 'Unknown error')}")
    
    return converted_files


def main():
    """Analyze and convert legacy pickle files."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FETCH Pickle Compatibility Tool')
    parser.add_argument('--analyze', type=str, help='Analyze a specific pickle file')
    parser.add_argument('--convert_dir', type=str, default='.', help='Convert all pickle files in directory')
    parser.add_argument('--output_dir', type=str, help='Output directory for converted files')
    
    args = parser.parse_args()
    
    if args.analyze:
        info = analyze_pickle_file(args.analyze)
        print("=== Pickle File Analysis ===")
        for key, value in info.items():
            print(f"{key}: {value}")
            
    else:
        print("=== Converting Legacy Pickle Files ===")
        converted = convert_legacy_pickles(args.convert_dir, args.output_dir)
        
        print(f"\n=== Conversion Complete ===")
        print(f"Converted {len(converted)} files:")
        for original, converted_path in converted.items():
            print(f"  {original} → {os.path.basename(converted_path)}")


if __name__ == "__main__":
    main()