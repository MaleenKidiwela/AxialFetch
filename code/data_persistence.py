"""
Data Persistence Module for FETCH StreamLine Analysis

This module handles saving and loading of processed data using various formats
including pickle files (for Python objects), CSV (for interoperability), 
and HDF5 (for large datasets with metadata).
"""

import os
from typing import Dict, Any, Optional, Union
import pandas as pd
import pickle
import logging

logger = logging.getLogger(__name__)


def save_dataframe_as_pickle(df: pd.DataFrame, filepath: str, metadata: Dict[str, Any] = None) -> None:
    """
    Save DataFrame as pickle file with optional metadata.
    
    Args:
        df: DataFrame to save
        filepath: Path to save the pickle file
        metadata: Optional metadata dictionary to store alongside the DataFrame
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if metadata:
            # Store DataFrame and metadata together
            data_package = {
                'dataframe': df,
                'metadata': metadata,
                'version': '1.0',
                'created_with': 'FETCH StreamLine Analysis Package'
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data_package, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # Just save the DataFrame directly (compatible with original format)
            df.to_pickle(filepath)
            
        logger.info(f"Successfully saved DataFrame to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {filepath}: {e}")
        raise


def load_dataframe_from_pickle(filepath: str) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Load DataFrame from pickle file, handling both direct DataFrame pickles 
    and packaged data with metadata.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        DataFrame if it's a direct pickle, or dictionary with 'dataframe' and 'metadata' keys
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, pd.DataFrame):
            # Direct DataFrame pickle (original format)
            logger.info(f"Loaded DataFrame from {filepath} (original format)")
            return data
        elif isinstance(data, dict) and 'dataframe' in data:
            # Packaged format with metadata
            logger.info(f"Loaded packaged data from {filepath} with metadata")
            return data
        else:
            # Try to treat as DataFrame anyway
            logger.warning(f"Unexpected data type in {filepath}, attempting to use as DataFrame")
            return data
            
    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {e}")
        raise


def save_processed_results(
    results: Dict[str, Any], 
    output_dir: str = '.',
    save_formats: list = ['pickle', 'csv']
) -> Dict[str, str]:
    """
    Save processed analysis results in multiple formats.
    
    Args:
        results: Dictionary containing analysis results with DataFrames
        output_dir: Directory to save files
        save_formats: List of formats to save ('pickle', 'csv', 'hdf5')
        
    Returns:
        Dictionary mapping result names to saved file paths
    """
    saved_files = {}
    
    for result_name, result_data in results.items():
        if isinstance(result_data, pd.DataFrame):
            # Save DataFrame in requested formats
            for fmt in save_formats:
                if fmt == 'pickle':
                    pickle_path = os.path.join(output_dir, f"{result_name}.pkl")
                    save_dataframe_as_pickle(result_data, pickle_path)
                    saved_files[f"{result_name}_pickle"] = pickle_path
                    
                elif fmt == 'csv':
                    csv_path = os.path.join(output_dir, f"{result_name}.csv")
                    result_data.to_csv(csv_path, index=True)
                    saved_files[f"{result_name}_csv"] = csv_path
                    logger.info(f"Saved CSV to {csv_path}")
                    
                elif fmt == 'hdf5':
                    hdf5_path = os.path.join(output_dir, f"{result_name}.h5")
                    result_data.to_hdf(hdf5_path, key='data', mode='w', format='table')
                    saved_files[f"{result_name}_hdf5"] = hdf5_path
                    logger.info(f"Saved HDF5 to {hdf5_path}")
                    
        elif isinstance(result_data, dict):
            # Save dictionary as pickle
            if 'pickle' in save_formats:
                pickle_path = os.path.join(output_dir, f"{result_name}.pkl")
                with open(pickle_path, 'wb') as f:
                    pickle.dump(result_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                saved_files[f"{result_name}_pickle"] = pickle_path
                logger.info(f"Saved dictionary to {pickle_path}")
                
    return saved_files


def save_baseline_data(
    baseline_data: Dict[str, pd.DataFrame], 
    output_dir: str = '.',
    instrument_pairs: list = None
) -> Dict[str, str]:
    """
    Save baseline/range data in the original FETCH format.
    
    Args:
        baseline_data: Dictionary mapping baseline names to DataFrames
        output_dir: Directory to save files
        instrument_pairs: List of instrument pair identifiers (e.g., ['2502_2503', '2502_2504'])
        
    Returns:
        Dictionary mapping baseline names to saved file paths
    """
    saved_files = {}
    
    for baseline_name, df in baseline_data.items():
        # Clean the DataFrame (remove empty columns like original)
        cleaned_df = df.dropna(axis=1, how='all')
        
        # Generate filename in original format (R{baseline}.pkl)
        if baseline_name.startswith('R'):
            filename = f"{baseline_name}.pkl"
        else:
            filename = f"R{baseline_name}.pkl"
            
        filepath = os.path.join(output_dir, filename)
        
        # Save as pickle (matching original format)
        cleaned_df.to_pickle(filepath)
        saved_files[baseline_name] = filepath
        
        logger.info(f"Saved baseline data for {baseline_name} to {filepath}")
        
    return saved_files


def save_instrument_data(
    instrument_data: Dict[str, pd.DataFrame],
    output_dir: str = '.'
) -> Dict[str, str]:
    """
    Save combined instrument data (like the original 2504.pkl format).
    
    Args:
        instrument_data: Dictionary mapping instrument IDs to combined DataFrames
        output_dir: Directory to save files
        
    Returns:
        Dictionary mapping instrument IDs to saved file paths
    """
    saved_files = {}
    
    for instrument_id, df in instrument_data.items():
        filename = f"{instrument_id}.pkl"
        filepath = os.path.join(output_dir, filename)
        
        # Save as pickle (matching original format)
        df.to_pickle(filepath)
        saved_files[instrument_id] = filepath
        
        logger.info(f"Saved combined data for instrument {instrument_id} to {filepath}")
        
    return saved_files


def load_existing_pickles(data_dir: str = '.') -> Dict[str, pd.DataFrame]:
    """
    Load all existing pickle files from a directory.
    
    Args:
        data_dir: Directory containing pickle files
        
    Returns:
        Dictionary mapping file names (without .pkl) to DataFrames
    """
    loaded_data = {}
    
    # Find all .pkl files
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    
    for pkl_file in pkl_files:
        filepath = os.path.join(data_dir, pkl_file)
        basename = os.path.splitext(pkl_file)[0]
        
        try:
            data = load_dataframe_from_pickle(filepath)
            if isinstance(data, pd.DataFrame):
                loaded_data[basename] = data
            elif isinstance(data, dict) and 'dataframe' in data:
                loaded_data[basename] = data['dataframe']
            else:
                loaded_data[basename] = data
                
            logger.info(f"Loaded {pkl_file}")
            
        except Exception as e:
            logger.warning(f"Could not load {pkl_file}: {e}")
            
    return loaded_data


def create_analysis_summary(
    processed_data: Dict[str, Any],
    output_dir: str = '.'
) -> str:
    """
    Create a summary report of the processed data and save it.
    
    Args:
        processed_data: Dictionary containing all processed results
        output_dir: Directory to save the summary
        
    Returns:
        Path to the saved summary file
    """
    summary_lines = [
        "# FETCH StreamLine Analysis Summary",
        f"Generated on: {pd.Timestamp.now()}",
        "",
        "## Processed Datasets:",
        ""
    ]
    
    for name, data in processed_data.items():
        if isinstance(data, pd.DataFrame):
            summary_lines.extend([
                f"### {name}",
                f"- Shape: {data.shape}",
                f"- Columns: {list(data.columns)}",
                f"- Date range: {data.index.min()} to {data.index.max()}" if hasattr(data.index, 'min') else "- Index type: non-datetime",
                f"- Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                ""
            ])
        elif isinstance(data, dict):
            summary_lines.extend([
                f"### {name} (Dictionary)",
                f"- Keys: {list(data.keys())}",
                ""
            ])
        else:
            summary_lines.extend([
                f"### {name}",
                f"- Type: {type(data).__name__}",
                ""
            ])
    
    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(output_dir, "analysis_summary.md")
    
    with open(summary_path, 'w') as f:
        f.write(summary_text)
        
    logger.info(f"Analysis summary saved to {summary_path}")
    return summary_path


def cleanup_temp_files(output_dir: str, keep_formats: list = ['pickle']) -> None:
    """
    Clean up temporary files, keeping only specified formats.
    
    Args:
        output_dir: Directory to clean
        keep_formats: List of file extensions to keep (without dots)
    """
    extensions_to_keep = [f".{fmt}" for fmt in keep_formats]
    
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        if os.path.isfile(filepath):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in extensions_to_keep and file_ext in ['.csv', '.h5', '.hdf5']:
                try:
                    os.remove(filepath)
                    logger.info(f"Removed temporary file: {filename}")
                except Exception as e:
                    logger.warning(f"Could not remove {filename}: {e}")