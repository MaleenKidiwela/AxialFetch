"""
Pipeline Orchestration Module for FETCH Range Calculation

Main workflow coordination for the complete analysis from raw data
to publication-ready outputs.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path

from .config import PipelineConfig, STATIONS, BASELINE_PAIRS, load_config
from .data_loading import (
    create_nested_dictionary,
    build_combined_dataframe,
    extract_inclinometer_data,
    extract_baseline_data,
    extract_sound_speed_data,
    ensure_datetime,
)
from .salinity import correct_salinity_drift, get_default_additional_gaps
from .pressure import load_tidal_predictions, remove_tidal_signal, load_all_node_pressures
from .velocity import (
    velocity_timeseries_teos10,
    interpolate_velocity,
    add_velocity_to_combined_df,
)
from .range_calculation import (
    compute_all_harmonic_velocities,
    build_baseline_perturbation,
    calculate_baseline_distances,
    apply_tilt_correction,
    save_distance_csv,
)
from .triangle import compute_triangle_metrics
from .figures import create_all_figures


def validate_inputs(config: PipelineConfig) -> List[str]:
    """
    Check all required input files exist.

    Args:
        config: Pipeline configuration

    Returns:
        List of missing file paths (empty if all exist)
    """
    missing = []

    # Check station files
    for filepath in config.station_files:
        if not Path(filepath).exists():
            missing.append(filepath)

    # Check CTD file
    if not Path(config.ctd_salinity_path).exists():
        missing.append(config.ctd_salinity_path)

    # Check tidal files
    for filepath in config.tidal_prediction_paths:
        if not Path(filepath).exists():
            missing.append(filepath)

    return missing


def run_full_pipeline(
    config: Optional[PipelineConfig] = None,
    config_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Execute complete analysis pipeline.

    Args:
        config: Pipeline configuration object
        config_path: Path to YAML configuration file
        verbose: Print progress messages

    Returns:
        Dictionary containing all analysis results
    """
    # Load configuration
    if config is None:
        config = load_config(config_path)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if verbose:
        print("=" * 60)
        print("FETCH Range Calculation Pipeline")
        print("=" * 60)

    # --- Step 1: Load raw station data ---
    if verbose:
        print("\n[1/9] Loading station data...")

    df_dict = create_nested_dictionary(config.station_files)
    results["df_dict"] = df_dict

    # Extract sound speed data
    result_dfs = {}
    for station_id in ["2502", "2503", "2504"]:
        try:
            ssp_df = extract_sound_speed_data(df_dict, station_id)
            result_dfs[station_id] = ssp_df
        except ValueError:
            if verbose:
                print(f"  Warning: No SSP data for station {station_id}")

    # Note: Station 2503 velocimeter was not working
    if "2503" in result_dfs:
        result_dfs["2503"]["SoundSpeed"] = np.nan

    results["result_dfs"] = result_dfs

    # --- Step 2: Salinity correction ---
    if verbose:
        print("\n[2/9] Correcting salinity...")

    salinity_df = correct_salinity_drift(
        filepath=config.ctd_salinity_path,
        bottle_times=config.bottle_times,
        bottle_values=config.bottle_values,
        gap_start=config.salinity_gap_start,
        gap_end=config.salinity_gap_end,
        start_time=config.start_time,
        end_time=config.end_time,
        additional_gaps=get_default_additional_gaps(),
        verbose=verbose,
    )
    results["salinity_df"] = salinity_df

    # --- Step 3: Build combined DataFrames ---
    if verbose:
        print("\n[3/9] Building combined datasets...")

    combined_dfs = {}
    temp_corrections = {"2502": 0.0, "2503": 0.0, "2504": config.temp_correction_2504}

    for station_id, temp_corr in temp_corrections.items():
        combined_dfs[station_id] = build_combined_dataframe(
            df_dict, station_id, salinity_df, temp_correction=temp_corr
        )

    results["combined_dfs"] = combined_dfs

    # --- Step 4: Tidal correction ---
    if verbose:
        print("\n[4/9] Applying tidal correction...")

    tidal_df = load_tidal_predictions(config.tidal_prediction_paths)

    for station_id in combined_dfs:
        combined_dfs[station_id], tidal_params = remove_tidal_signal(
            combined_dfs[station_id], tidal_df
        )
        if verbose:
            print(f"  Station {station_id}: amplitude={tidal_params[0]:.4f}, rho={tidal_params[1]:.1f}")

    # --- Step 5: Calculate velocities ---
    if verbose:
        print("\n[5/9] Calculating TEOS-10 velocities...")

    for station_id in combined_dfs:
        df = combined_dfs[station_id]

        # Calculate TEOS-10 velocity
        T_series = np.array(df["Temperature Deg C"])
        P_series = np.array(df["Pressure (kPa)"])
        S_series = np.array(df["Salinity"])

        velocities = velocity_timeseries_teos10(T_series, P_series, S_series)
        df["velocity"] = velocities

        # Interpolate recorded velocity if available
        if station_id in result_dfs and not result_dfs[station_id]["SoundSpeed"].isna().all():
            df["interp_v"] = interpolate_velocity(df["Record Time"], result_dfs[station_id])

    # Compute harmonic velocities
    harmonic_dfs = compute_all_harmonic_velocities(combined_dfs)
    results["harmonic_dfs"] = harmonic_dfs

    # --- Step 6: Build tilt corrections ---
    if verbose:
        print("\n[6/9] Computing tilt corrections...")

    inc_dfs = {}
    for station_id in ["2502", "2503", "2504"]:
        inc_dfs[station_id] = extract_inclinometer_data(df_dict, station_id)

    # Only compute 3 perturbation pairs (matching notebook behavior)
    # The notebook uses these 3 pairs for all 6 baselines (bidirectional)
    baseline_pairs = [
        ("2504", "2502"),  # Central-East: used for CE and EC
        ("2504", "2503"),  # Central-West: used for CW and WC
        ("2502", "2503"),  # East-West: used for EW and WE
    ]
    baseline_perturb = build_baseline_perturbation(inc_dfs, STATIONS, baseline_pairs)
    results["baseline_perturb"] = baseline_perturb

    # --- Step 7: Calculate baseline distances ---
    if verbose:
        print("\n[7/9] Calculating baseline distances...")

    distance_dfs = {}

    # Mapping from (source, target) to the tilt correction tag used in notebook
    # The notebook uses the same perturbation column for bidirectional baselines
    TILT_CORRECTION_TAGS = {
        ("2504", "2502"): "2504-2502",  # CE uses 2504-2502
        ("2504", "2503"): "2504-2503",  # CW uses 2504-2503
        ("2502", "2504"): "2504-2502",  # EC uses 2504-2502 (same as CE)
        ("2502", "2503"): "2502-2503",  # EW uses 2502-2503
        ("2503", "2502"): "2502-2503",  # WE uses 2502-2503 (same as EW)
        ("2503", "2504"): "2504-2503",  # WC uses 2504-2503 (same as CW)
    }

    # Define all baselines to calculate
    # Filtering is now handled by BASELINE_FILTERS in range_calculation.py
    baselines = [
        ("2504", "2502", "2502_2504"),  # Central -> East (IQR 0.15, 0.85)
        ("2504", "2503", "2503_2504"),  # Central -> West (hard filter >= 2618)
        ("2502", "2504", "2502_2504"),  # East -> Central (hard filter >= 2400)
        ("2502", "2503", "2502_2503"),  # East -> West (hard filter 4636.2-4638)
        ("2503", "2502", "2502_2503"),  # West -> East (IQR 0.25, 0.75)
        ("2503", "2504", "2503_2504"),  # West -> Central (IQR 0.25, 0.75)
    ]

    for source, target, harmonic_key in baselines:
        try:
            bsl_df = df_dict[source]["BSL"].copy()
            harmonic_df = harmonic_dfs.get(harmonic_key)

            if harmonic_df is None:
                # Try reversed key
                reversed_key = "_".join(harmonic_key.split("_")[::-1])
                harmonic_df = harmonic_dfs.get(reversed_key)

            if harmonic_df is not None:
                # Use baseline-specific filtering from BASELINE_FILTERS config
                dist_df = calculate_baseline_distances(
                    bsl_df, harmonic_df, source, target,
                    use_baseline_config=True,
                )

                # Apply tilt correction using notebook's tag mapping
                baseline_tag = TILT_CORRECTION_TAGS[(source, target)]
                dist_df = apply_tilt_correction(dist_df, baseline_tag, baseline_perturb)

                key = f"{source}_{target}"
                distance_dfs[key] = dist_df

                if verbose:
                    print(f"  {source} -> {target}: {len(dist_df)} measurements")
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed {source} -> {target}: {e}")

    results["distance_dfs"] = distance_dfs

    # Save distance CSVs
    output_names = {
        "2504_2502": "CE_distance_corrected.csv",
        "2504_2503": "CW_distance_corrected.csv",
        "2502_2504": "EC_distance_corrected.csv",
        "2502_2503": "EW_distance_corrected.csv",
        "2503_2502": "WE_distance_corrected.csv",
        "2503_2504": "WC_distance_corrected.csv",
    }

    for key, filename in output_names.items():
        if key in distance_dfs:
            save_distance_csv(
                distance_dfs[key],
                str(output_dir / filename),
                exclude_first_days=config.exclude_first_days,
            )

    # --- Step 8: Calculate triangle area ---
    if verbose:
        print("\n[8/9] Computing triangle area...")

    try:
        # Need West-Central, West-East, Central-East
        # Use the bidirectional baselines
        wc_key = "2504_2503" if "2504_2503" in distance_dfs else "2503_2504"
        we_key = "2502_2503" if "2502_2503" in distance_dfs else "2503_2502"
        ce_key = "2504_2502" if "2504_2502" in distance_dfs else "2502_2504"

        wc_df = distance_dfs.get(wc_key)
        we_df = distance_dfs.get(we_key)
        ce_df = distance_dfs.get(ce_key)

        if wc_df is not None and we_df is not None and ce_df is not None:
            # Prepare for triangle calculation
            wc_prep = pd.DataFrame({
                "Record Time": wc_df["Record Time"].values,
                "Distance (cm)": (wc_df["Calculated Distance (m)"] * 100).values,
            })
            we_prep = pd.DataFrame({
                "Record Time": we_df["Record Time"].values,
                "Distance (cm)": (we_df["Calculated Distance (m)"] * 100).values,
            })
            ce_prep = pd.DataFrame({
                "Record Time": ce_df["Record Time"].values,
                "Distance (cm)": (ce_df["Calculated Distance (m)"] * 100).values,
            })

            triangle_df = compute_triangle_metrics(
                wc_prep, we_prep, ce_prep,
                exclude_first_days=config.exclude_first_days,
                output_path=str(output_dir / "triangle_area_timeseries.csv"),
            )
            results["triangle_df"] = triangle_df
        else:
            if verbose:
                missing = []
                if wc_df is None:
                    missing.append("WC")
                if we_df is None:
                    missing.append("WE")
                if ce_df is None:
                    missing.append("CE")
                print(f"  Warning: Missing baselines for triangle: {missing}")
    except Exception as e:
        if verbose:
            print(f"  Warning: Triangle calculation failed: {e}")

    # --- Step 9: Load node pressure data ---
    if verbose:
        print("\n[9/9] Loading node pressure data...")

    try:
        node_dfs = load_all_node_pressures(config.node_pressure_pattern)
        results["node_dfs"] = node_dfs
    except Exception as e:
        if verbose:
            print(f"  Warning: Node pressure loading failed: {e}")

    # --- Generate figures ---
    if verbose:
        print("\n[*] Generating figures...")

    try:
        generated_figures = create_all_figures(results, str(output_dir))
        results["figures"] = generated_figures
    except Exception as e:
        if verbose:
            print(f"  Warning: Figure generation failed: {e}")

    if verbose:
        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print(f"Output directory: {output_dir}")
        print("=" * 60)

    return results


def run_quick_test(config: Optional[PipelineConfig] = None) -> bool:
    """
    Run a quick test to verify the pipeline can load data.

    Args:
        config: Pipeline configuration

    Returns:
        True if test passes, False otherwise
    """
    if config is None:
        config = PipelineConfig()

    print("Running quick pipeline test...")

    # Check files exist
    missing = validate_inputs(config)
    if missing:
        print(f"Missing files: {missing}")
        return False

    # Try loading station data
    try:
        df_dict = create_nested_dictionary(config.station_files)
        print(f"  Loaded {len(df_dict)} stations")

        for station_id, data in df_dict.items():
            print(f"    Station {station_id}: {list(data.keys())}")

        print("Quick test passed!")
        return True
    except Exception as e:
        print(f"Quick test failed: {e}")
        return False
