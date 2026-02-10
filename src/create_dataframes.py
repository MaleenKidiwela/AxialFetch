"""
Main pipeline: reproduce the Create_Dataframes.ipynb workflow.

Outputs stored as pickle files in src/output/:
  - combined_df4.pkl  (Northern station / 2504)
  - combined_df3.pkl  (Western station / 2503)
  - combined_df2.pkl  (Eastern station / 2502)
  - harmonic_mean_dfs.pkl  (SSP-based harmonic mean velocities between station pairs)
  - baseline_perturb.pkl  (tilt-induced baseline perturbations)

Usage:
    python -m src.create_dataframes
"""
import os
import pickle
import pandas as pd

from .config import OUTPUT_DIR
from .data_loader import load_all_data
from .salinity import compute_corrected_salinity
from .station_data import (
    build_combined_df4,
    build_combined_df3,
    build_combined_df2,
    build_inc_dataframes,
)
from .tidal import load_tidal_df, apply_tidal_correction
from .velocity import add_velocities
from .pressure import apply_pressure_postprocessing
from .harmonic import build_final_harmonic_mean_dfs
from .tilt import compute_baseline_perturb


def main():
    # ------------------------------------------------------------------
    # Step 1: Load FETCH CSV data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading FETCH instrument data...")
    print("=" * 60)
    df_dict, data_extracted, result_dfs = load_all_data()

    # ------------------------------------------------------------------
    # Step 2: Salinity correction
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Computing corrected salinity...")
    print("=" * 60)
    salinity_df = compute_corrected_salinity()

    # ------------------------------------------------------------------
    # Step 3: Build combined DataFrames (pressure + temperature + salinity)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Building combined station DataFrames...")
    print("=" * 60)
    combined_df4 = build_combined_df4(data_extracted, salinity_df)
    combined_df3 = build_combined_df3(data_extracted, salinity_df)
    combined_df2 = build_combined_df2(data_extracted, salinity_df)

    # Build INC dataframes and compute baseline perturbations
    INC_df4, INC_df3, INC_df2 = build_inc_dataframes(data_extracted)
    baseline_perturb = compute_baseline_perturb(INC_df4, INC_df3, INC_df2)

    print(f"  combined_df4 (Northern/2504): {combined_df4.shape}")
    print(f"  combined_df3 (Western/2503):  {combined_df3.shape}")
    print(f"  combined_df2 (Eastern/2502):  {combined_df2.shape}")
    print(f"  baseline_perturb:             {baseline_perturb.shape}, "
          f"columns={list(baseline_perturb.columns)}")

    # ------------------------------------------------------------------
    # Step 4: Tidal pressure correction
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Applying tidal pressure correction...")
    print("=" * 60)
    tidal_df = load_tidal_df()

    combined_df4 = apply_tidal_correction(
        combined_df4, tidal_df, adjusted_col='Adjusted Value'
    )
    combined_df3 = apply_tidal_correction(
        combined_df3, tidal_df, adjusted_col='Adjusted Value3'
    )
    combined_df2 = apply_tidal_correction(
        combined_df2, tidal_df, adjusted_col='Adjusted Value2'
    )

    print(f"  combined_df4 columns: {list(combined_df4.columns)}")
    print(f"  combined_df3 columns: {list(combined_df3.columns)}")
    print(f"  combined_df2 columns: {list(combined_df2.columns)}")

    # ------------------------------------------------------------------
    # Step 5: Velocity computation (interpolated SSP + TEOS-10)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Computing velocities...")
    print("=" * 60)
    combined_df4 = add_velocities(combined_df4, result_dfs, '2504')
    combined_df3 = add_velocities(combined_df3, result_dfs, '2503')
    combined_df2 = add_velocities(combined_df2, result_dfs, '2502')

    print(f"  combined_df4 velocity range: "
          f"{combined_df4['velocity'].min():.2f} - {combined_df4['velocity'].max():.2f}")

    # ------------------------------------------------------------------
    # Step 6: Pressure post-processing (demeaning + 15-day MA)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Pressure post-processing...")
    print("=" * 60)
    combined_df4, combined_df3, combined_df2 = apply_pressure_postprocessing(
        combined_df4, combined_df3, combined_df2
    )

    # ------------------------------------------------------------------
    # Step 7: Harmonic mean computation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 7: Computing harmonic means...")
    print("=" * 60)
    harmonic_mean_dfs, harmonic_df_dict = build_final_harmonic_mean_dfs(
        result_dfs, combined_df4, combined_df3, combined_df2
    )

    for key, df in harmonic_mean_dfs.items():
        print(f"  harmonic_mean_dfs['{key}']: {df.shape}, columns={list(df.columns)}")

    # ------------------------------------------------------------------
    # Step 8: Store outputs
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 8: Storing output DataFrames...")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set Record Time back as index for combined dfs (matching notebook Cell 23 state)
    def set_record_time_index(df):
        df = df.copy()
        if 'Record Time' in df.columns:
            df['Record Time'] = pd.to_datetime(df['Record Time'])
            df.set_index('Record Time', inplace=True)
        return df

    combined_df4_out = set_record_time_index(combined_df4)
    combined_df3_out = set_record_time_index(combined_df3)
    combined_df2_out = set_record_time_index(combined_df2)

    combined_df4_out.to_pickle(os.path.join(OUTPUT_DIR, "combined_df4.pkl"))
    combined_df3_out.to_pickle(os.path.join(OUTPUT_DIR, "combined_df3.pkl"))
    combined_df2_out.to_pickle(os.path.join(OUTPUT_DIR, "combined_df2.pkl"))

    with open(os.path.join(OUTPUT_DIR, "harmonic_mean_dfs.pkl"), 'wb') as f:
        pickle.dump(harmonic_mean_dfs, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(OUTPUT_DIR, "harmonic_df_dict.pkl"), 'wb') as f:
        pickle.dump(harmonic_df_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    baseline_perturb.to_pickle(os.path.join(OUTPUT_DIR, "baseline_perturb.pkl"))

    print(f"\n  Outputs saved to: {OUTPUT_DIR}/")
    print(f"    - combined_df4.pkl  (Northern station / 2504)")
    print(f"    - combined_df3.pkl  (Western station / 2503)")
    print(f"    - combined_df2.pkl  (Eastern station / 2502)")
    print(f"    - harmonic_mean_dfs.pkl  (SSP-based, column 'Harmonic Mean')")
    print(f"    - harmonic_df_dict.pkl   (TEOS-10 velocity-based, column 'HMean')")
    print(f"    - baseline_perturb.pkl")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)
    print(f"\ncombined_df4 (Northern): {combined_df4.shape}")
    print(combined_df4.head(3))
    print(f"\ncombined_df3 (Western): {combined_df3.shape}")
    print(combined_df3.head(3))
    print(f"\ncombined_df2 (Eastern): {combined_df2.shape}")
    print(combined_df2.head(3))
    print(f"\nharmonic_mean_dfs keys: {list(harmonic_mean_dfs.keys())}")
    print(f"\nbaseline_perturb: {baseline_perturb.shape}")
    print(baseline_perturb.head(3))

    return combined_df4, combined_df3, combined_df2, harmonic_mean_dfs, baseline_perturb


if __name__ == "__main__":
    main()
