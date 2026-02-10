"""
Range calculation pipeline: reproduce the Calculate_Range.ipynb workflow.

Loads pre-computed pickles from src/output/ (produced by create_dataframes.py),
extracts baseline range data from raw FETCH CSVs, computes distances using
harmonic-mean velocities, applies tilt and perturbation corrections, and
generates all figures.

Outputs:
  - src/output/distances/*.csv            (6 distance CSVs)
  - src/output/distances/*_corrected.csv  (6 corrected CSVs)
  - src/output/figures/*.png              (all figures)

Usage:
    python -m src.calculate_range
"""
import os
import pickle
import pandas as pd

from .config import OUTPUT_DIR, DISTANCES_DIR, FIGURES_DIR
from .data_loader import load_all_data
from .range_filter import build_all_baselines
from .range_correction import save_distance_csvs, apply_perturbation_correction
from .range_figures import generate_all_figures


def _load_pickles():
    """Load pre-computed pickles from src/output/."""
    baseline_perturb = pd.read_pickle(
        os.path.join(OUTPUT_DIR, "baseline_perturb.pkl")
    )
    harmonic_df_dict = pd.read_pickle(
        os.path.join(OUTPUT_DIR, "harmonic_df_dict.pkl")
    )
    with open(os.path.join(OUTPUT_DIR, "harmonic_mean_dfs.pkl"), 'rb') as f:
        harmonic_mean_dfs = pickle.load(f)

    return baseline_perturb, harmonic_df_dict, harmonic_mean_dfs


def main():
    # ------------------------------------------------------------------
    # Step 1: Load raw data (for BSL extraction)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading FETCH instrument data...")
    print("=" * 60)
    df_dict, _data_extracted, _result_dfs = load_all_data()

    # ------------------------------------------------------------------
    # Step 2: Load pre-computed pickles
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Loading pre-computed pickles...")
    print("=" * 60)
    baseline_perturb, harmonic_df_dict, harmonic_mean_dfs = _load_pickles()
    print(f"  baseline_perturb: {baseline_perturb.shape}")
    print(f"  harmonic_df_dict keys: {list(harmonic_df_dict.keys())}")
    print(f"  harmonic_mean_dfs keys: {list(harmonic_mean_dfs.keys())}")

    # ------------------------------------------------------------------
    # Step 3: Build all 6 directed baselines
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Extracting BSL data and computing distances...")
    print("=" * 60)
    baselines = build_all_baselines(
        df_dict, harmonic_df_dict, harmonic_mean_dfs, baseline_perturb
    )

    for name, df in baselines.items():
        print(f"  {name}: {df.shape[0]} measurements, "
              f"columns={list(df.columns)}")

    # ------------------------------------------------------------------
    # Step 4: Save distance CSVs
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Saving distance CSVs...")
    print("=" * 60)
    os.makedirs(DISTANCES_DIR, exist_ok=True)
    save_distance_csvs(baselines, DISTANCES_DIR)

    # ------------------------------------------------------------------
    # Step 5: Apply perturbation correction
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Applying perturbation corrections...")
    print("=" * 60)
    apply_perturbation_correction(
        baseline_perturb, DISTANCES_DIR, DISTANCES_DIR
    )

    # ------------------------------------------------------------------
    # Step 6: Generate figures
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Generating figures...")
    print("=" * 60)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    generate_all_figures(baselines, DISTANCES_DIR, FIGURES_DIR)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)
    print(f"\nDistance CSVs:     {DISTANCES_DIR}/")
    print(f"Corrected CSVs:   {DISTANCES_DIR}/*_corrected.csv")
    print(f"Figures:          {FIGURES_DIR}/")

    return baselines


if __name__ == "__main__":
    main()
