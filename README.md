<div align="center">

# AxialFetch

**Seafloor Geodetic Monitoring at Axial Seamount**

*Processing multi-year acoustic ranging data from cabled FETCH instruments to measure tectonic baseline distances on the Juan de Fuca Ridge*

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?logo=plotly&logoColor=white)
![GSW](https://img.shields.io/badge/GSW-TEOS--10-0077B6)

</div>

---

## Overview

AxialFetch processes pressure, temperature, salinity, sound speed, and acoustic travel-time time series from three ocean-bottom stations at the summit caldera of Axial Seamount. The two-stage pipeline produces calibrated baseline distance measurements between all station pairs, enabling detection of volcanic deformation at centimeter-scale precision.

```
         Northern (2504)
         /            \
    NW  /              \  NE
       /                \
 Western (2503) ---- Eastern (2502)
           WE / EW
```

### Station Network

| Station | ID | Coordinates | Caldera Position |
|:--------|:--:|:------------|:-----------------|
| **Northern** | 2504 | 45.959 N, 130.011 W | North side |
| **Western** | 2503 | 45.947 N, 130.027 W | West side |
| **Eastern** | 2502 | 45.963 N, 129.991 W | East side |

### Acoustic Baselines

| Baseline | Direction | Length |
|:---------|:----------|-------:|
| **NW** / **WN** | North &harr; West | 1.765 km |
| **NE** / **EN** | North &harr; East | 1.642 km |
| **WE** / **EW** | West &harr; East | 3.260 km |

---

## Quick Start

```bash
# Step 1 — Build calibrated station DataFrames and harmonic velocities
python -m src.create_dataframes

# Step 2 — Compute distances, apply corrections, generate figures
python -m src.calculate_range
```

> Pipeline 1 must complete before Pipeline 2 can run (Pipeline 2 loads pickle files produced by Pipeline 1).

---

## Processing Pipelines

### Pipeline 1 &mdash; `create_dataframes`

Ingests raw instrument data and produces calibrated, corrected per-station DataFrames.

<details>
<summary><b>Processing Steps</b></summary>

| Step | Description |
|:-----|:------------|
| 1. Load FETCH CSVs | Parse multi-code instrument files (TMP, DQZ, SSP, BSL, INC) for all three stations |
| 2. Salinity correction | Align CTD salinity from OOI cabled array (MJ03F), apply calibration residuals, correct data gaps via mean-shift |
| 3. Build combined DataFrames | Merge pressure, temperature (+0.351 &deg;C offset for station 2504), and salinity per station |
| 4. Tidal pressure correction | Fit tidal predictions via least-squares optimization, remove tidal signal from pressure |
| 5. Velocity computation | Compute two sound speed estimates: **SSP** (onboard velocimeter) and **TEOS-10** (via `gsw`) |
| 6. Pressure post-processing | Demean pressure and compute 15-day moving average, incorporating external NANO sensors |
| 7. Harmonic mean velocities | Compute pairwise harmonic means for both SSP-based and TEOS-10-based velocities |
| 8. Tilt perturbation | Project inclinometer pitch/roll through instrument headings onto baseline directions |

</details>

<details>
<summary><b>Outputs</b> &mdash; <code>src/output/</code></summary>

| File | Contents |
|:-----|:---------|
| `combined_df4.pkl` | Northern station (2504) &mdash; pressure, temperature, salinity, velocity, tidal correction |
| `combined_df3.pkl` | Western station (2503) |
| `combined_df2.pkl` | Eastern station (2502) |
| `harmonic_mean_dfs.pkl` | SSP-based harmonic mean velocities (column: `Harmonic Mean`) |
| `harmonic_df_dict.pkl` | TEOS-10-based harmonic mean velocities (column: `HMean`) |
| `baseline_perturb.pkl` | Tilt-induced baseline perturbations (`2504-2502_dL`, `2504-2503_dL`, `2502-2503_dL`) |

</details>

### Pipeline 2 &mdash; `calculate_range`

Extracts acoustic range measurements, computes inter-station distances, applies corrections, and generates figures.

<details>
<summary><b>Processing Steps</b></summary>

| Step | Description |
|:-----|:------------|
| 1. Load inputs | Re-parse FETCH CSVs for BSL data; load pickle files from Pipeline 1 |
| 2. Extract & filter BSL | Apply station-specific outlier filtering (see table below) |
| 3. Compute distances | Interpolate harmonic-mean velocity onto timestamps, calculate one-way distance |
| 4. Tilt correction | Subtract tilt-induced path-length perturbation from each distance |
| 5. 30-day moving average | Compute rolling 30-day mean on each baseline time series |
| 6. Save distance CSVs | Export 6 CSVs with `Record Time`, `Distance (cm)`, `Moving Average (cm)` |
| 7. Perturbation correction | Apply tilt correction via `merge_asof` (nearest, 1-hour tolerance), save corrected CSVs |
| 8. Generate figures | Produce 21 PNG plots: time series, binned stats, bidirectional comparisons, precision analysis |

**Outlier Filtering Rules:**

| Baseline | Method | Parameters |
|:---------|:-------|:-----------|
| NE (2504 &rarr; 2502) | IQR | Quantiles (0.15, 0.85) |
| NW (2504 &rarr; 2503) | Hard threshold | Range(ms) &ge; 2618 |
| WE (2503 &rarr; 2502) | IQR | Quantiles (0.25, 0.75) |
| WN (2503 &rarr; 2504) | IQR | Quantiles (0.25, 0.75) |
| EW (2502 &rarr; 2503) | Hard threshold | Range(ms) &isin; [4636.2, 4638] |
| EN (2502 &rarr; 2504) | Hard threshold | Range(ms) &ge; 2400 |

</details>

<details>
<summary><b>Outputs</b> &mdash; Distance CSVs</summary>

All saved to `src/output/distances/`:

| Raw | Corrected | Baseline |
|:----|:----------|:---------|
| `NW_distance.csv` | `NW_distance_corrected.csv` | North &rarr; West |
| `NE_distance.csv` | `NE_distance_corrected.csv` | North &rarr; East |
| `WN_distance.csv` | `WN_distance_corrected.csv` | West &rarr; North |
| `WE_distance.csv` | `WE_distance_corrected.csv` | West &rarr; East |
| `EN_distance.csv` | `EN_distance_corrected.csv` | East &rarr; North |
| `EW_distance.csv` | `EW_distance_corrected.csv` | East &rarr; West |

</details>

<details>
<summary><b>Outputs</b> &mdash; Figures</summary>

All saved to `src/output/figures/`:

| Figure | Description |
|:-------|:------------|
| `{NW,NE,WE,WN,EW,EN}_distance.png` | Distance time series with scatter + 30-day moving average |
| `{NW,NE,WE,WN,EW,EN}_binned.png` | 30-day binned mean &plusmn; 1 std error bars |
| `North_West_vs_West_North.png` | Bidirectional comparison: NW vs WN |
| `East_North_vs_North_East.png` | Bidirectional comparison: EN vs NE |
| `West_East_vs_East_West.png` | Bidirectional comparison: WE vs EW |
| `10day_std_all.png` | 10-day binned standard deviation, all 6 baselines |
| `10day_std_normalized.png` | 10-day STD normalized by baseline length (cm/km) |
| `10day_std_subplots.png` | 10-day STD in 3&times;1 subplot panels |
| `10day_std_normalized_subplots.png` | Normalized 10-day STD in 3&times;1 subplot panels |
| `rolling_std_normalized.png` | 30-day rolling STD normalized by baseline length |
| `rolling_std_3baselines.png` | 30-day rolling STD for WN, NE, WE |

</details>

---

## Project Structure

```
AxialFetch/
├── src/
│   ├── __init__.py
│   ├── config.py               # Paths, constants, station metadata, baseline mappings
│   ├── data_loader.py          # FETCH CSV parsing (multi-code instrument files)
│   ├── salinity.py             # CTD salinity correction pipeline
│   ├── station_data.py         # Per-station combined DataFrame assembly
│   ├── tidal.py                # Tidal pressure correction (least-squares fit)
│   ├── velocity.py             # Sound speed: SSP interpolation + TEOS-10 via gsw
│   ├── pressure.py             # Pressure demeaning + 15-day moving average
│   ├── harmonic.py             # Harmonic mean computation (SSP + TEOS-10 based)
│   ├── tilt.py                 # Inclinometer → baseline perturbation + tilt correction
│   ├── create_dataframes.py    # Pipeline 1 orchestration
│   ├── range_filter.py         # BSL extraction, outlier filtering, distance computation
│   ├── range_correction.py     # Moving average, CSV export, perturbation correction
│   ├── range_figures.py        # All figure generation (21 plots)
│   ├── calculate_range.py      # Pipeline 2 orchestration
│   └── output/
│       ├── *.pkl               # Pickle files from Pipeline 1
│       ├── distances/          # CSV files from Pipeline 2
│       └── figures/            # PNG plots from Pipeline 2
├── Data_*_2502_.csv            # FETCH instrument data (Eastern)
├── Data_*_2503_.csv            # FETCH instrument data (Western)
├── Data_*_2504_.csv            # FETCH instrument data (Northern)
└── README.md
```

---

## Technical Reference

<details>
<summary><b>Sound Speed Computation</b></summary>

Two independent velocity estimates are computed for each station:

**1. SSP (onboard velocimeter)**
Direct measurement from the FETCH instrument's sound speed profiler.
> Station 2503 SSP data is unreliable and set to NaN.

**2. TEOS-10 (derived)**
Calculated from temperature, pressure, and salinity using the Gibbs SeaWater library:

```python
SA = gsw.SA_from_SP(S, P_dbar, lon, lat)   # Absolute Salinity
CT = gsw.CT_from_t(SA, T, P_dbar)           # Conservative Temperature
V  = gsw.sound_speed(SA, CT, P_dbar)        # Sound speed (m/s)
```

Pairwise **harmonic means** of velocities are used for distance calculation, representing the average velocity along the acoustic path between two stations.

</details>

<details>
<summary><b>Distance Formula</b></summary>

```
Distance (m) = V_harmonic × (Range_ms − TAT_ms) / 2000
```

| Symbol | Description |
|:-------|:------------|
| `V_harmonic` | Pairwise harmonic mean sound speed (m/s) |
| `Range_ms` | Measured two-way acoustic travel time (ms) |
| `TAT_ms` | Instrument turnaround time (ms) |
| `/ 2000` | Converts ms &rarr; s (&div; 1000) and two-way &rarr; one-way (&div; 2) |

</details>

<details>
<summary><b>Special Handling</b></summary>

| Condition | Treatment |
|:----------|:----------|
| Station 2503 SSP | Set to NaN (no reliable velocimeter). `harmonic_mean_dfs['2502_2503']` uses `2502_2504` data; `harmonic_mean_dfs['2503_2504']` uses only station 2504 SSP |
| Station 2504 temperature | +0.351 &deg;C offset applied to TMP sensor readings |
| Tilt correction | Inclinometer pitch/roll projected onto baseline directions to remove apparent distance changes from instrument tilting |

</details>

---

## Data Sources

| Data | Source | Format |
|:-----|:-------|:-------|
| FETCH instrument data | 3 CSV files in project root | Multi-header CSV with codes (TMP, DQZ, SSP, BSL, INC) |
| CTD salinity | OOI Cabled Array (RS03CCAL, MJ03F) | CSV |
| Tidal predictions | Pre-computed (2022&ndash;2025) | Text (Year, Month, Day, Hour, Min, Sec, Value) |
| NANO pressure | Bottom pressure recorders (MJ03F, MJ03E, MJ03B) | `.Data` files |

---

## Requirements

```
python >= 3.8
pandas
numpy
scipy
matplotlib
gsw
```
