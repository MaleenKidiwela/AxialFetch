"""
Configuration: file paths, station metadata, and constants used throughout the pipeline.
"""
import os

# ---------------------------------------------------------------------------
# Project root (one level up from src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw instrument data, model files, and other static inputs
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

# ---------------------------------------------------------------------------
# FETCH instrument CSV files
# ---------------------------------------------------------------------------
FETCH_CSV_FILES = [
    os.path.join(DATA_DIR, "Data_230912111909_East_006870_2502_.csv"),
    os.path.join(DATA_DIR, "Data_230912112033_West_006874_2503_.csv"),
    os.path.join(DATA_DIR, "Data_230913091309_North_00687A_2504_.csv"),
]

# ---------------------------------------------------------------------------
# CTD salinity file
# ---------------------------------------------------------------------------
SALINITY_CSV = "/data/wsd02/maleen_data/ooi-rs03ccal-mj03f-12-ctdpfb305_2f21_522e_6ceb.csv"

# ---------------------------------------------------------------------------
# Tidal prediction files
# ---------------------------------------------------------------------------
TIDAL_FILES = [
    "/data/wsd02/maleen_data/pred_F_2022.txt",
    "/data/wsd02/maleen_data/pred_F_2023.txt",
    "/data/wsd02/maleen_data/pred_F_2024.txt",
    "/data/wsd02/maleen_data/pred_F_2025.txt",
]

# ---------------------------------------------------------------------------
# External pressure data (NANO files)
# ---------------------------------------------------------------------------
NANO_MJ03F_PATTERN = "/data/wsd02/maleen_data/MJ03F*.Data"
NANO_MJ03E_PATTERN = "/data/wsd02/maleen_data/MJ03E*.Data"
NANO_MJ03B_PATTERN = "/data/wsd02/maleen_data/MJ03B*.Data"

# ---------------------------------------------------------------------------
# Station identifiers
# ---------------------------------------------------------------------------
IDENTIFIERS = ["2504", "2503", "2502"]

# Station name mapping
STATION_NAMES = {
    "2504": "Northern",
    "2503": "Western",
    "2502": "Eastern",
}

# ---------------------------------------------------------------------------
# Station metadata (lat, lon, heading for tilt correction)
# ---------------------------------------------------------------------------
STATION_META = {
    "2502": dict(lat=45.96257971, lon=-129.9910622, heading=120),
    "2503": dict(lat=45.94700845, lon=-130.0265777, heading=195),
    "2504": dict(lat=45.95882833, lon=-130.0114997, heading=127),
}

# ---------------------------------------------------------------------------
# Geophysical constants
# ---------------------------------------------------------------------------
LONGITUDE = -130.01149966
LATITUDE = 45.95882833
R_EARTH = 6_371_000.0       # metres
H_TXR = 0.30                # transducer-tilt lever arm (m)

# ---------------------------------------------------------------------------
# Temperature offset for station 2504
# ---------------------------------------------------------------------------
TEMP_OFFSET_2504 = 0.351

# ---------------------------------------------------------------------------
# Output directory for stored DataFrames
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "output")
DISTANCES_DIR = os.path.join(OUTPUT_DIR, "distances")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# ---------------------------------------------------------------------------
# Baseline lengths in km (for STD normalization in plots)
# ---------------------------------------------------------------------------
BASELINE_LENGTH_KM = {
    "NW": 1.765, "WN": 1.765,
    "NE": 1.642, "EN": 1.642,
    "WE": 3.26,  "EW": 3.26,
}

# ---------------------------------------------------------------------------
# Perturbation correction mapping: perturb column -> CSV files that use it
# ---------------------------------------------------------------------------
PERTURBATION_CORRECTIONS = {
    "2504-2502_dL": ["NE_distance.csv", "EN_distance.csv"],
    "2504-2503_dL": ["NW_distance.csv", "WN_distance.csv"],
    "2502-2503_dL": ["EW_distance.csv", "WE_distance.csv"],
}

# ---------------------------------------------------------------------------
# Baseline CSV name mapping: internal key -> output filename and short label
# ---------------------------------------------------------------------------
BASELINE_CSV_MAP = {
    "R2504_2502": ("NE_distance.csv", "NE"),
    "R2504_2503": ("NW_distance.csv", "NW"),
    "R2503_2502": ("WE_distance.csv", "WE"),
    "R2503_2504": ("WN_distance.csv", "WN"),
    "R2502_2503": ("EW_distance.csv", "EW"),
    "R2502_2504": ("EN_distance.csv", "EN"),
}
