"""
Configuration and Constants Module for FETCH Range Calculation

Contains station metadata, physical constants, file path patterns,
and default parameters used throughout the analysis pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
from pathlib import Path


# Physical Constants
R_EARTH = 6_371_000.0  # Earth radius in meters
H_TXR = 0.30  # Transducer to tilt-sensor lever arm in meters
PSI_TO_KPA = 6.894757  # Conversion factor from PSI to kPa
GRAVITY = 9.81  # Gravitational acceleration m/s^2

# Default coordinates for FETCH project
DEFAULT_LONGITUDE = -130.01149966
DEFAULT_LATITUDE = 45.95882833


@dataclass
class StationMetadata:
    """Metadata for a single FETCH station."""
    name: str
    lat: float
    lon: float
    heading: float
    identifier: str = ""

    def __post_init__(self):
        if not self.identifier:
            self.identifier = self.name


# Station definitions
STATIONS: Dict[str, StationMetadata] = {
    "2502": StationMetadata(
        name="East",
        lat=45.96257971,
        lon=-129.9910622,
        heading=120,
        identifier="2502"
    ),
    "2503": StationMetadata(
        name="West",
        lat=45.94700845,
        lon=-130.0265777,
        heading=195,
        identifier="2503"
    ),
    "2504": StationMetadata(
        name="North",
        lat=45.95882833,
        lon=-130.0114997,
        heading=127,
        identifier="2504"
    ),
}

# Baseline pairs for range calculation (transmitter, receiver)
BASELINE_PAIRS = [
    ("2504", "2502"),  # North -> East
    ("2504", "2503"),  # North -> West
    ("2502", "2503"),  # East -> West
    ("2502", "2504"),  # East -> North
    ("2503", "2502"),  # West -> East
    ("2503", "2504"),  # West -> North
]

# Default input file patterns
DEFAULT_STATION_FILES = [
    "Data_230912111909_East_006870_2502_.csv",
    "Data_230912112033_West_006874_2503_.csv",
    "Data_230913091309_North_00687A_2504_.csv",
]

DEFAULT_CTD_SALINITY_PATH = "/data/wsd02/maleen_data/ooi-rs03ccal-mj03f-12-ctdpfb305_2f21_522e_6ceb.csv"
DEFAULT_NODE_PRESSURE_PATTERN = "/data/wsd02/maleen_data/MJ03{node}*.Data"
DEFAULT_TIDAL_PREDICTION_PATHS = [
    "/data/wsd02/maleen_data/pred_F_2022.txt",
    "/data/wsd02/maleen_data/pred_F_2023.txt",
    "/data/wsd02/maleen_data/pred_F_2024.txt",
    "/data/wsd02/maleen_data/pred_F_2025.txt",
]


@dataclass
class PipelineConfig:
    """Configuration for the full analysis pipeline."""

    # Station data files
    station_files: List[str] = field(default_factory=lambda: DEFAULT_STATION_FILES.copy())

    # External data files
    ctd_salinity_path: str = DEFAULT_CTD_SALINITY_PATH
    node_pressure_pattern: str = DEFAULT_NODE_PRESSURE_PATTERN
    tidal_prediction_paths: List[str] = field(
        default_factory=lambda: DEFAULT_TIDAL_PREDICTION_PATHS.copy()
    )

    # Output directory
    output_dir: str = "results/"

    # Processing parameters
    exclude_first_days: int = 30
    moving_average_window: str = "30D"
    outlier_quantiles: tuple = (0.15, 0.85)
    smoothing_window: int = 250

    # Salinity correction parameters
    bottle_times: List[str] = field(default_factory=lambda: [
        "2022-08-14 05:49:53",
        "2022-08-30 19:37:13",
        "2023-09-17 03:35:09",
    ])
    bottle_values: List[float] = field(default_factory=lambda: [
        34.52543602,
        34.52720340,
        34.52700000,
    ])

    # Gap handling for salinity
    salinity_gap_start: str = "2023-09-07 00:00:00"
    salinity_gap_end: str = "2023-09-14 00:00:00"

    # Temperature correction for station 2504
    temp_correction_2504: float = 0.351

    # Analysis time window
    start_time: str = "2022-08-14 05:49:53"
    end_time: str = "2025-09-05 23:59:00"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()

        # Map YAML fields to config attributes
        if 'input_files' in data:
            if 'station_data' in data['input_files']:
                config.station_files = data['input_files']['station_data']
            if 'ctd_salinity' in data['input_files']:
                config.ctd_salinity_path = data['input_files']['ctd_salinity']
            if 'node_pressure_pattern' in data['input_files']:
                config.node_pressure_pattern = data['input_files']['node_pressure_pattern']
            if 'tidal_predictions' in data['input_files']:
                config.tidal_prediction_paths = data['input_files']['tidal_predictions']

        if 'output_dir' in data:
            config.output_dir = data['output_dir']

        if 'parameters' in data:
            params = data['parameters']
            if 'exclude_first_days' in params:
                config.exclude_first_days = params['exclude_first_days']
            if 'moving_average_window' in params:
                config.moving_average_window = params['moving_average_window']
            if 'outlier_quantiles' in params:
                config.outlier_quantiles = tuple(params['outlier_quantiles'])
            if 'smoothing_window' in params:
                config.smoothing_window = params['smoothing_window']

        if 'salinity' in data:
            sal = data['salinity']
            if 'bottle_times' in sal:
                config.bottle_times = sal['bottle_times']
            if 'bottle_values' in sal:
                config.bottle_values = sal['bottle_values']
            if 'gap_start' in sal:
                config.salinity_gap_start = sal['gap_start']
            if 'gap_end' in sal:
                config.salinity_gap_end = sal['gap_end']

        if 'temperature' in data:
            if 'correction_2504' in data['temperature']:
                config.temp_correction_2504 = data['temperature']['correction_2504']

        if 'time_window' in data:
            if 'start' in data['time_window']:
                config.start_time = data['time_window']['start']
            if 'end' in data['time_window']:
                config.end_time = data['time_window']['end']

        return config

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        data = {
            'input_files': {
                'station_data': self.station_files,
                'ctd_salinity': self.ctd_salinity_path,
                'node_pressure_pattern': self.node_pressure_pattern,
                'tidal_predictions': self.tidal_prediction_paths,
            },
            'output_dir': self.output_dir,
            'parameters': {
                'exclude_first_days': self.exclude_first_days,
                'moving_average_window': self.moving_average_window,
                'outlier_quantiles': list(self.outlier_quantiles),
                'smoothing_window': self.smoothing_window,
            },
            'salinity': {
                'bottle_times': self.bottle_times,
                'bottle_values': self.bottle_values,
                'gap_start': self.salinity_gap_start,
                'gap_end': self.salinity_gap_end,
            },
            'temperature': {
                'correction_2504': self.temp_correction_2504,
            },
            'time_window': {
                'start': self.start_time,
                'end': self.end_time,
            },
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load pipeline configuration from YAML file or return defaults.

    Args:
        config_path: Path to YAML configuration file. If None, returns defaults.

    Returns:
        PipelineConfig object with loaded or default settings.
    """
    if config_path is None:
        return PipelineConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    return PipelineConfig.from_yaml(config_path)
