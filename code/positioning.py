"""
Positioning and Tilt Correction Module for FETCH StreamLine Analysis

This module contains functions for handling coordinate transformations,
tilt corrections, and positioning calculations for underwater measurements.
"""

from typing import Tuple
import pandas as pd
import numpy as np


# Earth's radius in meters
R_EARTH = 6371000.0

# Default height for transducer (TXR) in meters
H_TXR = 1.0


def interp_unique(df: pd.DataFrame, t: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Interpolate Pitch/Roll onto timeline t, after dropping duplicate timestamps.
    
    Args:
        df: DataFrame containing 'Record Time', 'Pitch', and 'Roll' columns
        t: Target timeline for interpolation
        
    Returns:
        DataFrame with interpolated Pitch and Roll values
    """
    tidy = (df.set_index("Record Time")
              .sort_index()
              .loc[lambda x: ~x.index.duplicated(keep="first")])
    
    return (tidy.reindex(t)
                .interpolate(limit_direction="both")
                .astype(float))


def local_xy(df: pd.DataFrame, h: float = H_TXR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate local x,y coordinates from pitch and roll angles.
    
    Args:
        df: DataFrame containing 'Pitch' and 'Roll' columns in degrees
        h: Height of transducer in meters
        
    Returns:
        Tuple of (dx, dy) arrays representing local coordinates
    """
    pitch, roll = np.radians(df["Pitch"]), np.radians(df["Roll"])
    dx = h * np.sin(roll)       # starboard (positive to right)
    dy = h * np.sin(pitch)      # forward (positive forward)
    return dx.values, dy.values


def to_enu(dx: np.ndarray, dy: np.ndarray, heading_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert local coordinates to East-North-Up (ENU) coordinate system.
    
    Args:
        dx: Local x coordinates (starboard)
        dy: Local y coordinates (forward)
        heading_deg: Heading in degrees
        
    Returns:
        Tuple of (east, north) coordinates in ENU system
    """
    h = np.radians(heading_deg)
    east = dx * np.sin(h) + dy * np.cos(h)
    north = dx * np.cos(h) - dy * np.sin(h)
    return east, north


def unit_vector(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
    """
    Calculate unit vector components between two geographic coordinate pairs.
    
    Args:
        lat1: First latitude in decimal degrees
        lon1: First longitude in decimal degrees
        lat2: Second latitude in decimal degrees
        lon2: Second longitude in decimal degrees
        
    Returns:
        Tuple of (east, north) components of unit vector
    """
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    mlat = np.radians((lat1 + lat2) / 2)
    
    de = dlon * np.cos(mlat) * R_EARTH
    dn = dlat * R_EARTH
    L = np.hypot(de, dn)
    
    if L == 0:
        return 0.0, 0.0
    
    return de / L, dn / L  # EN components of unit vector


def apply_tilt_correction(
    df_range: pd.DataFrame, 
    baseline_tag: str, 
    perturb_df: pd.DataFrame,
    dist_col: str = "Calculated Distance (m)"
) -> pd.DataFrame:
    """
    Apply tilt correction to range measurements.
    
    Args:
        df_range: DataFrame containing range measurements
        baseline_tag: Tag identifying the baseline for correction
        perturb_df: DataFrame containing perturbation corrections
        dist_col: Name of the distance column to correct
        
    Returns:
        DataFrame with new tilt-corrected distance column 'Distance_tiltcorr(m)'
    """
    col = f"{baseline_tag}_dL"
    merged = df_range.merge(
        perturb_df[[col]],
        left_on="Record Time", 
        right_index=True, 
        how="left"
    )
    merged["Distance_tiltcorr(m)"] = merged[dist_col] - merged[col]
    return merged


def calculate_baseline_perturbations(
    attitude_df: pd.DataFrame,
    baseline_coords: dict,
    h_txr: float = H_TXR
) -> pd.DataFrame:
    """
    Calculate baseline perturbations due to platform attitude changes.
    
    Args:
        attitude_df: DataFrame with 'Pitch', 'Roll', 'Heading' in degrees
        baseline_coords: Dictionary mapping baseline names to (lat, lon) coordinates
        h_txr: Transducer height in meters
        
    Returns:
        DataFrame with perturbation corrections for each baseline
    """
    perturbations = {}
    
    # Calculate local displacements from attitude
    dx, dy = local_xy(attitude_df, h_txr)
    
    # Convert to ENU coordinates
    east, north = to_enu(dx, dy, attitude_df['Heading'].values)
    
    # Calculate perturbations for each baseline
    for baseline_name, (lat, lon) in baseline_coords.items():
        # Get unit vector for this baseline (simplified - would need reference coordinates)
        # This is a placeholder - in practice you'd need the reference station coordinates
        ue, un = 1.0, 0.0  # Example unit vector (east-west baseline)
        
        # Calculate along-baseline perturbation
        dL = east * ue + north * un
        perturbations[f"{baseline_name}_dL"] = dL
    
    # Create DataFrame with same index as attitude_df
    perturb_df = pd.DataFrame(perturbations, index=attitude_df.index)
    
    return perturb_df


def build_baseline_perturbations(
    stations: dict,
    pairs: list,
    h_txr: float = H_TXR,
) -> pd.DataFrame:
    """
    Build a baseline perturbation table for multiple station pairs.

    Args:
        stations: Mapping of station IDs to dicts with keys:
            'inc' (DataFrame with Record Time, Pitch, Roll),
            'lat', 'lon', and 'heading'.
        pairs: List of (tx, rx) tuples specifying baseline directions.
        h_txr: Transducer-to-tilt-sensor lever arm in meters.

    Returns:
        DataFrame indexed by timestamp with per-baseline perturbation columns.
    """
    series_list = []

    for tx, rx in pairs:
        station_tx = stations[tx]
        station_rx = stations[rx]

        timeline = pd.Index(
            sorted(set(station_tx["inc"]["Record Time"]) | set(station_rx["inc"]["Record Time"]))
        )

        inc_tx = interp_unique(station_tx["inc"], timeline)
        inc_rx = interp_unique(station_rx["inc"], timeline)

        e_tx, n_tx = to_enu(*local_xy(inc_tx, h_txr), station_tx["heading"])
        e_rx, n_rx = to_enu(*local_xy(inc_rx, h_txr), station_rx["heading"])

        ue, un = unit_vector(
            station_tx["lat"],
            station_tx["lon"],
            station_rx["lat"],
            station_rx["lon"],
        )
        dL_oneway = (e_tx - e_rx) * ue + (n_tx - n_rx) * un

        series_list.append(
            pd.DataFrame({f"{tx}-{rx}_dL": dL_oneway}, index=timeline)
        )

    return pd.concat(series_list, axis=1).sort_index()


def geodetic_to_enu(
    lat: float, 
    lon: float, 
    alt: float,
    ref_lat: float, 
    ref_lon: float, 
    ref_alt: float
) -> Tuple[float, float, float]:
    """
    Convert geodetic coordinates (lat/lon/alt) to local ENU coordinates.
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees  
        alt: Altitude in meters
        ref_lat: Reference latitude in decimal degrees
        ref_lon: Reference longitude in decimal degrees
        ref_alt: Reference altitude in meters
        
    Returns:
        Tuple of (east, north, up) coordinates in meters
    """
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)
    
    # Differences
    dlat = lat_rad - ref_lat_rad
    dlon = lon_rad - ref_lon_rad
    dalt = alt - ref_alt
    
    # Calculate ENU coordinates
    mean_lat = (lat_rad + ref_lat_rad) / 2
    
    east = dlon * np.cos(mean_lat) * R_EARTH
    north = dlat * R_EARTH
    up = dalt
    
    return east, north, up


def calculate_range_from_coordinates(
    lat1: float, lon1: float, alt1: float,
    lat2: float, lon2: float, alt2: float
) -> float:
    """
    Calculate 3D range between two geographic positions.
    
    Args:
        lat1: First latitude in decimal degrees
        lon1: First longitude in decimal degrees
        alt1: First altitude in meters
        lat2: Second latitude in decimal degrees
        lon2: Second longitude in decimal degrees
        alt2: Second altitude in meters
        
    Returns:
        3D range in meters
    """
    # Convert first point to ENU relative to second point
    east, north, up = geodetic_to_enu(lat1, lon1, alt1, lat2, lon2, alt2)
    
    # Calculate 3D distance
    range_3d = np.sqrt(east**2 + north**2 + up**2)
    
    return range_3d


def correct_sound_path(
    measured_range: float,
    transmitter_coords: Tuple[float, float, float],
    receiver_coords: Tuple[float, float, float],
    sound_speed_profile: callable = None
) -> float:
    """
    Correct acoustic range measurements for sound speed variations.
    
    Args:
        measured_range: Raw measured acoustic range in meters
        transmitter_coords: (lat, lon, depth) of transmitter
        receiver_coords: (lat, lon, depth) of receiver
        sound_speed_profile: Function that returns sound speed at given depth
        
    Returns:
        Corrected range in meters
    """
    if sound_speed_profile is None:
        # Use standard sound speed if no profile provided
        return measured_range
    
    # Calculate geometric range
    geometric_range = calculate_range_from_coordinates(*transmitter_coords, *receiver_coords)
    
    # For more complex ray tracing, this would involve integration
    # For now, apply a simple average sound speed correction
    avg_depth = (transmitter_coords[2] + receiver_coords[2]) / 2
    sound_speed = sound_speed_profile(avg_depth)
    
    # Assume measured range was calculated with standard sound speed (1500 m/s)
    standard_sound_speed = 1500.0
    corrected_range = measured_range * (standard_sound_speed / sound_speed)
    
    return corrected_range
