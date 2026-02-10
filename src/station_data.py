"""
Station DataFrame assembly: merge pressure, temperature, and salinity for each
station into combined DataFrames.
"""
import pandas as pd
from .config import TEMP_OFFSET_2504


def build_combined_df4(data_extracted, salinity_df):
    """
    Build combined_df4 for station 2504 (Northern).
    Includes DQZ temperature as TempDQZ and applies +0.351 offset to TMP temperature.
    """
    pressure_df4 = pd.DataFrame({
        'Record Time': pd.to_datetime(data_extracted['2504']['DQZ']['Record Time']),
        'Pressure (kPa)': data_extracted['2504']['DQZ']['Pressure (kPa)'],
    }).set_index('Record Time')

    DQZtemp_df4 = pd.DataFrame({
        'Record Time': pd.to_datetime(data_extracted['2504']['DQZ']['Record Time']),
        'Temperature Deg C': data_extracted['2504']['DQZ']['Temperature (Deg C)'],
    }).set_index('Record Time')

    temperature_df4 = pd.DataFrame({
        'Record Time': pd.to_datetime(data_extracted['2504']['TMP']['Record Time']),
        'Temperature Deg C': data_extracted['2504']['TMP']['Temperature Deg C'] + TEMP_OFFSET_2504,
    }).set_index('Record Time')

    combined_df4 = pressure_df4.join(temperature_df4, how='left')

    DQZtemp_df4 = DQZtemp_df4[~DQZtemp_df4.index.duplicated(keep='first')]
    combined_df4['TempDQZ'] = DQZtemp_df4['Temperature Deg C']

    salinity_dfs4 = pd.DataFrame({
        'time': pd.to_datetime(salinity_df['time']),
        'adjusted_salinity': salinity_df['corrected_salinity'],
    }).set_index('time')

    aligned_salinity_df4 = salinity_dfs4.reindex(combined_df4.index, method='nearest')
    combined_df4['Salinity'] = aligned_salinity_df4['adjusted_salinity']

    combined_df4.reset_index(inplace=True)
    combined_df4.rename(columns={'index': 'Record Time'}, inplace=True)

    return combined_df4


def build_combined_df3(data_extracted, salinity_df):
    """
    Build combined_df3 for station 2503 (Western).
    """
    pressure_df3 = pd.DataFrame({
        'Record Time': pd.to_datetime(data_extracted['2503']['DQZ']['Record Time']),
        'Pressure (kPa)': data_extracted['2503']['DQZ']['Pressure (kPa)'],
    }).set_index('Record Time')

    temperature_df3 = pd.DataFrame({
        'Record Time': pd.to_datetime(data_extracted['2503']['TMP']['Record Time']),
        'Temperature Deg C': data_extracted['2503']['TMP']['Temperature Deg C'],
    }).set_index('Record Time')

    combined_df3 = pressure_df3.join(temperature_df3, how='left')

    salinity_dfs3 = pd.DataFrame({
        'time': pd.to_datetime(salinity_df['time']),
        'adjusted_salinity': salinity_df['corrected_salinity'],
    }).set_index('time')

    aligned_salinity_df3 = salinity_dfs3.reindex(combined_df3.index, method='nearest')
    combined_df3['Salinity'] = aligned_salinity_df3['adjusted_salinity']

    combined_df3.reset_index(inplace=True)
    combined_df3.rename(columns={'index': 'Record Time'}, inplace=True)

    return combined_df3


def build_combined_df2(data_extracted, salinity_df):
    """
    Build combined_df2 for station 2502 (Eastern).
    """
    pressure_df = pd.DataFrame({
        'Record Time': pd.to_datetime(data_extracted['2502']['DQZ']['Record Time']),
        'Pressure (kPa)': data_extracted['2502']['DQZ']['Pressure (kPa)'],
    }).set_index('Record Time')

    temperature_df1 = pd.DataFrame({
        'Record Time': pd.to_datetime(data_extracted['2502']['TMP']['Record Time']),
        'Temperature Deg C': data_extracted['2502']['TMP']['Temperature Deg C'],
    }).set_index('Record Time')

    combined_df2 = pressure_df.join(temperature_df1, how='left')

    salinity_dfs2 = pd.DataFrame({
        'time': pd.to_datetime(salinity_df['time']),
        'adjusted_salinity': salinity_df['corrected_salinity'],
    }).set_index('time')

    aligned_salinity_df2 = salinity_dfs2.reindex(combined_df2.index, method='nearest')
    combined_df2['Salinity'] = aligned_salinity_df2['adjusted_salinity']

    combined_df2.reset_index(inplace=True)
    combined_df2.rename(columns={'index': 'Record Time'}, inplace=True)

    return combined_df2


def build_inc_dataframes(data_extracted):
    """
    Build inclinometer DataFrames for each station and apply filters.
    Returns INC_df4, INC_df3, INC_df2.
    """
    INC_df4 = pd.DataFrame({
        'Record Time': pd.to_datetime(data_extracted['2504']['INC']['Record Time']),
        'Pitch': data_extracted['2504']['INC']['Pitch (deg)'],
        'Roll': data_extracted['2504']['INC']['Roll (deg)'],
    })

    INC_df3 = pd.DataFrame({
        'Record Time': pd.to_datetime(data_extracted['2503']['INC']['Record Time']),
        'Pitch': data_extracted['2503']['INC']['Pitch (deg)'],
        'Roll': data_extracted['2503']['INC']['Roll (deg)'],
    })

    INC_df2 = pd.DataFrame({
        'Record Time': pd.to_datetime(data_extracted['2502']['INC']['Record Time']),
        'Pitch': data_extracted['2502']['INC']['Pitch (deg)'],
        'Roll': data_extracted['2502']['INC']['Roll (deg)'],
    })

    # Apply filters (Cell 10 of notebook)
    INC_df3 = INC_df3[(INC_df3['Pitch'] < -3) & (INC_df3['Pitch'] > -3.5)]
    INC_df3 = INC_df3[(INC_df3['Roll'] < 0)]
    INC_df2 = INC_df2[(INC_df2['Roll'] < 0) & (INC_df2['Roll'] > -0.35)]

    return INC_df4, INC_df3, INC_df2
