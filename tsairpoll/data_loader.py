import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#pd.set_option('display.max_rows', 500)


def compute_dew_point(temp_c, rh_percent):
    a = 17.625
    b = 243.04  # in °C
    gamma = (a * temp_c) / (b + temp_c) + np.log(rh_percent / 100.0)
    dew_point = (b * gamma) / (a - gamma)
    return dew_point


def haversine(lat1, lon1, lat2, lon2):
    """Compute Haversine distance in meters. Supports both scalars and pandas Series."""

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Differences in coordinates
    dLat = lat2 - lat1
    dLon = lon2 - lon1

    # Haversine formula
    a = np.sin(dLat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth's radius in meters
    R = 6371000
    return R * c  # Distance in meters


def get_station_coordinates(file_path, station_name):
    """Reads a CSV file and retrieves the latitude & longitude of a given station."""
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Search for the station name (case-sensitive exact match)
    station_row = df[df["Stazione"] == station_name]
    
    if station_row.empty:
        raise ValueError(f"Station '{station_name}' not found.")
    
    # Extract coordinates
    lon = float(station_row["Lon"].values[0])
    lat = float(station_row["Lat"].values[0])
    
    return lat, lon  # Return as (latitude, longitude)


def merge_arpa_csv_files(folder_path, output_file):
    """Merge all CSV files in a folder by concatenating rows, assuming they have the same headers."""
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not csv_files:
        raise ValueError("No CSV files found in the specified folder.")

    df_list = []
    for file in csv_files:
        this_df = pd.read_csv(os.path.join(folder_path, file), delimiter=",", decimal=",")
        this_df.columns = ["timestamp", "station", "parameter", "value", "unit", "validation"]
        this_df["value"] = this_df["value"].astype(str).str.replace(',', '.')
        # Convert timestamp
        this_df["timestamp"] = this_df["timestamp"].str.split(".").str[0]  # Keep only the part before the dot
        this_df["timestamp"] = pd.to_datetime(this_df["timestamp"], format="%d/%m/%Y %H:%M:%S")
        this_df["timestamp"] = pd.to_datetime(this_df["timestamp"], format="%Y-%m-%d %H:%M:%S")

        df_list.append(this_df)

    merged_df = pd.concat(df_list, ignore_index=True).sort_values(by=['timestamp'], ascending=True)

    merged_df.to_csv(output_file, index=False)
    print(f"Merged {len(csv_files)} files into {output_file}")


def merge_cocal_csv_files(folder_path, output_file):
    """Merge all CSV files in a folder by concatenating rows, assuming they have the same headers."""
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not csv_files:
        raise ValueError("No CSV files found in the specified folder.")

    df_list = [pd.read_csv(os.path.join(folder_path, file), delimiter=",", decimal=".") for file in csv_files]

    merged_df = pd.concat(df_list, ignore_index=True).sort_values(by=['time'], ascending=True)

    merged_df.to_csv(output_file, index=False)
    print(f"Merged {len(csv_files)} files into {output_file}")


def load_arpa_data(file_path, station_name, parameter_name):
    """Load and process ARPA data."""

    if 'PM10' in parameter_name.upper():
        label = 'PM10_ARPA'
    elif 'PM25' in parameter_name.upper() or 'PM2.5' in parameter_name.upper():
        label = 'PM25_ARPA'
    else:
        raise AttributeError(f'The parameter {parameter_name} is neither PM10 nor PM25/PM2.5.')

    # Load dataset
    df = pd.read_csv(file_path, delimiter=",", decimal=".")
    
    # Rename columns for easier handling
    df.columns = ["timestamp", "station", "parameter", "value", "unit", "validation"]

    # Convert timestamp
    df["timestamp"] = df["timestamp"].str.split(".").str[0]  # Keep only the part before the dot
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")

    # Convert to datetime and set timezone to UTC+1
    df['timestamp'] = df['timestamp'].dt.tz_localize('Etc/GMT-1')  # UTC+1

    # Convert to UTC
    df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")

    # Remove rows with "N/D" in value
    df = df[df["value"] != "N/D"]
    
    # Convert PM10/PM25 values to float
    df["value"] = df["value"].str.replace(',', '.').astype(float)

    # Keep only PM10/PM25 measurements from station_name
    df = df[(df["station"] == station_name) & (df["parameter"] == parameter_name)]
    df = df.drop(columns=['station', 'unit', 'validation', 'parameter'])

    return df.rename(columns={"value": label})


def load_cocal_data(file_path, ref_lat, ref_lon, max_distance, device_type):
    """Load, filter, and process COCAL data to match ARPA timestamps."""

    # Load dataset
    df = pd.read_csv(file_path, delimiter=",", decimal=".")

    # Convert timestamp
    df = df.rename(columns={"time": "timestamp"})    
    df["timestamp"] = df["timestamp"].str.split(".").str[0]  # Keep only the part before the dot
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")

    # Convert longitude & latitude to float
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)

    # Convert all the other numerical columns to float or int
    df["val"] = df["val"].astype(float)
    df["gps_kmh"] = df["gps_kmh"].astype(float)
    df["gps_dir"] = df["gps_dir"].astype(float)
    df["gps_alt"] = df["gps_alt"].astype(float)
    # df["gps_nsat"] = df["gps_nsat"].astype(int)

    # Compute distances and filter nearby rows
    df["distance"] = haversine(df["lat"], df["lon"], ref_lat, ref_lon)
    df = df[df["distance"] <= max_distance].drop(columns=["distance"])

    # Pivot to create a single row per timestamp with sensor values as columns
    df = df.pivot_table(index=["timestamp", "lat", "lon", "device"] + ["gps_kmh", "gps_dir", "gps_alt"], columns="sensor", values="val", aggfunc='mean').reset_index()
    for to_remove_col in ["AN_CO2", "DF_O3", "WI_O2"]:
        if to_remove_col in df.columns:
            df = df.drop(columns=[to_remove_col])
    df = df.dropna(axis=0)

    # Flatten MultiIndex columns and remove the "sensor" name
    df.columns.name = None  # Remove the name of the columns index
    df.columns = [str(col) for col in df.columns]  # Convert to flat structure

    # Round timestamp to previous hour
    df["timestamp"] = df["timestamp"].dt.floor("h")
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")

    # Compute dew point
    df["dew_point"] = compute_dew_point(df["BM_T"], df["BM_H"])
    cols = df.columns.tolist()  # Get current column order
    cols.insert(cols.index("timestamp") + 1, cols.pop(cols.index("dew_point")))  # Move "dew_point"
    df = df[cols]  # Apply new order

    # Aggregating by mean
    #df = df.groupby(['timestamp', 'device'], as_index=False).mean(numeric_only=True)

    # Group by timestamp and device, computing mean, max, min, and count
    aggregated = df.groupby(['timestamp', 'device'], as_index=False).agg(['mean', 'max', 'min', 'count'])

    # Flatten MultiIndex columns
    aggregated.columns = ['_'.join(col).upper() for col in aggregated.columns]
    aggregated = aggregated.reset_index(drop=True)

    # Rename count column to NUM_MEASUREMENTS for one representative variable
    count_cols = [col for col in aggregated.columns if col.endswith('_COUNT')]
    if count_cols:
        aggregated["NUM_MEASUREMENTS"] = aggregated[count_cols[0]]
        aggregated = aggregated.drop(columns=count_cols)

    df = aggregated

    # Reset index (flatten the index structure)
    df = df.reset_index(drop=True)

    df.rename(columns={"TIMESTAMP_": "timestamp", "DEVICE_": "device"}, inplace=True)

    # Add a new column "device_type"
    df["device_type"] = df["device"].apply(lambda x: "static" if x == "CAMO004" else "dynamic")
    df = df.drop(columns=["device"])
    # Order columns
    cols = df.columns.tolist()  # Get current column order
    cols.insert(cols.index("timestamp") + 1, cols.pop(cols.index("device_type")))  # Move "device_type"
    df = df[cols]  # Apply new order

    if device_type.strip().lower() == 'static':
        df = df[df["device_type"] == "static"]
        df = df.drop(columns=["device_type"])
    elif device_type.strip().lower() == 'dynamic':
        df = df[df["device_type"] == "dynamic"]
        df = df.drop(columns=["device_type"])
    else:
        print('ALL DEVICES, BOTH STATIC AND DYNAMIC, ARE BEING CONSIDERED!')

    df = df.dropna(axis=0)

    return df


def derive_time_attributes(df):
    """Derive time-related attributes and place them at the beginning of the DataFrame."""

    # Convert timestamp to datetime if it's not already
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")

    # Extract time-based features
    df["month"] = df["timestamp"].dt.month
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # Monday=0, ..., Sunday=6

    # Define time of day (morning, afternoon, evening, night)
    def categorize_time_of_day(hour):
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 24:
            return "evening"
        else:
            return "night"

    # Define season
    def categorize_season(month):
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    # Define weekend or not
    def categorize_weekend(day_of_week):
        return "weekend" if day_of_week in [5, 6] else "weekday"

    df["season"] = df["month"].apply(categorize_season)
    df["weekend"] = df["day_of_week"].apply(categorize_weekend)
    df["time_of_day"] = df["hour"].apply(categorize_time_of_day)
    df["timestamp"] = df["timestamp"].astype(str)
    df["timestamp"] = df["timestamp"].str.split(".").str[0]

    # Drop unnecessary columns
    df = df.drop(columns=['hour', 'month', 'day_of_week'])

    # Move new columns to the beginning
    cols = list(df.columns)
    df = df[cols[-3:] + cols[:-3]] # Ensures new attributes are placed first

    return df


def integrate_arpa_cocal(stations_file, arpa_file, cocal_file, station_name, parameter_name, max_distance, device_type, use_delta_as_target):
    """Merge ARPA and COCAL datasets and compute PM10 delta."""

    # Load and process datasets
    df_arpa = load_arpa_data(arpa_file, station_name, parameter_name)
    ref_lat, ref_lon = get_station_coordinates(stations_file, station_name=station_name)
    df_cocal = load_cocal_data(cocal_file, ref_lat=ref_lat, ref_lon=ref_lon, max_distance=max_distance, device_type=device_type)

    # Merge on hourly timestamps
    df_cocal['timestamp'] = df_cocal['timestamp'].dt.tz_localize(None)
    df_arpa['timestamp'] = df_arpa['timestamp'].dt.tz_localize(None)
    df = df_cocal.merge(df_arpa, on="timestamp")
    if 'ARPA' not in list(df.columns)[-1].upper():
        raise ValueError(f'The last column here must be the PM from ARPA.')
    df = df.drop(columns=['PM25_MEAN', 'PM25_MIN', 'PM25_MAX']) if 'PM10' in list(df.columns)[-1].upper() else df.drop(columns=['PM10_MEAN', 'PM10_MIN', 'PM10_MAX'])

    # Derive time-related attributes
    df = derive_time_attributes(df)

    # Drop eventual rows with null or NaN
    df.dropna(axis='index', how='any', inplace=True, ignore_index=True)

    # Drop duplicate rows
    df.drop_duplicates(keep='first', inplace=True, ignore_index=True)

    # Compute delta as target variable by doing ARPA PM - COCAL PM
    if use_delta_as_target:
        if 'PM10' in list(df.columns)[-1].upper():
            df['delta'] = df['PM10_ARPA'] - df['PM10_MEAN']
            df = df.drop(columns=['PM10_ARPA'])
        elif 'PM25' in list(df.columns)[-1].upper():
            df['delta'] = df['PM25_ARPA'] - df['PM25_MEAN']
            df = df.drop(columns=['PM25_ARPA'])
        else:
            raise ValueError(f'The target column does not contain any PM-related value.')

    # Retrieve timestamps and drop them
    timestamps = df['timestamp']
    df = df.drop(columns=['timestamp'])

    # Move PM10_MEAN and PM25_MEAN just before the last column (usually 'delta' or ARPA column)
    last_col = df.columns[-1]
    cols = df.columns.tolist()

    # Remove PM10_MEAN and PM25_MEAN if present
    for col in ['PM10_MEAN', 'PM25_MEAN']:
        if col in cols:
            cols.remove(col)

    # Insert them before the last column
    insert_index = cols.index(last_col)
    for col in ['PM25_MEAN', 'PM10_MEAN'][::-1]:  # Reverse to maintain order
        if col in df.columns:
            cols.insert(insert_index, col)

    df = df[cols]

    return df, timestamps


def save_csv_data(df, path):
    """Save the DataFrame to a .csv file."""
    df.to_csv(path, sep=',', index=False)


def load_data(file_path):    
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path, delimiter=',', decimal='.') 
    return df


def dataframe_filtered_to_numpy(df, timestamps, selected_features):
    target = df.columns[-1]
    if len(selected_features) == 0:
        df = df.copy()
    else:
        df = df[selected_features + [target]].copy()

    # Ensure timestamps match with the data
    assert len(df) == len(timestamps), "Mismatch in the number of rows between data and timestamps"

    # Merge timestamps into the data
    df['timestamp'] = timestamps['timestamp']

    df = df.reset_index(drop=True)
    df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
    df = df.drop(columns=['timestamp'])

    X_test = df.iloc[:, :-1]
    y_test = df[target].to_numpy()

    return X_test, y_test


def split_data_from_path(data_path, timestamps_path, test_size, random_state):
    # Load the data and timestamps
    df = load_data(data_path)
    timestamps = load_data(timestamps_path)
    return split_data(df=df, timestamps=timestamps, test_size=test_size, random_state=random_state, sort=True)


def split_data(df, timestamps, test_size, random_state, sort=True):
    # Ensure timestamps match with the data
    assert len(df) == len(timestamps), "Mismatch in the number of rows between data and timestamps"

    # Merge timestamps into the data
    df['timestamp'] = timestamps['timestamp']

    # Perform train-test split, stratifying on the last column (assumed target)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Sort both sets by timestamp
    if sort:
        train_df = train_df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
    test_df = test_df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)

    # Extract timestamps
    train_df["timestamp"] = train_df["timestamp"].astype(str)
    train_df["timestamp"] = train_df["timestamp"].str.split(".").str[0]
    test_df["timestamp"] = test_df["timestamp"].astype(str)
    test_df["timestamp"] = test_df["timestamp"].str.split(".").str[0]
    train_timestamp = train_df['timestamp']
    test_timestamp = test_df['timestamp']
    train_df = train_df.drop(columns=['timestamp'])
    test_df = test_df.drop(columns=['timestamp'])

    return train_df, train_timestamp, test_df, test_timestamp


def create_dir_path_results(base_path, dataset, features, encoding, scaling, augmentation, model, test_size, n_iter, cv, linear_scaling, log_scale_target, n_train_records):
    s = base_path.strip()
    test_size = str(test_size).replace('.', 'd')
    augmentation = str(augmentation).replace('.', 'd')
    linear_scaling = linear_scaling != 0
    log_scale_target = log_scale_target != 0
    s = os.path.join(
        s,
        dataset,
        features,
        f'{encoding}_{scaling}_{augmentation}_{model}',
        f'test{test_size}cviter{n_iter}cvfold{cv}linscale{int(linear_scaling)}logscaletarget{int(log_scale_target)}ntrain{n_train_records}',
        f''
    )

    return s
