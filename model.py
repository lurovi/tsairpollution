import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


# =============================================================================
# Constants
# =============================================================================

REF_LAT, REF_LON = 45.6544, 13.7890  # Volontari ARPA's coordinates in degrees
DISTANCE = 4000  # distance threshold in meters

# =============================================================================
# Helper Functions
# =============================================================================

def compute_dew_point(temp_c, rh_percent):
    a = 17.625
    b = 243.04  # in °C
    gamma = (a * temp_c) / (b + temp_c) + np.log(rh_percent / 100.0) # temp_c is in °C and rh_percent is in %
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



# =============================================================================
# Models
# =============================================================================

def sr_center_model(df_center: pd.DataFrame) -> pd.Series:
    # if coordinates of sensors are within 4 km from Volontari ARPA's coordinates
    preds_center_sr = 7.121288**(0.086461845873451697*df_center['dew_point'] - 1.0625816000264272)*(0.087804074962728641*df_center['BM_P'] - 86.63514053991303) \
                        + np.abs(
                            0.890044936950707*df_center['PM10'] - np.abs(
                                    - 0.23230223138202909*df_center['BM_P'] + 0.13966812947870454*df_center['gps_kmh'] \
                                    - 0.404384944593045*df_center['PM10'] + df_center['season_autumn'] + 242.64444926737826
                                ) + 5.7862680445378522
                            ) + 4.1231313
    return preds_center_sr


def el_far_model(df_far: pd.DataFrame) -> pd.Series:
    # if coordinates of sensors are further than 4 km from Volontari ARPA's coordinates
    # NUM_MEASUREMENTS is used as a representative variable for the number of measurements and
    # it is always 1 in real cases when no aggregation is performed
    preds_far_el = 0.21922735688784111*df_far['BM_P'] + 0.02326372850401895*df_far['gps_alt'] \
                    - 0.0028953559222595437*df_far['gps_dir'] - 0.074144178352954233*df_far['gps_kmh'] \
                    - 0.1716236221740934*df_far['NUM_MEASUREMENTS'] + 0.80541566340406936*df_far['PM10'] \
                    + 0.51015088847029162*df_far['dew_point'] - 1.5437585049381666*df_far['season_autumn'] \
                    + 1.800796053315133*df_far['season_spring'] - 0.8668227841881248*df_far['season_summer'] \
                    + 0.6006034360228186*df_far['season_winter'] + 0.8874388174662594*df_far['time_of_day_afternoon'] \
                    + 0.004663331919003846*df_far['time_of_day_morning'] - 2.844773187462841*df_far['time_of_day_night'] - 217.23136634033793
    return preds_far_el


# =============================================================================
# Data Loading and Processing
# =============================================================================

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
    df['timestamp'] = df['timestamp'].dt.tz_localize('Etc/GMT-1')  # UTC+1 # type: ignore

    # Convert to UTC
    df['timestamp'] = df['timestamp'].dt.tz_convert('UTC') # type: ignore
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")

    # Remove rows with "N/D" in value
    df = df[df["value"] != "N/D"]
    
    # Convert PM10/PM25 values to float
    df["value"] = df["value"].str.replace(',', '.').astype(float)

    # Keep only PM10/PM25 measurements from station_name
    df = df[(df["station"] == station_name) & (df["parameter"] == parameter_name)]
    df = df.drop(columns=['station', 'unit', 'validation', 'parameter'])

    return df.rename(columns={"value": label})


def load_cocal_data(file_path, aggregate=False, filter=False, coordinates=None):
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
    if filter and coordinates is not None:
        df["distance"] = haversine(df["lat"], df["lon"], coordinates[0], coordinates[1])
        df = df[df["distance"] <= 100].drop(columns=["distance"])

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
    if aggregate:
        df["timestamp"] = df["timestamp"].dt.floor("h")
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")

    # Compute dew point
    df["dew_point"] = compute_dew_point(df["BM_T"], df["BM_H"])
    cols = df.columns.tolist()  # Get current column order # type: ignore
    cols.insert(cols.index("timestamp") + 1, cols.pop(cols.index("dew_point")))  # Move "dew_point"
    df = df[cols]  # Apply new order

    if aggregate:
        # Group by timestamp and device, computing mean, max, min, and count
        aggregated = df.groupby(['timestamp', 'device'], as_index=False).agg(['mean', 'count'])

        # Flatten MultiIndex columns
        aggregated.columns = ['_'.join(col).upper() for col in aggregated.columns]
        aggregated = aggregated.reset_index(drop=True)

        # Rename count column to NUM_MEASUREMENTS for one representative variable
        count_cols = [col for col in aggregated.columns if col.endswith('_COUNT')]
        if count_cols:
            aggregated["NUM_MEASUREMENTS"] = aggregated[count_cols[0]]
            aggregated = aggregated.drop(columns=count_cols)

        df = aggregated
    else:
        df['NUM_MEASUREMENTS'] = 1

    # Reset index (flatten the index structure)
    df = df.reset_index(drop=True)

    if aggregate:
        df.rename(columns={"TIMESTAMP_": "timestamp", "DEVICE_": "device"}, inplace=True)
        df.rename(columns={'DEW_POINT_MEAN': 'dew_point'}, inplace=True)
        df.rename(columns={'GPS_KMH_MEAN': 'gps_kmh', 'GPS_DIR_MEAN': 'gps_dir', 'GPS_ALT_MEAN': 'gps_alt', 'BM_P_MEAN': 'BM_P'}, inplace=True)
        df.rename(columns={'PM10_MEAN': 'PM10'}, inplace=True)
        df.rename(columns={'PM25_MEAN': 'PM25'}, inplace=True)
        df.rename(columns={'LAT_MEAN': 'lat', 'LON_MEAN': 'lon'}, inplace=True)

    # Add a new column "device_type"
    df["device_type"] = df["device"].apply(lambda x: "static" if x == "CAMO004" else "dynamic")
    #df = df.drop(columns=["device"])
    # Order columns
    cols = df.columns.tolist()  # Get current column order
    cols.insert(cols.index("timestamp") + 1, cols.pop(cols.index("device_type")))  # Move "device_type"
    df = df[cols]  # Apply new order
    
    if filter:
        df = df[df["device_type"] == "dynamic"]
        df = df.drop(columns=["device_type"])

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
    df["day_of_month"] = df["timestamp"].dt.day
    
    # Week of month: only 4 values based on 7-day intervals
    def get_week_of_month(day):
        if day <= 7:
            return 1
        elif day <= 14:
            return 2
        elif day <= 21:
            return 3
        else:
            return 4
    
    df["week_of_month"] = df["day_of_month"].apply(get_week_of_month)

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
    
    # Categorize day of week
    def categorize_day_of_week(day_of_week):
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        return days[day_of_week]
    
    def categorize_week_of_month(week_of_month):
        return f"week{week_of_month}"
    
    def categorize_month(month):
        months = ["january", "february", "march", "april", "may", "june", 
                  "july", "august", "september", "october", "november", "december"]
        return months[month - 1]

    df["season"] = df["month"].apply(categorize_season)
    df["weekend"] = df["day_of_week"].apply(categorize_weekend)
    df["time_of_day"] = df["hour"].apply(categorize_time_of_day)
    df["day_of_week"] = df["day_of_week"].apply(categorize_day_of_week)
    df["week_of_month"] = df["week_of_month"].apply(categorize_week_of_month)
    df["month"] = df["month"].apply(categorize_month)
    df["timestamp"] = df["timestamp"].astype(str)
    df["timestamp"] = df["timestamp"].str.split(".").str[0]

    # Drop unnecessary columns
    #df = df.drop(columns=['hour', 'month', 'day_of_week', 'day_of_month', 'week_of_month'])

    # Move new columns to the beginning
    cols = list(df.columns)
    df = df[cols[-8:] + cols[:-8]] # Ensures new attributes are placed first

    return df


def eventually_merge_and_extract_time_categories(cocal_file, aggregate=False, filter=False, arpa_file=None, station_name=None, coordinates=None, parameter_name=None):
    """Merge ARPA and COCAL datasets and compute PM10 delta."""

    is_arpa_provided = arpa_file is not None and station_name is not None and coordinates is not None and parameter_name is not None
    # Load and process datasets
    if is_arpa_provided:
        df_arpa = load_arpa_data(arpa_file, station_name, parameter_name)
    else:
        df_arpa = None
    df_cocal = load_cocal_data(cocal_file, aggregate=aggregate, filter=filter, coordinates=coordinates)

    # Merge on hourly timestamps
    df_cocal['timestamp'] = df_cocal['timestamp'].dt.tz_localize(None)
    if df_arpa is not None:
        df_arpa['timestamp'] = df_arpa['timestamp'].dt.tz_localize(None)
        df = df_cocal.merge(df_arpa, on="timestamp")
        if 'ARPA' not in list(df.columns)[-1].upper():
            raise ValueError(f'The last column here must be the PM from ARPA.')
    else:
        df = df_cocal
    #df = df.drop(columns=['PM25_MEAN', 'PM25_MIN', 'PM25_MAX']) if 'PM10' in list(df.columns)[-1].upper() else df.drop(columns=['PM10_MEAN', 'PM10_MIN', 'PM10_MAX'])

    # Derive time-related attributes
    df = derive_time_attributes(df)
    
    df['season'] = pd.Categorical(df['season'], categories=sorted(['autumn', 'spring', 'summer', 'winter']), ordered=False)
    df['time_of_day'] = pd.Categorical(df['time_of_day'], categories=sorted(['afternoon', 'evening', 'morning', 'night']), ordered=False)
    categorical_cols = df.select_dtypes(include=['category']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, dtype=int)

    # Drop eventual rows with null or NaN
    df.dropna(axis='index', how='any', inplace=True, ignore_index=True)

    # Drop duplicate rows
    df.drop_duplicates(keep='first', inplace=True, ignore_index=True)

    # Retrieve timestamps and drop them
    #timestamps = df['timestamp']
    #df = df.drop(columns=['timestamp'])

    # Move PM10_MEAN and PM25_MEAN just before the last column (usually 'delta' or ARPA column)
    last_col = df.columns[-1]
    cols = df.columns.tolist()

    # Remove PM10 and PM25 if present
    for col in ['PM10', 'PM25']:
        if col in cols:
            cols.remove(col)

    # Insert them before the last column
    insert_index = cols.index(last_col)
    for col in ['PM25', 'PM10'][::-1]:  # Reverse to maintain order
        if col in df.columns:
            cols.insert(insert_index, col)

    df = df[cols]

    return df

# =============================================================================
# CSV Data Saving and Loading
# =============================================================================

def save_csv_data(df, path):
    """Save the DataFrame to a .csv file."""
    df.to_csv(path, sep=',', index=False)


def load_data(file_path):    
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path, delimiter=',', decimal='.') 
    return df

# =============================================================================
# Model Application and Evaluation
# =============================================================================


def apply_models(df: pd.DataFrame, ref_lat: float, ref_lon: float, max_distance: float) -> pd.DataFrame:
    df["distance"] = haversine(df["lat"], df["lon"], ref_lat, ref_lon)
    center_mask = df["distance"] <= max_distance
    far_mask = df["distance"] > max_distance
    
    # Split data based on distance
    df_center = df[center_mask].copy()
    df_far = df[far_mask].copy()
    df_center.reset_index(drop=True, inplace=True)
    df_far.reset_index(drop=True, inplace=True)

    # Apply models
    preds_center_sr = sr_center_model(df_center) if len(df_center) > 0 else np.array([])
    preds_far_el = el_far_model(df_far) if len(df_far) > 0 else np.array([])

    # Concatenate predictions
    preds = np.zeros(len(df))
    preds[center_mask] = preds_center_sr
    preds[far_mask] = preds_far_el

    df['PM10_SR_EN'] = preds
    
    return df

def evaluate_models(df: pd.DataFrame) -> dict:
    y_true = df['PM10_ARPA']
    y_pred = df['PM10_SR_EN']

    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    

# =============================================================================
# Augmenting Original Data with Predictions
# =============================================================================

def convert_data_back_to_cocal_format_with_predictions(cocal_file: str, df_aggregated_with_preds: pd.DataFrame) -> pd.DataFrame:
    # Make a copy of the original cocal data
    cocal_df = load_data(cocal_file)

    sample_df = cocal_df[cocal_df['sensor'] == 'PM10'].copy()
    # Ensure time column is datetime in both dataframes
    sample_df["time"] = sample_df["time"].str.split(".").str[0]  # Keep only the part before the dot
    sample_df["time"] = pd.to_datetime(sample_df["time"], format="%Y-%m-%d %H:%M:%S")

    sample_df.drop(columns=['val', 'sensor'], inplace=True)
    sample_df['sensor'] = 'PM10_SR_EN'

    # Ensure time in aggregated data is datetime
    df_aggregated_with_preds['timestamp'] = pd.to_datetime(df_aggregated_with_preds['timestamp'], format="%Y-%m-%d %H:%M:%S")
    df_aggregated_with_preds = df_aggregated_with_preds[['timestamp', 'lat', 'lon', 'device', 'PM10_SR_EN']].copy()
    df_aggregated_with_preds.rename(columns={"timestamp": "time"}, inplace=True)

    # Join to get predictions
    merged_df = sample_df.merge(df_aggregated_with_preds, on=['time', 'lat', 'lon', 'device'])
    merged_df.rename(columns={"PM10_SR_EN": "val"}, inplace=True)

    print(cocal_df.head(20))
    print(merged_df.head(20))

    # Add the rows from merged_df to cocal_df and then order by timestamp asceindingly
    cocal_df = pd.concat([cocal_df, merged_df], ignore_index=True)
    # ensure the "time" column is in the correct datetime format and sort by it
    cocal_df['time'] = cocal_df['time'].astype(str)  # Ensure it's string before splitting
    cocal_df["time"] = cocal_df["time"].str.split(".").str[0]
    cocal_df['time'] = pd.to_datetime(cocal_df['time'], format="%Y-%m-%d %H:%M:%S")
    merged_df['time'] = merged_df['time'].astype(str)  # Ensure it's string before splitting
    merged_df['time'] = merged_df['time'].str.split(".").str[0]
    merged_df['time'] = pd.to_datetime(merged_df['time'], format="%Y-%m-%d %H:%M:%S")
    cocal_df = cocal_df.sort_values(by=['time', 'device']).reset_index(drop=True)
    print(cocal_df.head(50))
    return cocal_df


def main():
    """
    "Stazione","Lon","Lat","Indirizzo" 
    "Trieste - via Carpineto","13.787500000","45.6232","via del Carpineto - Trieste (TS)"
    "Trieste - piazza Carlo Alberto","13.756100000","45.6423","piazza Carlo Alberto - Trieste (TS)"
    "Trieste - Sincrotrone","13.855000000","45.6473","Basovizza - Trieste (TS)"
    "Trieste - P.zza Volontari Giuliani","13.789000000","45.6544","Piazza Volontari Giuliani - Trieste (TS)"
    "Trieste - Via del Ponticello","13.783800000","45.6206","via del Ponticello - Trieste (TS)"
    "Trieste - via Pitacco-ARPA","13.780300000","45.6242","VIA PITACCO - Trieste (TS)"
    "Trieste - p.le Rosmini","13.766100000","45.6405","p.le Rosmini - Trieste (TS)"
    "Trieste - RFI","13.779998","45.622565","Trieste RFI - Servola"
    """
    zone_to_coordinates = {
        'volontari': (45.6544, 13.7890),  # Trieste - P.zza Volontari Giuliani
        'carpineto': (45.6232, 13.7875),  # Trieste - via Carpineto
        'sincrotrone': (45.6473, 13.8550)  # Trieste - Sincrotrone
    }
    dict_zone_station = {'volontari': "Trieste - P.zza Volontari Giuliani",
                         'carpineto': "Trieste - via Carpineto",
                         'sincrotrone': "Trieste - Sincrotrone"}
    # Process data (aggregation, feature engineering)
    zone = 'carpineto'
    #df_aggregated = eventually_merge_and_extract_time_categories(cocal_file=f'data_pre/raw/cocal/2025/{zone}/cocal.csv', aggregate=True, filter=True, arpa_file=f'data_pre/raw/arpa/2025/pm10/{zone}/arpa.csv', station_name=dict_zone_station[zone], coordinates=zone_to_coordinates[zone], parameter_name="Particelle sospese PM10")
    df_aggregated = eventually_merge_and_extract_time_categories(cocal_file=f'data_pre/raw/cocal/2025/{zone}/cocal.csv')
    # Apply ML models to get predictions
    df_with_predictions = apply_models(df_aggregated, ref_lat=REF_LAT, ref_lon=REF_LON, max_distance=DISTANCE)
    
    #print(evaluate_models(df_with_predictions))
    
    # Convert back to COCAL format with predictions
    df_cocal_with_preds = convert_data_back_to_cocal_format_with_predictions(cocal_file=f'data_pre/raw/cocal/2025/{zone}/cocal.csv', df_aggregated_with_preds=df_with_predictions)
    
    # Save the result
    #save_csv_data(df_with_predictions, 'data_pre/processed/with_predictions.csv')


if __name__ == "__main__":
    main()
