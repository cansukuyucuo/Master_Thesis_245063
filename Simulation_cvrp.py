# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:32:06 2024

@author: CansuKuyucuoglu
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm 
import seaborn as sns
from datetime import datetime, timedelta

# Load the data from the given path
file_path = r'C:\Users\18360039068941666201\OneDrive - HelloFresh Group\Desktop\Uni_Cansu\Master_Thesis\database.csv'
data = pd.read_csv(file_path, low_memory=False)

# Limit data to the year 2020 before preprocessing
data['departure'] = pd.to_datetime(data['departure'])
data = data[data['departure'].dt.year == 2020]

# Function to get sunrise and sunset times (using placeholder values for simplicity)
def get_daylight_hours(departure_time):
    sunrise_hour = 6  # Placeholder sunrise time
    sunset_hour = 18  # Placeholder sunset time
    return 1 if sunrise_hour <= departure_time.hour <= sunset_hour else 0

# Function to remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def preprocess_data(data):
    data['departure_id'] = data['departure_id'].astype(str).str.replace('.0', '', regex=False)
    data['return_id'] = data['return_id'].astype(str).str.replace('.0', '', regex=False)
    data['departure'] = pd.to_datetime(data['departure'])
    data['return'] = pd.to_datetime(data['return'])
    data['distance (km)'] = data['distance (m)'] / 1000.0
    data['duration (min)'] = data['duration (sec.)'] / 60.0
    data.drop(columns=['distance (m)', 'duration (sec.)'], inplace=True)
    data = data[(data['distance (km)'] >= 0) & (data['duration (min)'] >= 0)]
    data = data.dropna(subset=['departure', 'return', 'departure_id', 'return_id', 'distance (km)', 'return_latitude', 'return_longitude', 'avg_speed (km/h)'])    
    data = remove_outliers(data, 'distance (km)')
    data = remove_outliers(data, 'duration (min)')    
    data['departure_year'] = data['departure'].dt.year
    data['departure_month'] = data['departure'].dt.month
    data['departure_day'] = data['departure'].dt.dayofweek
    data['departure_day_of_month'] = data['departure'].dt.day
    data['departure_hour'] = data['departure'].dt.hour   
    top_stations_d = data['departure_id'].value_counts().head(20).index
    data['is_top_station_d'] = data['departure_id'].apply(lambda x: 1 if x in top_stations_d else 0)
      
    mask = data['departure_id'] == data['return_id']
    data = data[~mask]
    
    # Add daylight information
    data['is_daylight_d'] = data['departure'].apply(get_daylight_hours)
        
    # Add cyclic features for departure_hour and departure_day
    data['departure_hour_sin'] = np.sin(2 * np.pi * data['departure_hour'] / 24)
    data['departure_hour_cos'] = np.cos(2 * np.pi * data['departure_hour'] / 24)
        
    # Add cyclic features for return_hour and return_day
    data['return_year'] = data['return'].dt.year
    data['return_month'] = data['return'].dt.month
    data['return_day'] = data['return'].dt.dayofweek
    data['return_day_of_month'] = data['return'].dt.day
    data['return_hour'] = data['return'].dt.hour
    data['is_daylight_r'] = data['return'].apply(get_daylight_hours)   
    top_stations_r = data['return_id'].value_counts().head(20).index
    data['is_top_station_r'] = data['return_id'].apply(lambda x: 1 if x in top_stations_r else 0)
    
    data['return_hour_sin'] = np.sin(2 * np.pi * data['return_hour'] / 24)
    data['return_hour_cos'] = np.cos(2 * np.pi * data['return_hour'] / 24)
        
    return data

# Preprocess the loaded data
preprocessed_data = preprocess_data(data)

# Function to aggregate demand data hourly
def aggregate_demand(data, id_col, hour_col, demand_col):
    data[id_col] = data[id_col].astype(str)
    demand_data = data.groupby([id_col, hour_col], observed=False).size().reset_index(name=demand_col)
    data = pd.merge(data, demand_data, on=[id_col, hour_col], how='left')
    return data

# Aggregate demand data for departures and returns
preprocessed_data = aggregate_demand(preprocessed_data, 'departure_id', 'departure_hour', 'departure_demand')
preprocessed_data = aggregate_demand(preprocessed_data, 'return_id', 'return_hour', 'return_demand')

# Function to compute the Haversine distance between two points in kilometers
def haversine_distance(lat1, lon1, lat2, lon2):
    # Radius of Earth in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Distance in kilometers
    return R * c

# Updated add_location_features function using custom Haversine function
def add_location_features(data):
    city_center_coords = (60.1708, 24.9375)  # Helsinki's coordinates
    data['departure_distance_to_center'] = haversine_distance(
        data['departure_latitude'], data['departure_longitude'], city_center_coords[0], city_center_coords[1]
    )
    data['return_distance_to_center'] = haversine_distance(
        data['return_latitude'], data['return_longitude'], city_center_coords[0], city_center_coords[1]
    )
    return data

preprocessed_data = add_location_features(preprocessed_data)

# Clustering stations based on their locations and usage patterns
kmeans = KMeans(n_clusters=5, random_state=42)
preprocessed_data['departure_station_cluster'] = kmeans.fit_predict(preprocessed_data[['departure_latitude', 'departure_longitude']])
preprocessed_data['return_station_cluster'] = kmeans.fit_predict(preprocessed_data[['return_latitude', 'return_longitude']])

# Features and target variable for departure demand prediction
features_departure = [
    'is_daylight_d', 'is_top_station_d', 'departure_station_cluster',
    'departure_hour_sin', 'departure_hour_cos'
]
y_departure = preprocessed_data['departure_demand']

# Features and target variable for return demand prediction
features_return = [
    'is_daylight_r', 'is_top_station_r', 'return_station_cluster',
    'return_hour_sin', 'return_hour_cos'
]
y_return = preprocessed_data['return_demand']

# Create polynomial features for departure and return
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# Polynomial features for departure
X_departure_poly = poly.fit_transform(preprocessed_data[['departure_hour', 'departure_distance_to_center']])
departure_poly_features = poly.get_feature_names_out(['departure_hour', 'departure_distance_to_center'])
departure_poly_df = pd.DataFrame(X_departure_poly, columns=departure_poly_features)

# Polynomial features for return
X_return_poly = poly.fit_transform(preprocessed_data[['return_hour', 'return_distance_to_center']])
return_poly_features = poly.get_feature_names_out(['return_hour', 'return_distance_to_center'])
return_poly_df = pd.DataFrame(X_return_poly, columns=return_poly_features)

# Drop the original columns used for polynomial features from the feature sets
X_departure = preprocessed_data[features_departure].copy()  # Copy to avoid altering the original data
X_return = preprocessed_data[features_return].copy()

# Join polynomial features with the original features
X_departure = pd.concat([X_departure, pd.DataFrame(X_departure_poly, columns=departure_poly_features)], axis=1)
X_return = pd.concat([X_return, pd.DataFrame(X_return_poly, columns=return_poly_features)], axis=1)

# Standardize the features
scaler = StandardScaler()
X_departure_scaled = scaler.fit_transform(X_departure) 
X_return_scaled = scaler.fit_transform(X_return)

# Step 6: Combine the original departure features with the polynomial features
X_departure_combined = pd.concat([X_departure, departure_poly_df], axis=1)
X_return_combined = pd.concat([X_return, return_poly_df], axis=1)

# Split the data for departure and return demand prediction
X_train_dep, X_test_dep, y_train_dep, y_test_dep = train_test_split(X_departure_scaled, y_departure, test_size=0.2, random_state=42)
X_train_ret, X_test_ret, y_train_ret, y_test_ret = train_test_split(X_return_scaled, y_return, test_size=0.2, random_state=42)

# Function to find the representative week
def find_representative_week(preprocessed_data):
    preprocessed_data['week_of_year'] = preprocessed_data['departure'].dt.isocalendar().week

    weekly_stats = preprocessed_data.groupby('week_of_year').agg(
        weekly_mean_departure=('departure_demand', 'mean'),
        weekly_std_departure=('departure_demand', 'std'),
        weekly_mean_return=('return_demand', 'mean'),
        weekly_std_return=('return_demand', 'std')
    ).reset_index()

    yearly_mean_departure = preprocessed_data['departure_demand'].mean()
    yearly_std_departure = preprocessed_data['departure_demand'].std()
    yearly_mean_return = preprocessed_data['return_demand'].mean()
    yearly_std_return = preprocessed_data['return_demand'].std()

    weekly_stats['departure_mean_diff'] = np.abs(weekly_stats['weekly_mean_departure'] - yearly_mean_departure)
    weekly_stats['departure_std_diff'] = np.abs(weekly_stats['weekly_std_departure'] - yearly_std_departure)
    weekly_stats['return_mean_diff'] = np.abs(weekly_stats['weekly_mean_return'] - yearly_mean_return)
    weekly_stats['return_std_diff'] = np.abs(weekly_stats['weekly_std_return'] - yearly_std_return)

    weekly_stats['total_diff'] = (
        weekly_stats['departure_mean_diff'] + 
        weekly_stats['departure_std_diff'] + 
        weekly_stats['return_mean_diff'] + 
        weekly_stats['return_std_diff']
    )

    representative_week = weekly_stats.loc[weekly_stats['total_diff'].idxmin()]

    print(f"Representative Week Number: {representative_week['week_of_year']}")
    return representative_week['week_of_year']

# Function to extract data for the representative week
def extract_week_data(preprocessed_data, week_number):
    return preprocessed_data[preprocessed_data['week_of_year'] == week_number]

# Step 1: Find the representative week
representative_week_number = find_representative_week(preprocessed_data)

# Step 2: Extract data for the representative week
week_data = extract_week_data(preprocessed_data, representative_week_number)

# Step 3: Extract the departure and return hour distributions from the historical data
departure_hour_dist = week_data['departure_hour'].value_counts(normalize=True).sort_index()
return_hour_dist = week_data['return_hour'].value_counts(normalize=True).sort_index()

# Step 4: Extract the mapping of departure hour to return hour based on historical data
departure_to_return_hour_mapping = week_data.groupby('departure_hour')['return_hour'].apply(lambda x: x.mode()[0])

# Define the start and end date for the future simulation
simulation_start_date = datetime(2020, 6, 1)
simulation_end_date = simulation_start_date + timedelta(days=7)

# Generate hourly timestamps
future_timestamps = pd.date_range(start=simulation_start_date, end=simulation_end_date, freq='H')

# Ensure that future data maintains the same structure as week data
stations = week_data['departure_id'].unique()

# Create future data DataFrame
future_data = pd.DataFrame({
    'departure_id': np.repeat(stations, len(future_timestamps)),
    'departure': np.tile(future_timestamps, len(stations))
})

# Step 5: Assign departure hours based on the departure timestamps (simulated data)
future_data['departure_hour'] = future_data['departure'].dt.hour

# Step 6: Assign return hour using the historical mapping from departure_hour to return_hour
future_data['return_hour'] = future_data['departure_hour'].map(departure_to_return_hour_mapping)

# Add cyclic features for departure and return hours
future_data['departure_hour_sin'] = np.sin(2 * np.pi * future_data['departure_hour'] / 24)
future_data['departure_hour_cos'] = np.cos(2 * np.pi * future_data['departure_hour'] / 24)

future_data['return_hour_sin'] = np.sin(2 * np.pi * future_data['return_hour'] / 24)
future_data['return_hour_cos'] = np.cos(2 * np.pi * future_data['return_hour'] / 24)

# Step 7: Simulate time difference between departure and return based on historical data
week_data['time_diff_hours'] = (week_data['return'] - week_data['departure']).dt.total_seconds() / 3600.0
time_diff_distribution = week_data['time_diff_hours'].describe()

# Simulate time difference based on historical data
future_data['time_diff_hours'] = np.random.normal(loc=time_diff_distribution['mean'], scale=time_diff_distribution['std'], size=len(future_data))
future_data['time_diff_hours'] = np.maximum(0, future_data['time_diff_hours'])  # Ensure time difference is non-negative

# Calculate return timestamp by adding the time difference to the departure timestamp
future_data['return'] = future_data['departure'] + pd.to_timedelta(future_data['time_diff_hours'], unit='h')

# Step 8: Add daylight information after 'return' is created
future_data['is_daylight_d'] = future_data['departure'].apply(get_daylight_hours)
future_data['is_daylight_r'] = future_data['return'].apply(get_daylight_hours)

# Merge station features from the historical data
station_features = week_data[['departure_id', 'is_top_station_d', 'departure_station_cluster', 'departure_distance_to_center']].drop_duplicates()
future_data = future_data.merge(station_features, on='departure_id', how='left')

# Step 9: Extract departure and return station relationship from the historical data
departure_return_pairs = week_data.groupby(['departure_id', 'return_id']).size().reset_index(name='counts')

# Calculate the probability of returning to each station given a departure station
departure_return_probs = departure_return_pairs.groupby('departure_id').apply(
    lambda x: x[['return_id', 'counts']].set_index('return_id')['counts'] / x['counts'].sum()
).reset_index()

# Function to sample return_id based on departure_id and historical probabilities
def sample_return_id(departure_id):
    possible_returns = departure_return_probs[departure_return_probs['departure_id'] == departure_id]
    if not possible_returns.empty:
        return np.random.choice(possible_returns['return_id'], p=possible_returns['counts'])
    else:
        # If no data available for this departure station, return a random station
        return np.random.choice(stations)

# Sample return_id for each departure_id
future_data['return_id'] = future_data['departure_id'].apply(sample_return_id)

# Merge return station features
station_features_return = week_data[['return_id', 'is_top_station_r', 'return_station_cluster', 'return_distance_to_center']].drop_duplicates()
station_features_return.rename(columns={'return_id': 'departure_id'}, inplace=True)
future_data = future_data.merge(station_features_return, on='departure_id', how='left')

# Handle missing values
future_data['is_top_station_r'].fillna(0, inplace=True)
future_data['return_station_cluster'].fillna(-1, inplace=True)
future_data['return_distance_to_center'].fillna(future_data['return_distance_to_center'].mean(), inplace=True)

# Step 10: Simulate departure and return demand based on historical patterns
departure_demand_hourly = week_data.groupby('departure_hour')['departure_demand'].agg(['mean', 'std'])
return_demand_hourly = week_data.groupby('return_hour')['return_demand'].agg(['mean', 'std'])

# Function to sample demand based on historical patterns
def sample_demand(hour, demand_distribution):
    mean = demand_distribution.loc[hour, 'mean']
    std = demand_distribution.loc[hour, 'std']
    return max(0, np.random.normal(mean, std))  # Ensure demand is non-negative

# Sample departure and return demand
future_data['simulated_departure_demand'] = future_data['departure_hour'].apply(
    lambda x: sample_demand(x, departure_demand_hourly)
)

future_data['simulated_return_demand'] = future_data['return_hour'].apply(
    lambda x: sample_demand(x, return_demand_hourly)
)

# Ensure demands are rounded to integers
future_data['simulated_departure_demand'] = future_data['simulated_departure_demand'].round().astype(int)
future_data['simulated_return_demand'] = future_data['simulated_return_demand'].round().astype(int)

# Display a summary of the future_data to check its structure
summary = future_data.describe(include='all')

# Check the unique values for 'departure_hour' and 'return_hour' to confirm they are covering all hours
departure_hours_unique = future_data['departure_hour'].unique()
return_hours_unique = future_data['return_hour'].unique()

# Group by 'day_identifier' and 'departure_hour' to see the number of records (stations) for each hour in each day
hourly_departure_summary = future_data.groupby(['day_identifier', 'departure_hour']).size().reset_index(name='departure_station_count')
hourly_return_summary = future_data.groupby(['day_identifier', 'return_hour']).size().reset_index(name='return_station_count')

# Group by day and hour, count the number of unique departure and return stations
hourly_summary = future_data.groupby(['day_identifier', 'departure_hour']).agg(
    total_departure_stations=('departure_id', 'nunique'),
    total_return_stations=('return_id', 'nunique')
).reset_index()

# Display the DataFrame
print(hourly_summary)
# Display the results for inspection
summary, departure_hours_unique, return_hours_unique

import pandas as pd

# Ensure the 'departure' column is in datetime format
week_data['departure'] = pd.to_datetime(week_data['departure'])

# Create 'day_identifier' and 'departure_hour' columns based on the 'departure' column
week_data['day_identifier'] = week_data['departure'].dt.date
week_data['departure_hour'] = week_data['departure'].dt.hour

# Group by 'day_identifier' and 'departure_hour' to count unique departure and return stations
historical_summary = week_data.groupby(['day_identifier', 'departure_hour']).agg(
    total_departure_stations=('departure_id', 'nunique'),
    total_return_stations=('return_id', 'nunique')
).reset_index()

# Display the summary to check the consistency
print(historical_summary)

# Function to adjust the number of unique stations in future data based on historical data
def adjust_future_data(future_data, historical_summary):
    for _, row in historical_summary.iterrows():
        # Filter future_data for the specific day and hour
        day = row['day_identifier']
        hour = row['departure_hour']
        dep_count = row['total_departure_stations']
        ret_count = row['total_return_stations']
        
        # Filter future data for the given day and hour
        mask = (future_data['day_identifier'] == day) & (future_data['departure_hour'] == hour)
        future_subset = future_data[mask]
        
        # Adjust the number of unique departure and return stations
        current_departure_ids = future_subset['departure_id'].unique()
        current_return_ids = future_subset['return_id'].unique()
        
        if len(current_departure_ids) > dep_count:
            # Reduce the number of departure stations if too many
            drop_count = len(current_departure_ids) - dep_count
            drop_ids = np.random.choice(current_departure_ids, drop_count, replace=False)
            future_data = future_data[~((future_data['day_identifier'] == day) & 
                                        (future_data['departure_hour'] == hour) & 
                                        (future_data['departure_id'].isin(drop_ids)))]
        elif len(current_departure_ids) < dep_count:
            # Add more rows to increase the number of departure stations if too few
            needed_count = dep_count - len(current_departure_ids)
            additional_rows = future_subset.sample(needed_count, replace=True).copy()
            additional_rows['departure_id'] = np.random.choice(current_departure_ids, needed_count)
            future_data = pd.concat([future_data, additional_rows], ignore_index=True)
        
        if len(current_return_ids) > ret_count:
            # Reduce the number of return stations if too many
            drop_count = len(current_return_ids) - ret_count
            drop_ids = np.random.choice(current_return_ids, drop_count, replace=False)
            future_data = future_data[~((future_data['day_identifier'] == day) & 
                                        (future_data['departure_hour'] == hour) & 
                                        (future_data['return_id'].isin(drop_ids)))]
        elif len(current_return_ids) < ret_count:
            # Add more rows to increase the number of return stations if too few
            needed_count = ret_count - len(current_return_ids)
            additional_rows = future_subset.sample(needed_count, replace=True).copy()
            additional_rows['return_id'] = np.random.choice(current_return_ids, needed_count)
            future_data = pd.concat([future_data, additional_rows], ignore_index=True)

    return future_data

# Adjust the future data based on the historical summary
future_data = adjust_future_data(future_data, historical_summary)

# Group the adjusted future data to verify the changes
adjusted_summary = future_data.groupby(['day_identifier', 'departure_hour']).agg(
    total_departure_stations=('departure_id', 'nunique'),
    total_return_stations=('return_id', 'nunique')
).reset_index()

# Display the adjusted summary to check if it matches the historical pattern
print(adjusted_summary)

# Visualize simulated vs historical demand
def plot_demand_comparison():
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=week_data['departure_hour'], y=week_data['departure_demand'], label='Historical Departure Demand', estimator='mean', ci=None, color='blue')
    sns.lineplot(x=future_data['departure_hour'], y=future_data['simulated_departure_demand'], label='Simulated Departure Demand', estimator='mean', ci=None, color='orange')
    plt.title('Comparison of Departure Demand (Historical vs Simulated)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Demand')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=week_data['return_hour'], y=week_data['return_demand'], label='Historical Return Demand', estimator='mean', ci=None, color='blue')
    sns.lineplot(x=future_data['return_hour'], y=future_data['simulated_return_demand'], label='Simulated Return Demand', estimator='mean', ci=None, color='orange')
    plt.title('Comparison of Return Demand (Historical vs Simulated)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Demand')
    plt.legend()
    plt.show()

plot_demand_comparison()

# Print the final structure of future_data
print(future_data[['departure_id', 'return_id', 'departure_hour', 'simulated_departure_demand', 'return_hour', 'simulated_return_demand']].head())

future_data.to_csv('future_data.csv', index=False)

# Define the range of weeks
week_number_start = week_data['week_of_year'].min() - 2
week_number_end = week_data['week_of_year'].max() + 2

# Filter the data to get weeks 2 weeks before and after
extended_historical_data = preprocessed_data[(preprocessed_data['week_of_year'] >= week_number_start) & 
                                             (preprocessed_data['week_of_year'] <= week_number_end)]

# Step 1: Extract the same features as used in historical data
features_departure = [
    'is_daylight_d', 'is_top_station_d', 'departure_station_cluster',
    'departure_hour_sin', 'departure_hour_cos'
]
features_return = [
    'is_daylight_r', 'is_top_station_r', 'return_station_cluster',
    'return_hour_sin', 'return_hour_cos'
]

# Step 2: Apply the same polynomial transformation to the future data
# Same as done for historical data
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# Polynomial features for departure in the future data
X_future_departure_poly = poly.fit_transform(future_data[['departure_hour', 'departure_distance_to_center']])
departure_poly_features = poly.get_feature_names_out(['departure_hour', 'departure_distance_to_center'])

# Polynomial features for return in the future data
X_future_return_poly = poly.fit_transform(future_data[['return_hour', 'return_distance_to_center']])
return_poly_features = poly.get_feature_names_out(['return_hour', 'return_distance_to_center'])

# Step 3: Prepare the future departure and return data (including the new polynomial features)
# Extract departure and return features
X_future_departure = future_data[features_departure].copy()
X_future_return = future_data[features_return].copy()

# Add polynomial features to the future data
X_future_departure = pd.concat([X_future_departure, pd.DataFrame(X_future_departure_poly, columns=departure_poly_features)], axis=1)
X_future_return = pd.concat([X_future_return, pd.DataFrame(X_future_return_poly, columns=return_poly_features)], axis=1)

# Step 4: Ensure the future data has the correct feature names (matching the training data)
# This ensures we have the same columns as in the historical data preparation
X_future_departure = X_future_departure.reindex(columns=X_departure_combined.columns, fill_value=0)
X_future_return = X_future_return.reindex(columns=X_return_combined.columns, fill_value=0)

# Step 5: Standardize the future data using the same scalers that were used on the historical data
departure_scaler = StandardScaler()
return_scaler = StandardScaler()

# Fit the scalers based on the current future data (or you can apply fitted scalers from historical data if available)
X_future_departure_scaled = departure_scaler.fit_transform(X_future_departure)  # Or use .transform if you already have fitted scalers
X_future_return_scaled = return_scaler.fit_transform(X_future_return)

# Combine the historical week_data with the entire simulated future_data
combined_data = pd.concat([extended_historical_data, future_data], ignore_index=True)
# Create polynomial features for departure and return using the combined dataset
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# Polynomial features for departure
X_combined_departure_poly = poly.fit_transform(combined_data[['departure_hour', 'departure_distance_to_center']])
departure_poly_features = poly.get_feature_names_out(['departure_hour', 'departure_distance_to_center'])

# Polynomial features for return
X_combined_return_poly = poly.fit_transform(combined_data[['return_hour', 'return_distance_to_center']])
return_poly_features = poly.get_feature_names_out(['return_hour', 'return_distance_to_center'])

# Prepare combined departure and return data
X_combined_departure = combined_data[features_departure].copy()
X_combined_return = combined_data[features_return].copy()

# Add polynomial features to the combined data
X_combined_departure = pd.concat([X_combined_departure, pd.DataFrame(X_combined_departure_poly, columns=departure_poly_features)], axis=1)
X_combined_return = pd.concat([X_combined_return, pd.DataFrame(X_combined_return_poly, columns=return_poly_features)], axis=1)

# Target variable (departure and return demand)
y_combined_departure = combined_data['departure_demand'].combine_first(combined_data['simulated_departure_demand'])
y_combined_return = combined_data['return_demand'].combine_first(combined_data['simulated_return_demand'])

# Standardize the features for both departure and return
departure_scaler = StandardScaler()
X_combined_departure_scaled = departure_scaler.fit_transform(X_combined_departure)

return_scaler = StandardScaler()
X_combined_return_scaled = return_scaler.fit_transform(X_combined_return)

# Use departure_day to split the data
combined_data['departure_day'] = combined_data['departure'].dt.day
last_day = combined_data['departure_day'].max()

# Split data into training (first 5 days) and test (last 2 days)
train_data = combined_data[combined_data['departure_day'] < (last_day - 1)]
test_data = combined_data[combined_data['departure_day'] >= (last_day - 1)]

# Split the scaled features and targets for departure and return
X_train_departure = X_combined_departure_scaled[train_data.index]
X_test_departure = X_combined_departure_scaled[test_data.index]
y_train_departure = y_combined_departure.loc[train_data.index]
y_test_departure = y_combined_departure.loc[test_data.index]

X_train_return = X_combined_return_scaled[train_data.index]
X_test_return = X_combined_return_scaled[test_data.index]
y_train_return = y_combined_return.loc[train_data.index]
y_test_return = y_combined_return.loc[test_data.index]

param_grids = {
    'Lasso Regression': {'alpha': [0.01, 0.1, 1, 10]},
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5]
    }
}

# Use GridSearchCV for Lasso as the param space is small
def retrain_and_tune_model_with_grid_cv(X_train, y_train, model_name, model, param_grid):
    print(f"Training and tuning {model_name} with GridSearchCV...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation score for {model_name}: {grid_search.best_score_}")
    return grid_search.best_estimator_

# Use RandomizedSearchCV for Random Forest and Gradient Boosting
def retrain_and_tune_model_with_random_cv(X_train, y_train, model_name, model, param_grid, n_iter=10):
    print(f"Training and tuning {model_name} with RandomizedSearchCV...")
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter, cv=3, n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X_train, y_train)
    print(f"Best parameters for {model_name}: {random_search.best_params_}")
    print(f"Best cross-validation score for {model_name}: {random_search.best_score_}")
    return random_search.best_estimator_

# Train Lasso using GridSearchCV since the parameter space is small
lasso_best_departure = retrain_and_tune_model_with_grid_cv(X_train_departure, y_train_departure, 'Lasso Regression', Lasso(), param_grids['Lasso Regression'])
random_forest_best_departure = retrain_and_tune_model_with_random_cv(X_train_departure, y_train_departure, 'Random Forest', RandomForestRegressor(), param_grids['Random Forest'], n_iter=10)
gradient_boosting_best_departure = retrain_and_tune_model_with_random_cv(X_train_departure, y_train_departure, 'Gradient Boosting', GradientBoostingRegressor(), param_grids['Gradient Boosting'], n_iter=10)

# Repeat the process for the return demand prediction
lasso_best_return = retrain_and_tune_model_with_grid_cv(X_train_return, y_train_return, 'Lasso Regression', Lasso(), param_grids['Lasso Regression'])
random_forest_best_return = retrain_and_tune_model_with_random_cv(X_train_return, y_train_return, 'Random Forest', RandomForestRegressor(), param_grids['Random Forest'], n_iter=10)
gradient_boosting_best_return = retrain_and_tune_model_with_random_cv(X_train_return, y_train_return, 'Gradient Boosting', GradientBoostingRegressor(), param_grids['Gradient Boosting'], n_iter=10)

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {model_name} ---")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R² Score: {r2}")
    print("-" * 50)

# Evaluate the models for departure demand prediction
y_pred_departure_lasso = lasso_best_departure.predict(X_test_departure)
y_pred_departure_rf = random_forest_best_departure.predict(X_test_departure)
y_pred_departure_gb = gradient_boosting_best_departure.predict(X_test_departure)

evaluate_model(y_test_departure, y_pred_departure_lasso, "Tuned Lasso (Departure)")
evaluate_model(y_test_departure, y_pred_departure_rf, "Tuned Random Forest (Departure)")
evaluate_model(y_test_departure, y_pred_departure_gb, "Tuned Gradient Boosting (Departure)")

# Evaluate the models for return demand prediction
y_pred_return_lasso = lasso_best_return.predict(X_test_return)
y_pred_return_rf = random_forest_best_return.predict(X_test_return)
y_pred_return_gb = gradient_boosting_best_return.predict(X_test_return)

evaluate_model(y_test_return, y_pred_return_lasso, "Tuned Lasso (Return)")
evaluate_model(y_test_return, y_pred_return_rf, "Tuned Random Forest (Return)")
evaluate_model(y_test_return, y_pred_return_gb, "Tuned Gradient Boosting (Return)")

# Check the data types of each column to identify non-numeric columns
non_numeric_columns = future_data.select_dtypes(include=['object']).columns
print("Non-numeric columns in future_data:")
print(non_numeric_columns)

# Display unique values from the non-numeric columns to investigate further
for col in non_numeric_columns:
    print(f"\nUnique values in column '{col}':")
    print(future_data[col].unique())

# Exclude non-numeric columns before performing the operation
numeric_future_data = future_data.select_dtypes(include=[np.number])

# Check for very large or small values in numeric columns
large_values = numeric_future_data[(numeric_future_data > 1e6) | (numeric_future_data < -1e6)]
print("\nRows with very large or small values (greater than 1e6 or less than -1e6):")
print(large_values.dropna())

# Print a sample of the data before scaling to ensure no NaNs or invalid values
print(future_data[features_departure].head())
print(future_data[features_return].head())
# Check which rows and columns have NaNs in the scaled dataset
import numpy as np

# Find indices of rows that contain NaNs in the scaled departure dataset
nan_indices_departure = np.where(np.isnan(X_future_departure_scaled))
print("Rows and columns with NaNs in X_future_departure_scaled:")
print(nan_indices_departure)

# Find indices of rows that contain NaNs in the scaled return dataset
nan_indices_return = np.where(np.isnan(X_future_return_scaled))
print("Rows and columns with NaNs in X_future_return_scaled:")
print(nan_indices_return)

# Inspect the rows in future_data that correspond to the identified indices
problematic_rows = future_data.iloc[34850:35258]
print("\nProblematic rows in future_data:")
print(problematic_rows)

# Investigate columns with constant values in the problematic rows
for col in future_data.columns:
    if problematic_rows[col].nunique() == 1:
        print(f"Column '{col}' has a constant value in the problematic rows.")

# Drop the 'day_identifier' column from future_data
future_data = future_data.drop(columns=['day_identifier'])
# Identify if there are NaNs in columns used for polynomial features
print("Checking for NaNs in columns used for polynomial features:")
print(f"NaNs in 'departure_hour': {future_data['departure_hour'].isnull().sum()}")
print(f"NaNs in 'departure_distance_to_center': {future_data['departure_distance_to_center'].isnull().sum()}")
print(f"NaNs in 'return_hour': {future_data['return_hour'].isnull().sum()}")
print(f"NaNs in 'return_distance_to_center': {future_data['return_distance_to_center'].isnull().sum()}")

# Check if there are any zero values (if they might cause issues)
print("\nChecking for zeros in columns used for polynomial features:")
print(f"Zeros in 'departure_hour': {(future_data['departure_hour'] == 0).sum()}")
print(f"Zeros in 'departure_distance_to_center': {(future_data['departure_distance_to_center'] == 0).sum()}")
print(f"Zeros in 'return_hour': {(future_data['return_hour'] == 0).sum()}")
print(f"Zeros in 'return_distance_to_center': {(future_data['return_distance_to_center'] == 0).sum()}")

# Step 2: Apply the same polynomial transformation to the future data
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# Polynomial features for departure in the future data
X_future_departure_poly = poly.fit_transform(future_data[['departure_hour', 'departure_distance_to_center']])
departure_poly_features = poly.get_feature_names_out(['departure_hour', 'departure_distance_to_center'])

X_future_return_poly = poly.fit_transform(future_data[['return_hour', 'return_distance_to_center']])
return_poly_features = poly.get_feature_names_out(['return_hour', 'return_distance_to_center'])

# Step 3: Prepare the future departure and return data (including the new polynomial features)
# Extract departure and return features
X_future_departure = future_data[features_departure].copy()
X_future_return = future_data[features_return].copy()

# Add polynomial features to the future data
X_future_departure = pd.concat([X_future_departure, pd.DataFrame(X_future_departure_poly, columns=departure_poly_features)], axis=1)
X_future_return = pd.concat([X_future_return, pd.DataFrame(X_future_return_poly, columns=return_poly_features)], axis=1)

# Step 4: Ensure the future data has the correct feature names (matching the training data)
# This ensures we have the same columns as in the historical data preparation
X_future_departure = X_future_departure.reindex(columns=X_departure_combined.columns, fill_value=0)
X_future_return = X_future_return.reindex(columns=X_return_combined.columns, fill_value=0)
# Drop duplicate columns in departure and return DataFrames
X_departure_combined = X_departure_combined.loc[:, ~X_departure_combined.columns.duplicated()]
X_return_combined = X_return_combined.loc[:, ~X_return_combined.columns.duplicated()]

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Create the SimpleImputer instance with strategy 'mean'
imputer = SimpleImputer(strategy='mean')

# Apply the imputer to the departure data
X_future_departure_imputed = imputer.fit_transform(X_future_departure)

# Apply the imputer to the return data
X_future_return_imputed = imputer.fit_transform(X_future_return)

# Step 3: Standardize the data after imputing
departure_scaler = StandardScaler()
return_scaler = StandardScaler()

# Fit and transform for departure and return features after imputing
X_future_departure_scaled = departure_scaler.fit_transform(X_future_departure_imputed)
X_future_return_scaled = return_scaler.fit_transform(X_future_return_imputed)

# Check if the NaN values persist after imputation and scaling
print("Checking for NaNs after imputing and scaling...")
print("NaNs in X_future_departure_scaled:", np.isnan(X_future_departure_scaled).sum())
print("NaNs in X_future_return_scaled:", np.isnan(X_future_return_scaled).sum())

# Trim X_future_departure_scaled to match the length of future_data if needed
if X_future_departure_scaled.shape[0] > future_data.shape[0]:
    X_future_departure_scaled = X_future_departure_scaled[:future_data.shape[0], :]
    
# Trim X_future_departure_scaled to match the length of future_data if needed
if X_future_return_scaled.shape[0] > future_data.shape[0]:
    X_future_return_scaled = X_future_return_scaled[:future_data.shape[0], :]

# Step 5: Predict departure demand using fine-tuned models
future_data['dep_predicted_demand_lasso'] = np.maximum(lasso_best_departure.predict(X_future_departure_scaled), 0)
future_data['dep_predicted_demand_random'] = np.maximum(random_forest_best_departure.predict(X_future_departure_scaled), 0)
future_data['dep_predicted_demand_gradient'] = np.maximum(gradient_boosting_best_departure.predict(X_future_departure_scaled), 0)

# Step 6: Predict return demand using fine-tuned models
future_data['ret_predicted_demand_lasso'] = np.maximum(lasso_best_return.predict(X_future_return_scaled), 0)
future_data['ret_predicted_demand_random'] = np.maximum(random_forest_best_return.predict(X_future_return_scaled), 0)
future_data['ret_predicted_demand_gradient'] = np.maximum(gradient_boosting_best_return.predict(X_future_return_scaled), 0)

# Step 7: Save the future data with predictions
future_data.to_csv('future_data_with_predictions.csv', index=False)

print("Predictions (including Lasso) saved successfully in the 'future_data_with_predictions.csv' file.")

'''SIMULATION CVRP'''

# Load the data from the given path
file_path1 = r'C:\Users\18360039068941666201\future_data_with_predictions.csv'
future_data = pd.read_csv(file_path1, low_memory=False)

# Step 1: Drop existing 'avg_historical' columns if they exist
future_data = future_data.drop(columns=[col for col in future_data.columns if 'historical' in col], errors='ignore')

# Ensure both columns are of the same type (string) for consistent merging
preprocessed_data['departure_id'] = preprocessed_data['departure_id'].astype(str)
preprocessed_data['return_id'] = preprocessed_data['return_id'].astype(str)
future_data['departure_id'] = future_data['departure_id'].astype(str)
future_data['return_id'] = future_data['return_id'].astype(str)

# Step 2: Calculate average historical demand for each station and hour in preprocessed_data
historical_demand_departure = preprocessed_data.groupby(['departure_id', 'departure_hour']).agg(
    historical_departure_demand=('departure_demand', 'mean')
).reset_index()

historical_demand_return = preprocessed_data.groupby(['return_id', 'return_hour']).agg(
    historical_return_demand=('return_demand', 'mean')
).reset_index()

print(future_data['departure'].dtype)
future_data['departure'] = pd.to_datetime(future_data['departure'], errors='coerce')

# Step 3: Ensure 'hour' column is available in future_data
future_data['hour'] = future_data['departure'].dt.hour

# Step 4: Merge the average historical demand for departures with future_data using 'departure_id'
future_data = pd.merge(
    future_data,
    historical_demand_departure.rename(columns={'departure_id': 'departure_id', 'departure_hour': 'hour'}),
    how='left',
    on=['departure_id', 'hour']
)

# Merge the average historical demand for returns with future_data using 'return_id'
future_data = pd.merge(
    future_data,
    historical_demand_return.rename(columns={'return_id': 'return_id', 'return_hour': 'hour'}),
    how='left',
    on=['return_id', 'hour']
)

# Step 5: Fill NaN values with 0 where no historical data is available
future_data['historical_departure_demand'].fillna(0, inplace=True)
future_data['historical_return_demand'].fillna(0, inplace=True)

# Display the merged data to verify
print(future_data[['departure_id', 'return_id', 'hour', 'historical_departure_demand', 'historical_return_demand']].head())

def calculate_surplus_deficit(data, hour, demand_type='simulated'):
    """
    Calculates the surplus and deficit for each station based on simulated, predicted, or historical demand for a specific hour.
    """
    if demand_type == 'simulated':
        dep_col = 'simulated_departure_demand'
        ret_col = 'simulated_return_demand'
    elif demand_type == 'historical':
        dep_col = 'historical_departure_demand'
        ret_col = 'historical_return_demand'
    else:
        # For predicted models, extract the model name from demand_type
        model_name = demand_type.split("_")[1]  # Extract model name part
        dep_col = f'dep_predicted_demand_{model_name}'
        ret_col = f'ret_predicted_demand_{model_name}'

    # Filter data for the given hour
    hourly_data = data[data['departure_hour'] == hour]

    # Aggregate departure and return demands
    departure_demand = hourly_data.groupby('departure_id')[dep_col].sum().reset_index().rename(columns={'departure_id': 'station_id', dep_col: 'departure_demand'})
    return_demand = hourly_data.groupby('return_id')[ret_col].sum().reset_index().rename(columns={'return_id': 'station_id', ret_col: 'return_demand'})

    # Ensure that 'station_id' columns are of the same type (e.g., int64)
    departure_demand['station_id'] = departure_demand['station_id'].astype(int)
    return_demand['station_id'] = return_demand['station_id'].astype(int)

    # Merge to get net demand
    demand_data = pd.merge(departure_demand, return_demand, on='station_id', how='outer').fillna(0)
    demand_data['net_demand'] = demand_data['return_demand'] - demand_data['departure_demand']
    demand_data['surplus'] = demand_data['net_demand'].apply(lambda x: x if x > 0 else 0)
    demand_data['deficit'] = demand_data['net_demand'].apply(lambda x: -x if x < 0 else 0)

    return demand_data[['station_id', 'surplus', 'deficit']]

# Function to calculate and save surplus and deficit for all hours
def calculate_and_save_surplus_deficit(data):
    """
    Calculate and save the surplus and deficit based on simulated, predicted, and historical demands.
    """
    model_names = ['lasso', 'random', 'gradient']  

    # Initialize columns for simulated surplus and deficit
    data['simulated_surplus'] = 0.0
    data['simulated_deficit'] = 0.0
    data['historical_surplus'] = 0.0
    data['historical_deficit'] = 0.0

    # Initialize columns for predicted surplus and deficit for each model
    for model_name in model_names:
        data[f'{model_name}_surplus'] = 0.0
        data[f'{model_name}_deficit'] = 0.0

    # Calculate surplus and deficit for each hour
    for hour in tqdm(data['departure_hour'].unique(), desc="Processing Hours"):
        print(f"Processing Hour: {hour}")
        
        # Calculate simulated surplus and deficit for this hour
        simulated_results = calculate_surplus_deficit(data, hour, 'simulated')
        simulated_results_dict = simulated_results.set_index('station_id')[['surplus', 'deficit']].to_dict(orient='index')
        data.loc[data['departure_hour'] == hour, 'simulated_surplus'] = data.loc[data['departure_hour'] == hour, 'departure_id'].map(lambda x: simulated_results_dict.get(x, {}).get('surplus', 0))
        data.loc[data['departure_hour'] == hour, 'simulated_deficit'] = data.loc[data['departure_hour'] == hour, 'departure_id'].map(lambda x: simulated_results_dict.get(x, {}).get('deficit', 0))

        # Calculate historical surplus and deficit for this hour
        historical_results = calculate_surplus_deficit(data, hour, 'historical')
        historical_results_dict = historical_results.set_index('station_id')[['surplus', 'deficit']].to_dict(orient='index')
        data.loc[data['departure_hour'] == hour, 'historical_surplus'] = data.loc[data['departure_hour'] == hour, 'departure_id'].map(lambda x: historical_results_dict.get(x, {}).get('surplus', 0))
        data.loc[data['departure_hour'] == hour, 'historical_deficit'] = data.loc[data['departure_hour'] == hour, 'departure_id'].map(lambda x: historical_results_dict.get(x, {}).get('deficit', 0))

        # Process each prediction model similarly
        for model_name in model_names:
            print(f"Calculating surplus/deficit for {model_name} model in hour {hour}...")
            predicted_results = calculate_surplus_deficit(data, hour, f'predicted_{model_name}')
            predicted_results_dict = predicted_results.set_index('station_id')[['surplus', 'deficit']].to_dict(orient='index')
            data.loc[data['departure_hour'] == hour, f'{model_name}_surplus'] = data.loc[data['departure_hour'] == hour, 'departure_id'].map(lambda x: predicted_results_dict.get(x, {}).get('surplus', 0))
            data.loc[data['departure_hour'] == hour, f'{model_name}_deficit'] = data.loc[data['departure_hour'] == hour, 'departure_id'].map(lambda x: predicted_results_dict.get(x, {}).get('deficit', 0))

    print("Surplus and deficit calculated and saved for simulated, historical, and all prediction models.")
    return data

# Calculate surplus and deficit for the future data
future_data = calculate_and_save_surplus_deficit(future_data)

# Ensure both 'departure_id' columns are of the same type before merging
future_data['departure_id'] = future_data['departure_id'].astype(str)
week_data['departure_id'] = week_data['departure_id'].astype(str)

location_columns = ['departure_id', 'departure_latitude', 'departure_longitude']

# Check if these columns exist in week_data
if all(col in week_data.columns for col in location_columns):
    # Merge latitude and longitude from week_data into future_data based on 'departure_id'
    future_data = future_data.merge(
        week_data[location_columns].drop_duplicates(),
        how='left',
        on='departure_id'
    )
else:
    print("week_data does not contain the required location columns: 'departure_latitude', 'departure_longitude'.")

# Verify if the merge was successful by checking the resulting future_data
print(future_data[['departure_id', 'departure_latitude', 'departure_longitude']].head())

# Create distance matrix using Haversine formula
def create_distance_matrix_optimized(stations):
    latitudes = np.radians(stations['latitude'].values)
    longitudes = np.radians(stations['longitude'].values)
    num_stations = len(stations)
    distance_matrix = np.zeros((num_stations, num_stations))

    for i in range(num_stations):
        for j in range(i + 1, num_stations):
            lat1, lon1 = latitudes[i], longitudes[i]
            lat2, lon2 = latitudes[j], longitudes[j]
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            # Haversine formula
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            R = 6371.0  # Radius of Earth in kilometers
            distance = R * c

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    print("Distance matrix created successfully!")
    return distance_matrix


# Precompute the total distance from each station to all others
def precompute_distance_sums(distance_matrix):
    distance_sums = np.sum(distance_matrix, axis=1)
    print(f"Precomputed distance sums:\n{distance_sums[:5]}")  # Print the first 5
    return distance_sums


# Calculate the total distance using a reference station
def calculate_reference(distance_sums, reference_index):
    total_distance = distance_sums[reference_index]
    print(f"Calculated total distance for reference index {reference_index}: {total_distance}")
    return total_distance


# Find the best reference station
def find_best_reference_station(stations, distance_sums, station_id_to_index):
    min_total_distance = float('inf')
    best_reference_station = None

    print("Calculating Optimal Reference Station Based on Simulated Demand Data...")
    for ref_station in tqdm(stations['station_id'], desc="Reference Station Optimization"):
        reference_index = station_id_to_index[ref_station]
        total_distance = calculate_reference(distance_sums, reference_index)

        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_reference_station = ref_station

    print(f"Best Reference Station: {best_reference_station}, with Total Distance: {min_total_distance:.2f} km")
    return best_reference_station

# Assuming future_data contains 'departure_id', 'departure_latitude', 'departure_longitude' columns
stations = future_data[['departure_id', 'departure_latitude', 'departure_longitude']].drop_duplicates().reset_index(drop=True)
stations.columns = ['station_id', 'latitude', 'longitude']

# Create distance matrix and precompute distances
distance_matrix = create_distance_matrix_optimized(stations)
distance_sums = precompute_distance_sums(distance_matrix)
station_id_to_index = {station_id: idx for idx, station_id in enumerate(stations['station_id'])}
# Convert the distance matrix into a DataFrame for better visualization
distance_matrix_df = pd.DataFrame(
    distance_matrix,
    index=stations['station_id'],
    columns=stations['station_id']
)

# Plotting the heatmap
plt.figure(figsize=(14, 12))
ax = sns.heatmap(distance_matrix_df, cmap='coolwarm', cbar_kws={'label': 'Distance (km)'})
plt.title("Distance Matrix Heatmap (in km)", fontsize=16)
plt.xlabel("Station ID", fontsize=12)
plt.ylabel("Station ID", fontsize=12)

# Rotate x-axis labels for better visibility
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

# Find the best reference station
best_reference_station = find_best_reference_station(stations, distance_sums, station_id_to_index)

def clarke_wright_vrp_optimized(demand_data, distance_matrix, truck_capacity, reference_station_id, station_id_to_index, max_distance_per_day):
    """
    Optimized Clarke-Wright Savings algorithm adapted for VRP with dynamic updates to demand and surplus/deficit.
    """
    num_stations = len(demand_data)
    savings = []
    routes = {i: [i] for i in range(num_stations)}
    route_load = {i: 0 for i in range(num_stations)}  # Tracks the current load of bikes in each  
    total_bikes_carried = 0

    # Ensure the reference index is correctly retrieved as an integer
    reference_index = station_id_to_index.get(reference_station_id)
    if reference_index is None:
        raise ValueError(f"Reference station {reference_station_id} not found in station_id_to_index.")

    # Calculate savings between station pairs
    for i in range(num_stations):
        for j in range(i + 1, num_stations):
            station_i_id = int(demand_data.iloc[i]['station_id'])
            station_j_id = int(demand_data.iloc[j]['station_id'])

            station_i_index = station_id_to_index.get(station_i_id)
            station_j_index = station_id_to_index.get(station_j_id)

            if station_i_index is None or station_j_index is None:
                continue  # Skip if either station index is not found

            save = (distance_matrix[station_i_index, reference_index] +
                    distance_matrix[reference_index, station_j_index] - 
                    distance_matrix[station_i_index, station_j_index])

            if np.isscalar(save):
                savings.append((save, i, j))

    # Sort savings in descending order
    savings.sort(reverse=True, key=lambda x: x[0])

    valid_routes = []

    # Merge routes based on savings and truck capacity
    for save, i, j in savings:
        if i in demand_data.index and j in demand_data.index:
            if demand_data.at[i, 'deficit'] > 0 and demand_data.at[j, 'surplus'] > 0:
                if route_load[i] + route_load[j] <= truck_capacity:
                    surplus_j = demand_data.at[j, 'surplus']
                    deficit_i = demand_data.at[i, 'deficit']
                    bikes_to_transfer = min(surplus_j, deficit_i, truck_capacity - route_load[i])

                    if bikes_to_transfer > 0:
                        total_bikes_carried += bikes_to_transfer
                        route_load[i] += bikes_to_transfer

                        # Make sure we safely modify the DataFrame
                        if j in demand_data.index:
                            demand_data.at[i, 'deficit'] -= bikes_to_transfer
                            demand_data.at[j, 'surplus'] -= bikes_to_transfer

                            # Merge routes and update availability
                            routes[i] = routes[i] + routes[j]
                            routes[j] = []
                            valid_routes.append(routes[i])
    
    # Calculate total distance for the valid routes
    total_distance = 0
    for route in valid_routes:
        for k in range(len(route) - 1):
            station_i_index = station_id_to_index.get(int(demand_data.iloc[route[k]]['station_id']))
            station_j_index = station_id_to_index.get(int(demand_data.iloc[route[k + 1]]['station_id']))

            if station_i_index is None or station_j_index is None:
                continue  # Skip if either station index is missing

            distance_km = distance_matrix[station_i_index, station_j_index]
            total_distance += distance_km

    return valid_routes, total_distance, total_bikes_carried

def run_vrp_all_hours_with_reference_dynamic(data, distance_matrix, demand_type='simulated', truck_capacity=50, reference_station_id=None, station_id_to_index=None, n_hours=24, max_distance_per_day=500):
    """
    Run VRP for 24 hours and update bike numbers dynamically for each station.
    Calculates fulfillment based on how much of the simulated demand can be fulfilled using the predicted surpluses.
    """
    all_distances = []
    detailed_routes = []
    fulfillment_ratios = []
    demand_fulfillment_log = []
    trucks_used_log = []  # Log the number of trucks used each hour
    
    all_station_ids = pd.concat([data['departure_id'], data['return_id']]).unique()
    bikes_available = {int(station): 0 for station in all_station_ids}  # Cast station IDs to integers
    total_fulfilled_demand = 0  # Track fulfilled demand for the specified demand type
    total_simulated_demand = 0  # Track total simulated demand across all hours

    for hour in range(n_hours):
        # Calculate surplus and deficit for the specified demand type
        demand_data = calculate_surplus_deficit(data, hour, demand_type)
        demand_data['station_id'] = demand_data['station_id'].astype(int)

        # Calculate the deficit based on the simulated demand to compare against other demand types
        simulated_deficit_data = calculate_surplus_deficit(data, hour, demand_type='simulated')
        simulated_deficit_data['station_id'] = simulated_deficit_data['station_id'].astype(int)
        simulated_demand = int(simulated_deficit_data['deficit'].sum())  # Total simulated demand for this hour
        total_simulated_demand += simulated_demand  # Accumulate total simulated demand over the day
        fulfilled_demand = 0  # Track fulfilled demand for this hour for the current demand type
        trucks_used = 0  # Track trucks used for the hour

        # Redistribute bikes to fulfill the simulated deficit
        simulated_deficit_data = simulated_deficit_data[simulated_deficit_data['deficit'] > 0].sort_values(by='deficit', ascending=False)

        for _, row in simulated_deficit_data.iterrows():
            station_id = int(row['station_id'])
            station_deficit = int(row['deficit'])

            if station_id not in bikes_available:
                bikes_available[station_id] = 0

            # Satisfy the simulated deficit with available surpluses for the given demand type
            best_surplus_station = None
            best_weighted_score = -float('inf')
            bikes_to_satisfy = 0

            for surplus_station, surplus_bikes in bikes_available.items():
                if surplus_bikes > 0 and station_id != surplus_station:
                    if surplus_station in station_id_to_index and station_id in station_id_to_index:
                        distance = distance_matrix[station_id_to_index[surplus_station], station_id_to_index[station_id]]
                        weighted_score = (0.5 * surplus_bikes) - (0.5 * distance)
                        if weighted_score > best_weighted_score:
                            best_weighted_score = weighted_score
                            best_surplus_station = surplus_station
                            bikes_to_satisfy = min(surplus_bikes, station_deficit)

            # Fulfill demand if a suitable surplus station was found
            if best_surplus_station is not None and bikes_to_satisfy > 0:
                bikes_available[best_surplus_station] -= bikes_to_satisfy
                fulfilled_demand += bikes_to_satisfy
                trucks_used += 1

                detailed_routes.append({
                    'hour': hour,
                    'truck_number': trucks_used,
                    'from_station': best_surplus_station,
                    'to_station': station_id,
                    'distance_km': distance,
                    'bikes_carried': bikes_to_satisfy
                })

        # Log trucks used for this hour
        trucks_used_log.append({'hour': hour, 'trucks_used': trucks_used})

        # Calculate hourly fulfillment ratio based on how much of the simulated demand was fulfilled
        fulfillment_ratio = (fulfilled_demand / simulated_demand) * 100 if simulated_demand > 0 else 0
        fulfillment_ratios.append(fulfillment_ratio)

        demand_fulfillment_log.append({
            'hour': hour,
            'total_simulated_demand': simulated_demand,
            'total_fulfilled_demand': fulfilled_demand,
            'fulfillment_ratio': fulfillment_ratio
        })

        # Calculate hourly distance with VRP optimization
        routes, total_distance, bikes_carried = clarke_wright_vrp_optimized(
            demand_data.sort_values(by='deficit', ascending=False),
            distance_matrix,
            truck_capacity,
            reference_station_id,
            station_id_to_index,
            max_distance_per_day=max_distance_per_day
        )

        # Update available bikes for each station based on surplus
        for _, row in demand_data.iterrows():
            station_id = int(row['station_id'])
            bikes_available[station_id] += int(row['surplus'])

        # Append hourly distance to total distances
        all_distances.append(total_distance)
        total_fulfilled_demand += fulfilled_demand

    # Calculate total daily fulfillment ratio based on simulated demand
    total_fulfillment_ratio = (total_fulfilled_demand / total_simulated_demand) * 100 if total_simulated_demand > 0 else 0

    # Convert logs to DataFrames
    detailed_routes_df = pd.DataFrame(detailed_routes) if detailed_routes else pd.DataFrame(columns=['hour', 'truck_number', 'from_station', 'to_station', 'distance_km', 'bikes_carried'])
    trucks_used_df = pd.DataFrame(trucks_used_log)
    demand_fulfillment_df = pd.DataFrame(demand_fulfillment_log)

    return all_distances, detailed_routes_df, fulfillment_ratios, demand_fulfillment_df, trucks_used_df, total_fulfillment_ratio

# Create station_id_to_index using unique departure IDs from future_data
station_ids = future_data['departure_id'].unique()
station_id_to_index = {int(station_id): index for index, station_id in enumerate(station_ids)}

# Define the reference station as an integer (using 81 as an example)
reference_station_id = 81

# Check if reference station is in the station_id_to_index dictionary
if reference_station_id in station_id_to_index:
    best_reference_station = reference_station_id
    best_reference_index = station_id_to_index[best_reference_station]
    print(f"Reference station {best_reference_station} found with index {best_reference_index}.")
else:
    print(f"Reference station {reference_station_id} not found in station_id_to_index.")
    # Optionally, set a default or choose a different station from station_id_to_index
    best_reference_station = list(station_id_to_index.keys())[0]  # Choose the first station as a fallback
    best_reference_index = station_id_to_index[best_reference_station]
    print(f"Defaulting to reference station {best_reference_station} with index {best_reference_index}.")

# Debug: Print out station mappings to confirm setup
print("Station mappings for station_id_to_index:")
print(station_id_to_index)

def bernoulli_redistribution_policy(day_data, station_id_to_index, distance_matrix, truck_capacity=50, trucks_per_hour=50, num_hours=24):
    """
    Bernoulli redistribution policy with a fixed probability distribution for all hours.
    Fulfillment is calculated based on how much of the simulated deficit is met each hour.
    """
    all_stations = list(station_id_to_index.keys())
    total_distance = 0
    detailed_routes = []
    hourly_fulfillment_ratios = []  # Store hourly fulfillment percentages

    print(f"Starting Bernoulli redistribution with {len(all_stations)} stations and {trucks_per_hour * num_hours} total trucks")

    # Generate a fixed probability distribution for each station (once at the start)
    station_prob_map = np.random.dirichlet(np.ones(len(all_stations)))

    for hour in range(num_hours):
        # Calculate simulated deficit for this hour
        simulated_deficit_data = calculate_surplus_deficit(day_data, hour, demand_type='simulated')
        total_simulated_deficit = simulated_deficit_data['deficit'].sum()
        bikes_moved = 0  # Reset for each hour

        for truck in range(trucks_per_hour):
            # Select stations randomly based on the fixed probability distribution
            from_station = np.random.choice(all_stations, p=station_prob_map)
            to_station = np.random.choice(all_stations, p=station_prob_map)
            
            if from_station == to_station:
                continue  # Skip if stations are the same
            
            from_index = station_id_to_index.get(from_station)
            to_index = station_id_to_index.get(to_station)
            
            if from_index is None or to_index is None:
                continue
            
            # Fetch distance from the distance matrix
            distance_km = distance_matrix[from_index, to_index]
            bikes_to_move = truck_capacity  # Use the truck's full capacity
            
            detailed_routes.append({
                'hour': hour,
                'truck': truck,
                'from_station': from_station,
                'to_station': to_station,
                'distance_km': distance_km,
                'bikes_moved': bikes_to_move
            })
            
            total_distance += distance_km
            bikes_moved += bikes_to_move
        
        # Calculate fulfillment ratio based on the actual simulated deficit for this hour
        fulfillment_ratio = (bikes_moved / total_simulated_deficit) * 100 if total_simulated_deficit > 0 else 0
        hourly_fulfillment_ratios.append(fulfillment_ratio)

    # Convert routes to DataFrame
    routes_df = pd.DataFrame(detailed_routes)
    return routes_df, total_distance, hourly_fulfillment_ratios

def run_vrp_for_all_models(day_to_run, future_data, distance_matrix, station_id_to_index, reference_station_id, truck_capacity=50):
    """
    Run VRP for different demand models (e.g., historical, simulated, predicted_lasso, etc.) and calculate daily metrics based on hourly metrics.
    Collects hourly distances, fulfillment ratios, and trucks used for each model, then aggregates for daily totals.
    """
    model_names = ['historical', 'simulated', 'predicted_lasso', 'predicted_random', 'predicted_gradient', 'bernoulli']
    total_distances = {}
    fulfillment_ratios_per_model = {}
    trucks_used_per_model = {}
    detailed_routes_by_model = {}
    hourly_fulfillment_ratios_dict = {}  # Store hourly fulfillment ratios for each model
    hourly_distances_list = []  # Store hourly distances for each model

    # Filter data for the specific day
    day_data = future_data[future_data['day_identifier'] == pd.to_datetime(day_to_run).date()].copy()
    if 'hour' not in day_data.columns:
        day_data['hour'] = day_data['departure'].dt.hour

    for model_name in model_names:
        print(f"Running VRP for {model_name} demand on {day_to_run}...")

        if model_name == 'bernoulli':
            # Run Bernoulli model and store hourly ratios and total distance
            routes_df, total_distance, hourly_fulfillment_ratios = bernoulli_redistribution_policy(
                day_data,
                station_id_to_index,
                distance_matrix,
                truck_capacity=truck_capacity
            )
            # Store Bernoulli data in dictionaries
            hourly_fulfillment_ratios_dict[model_name] = hourly_fulfillment_ratios
            total_distances[model_name] = total_distance
            trucks_used_per_model[model_name] = 24 * 50  # Assuming constant truck usage of 50 per hour for Bernoulli

            # Append hourly distances for Bernoulli
            for hour, distance in enumerate([total_distance / 24] * 24):  # Distribute total distance evenly
                hourly_distances_list.append({'Hour': hour, 'Model': model_name, 'Total_Distance': distance})

        else:
            # Run VRP model for the specified demand type and gather hourly metrics
            distances, routes_df, hourly_fulfillment_ratios, demand_fulfillment_df, trucks_used_df, total_fulfillment_ratio = run_vrp_all_hours_with_reference_dynamic(
                day_data,
                distance_matrix,
                demand_type=model_name,
                truck_capacity=truck_capacity,
                reference_station_id=reference_station_id,
                station_id_to_index=station_id_to_index,
                n_hours=24
            )
            
            # Calculate total distance and trucks used across all hours
            total_distance = sum(distances)
            trucks_used_total = trucks_used_df['trucks_used'].sum()
            
            # Store metrics
            total_distances[model_name] = total_distance
            trucks_used_per_model[model_name] = trucks_used_total
            fulfillment_ratios_per_model[model_name] = total_fulfillment_ratio
            hourly_fulfillment_ratios_dict[model_name] = hourly_fulfillment_ratios
            detailed_routes_by_model[model_name] = routes_df

            # Append hourly distances for this model
            for hour, distance in enumerate(distances):
                hourly_distances_list.append({'Hour': hour, 'Model': model_name, 'Total_Distance': distance})

    # Convert hourly_distances_list to DataFrame
    hourly_distances_df = pd.DataFrame(hourly_distances_list)

    return total_distances, fulfillment_ratios_per_model, trucks_used_per_model, detailed_routes_by_model, hourly_fulfillment_ratios_dict, hourly_distances_df

# Prepare future_data and set day_to_run
future_data['departure'] = pd.to_datetime(future_data['departure'])
future_data['day_identifier'] = future_data['departure'].dt.date
day_to_run = '2020-06-03'

# Run VRP for all models and get the hourly distances DataFrame
total_distances, fulfillment_ratios_per_model, trucks_used_per_model, detailed_routes_by_model, hourly_fulfillment_ratios_dict, hourly_distances_df = run_vrp_for_all_models(
    day_to_run=day_to_run,
    future_data=future_data,
    distance_matrix=distance_matrix,
    station_id_to_index=station_id_to_index,
    reference_station_id=best_reference_station,
    truck_capacity=50
)

# Print Total Distances for Each Model
print("Total Distances per Model:")
for model, distance in total_distances.items():
    print(f"  {model}: {distance:.2f} km")

print("\nFulfillment Ratios per Model (in %):")
for model, ratio in fulfillment_ratios_per_model.items():
    print(f"  {model}: {ratio:.2f}%")

print("\nTotal Trucks Used per Model:")
for model, trucks_used in trucks_used_per_model.items():
    print(f"  {model}: {trucks_used} trucks")

print("\nHourly Fulfillment Ratios per Model:")
for model, hourly_ratios in hourly_fulfillment_ratios_dict.items():
    print(f"\nModel: {model}")
    for hour, ratio in enumerate(hourly_ratios):
        print(f"  Hour {hour}: {ratio:.2f}%")

# Define model name mappings for readability
model_name_map = {
    'historical': 'CVRP without\nPrediction',
    'simulated': 'CVRP with\nPerfect Information',
    'predicted_lasso': 'CVRP with\nLasso',
    'predicted_random': 'CVRP with\nRandom Forest',
    'predicted_gradient': 'CVRP with\nGradient Boosting',
    'bernoulli': 'CVRP with\nBernoulli (Baseline)'
}

# Map model names for hourly distances
hourly_distances_df['Model'] = hourly_distances_df['Model'].map(model_name_map)

# 1. Hourly Distances Box Plot (Including Bernoulli)
plt.figure(figsize=(14, 6))
sns.boxplot(data=hourly_distances_df, x='Model', y='Total_Distance')
plt.title(f"Hourly Total Distance Distribution for Each Model on {day_to_run}")
plt.xlabel("Model")
plt.ylabel("Total Distance (km)")
plt.xticks(rotation=0, ha='center')
plt.tight_layout()
plt.show()

# 1. Hourly Distances Box Plot (Excluding Bernoulli)
hourly_distances_no_bernoulli_df = hourly_distances_df[hourly_distances_df['Model'] != 'CVRP with\nBernoulli (Baseline)']
plt.figure(figsize=(12, 6))
sns.boxplot(data=hourly_distances_no_bernoulli_df, x='Model', y='Total_Distance')
plt.title(f"Hourly Total Distance Distribution for Each Model on {day_to_run} (Excluding Bernoulli)")
plt.xlabel("Model")
plt.ylabel("Total Distance (km)")
plt.show()
# Calculate the total distance for Bernoulli
bernoulli_total_distance = hourly_distances_df[hourly_distances_df['Model'] == 'CVRP with\nBernoulli (Baseline)']['Total_Distance'].sum()

# Add the Bernoulli data to total_distances if not already present
if 'bernoulli' not in total_distances:
    total_distances['bernoulli'] = bernoulli_total_distance

# Ensure Bernoulli entry exists in all dictionaries; if not, set a placeholder value
if 'bernoulli' not in trucks_used_per_model:
    trucks_used_per_model['bernoulli'] = 1200  # Example placeholder, adjust as needed

# Calculate the average fulfillment ratio for Bernoulli if it exists in the hourly_fulfillment_ratios_dict
if 'bernoulli' not in fulfillment_ratios_per_model:
    if 'bernoulli' in hourly_fulfillment_ratios_dict:
        fulfillment_ratios_per_model['bernoulli'] = sum(hourly_fulfillment_ratios_dict['bernoulli']) / len(hourly_fulfillment_ratios_dict['bernoulli'])
    else:
        fulfillment_ratios_per_model['bernoulli'] = 0  # Default value if no data is available

# Now synchronize dictionaries by filtering with common keys
common_keys = set(total_distances.keys()) & set(trucks_used_per_model.keys()) & set(fulfillment_ratios_per_model.keys())

# Filter dictionaries to include only common keys
total_distances = {k: total_distances[k] for k in common_keys}
trucks_used_per_model = {k: trucks_used_per_model[k] for k in common_keys}
fulfillment_ratios_per_model = {k: fulfillment_ratios_per_model[k] for k in common_keys}

# Create the DataFrame including Bernoulli
results_df = pd.DataFrame({
    'Model': [model_name_map.get(name, name) for name in common_keys],
    'Total_Distance': [total_distances[name] for name in common_keys],
    'Trucks_Used': [trucks_used_per_model[name] for name in common_keys],
    'Fulfillment_Ratio': [fulfillment_ratios_per_model[name] for name in common_keys]
})

# 2. Total Distance Bar Plot (Including Bernoulli)
def plot_total_distance(data, title_suffix):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x='Model', y='Total_Distance', color='skyblue')
    plt.ylabel("Total Distance (km)", color='blue', fontsize=14)
    plt.title(f"Total Distance per Model on {day_to_run} {title_suffix}", fontsize=16)
    for index, row in data.iterrows():
        plt.text(index, row['Total_Distance'] + 40, f"{row['Total_Distance']:.0f} km", 
                 color='blue', ha="center", fontsize=12, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

# Plot total distance including Bernoulli
plot_total_distance(results_df, "(Including Bernoulli)")

# 3. Combined Total Distance and Truck Number Plot (Excluding Bernoulli)
results_df_no_bernoulli = results_df[results_df['Model'] != 'CVRP with\nBernoulli (Baseline)']
def plot_total_distance_and_truck_number(data, title_suffix):
    fig, ax1 = plt.subplots(figsize=(14, 8))
    sns.barplot(data=data, x='Model', y='Total_Distance', color='skyblue', label='Total Distance', ax=ax1)
    ax1.set_ylabel("Total Distance (km)", color='blue', fontsize=14)
    ax1.set_title(f"Total Distance and Truck Number per Model on {day_to_run} {title_suffix}", fontsize=16)
    for index, row in data.iterrows():
        ax1.text(index, row['Total_Distance'] + 20, f"{row['Total_Distance']:.0f} km", 
                 color='blue', ha="center", va="bottom", fontsize=12, fontweight='bold')
    ax2 = ax1.twinx()
    sns.scatterplot(data=data, x='Model', y='Trucks_Used', color='red', marker='o', s=150)
    ax2.set_ylabel("Truck Number", color='red', fontsize=14)
    for index, row in data.iterrows():
        ax2.text(index, row['Trucks_Used'] + 6, f"{row['Trucks_Used']:.0f}", 
                 color='red', ha="center", va="bottom", fontsize=12, fontweight='bold')
    ax1.grid(False)
    ax2.grid(False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

# Plot total distance and truck number excluding Bernoulli
plot_total_distance_and_truck_number(results_df_no_bernoulli, "(Excluding Bernoulli)")

# Prepare data for hourly fulfillment percentages
hourly_fulfillment_data = []
for model_name, hourly_ratios in hourly_fulfillment_ratios_dict.items():
    for hour, ratio in enumerate(hourly_ratios):
        hourly_fulfillment_data.append({'Hour': hour, 'Model': model_name, 'Fulfillment_Percentage': ratio})

hourly_fulfillment_df = pd.DataFrame(hourly_fulfillment_data)
hourly_fulfillment_df['Model'] = hourly_fulfillment_df['Model'].map(model_name_map)

# Calculate statistics (mean and std) for each model
model_stats = hourly_fulfillment_df.groupby('Model')['Fulfillment_Percentage'].agg(['mean', 'std']).reset_index()

# Define consistent color palette for models
palette = {
    'CVRP without\nPrediction': '#1f77b4',
    'CVRP with\nPerfect Information': '#ff7f0e',
    'CVRP with\nLasso': '#2ca02c',
    'CVRP with\nRandom Forest': '#d62728',
    'CVRP with\nGradient Boosting': '#9467bd',
    'CVRP with\nBernoulli (Baseline)': '#8c564b'
}

# Plot 1: Hourly Fulfillment Rates including Bernoulli with mean and std in legend
plt.figure(figsize=(12, 6))

for model_name, model_data in hourly_fulfillment_df.groupby('Model'):
    # Retrieve mean and std for the model
    mean_value = model_stats.loc[model_stats['Model'] == model_name, 'mean'].values[0]
    std_value = model_stats.loc[model_stats['Model'] == model_name, 'std'].values[0]
    
    # Format label with mean and standard deviation
    label = f"{model_name} (Mean: {mean_value:.2f}%, Std: {std_value:.2f}%)"
    
    # Plot each model's data with line and markers
    plt.plot(model_data['Hour'], model_data['Fulfillment_Percentage'], label=label, marker='o', color=palette.get(model_name, 'gray'))

plt.xlabel('Hour')
plt.ylabel('Fulfillment Percentage (%)')
plt.title(f'Hourly Fulfillment Rates for Different Models on {day_to_run} (Including Bernoulli)')
plt.legend(title='Model Performance', loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot 2: Hourly Fulfillment Rates excluding Bernoulli with mean and std in legend
hourly_fulfillment_no_bernoulli_df = hourly_fulfillment_df[hourly_fulfillment_df['Model'] != 'CVRP with\nBernoulli (Baseline)']

plt.figure(figsize=(12, 6))

for model_name, model_data in hourly_fulfillment_no_bernoulli_df.groupby('Model'):
    # Retrieve mean and std for the model
    mean_value = model_stats.loc[model_stats['Model'] == model_name, 'mean'].values[0]
    std_value = model_stats.loc[model_stats['Model'] == model_name, 'std'].values[0]
    
    # Format label with mean and standard deviation
    label = f"{model_name} (Mean: {mean_value:.2f}%, Std: {std_value:.2f}%)"
    
    # Plot each model's data with line and markers
    plt.plot(model_data['Hour'], model_data['Fulfillment_Percentage'], label=label, marker='o', color=palette.get(model_name, 'gray'))

plt.xlabel('Hour')
plt.ylabel('Fulfillment Percentage (%)')
plt.title(f'Hourly Fulfillment Rates for Different Models on {day_to_run} (Excluding Bernoulli)')

# Set legend to bottom-right corner
plt.legend(title='Model Performance', loc='lower right', bbox_to_anchor=(1, 0))
plt.grid(True)
plt.tight_layout()
plt.show()

import time

def measure_optimization_times(day_to_run, future_data, distance_matrix, station_id_to_index, reference_station_id, truck_capacity=50):
    """
    Measure the optimization times for different demand models.
    """
    # Mapping model names to match graph labels
    model_names_mapping = {
        "historical": "CVRP without\nPrediction",
        "simulated": "CVRP with\nPerfect Information",
        "predicted_lasso": "CVRP with\nLasso",
        "predicted_random": "CVRP with\nRandom Forest",
        "predicted_gradient": "CVRP with\nGradient Boosting",
        "bernoulli": "CVRP with\nBernoulli (Baseline)"
    }

    optimization_times = {}

    # Filter data for the specific day
    day_data = future_data[future_data['day_identifier'] == pd.to_datetime(day_to_run).date()].copy()
    if 'hour' not in day_data.columns:
        day_data['hour'] = day_data['departure'].dt.hour

    for model_key, model_label in model_names_mapping.items():
        print(f"Measuring optimization time for {model_label} demand on {day_to_run}...")

        start_time = time.time()

        if model_key == 'bernoulli':
            # Measure time for Bernoulli model
            _, _, _ = bernoulli_redistribution_policy(
                day_data,
                station_id_to_index,
                distance_matrix,
                truck_capacity=truck_capacity
            )
        else:
            # Measure time for VRP model
            _, _, _, _, _, _ = run_vrp_all_hours_with_reference_dynamic(
                day_data,
                distance_matrix,
                demand_type=model_key,
                truck_capacity=truck_capacity,
                reference_station_id=reference_station_id,
                station_id_to_index=station_id_to_index,
                n_hours=24
            )
        
        end_time = time.time()
        optimization_times[model_label] = end_time - start_time

    return optimization_times

# Measure optimization times for all models
optimization_times = measure_optimization_times(
    day_to_run=day_to_run,
    future_data=future_data,
    distance_matrix=distance_matrix,
    station_id_to_index=station_id_to_index,
    reference_station_id=best_reference_station,
    truck_capacity=50
)

# Convert optimization times to a DataFrame for visualization
optimization_times_df = pd.DataFrame(list(optimization_times.items()), columns=['Model', 'Optimization Time (s)'])

# Plot optimization times
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Optimization Time (s)', data=optimization_times_df, palette='coolwarm')
plt.xlabel('Model', fontsize=14)
plt.ylabel('Optimization Time (s)', fontsize=14)
plt.title('Optimization Times for Different CVRP Models', fontsize=16)
plt.xticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.show()


def run_vrp_for_week(start_date, future_data, distance_matrix, station_id_to_index, reference_station_id, truck_capacity=50):
    """
    Run VRP for each day in a week, starting from start_date.
    Collects and compares total distances, truck numbers, and fulfillment ratios of each model for each day.
    """
    weekly_metrics = []  # List to store weekly metrics for each model

    # Loop over each day in the week
    for day_offset in range(7):  # 7 days in a week
        day_to_run = start_date + timedelta(days=day_offset)
        print(f"\nRunning VRP for {day_to_run}...")

        # Run VRP for each model for the current day
        total_distances, fulfillment_ratios_per_model, trucks_used_per_model, _, hourly_fulfillment_ratios_dict, hourly_distances_df = run_vrp_for_all_models(
            day_to_run=day_to_run,
            future_data=future_data,
            distance_matrix=distance_matrix,
            station_id_to_index=station_id_to_index,
            reference_station_id=reference_station_id,
            truck_capacity=truck_capacity
        )

        # Store results for each model for the day
        for model_name_key, distance in total_distances.items():
            # Map the model name using the model_name_map dictionary
            model_name = model_name_map.get(model_name_key, model_name_key)  # Fallback to original key if not found in map
            
            # Ensure fulfillment ratio for Bernoulli is correctly calculated as an average of hourly ratios
            if model_name_key == 'bernoulli':
                fulfillment_ratio = np.mean(hourly_fulfillment_ratios_dict[model_name_key]) if hourly_fulfillment_ratios_dict[model_name_key] else 0
            else:
                fulfillment_ratio = fulfillment_ratios_per_model.get(model_name_key, 0)
            
            weekly_metrics.append({
                'Date': day_to_run,
                'Model': model_name,
                'Total_Distance': distance,
                'Trucks_Used': trucks_used_per_model[model_name_key],
                'Fulfillment_Ratio': fulfillment_ratio,
                'Hourly_Fulfillment_Ratios': hourly_fulfillment_ratios_dict[model_name_key]  # Store hourly ratios as list
            })

    # Convert weekly metrics to DataFrame
    weekly_metrics_df = pd.DataFrame(weekly_metrics)
    return weekly_metrics_df

# Define the start date for the week
start_date = pd.to_datetime('2020-06-01').date()  # Change as needed

# Run the VRP for each day in the week, starting from start_date
weekly_metrics_df = run_vrp_for_week(
    start_date=start_date,
    future_data=future_data,
    distance_matrix=distance_matrix,
    station_id_to_index=station_id_to_index,
    reference_station_id=best_reference_station,
    truck_capacity=50
)

# Verify that model names were mapped correctly
print(weekly_metrics_df['Model'].unique())

# Display the resulting DataFrame to check the weekly metrics
weekly_metrics_df.head()
weekly_metrics_df.to_csv('weekly_metrics_df.csv', index=False)


''' GRAPHS '''
file_path2 = r'C:\Users\18360039068941666201\weekly_metrics_df.csv'
weekly_metrics_df = pd.read_csv(file_path2, low_memory=False)

# Define a consistent color palette for models
custom_palette = {
    'CVRP without\nPrediction': '#1f77b4',
    'CVRP with\nPerfect Information': '#ff7f0e',
    'CVRP with\nLasso': '#2ca02c',
    'CVRP with\nRandom Forest': '#d62728',
    'CVRP with\nGradient Boosting': '#9467bd',
    'CVRP with\nBernoulli (Baseline)': '#8c564b'
}

# Ensure correct mappings for your data based on available models
unique_models = weekly_metrics_df['Model'].unique()
model_palette = {model: custom_palette.get(model.split(" (")[0], 'grey') for model in unique_models}

# 1. Fulfillment Ratio per Model Across Days (excluding Bernoulli)
weekly_metrics_df_no_bernoulli = weekly_metrics_df[weekly_metrics_df['Model'] != 'CVRP with\nBernoulli (Baseline)']
plt.figure(figsize=(12, 6))
sns.lineplot(data=weekly_metrics_df_no_bernoulli, x='Date', y='Fulfillment_Ratio', hue='Model', marker="o", palette=model_palette)
plt.title("Fulfillment Ratio per Model Across Days Excluding Bernoulli")
plt.xlabel("Date")
plt.ylabel("Fulfillment Ratio (%)")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=weekly_metrics_df, x='Date', y='Fulfillment_Ratio', hue='Model', marker="o", palette=model_palette)
plt.title("Fulfillment Ratio per Model Across Days Excluding Bernoulli")
plt.xlabel("Date")
plt.ylabel("Fulfillment Ratio (%)")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
plt.tight_layout()
plt.show()
# 2. Aggregate Fulfillment Ratio Distribution Across the Week (excluding Bernoulli)
plt.figure(figsize=(12, 6))
sns.boxplot(data=weekly_metrics_df_no_bernoulli, x='Model', y='Fulfillment_Ratio', palette=model_palette)
plt.title("Aggregate Fulfillment Ratio Distribution Across the Week for Each Model (Excluding Bernoulli)")
plt.xlabel("Model")
plt.ylabel("Fulfillment Ratio (%)")
plt.xticks(rotation=0)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# 3. Trucks Used per Model Across Days
mean_trucks_used = weekly_metrics_df.groupby('Model')['Trucks_Used'].mean()
legend_labels = [f"{model} (Mean: {int(mean_value)})" for model, mean_value in mean_trucks_used.items()]
plt.figure(figsize=(12, 6))
sns.barplot(data=weekly_metrics_df, x='Date', y='Trucks_Used', hue='Model', palette=model_palette)
plt.legend(title="Model (Mean)", bbox_to_anchor=(1.05, 1), loc='upper left', labels=legend_labels)
plt.title("Trucks Used per Model Across Days")
plt.xlabel("Date")
plt.ylabel("Trucks Used")
plt.xticks(rotation=0)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# 4. Daily Total Distance Distribution by Model (excluding Bernoulli)
plt.figure(figsize=(12, 6))
sns.boxplot(data=weekly_metrics_df_no_bernoulli, x='Model', y='Total_Distance', palette=model_palette)
plt.title("Daily Total Distance Distribution by Model (Excluding Bernoulli)")
plt.xlabel("Model")
plt.ylabel("Total Distance (km)")
plt.xticks(rotation=0)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

import ast

# Ensure `Hourly_Fulfillment_Ratios` is a list of numeric values
def parse_ratios(row):
    if isinstance(row['Hourly_Fulfillment_Ratios'], str):
        # Parse stringified lists
        try:
            return [float(x) for x in ast.literal_eval(row['Hourly_Fulfillment_Ratios'])]
        except (ValueError, SyntaxError):
            print(f"Failed to parse: {row['Hourly_Fulfillment_Ratios']}")
            return []
    return row['Hourly_Fulfillment_Ratios']

weekly_metrics_df['Hourly_Fulfillment_Ratios'] = weekly_metrics_df.apply(parse_ratios, axis=1)

# Extract hourly fulfillment data
hourly_fulfillment_data = []
for _, row in weekly_metrics_df.iterrows():
    if isinstance(row['Hourly_Fulfillment_Ratios'], list):
        for hour, ratio in enumerate(row['Hourly_Fulfillment_Ratios']):
            hourly_fulfillment_data.append({
                'Hour': hour,
                'Model': row['Model'],
                'Fulfillment_Percentage': ratio
            })

hourly_fulfillment_df = pd.DataFrame(hourly_fulfillment_data)

# Ensure `Fulfillment_Percentage` is numeric
hourly_fulfillment_df['Fulfillment_Percentage'] = pd.to_numeric(hourly_fulfillment_df['Fulfillment_Percentage'], errors='coerce')

plt.figure(figsize=(14, 8))
for model, data in hourly_fulfillment_df.groupby('Model'):
    hourly_mean = data.groupby('Hour')['Fulfillment_Percentage'].mean()
    hourly_std = data.groupby('Hour')['Fulfillment_Percentage'].std()
    plt.plot(hourly_mean.index, hourly_mean, marker="o", label=model, color=model_palette.get(model, 'grey'), linewidth=2)
    plt.fill_between(hourly_mean.index, hourly_mean - hourly_std, hourly_mean + hourly_std, 
                     color=model_palette.get(model, 'grey'), alpha=0.2)
plt.title("Average Fulfillment Ratio per Model for Each Hour Across the Week")
plt.xlabel("Hour")
plt.ylabel("Fulfillment Ratio (%)")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Filter out Bernoulli data if it's in the DataFrame
weekly_metrics_df_no_bernoulli = weekly_metrics_df[weekly_metrics_df['Model'] != 'CVRP with\nBernoulli (Baseline)']

# Calculate mean and standard deviation for each model
model_stats = weekly_metrics_df_no_bernoulli.groupby('Model')['Fulfillment_Ratio'].agg(['mean', 'std']).reset_index()

# Format model labels with mean and standard deviation
model_labels = {
    row['Model']: f"{row['Model']} (Mean: {row['mean']:.2f}%, Std: {row['std']:.2f}%)"
    for _, row in model_stats.iterrows()
}

# Update the model labels in the DataFrame for the legend
weekly_metrics_df_no_bernoulli = weekly_metrics_df_no_bernoulli.copy()
weekly_metrics_df_no_bernoulli['Model'] = weekly_metrics_df_no_bernoulli['Model'].map(model_labels)

# Define custom colors for each model, including all expected models
palette = {
    "CVRP without\nPrediction (Mean: 63.52%, Std: 1.86%)": "#1f77b4",            # Blue
    "CVRP with\nPerfect Information (Mean: 90.15%, Std: 3.08%)": "#ff7f0e",      # Orange
    "CVRP with\nLasso (Mean: 82.94%, Std: 1.85%)": "#2ca02c",                   # Green
    "CVRP with\nRandom Forest (Mean: 62.51%, Std: 1.48%)": "#d62728",           # Red
    "CVRP with\nGradient Boosting (Mean: 62.38%, Std: 1.04%)": "#9467bd"        # Purple
}

# Plot the fulfillment ratio per model across days
plt.figure(figsize=(12, 6))
sns.lineplot(data=weekly_metrics_df_no_bernoulli, x='Date', y='Fulfillment_Ratio', hue='Model', marker="o", palette=palette)
plt.title("Fulfillment Ratio per Model Across Days Excluding Bernoulli")
plt.xlabel("Date")
plt.ylabel("Fulfillment Ratio (%)")
plt.legend(title="Model (Mean, Std Dev)", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Plot the fulfillment ratio per model across days
plt.figure(figsize=(12, 6))
sns.lineplot(data=weekly_metrics_df_no_bernoulli, x='Date', y='Fulfillment_Ratio', hue='Model', marker="o", palette=palette)
plt.title("Fulfillment Ratio per Model Across Days Excluding Bernoulli")
plt.xlabel("Date")
plt.ylabel("Fulfillment Ratio (%)")
plt.legend(title="Model (Mean, Std Dev)", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=weekly_metrics_df_no_bernoulli, x='Model', y='Total_Distance')
plt.title("Daily Total Distance Distribution by Model (Excluding Bernoulli)")
plt.xlabel("Model")
plt.ylabel("Total Distance (km)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
sns.barplot(data=weekly_metrics_df, x='Date', y='Total_Distance', hue='Model')
plt.title("Total Distance per Model Across Days")
plt.xlabel("Date")
plt.ylabel("Total Distance (km)")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
sns.barplot(data=weekly_metrics_df_no_bernoulli, x='Date', y='Total_Distance', hue='Model')
plt.title("Total Distance per Model Across Days (Excluding Bernoulli)")
plt.xlabel("Date")
plt.ylabel("Total Distance (km)")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Filter out "Bernoulli (Baseline)" model
weekly_metrics_df_no_bernoulli = weekly_metrics_df[weekly_metrics_df['Model'] != 'CVRP with\nBernoulli (Baseline)']

# Flatten hourly fulfillment ratios and associate with each hour and model
hourly_fulfillment_ratios_df = pd.DataFrame(columns=['Hour', 'Model', 'Fulfillment_Ratio'])
for _, row in weekly_metrics_df_no_bernoulli.iterrows():
    model_name = row['Model']
    hourly_ratios = row['Hourly_Fulfillment_Ratios']
    
    # Check if hourly_ratios has 24 values, if not, skip this row or handle as needed
    if len(hourly_ratios) == 24:
        temp_df = pd.DataFrame({
            'Hour': range(24),
            'Model': model_name,
            'Fulfillment_Ratio': hourly_ratios
        })
        hourly_fulfillment_ratios_df = pd.concat([hourly_fulfillment_ratios_df, temp_df], ignore_index=True)

# Define a color palette for each model (excluding "Bernoulli")
palette = {
    "CVRP without\nPrediction": "#1f77b4",
    "CVRP with\nPerfect Information": "#ff7f0e",
    "CVRP with\nLasso": "#2ca02c",
    "CVRP with\nRandom Forest": "#d62728",
    "CVRP with\nGradient Boosting": "#9467bd"
}

# Plot: Average Fulfillment Ratio per Model for Each Hour Across the Week
plt.figure(figsize=(14, 8))
sns.lineplot(data=hourly_fulfillment_ratios_df, x='Hour', y='Fulfillment_Ratio', hue='Model', marker="o", palette=palette)
plt.title("Average Fulfillment Ratio per Model for Each Hour Across the Week (Excluding Bernoulli)")
plt.xlabel("Hour")
plt.ylabel("Fulfillment Ratio (%)")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# Filter out "Bernoulli (Baseline)" model
weekly_metrics_df_no_bernoulli = weekly_metrics_df[weekly_metrics_df['Model'] != 'CVRP with\nBernoulli (Baseline)']

# Flatten hourly fulfillment ratios and associate with each hour and model
hourly_fulfillment_ratios_df = pd.DataFrame(columns=['Hour', 'Model', 'Fulfillment_Ratio'])
for _, row in weekly_metrics_df_no_bernoulli.iterrows():
    model_name = row['Model']
    hourly_ratios = row['Hourly_Fulfillment_Ratios']
    
    # Check if hourly_ratios has 24 values, if not, skip this row or handle as needed
    if len(hourly_ratios) == 24:
        temp_df = pd.DataFrame({
            'Hour': range(24),
            'Model': model_name,
            'Fulfillment_Ratio': hourly_ratios
        })
        hourly_fulfillment_ratios_df = pd.concat([hourly_fulfillment_ratios_df, temp_df], ignore_index=True)

# Define a color palette for each model (excluding "Bernoulli")
palette = {
    "CVRP without\nPrediction": "#1f77b4",
    "CVRP with\nPerfect Information": "#ff7f0e",
    "CVRP with\nLasso": "#2ca02c",
    "CVRP with\nRandom Forest": "#d62728",
    "CVRP with\nGradient Boosting": "#9467bd"
}

# Plot: Average Fulfillment Ratio per Model for Each Hour Across the Week without fill
plt.figure(figsize=(14, 8))
sns.lineplot(data=hourly_fulfillment_ratios_df, x='Hour', y='Fulfillment_Ratio', hue='Model', marker="o", palette=palette, ci=None)
plt.title("Average Fulfillment Ratio per Model for Each Hour Across the Week (Excluding Bernoulli)")
plt.xlabel("Hour")
plt.ylabel("Fulfillment Ratio (%)")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

