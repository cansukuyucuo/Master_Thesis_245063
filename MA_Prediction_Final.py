# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 14:18:32 2024

@author: CansuKuyucuoglu
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
import shap
import time
import seaborn as sns

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

''' GRAPHS EXPLANATORY ANALYSIS'''

# Calculate the average demand for each hour of the day for both departures and returns
hourly_departure_demand = preprocessed_data.groupby('departure_hour')['departure_demand'].mean()
hourly_return_demand = preprocessed_data.groupby('return_hour')['return_demand'].mean()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(hourly_departure_demand.index, hourly_departure_demand.values, marker='o', color='b', linestyle='-', label='Departure Demand')
plt.plot(hourly_return_demand.index, hourly_return_demand.values, marker='o', color='r', linestyle='-', label='Return Demand')
plt.title("Average Hourly Bike Demand: Departures vs. Returns")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Number of Rides")
plt.xticks(range(0, 24))  # Show every hour on the x-axis
plt.legend()
plt.grid(True)
plt.show()

# 2. Histogram: Distribution of Distances
plt.figure(figsize=(10, 6))
sns.histplot(preprocessed_data['distance (km)'], bins=30, kde=True)
plt.xlabel('Distance (km)')
plt.ylabel('Frequency')
plt.title('Distribution of Trip Distances')
plt.show()

# 3. Histogram: Distribution of Durations
plt.figure(figsize=(10, 6))
sns.histplot(preprocessed_data['duration (min)'], bins=30, kde=True)
plt.xlabel('Duration (min)')
plt.ylabel('Frequency')
plt.title('Distribution of Trip Durations')
plt.show()

# Calculate average departure and return demand for daylight and non-daylight hours
average_demand_daylight = preprocessed_data.groupby('is_daylight_d')['departure_demand'].mean()
average_return_daylight = preprocessed_data.groupby('is_daylight_r')['return_demand'].mean()

# Plotting the bar chart
plt.figure(figsize=(8, 6))
bar_width = 0.35
index = [0, 1]  # Daylight (0 = No, 1 = Yes)

# Bars for departure demand
plt.bar(index, average_demand_daylight, bar_width, color='orange', alpha=0.6, label='Departure Demand')

# Bars for return demand
plt.bar([i + bar_width for i in index], average_return_daylight, bar_width, color='blue', alpha=0.6, label='Return Demand')

# Labels and title
plt.xlabel('Daylight (1=Yes, 0=No)')
plt.ylabel('Average Demand')
plt.title('Average Bike Demand During Daylight and Non-Daylight Hours')
plt.xticks([i + bar_width / 2 for i in index], ['Non-Daylight', 'Daylight'])
plt.legend()

plt.show()

# 5. Bar Chart: Demand by Top 20 Stations
top_20_stations = preprocessed_data[preprocessed_data['is_top_station_d'] == 1]
top_station_demand = top_20_stations['departure_id'].value_counts().head(20)
plt.figure(figsize=(12, 6))
top_station_demand.plot(kind='bar', color='skyblue')
plt.xlabel('Station ID')
plt.ylabel('Demand Count')
plt.title('Demand at Top 20 Stations')
plt.xticks(rotation=45)
plt.show()

# 6. Scatter Plot: Distance to City Center vs. Demand
plt.figure(figsize=(10, 6))
sns.scatterplot(data=preprocessed_data, x='departure_distance_to_center', y='departure_demand', hue='departure_station_cluster', palette='viridis')
plt.xlabel('Distance to City Center (km)')
plt.ylabel('Departure Demand')
plt.title('Demand vs. Distance to City Center')
plt.show()

# 7. Map with Clusters: Clustering of Stations
plt.figure(figsize=(10, 8))
sns.scatterplot(data=preprocessed_data, x='departure_longitude', y='departure_latitude', hue='departure_station_cluster', palette='Set1', s=50)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographical Clustering of Stations')
plt.legend(title='Cluster')
plt.show()

# Features and target variable for departure demand prediction
features_departure = [
    'is_daylight_d', 'is_top_station_d', 'departure_station_cluster',
    'departure_hour_sin', 'departure_hour_cos',
    'departure_hour','departure_distance_to_center'
]
X_departure = preprocessed_data[features_departure]
#X_departure = pd.get_dummies(X_departure, columns=['time_window_d'], drop_first=True)
y_departure = preprocessed_data['departure_demand']

# Features and target variable for return demand prediction
features_return = [
    'is_daylight_r', 'is_top_station_r', 'return_station_cluster',
    'return_hour_sin', 'return_hour_cos',
    'return_hour', 'return_distance_to_center']

X_return = preprocessed_data[features_return]
#X_return = pd.get_dummies(X_return, columns=['time_window_r'], drop_first=True)
y_return = preprocessed_data['return_demand']

# Add interaction terms and polynomial features for non-linear models
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_departure_poly = poly.fit_transform(X_departure[['departure_hour', 'departure_distance_to_center']])
X_departure_poly = pd.DataFrame(X_departure_poly, columns=poly.get_feature_names_out(['departure_hour', 'departure_distance_to_center']))

X_return_poly = poly.fit_transform(X_return[['return_hour', 'return_distance_to_center']])
X_return_poly = pd.DataFrame(X_return_poly, columns=poly.get_feature_names_out(['return_hour', 'return_distance_to_center']))

# Drop the original columns used for polynomial features
X_departure = X_departure.drop(columns=['departure_hour', 'departure_distance_to_center'])
X_return = X_return.drop(columns=['return_hour', 'return_distance_to_center'])

# Combine with original features
X_departure = X_departure.join(X_departure_poly)
X_return = X_return.join(X_return_poly)

# Standardize the features
scaler = StandardScaler()
X_departure_scaled = scaler.fit_transform(X_departure) 
X_return_scaled = scaler.fit_transform(X_return)

# Split the data for departure and return demand prediction
X_train_dep, X_test_dep, y_train_dep, y_test_dep = train_test_split(X_departure_scaled, y_departure, test_size=0.2, random_state=42)
X_train_ret, X_test_ret, y_train_ret, y_test_ret = train_test_split(X_return_scaled, y_return, test_size=0.2, random_state=42)

print("Starting model evaluation, hyperparameter tuning, and cross-validation process...")

# Custom scorers
def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae_scorer(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Custom scorers for cross_val_score
rmse_scorer_cv = make_scorer(rmse_scorer, greater_is_better=False)
mae_scorer_cv = make_scorer(mae_scorer, greater_is_better=False)

# Function to perform hyperparameter tuning on a subset of the data
def perform_hyperparameter_tuning(model, param_distributions, X_train, y_train, n_iter=10):
    if not param_distributions:
        # No hyperparameters to tune, return the model as is
        return model, None
    print("Performing hyperparameter tuning...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=rmse_scorer_cv,
        cv=3,  # Using 3-fold cross-validation for hyperparameter tuning
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    print(f"Best parameters found: {random_search.best_params_}")
    return random_search.best_estimator_, random_search.best_params_

# Function to perform cross-validation on a subset of the data
def perform_cross_validation(model, X_train, y_train, cv=10, subset_fraction=0.1):
    print("Performing cross-validation...")
    X_train_cv, _, y_train_cv, _ = train_test_split(X_train, y_train, test_size=1-subset_fraction, random_state=42)
    kf = KFold(n_splits=cv)
    cv_scores = cross_val_score(model, X_train_cv, y_train_cv, cv=kf, scoring=rmse_scorer_cv, n_jobs=-1)
    print("Cross-validation scores (RMSE): ", cv_scores)
    return cv_scores

# List of models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(random_state=42),
    'Ridge Regression': Ridge(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Define parameter distributions for each model
param_distributions = {
    'Linear Regression': {},  # No hyperparameters to tune for plain linear regression
    'Lasso Regression': {'alpha': uniform(0.01, 0.1)},
    'Ridge Regression': {'alpha': uniform(0.01, 0.1)},
    'Random Forest': {
        'n_estimators': randint(50, 100),
        'max_depth': randint(5, 15),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2']
    },
    'Gradient Boosting': {
        'n_estimators': randint(50, 100),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.1),
        'subsample': uniform(0.7, 0.3)
    }
}

# Sampling a small subset of the data for hyperparameter tuning
subset_fraction_quick = 0.1 # Use 10% of the data for quick evaluation
X_train_dep_quick, _, y_train_dep_quick, _ = train_test_split(X_train_dep, y_train_dep, test_size=1-subset_fraction_quick, random_state=42)
X_train_ret_quick, _, y_train_ret_quick, _ = train_test_split(X_train_ret, y_train_ret, test_size=1-subset_fraction_quick, random_state=42)

# Perform hyperparameter tuning on all models
best_models_dep = {}
best_models_ret = {}
best_params_dep = {}
best_params_ret = {}

for model_name, model in models.items():
    print(f"Tuning {model_name} for departure demand...")
    best_models_dep[model_name], best_params_dep[model_name] = perform_hyperparameter_tuning(model, param_distributions[model_name], X_train_dep_quick, y_train_dep_quick)

    print(f"Tuning {model_name} for return demand...")
    best_models_ret[model_name], best_params_ret[model_name] = perform_hyperparameter_tuning(model, param_distributions[model_name], X_train_ret_quick, y_train_ret_quick)

# Perform final cross-validation on the full dataset using the best models
cv_results_dep = {}
cv_results_ret = {}

for model_name in models.keys():
    print(f"Cross-validating {model_name} for departure demand...")
    cv_results_dep[model_name] = perform_cross_validation(best_models_dep[model_name], X_train_dep, y_train_dep)

for model_name in models.keys():
    print(f"Cross-validating {model_name} for return demand...")
    cv_results_ret[model_name] = perform_cross_validation(best_models_ret[model_name], X_train_ret, y_train_ret)

# Plotting box plots of RMSE values for high and low RMSE models (departure demand)
high_rmse_models = ['Linear Regression', 'Lasso Regression', 'Ridge Regression']
low_rmse_models = ['Random Forest', 'Gradient Boosting']

# Plotting improved box plots of RMSE values for high and low RMSE models (departure demand)
colors_high_rmse = ['lightblue', 'lightgreen', 'lightcoral']
colors_low_rmse = ['lightblue', 'lightgreen']

# High RMSE models for departure demand
plt.figure(figsize=(12, 6))
bp = plt.boxplot([np.abs(cv_results_dep[model]) for model in high_rmse_models], 
                 labels=high_rmse_models, patch_artist=True, showmeans=True,
                 meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
                 medianprops={"color": "red", "linewidth": 1.5})

# Coloring each box with distinct colors
for patch, color in zip(bp['boxes'], colors_high_rmse):
    patch.set_facecolor(color)

# Adding mean value labels
for i in range(len(high_rmse_models)):
    mean_val = np.mean(np.abs(cv_results_dep[high_rmse_models[i]]))
    plt.text(i+1, mean_val, f'{mean_val:.2f}', ha='center', va='bottom', fontsize=10, color='black')

plt.title('RMSE Distribution (Departure Demand - High RMSE Models)', fontsize=16)
plt.ylabel('Root Mean Squared Error', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.show()

# Low RMSE models for departure demand
plt.figure(figsize=(12, 6))
bp = plt.boxplot([np.abs(cv_results_dep[model]) for model in low_rmse_models], 
                 labels=low_rmse_models, patch_artist=True, showmeans=True,
                 meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
                 medianprops={"color": "red", "linewidth": 1.5})

# Coloring each box with distinct colors
for patch, color in zip(bp['boxes'], colors_low_rmse):
    patch.set_facecolor(color)

# Adding mean value labels
for i in range(len(low_rmse_models)):
    mean_val = np.mean(np.abs(cv_results_dep[low_rmse_models[i]]))
    plt.text(i+1, mean_val, f'{mean_val:.2f}', ha='center', va='bottom', fontsize=10, color='black')

plt.title('RMSE Distribution (Departure Demand - Low RMSE Models)', fontsize=16)
plt.ylabel('Root Mean Squared Error', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.show()

# High RMSE models for return demand
plt.figure(figsize=(12, 6))
bp = plt.boxplot([np.abs(cv_results_ret[model]) for model in high_rmse_models], 
                 labels=high_rmse_models, patch_artist=True, showmeans=True,
                 meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
                 medianprops={"color": "red", "linewidth": 1.5})

# Coloring each box with distinct colors
for patch, color in zip(bp['boxes'], colors_high_rmse):
    patch.set_facecolor(color)

# Adding mean value labels
for i in range(len(high_rmse_models)):
    mean_val = np.mean(np.abs(cv_results_ret[high_rmse_models[i]]))
    plt.text(i+1, mean_val, f'{mean_val:.2f}', ha='center', va='bottom', fontsize=10, color='black')

plt.title('RMSE Distribution (Return Demand - High RMSE Models)', fontsize=16)
plt.ylabel('Root Mean Squared Error', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.show()

# Low RMSE models for return demand
plt.figure(figsize=(12, 6))
bp = plt.boxplot([np.abs(cv_results_ret[model]) for model in low_rmse_models], 
                 labels=low_rmse_models, patch_artist=True, showmeans=True,
                 meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
                 medianprops={"color": "red", "linewidth": 1.5})

# Coloring each box with distinct colors
for patch, color in zip(bp['boxes'], colors_low_rmse):
    patch.set_facecolor(color)

# Adding mean value labels
for i in range(len(low_rmse_models)):
    mean_val = np.mean(np.abs(cv_results_ret[low_rmse_models[i]]))
    plt.text(i+1, mean_val, f'{mean_val:.2f}', ha='center', va='bottom', fontsize=10, color='black')

plt.title('RMSE Distribution (Return Demand - Low RMSE Models)', fontsize=16)
plt.ylabel('Root Mean Squared Error', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.show()

# Initialize dictionaries to store training and prediction times
training_times_dep, prediction_times_dep = {}, {}
training_times_ret, prediction_times_ret = {}, {}

# Function to measure training and prediction time with the full dataset
def measure_time_on_full_data(model, X_train, y_train, X_test):
    # Measure training time on the full dataset
    start_train = time.time()
    model.fit(X_train, y_train)  # Fit on the full dataset
    training_time = time.time() - start_train
    
    # Measure prediction time on 100 predictions
    start_pred = time.time()
    model.predict(X_test[:100])  # Predict on a subset of 100 samples
    prediction_time = time.time() - start_pred
    
    return training_time, prediction_time

# Modified evaluate function to include time measurements
def evaluate_and_save_predictions_with_best_params(models, best_params, X_train, X_test, y_train, y_test, X_full, prefix):
    eval_results = []
    predictions = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name} for {prefix} demand...")
        
        # Initialize models with best parameters if available, otherwise use default for Linear Regression
        if model_name == 'Linear Regression':
            tuned_model = LinearRegression()  # No hyperparameters to tune
        elif model_name == 'Lasso Regression':
            tuned_model = Lasso(**best_params[model_name])
        elif model_name == 'Ridge Regression':
            tuned_model = Ridge(**best_params[model_name])
        elif model_name == 'Random Forest':
            tuned_model = RandomForestRegressor(**best_params[model_name], random_state=42, n_jobs=-1)
        elif model_name == 'Gradient Boosting':
            tuned_model = GradientBoostingRegressor(**best_params[model_name], random_state=42)
        else:
            print(f"Model {model_name} is not recognized.")
            continue
        
        # Measure training and prediction times
        training_time, prediction_time = measure_time_on_full_data(tuned_model, X_train, y_train, X_test)
        
        # Store training and prediction times
        if prefix == 'dep':
            training_times_dep[model_name] = training_time
            prediction_times_dep[model_name] = prediction_time
        else:
            training_times_ret[model_name] = training_time
            prediction_times_ret[model_name] = prediction_time
        
        # Predict on test data
        y_pred = tuned_model.predict(X_test)
        
        # Predict on entire data for final_combined
        full_pred = tuned_model.predict(X_full)
        predictions[f'{prefix}_predicted_demand_{model_name}'] = full_pred
        
        # Evaluate performance
        rmse = rmse_scorer(y_test, y_pred)
        mae = mae_scorer(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        eval_results.append({
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Training Time (s)': training_time,
            'Prediction Time (s)': prediction_time
        })
        
        print(f'{model_name} - RMSE: {rmse}, MAE: {mae}, R²: {r2}, Training Time: {training_time}s, Prediction Time: {prediction_time}s')
        
        # Save predictions to final_combined
        preprocessed_data[f'{prefix}_predicted_demand_{model_name}'] = full_pred
    
    return pd.DataFrame(eval_results)

# Evaluate models on departure demand and save predictions with best parameters
eval_results_dep = evaluate_and_save_predictions_with_best_params(
    models, best_params_dep, X_train_dep, X_test_dep, y_train_dep, y_test_dep, X_departure_scaled, 'dep'
)
print("Evaluation results for departure demand with best parameters:")
print(eval_results_dep)
eval_results_dep.to_csv('eval_results_dep.csv')

# Evaluate models on return demand and save predictions with best parameters
eval_results_ret = evaluate_and_save_predictions_with_best_params(
    models, best_params_ret, X_train_ret, X_test_ret, y_train_ret, y_test_ret, X_return_scaled, 'ret'
)
print("Evaluation results for return demand with best parameters:")
print(eval_results_ret)
eval_results_ret.to_csv('eval_results_ret.csv')

# Save training and prediction times for further analysis if needed
print("Training and Prediction times for Departure Demand Models:")
print(training_times_dep)
print(prediction_times_dep)

print("Training and Prediction times for Return Demand Models:")
print(training_times_ret)
print(prediction_times_ret)

# Create a copy of the relevant columns to avoid SettingWithCopyWarning
final_combined = preprocessed_data[['departure_id', 'return_id', 'departure_hour', 'return_hour', 
                                    'departure', 'return', 'departure_demand', 'return_demand', 
                                    'departure_latitude', 'departure_longitude', 'return_latitude', 
                                    'return_longitude']].copy()

# Update final_combined with predicted values for each model using .loc
for model_name in models.keys():
    final_combined.loc[:, f'dep_predicted_demand_{model_name}'] = preprocessed_data[f'dep_predicted_demand_{model_name}'].values
    final_combined.loc[:, f'ret_predicted_demand_{model_name}'] = preprocessed_data[f'ret_predicted_demand_{model_name}'].values

# Save the final dataset to a CSV file
final_combined.to_csv('final_combined_predicted_outcomes.csv', index=False)
print("All predictions for all models have been saved to 'final_combined_predicted_outcomes.csv'.")
preprocessed_data.to_csv('preprocessed_final_data.csv')

print("Computing SHAP values for the best model (Gradient Boosting) on the full dataset...")

# Get the best model
best_model_dep = best_models_dep["Gradient Boosting"]

# Ensure X_train_dep is a DataFrame with correct columns
if isinstance(X_train_dep, np.ndarray):
    X_train_dep = pd.DataFrame(X_train_dep, columns=X_departure.columns)

# Align columns of X_train_dep to match X_departure
X_train_dep_corrected = X_train_dep[X_departure.columns]

# Initialize the SHAP explainer
explainer = shap.Explainer(best_model_dep, X_train_dep_corrected)

# Compute SHAP values for the entire dataset
shap_values = explainer(X_train_dep_corrected)

# Plot SHAP summary using the full dataset
shap.summary_plot(shap_values, X_train_dep_corrected, feature_names=X_departure.columns)



# Display SHAP values for the best model
print("Computing SHAP values for (Lasso)...")
best_model_dep = best_models_dep["Lasso Regression"]
explainer = shap.Explainer(best_model_dep, X_train_dep)
shap_values = explainer(X_train_dep)
shap.summary_plot(shap_values, X_train_dep, feature_names=X_departure.columns)

# Colors for the bars
colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink']

# ==============================
# Plot Training Time (Departure Demand)
# ==============================
plt.figure(figsize=(10, 6))
plt.bar(training_times_dep.keys(), training_times_dep.values(), color=colors)
plt.title('Training Time for Prediction Models (Departure)', fontsize=14)
plt.ylabel('Training Time (seconds)', fontsize=12)
plt.xlabel('Models')
for i, v in enumerate(training_times_dep.values()):
    plt.text(i, v + 0.01, f'{v:.2f}s', ha='center', va='bottom', fontsize=10)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ==============================
# Plot Prediction Time (Departure Demand - 100 Samples)
# ==============================
plt.figure(figsize=(10, 6))
plt.bar(prediction_times_dep.keys(), prediction_times_dep.values(), color=colors)
plt.title('Prediction Time for 100 Samples (Departure Demand)', fontsize=14)
plt.ylabel('Prediction Time (seconds)', fontsize=12)
plt.xlabel('Models')
for i, v in enumerate(prediction_times_dep.values()):
    plt.text(i, v + 0.001, f'{v:.4f}s', ha='center', va='bottom', fontsize=10)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ==============================
# Plot Training Time (Return Demand)
# ==============================
plt.figure(figsize=(10, 6))
plt.bar(training_times_ret.keys(), training_times_ret.values(), color=colors)
plt.title('Training Time for Prediction Models (Return)', fontsize=14)
plt.ylabel('Training Time (seconds)', fontsize=12)
plt.xlabel('Models')
for i, v in enumerate(training_times_ret.values()):
    plt.text(i, v + 0.01, f'{v:.2f}s', ha='center', va='bottom', fontsize=10)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ==============================
# Plot Prediction Time (Return Demand - 100 Samples)
# ==============================
plt.figure(figsize=(10, 6))
plt.bar(prediction_times_ret.keys(), prediction_times_ret.values(), color=colors)
plt.title('Prediction Time for 100 Samples (Return Demand)', fontsize=14)
plt.ylabel('Prediction Time (seconds)', fontsize=12)
plt.xlabel('Models')
for i, v in enumerate(prediction_times_ret.values()):
    plt.text(i, v + 0.001, f'{v:.4f}s', ha='center', va='bottom', fontsize=10)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

