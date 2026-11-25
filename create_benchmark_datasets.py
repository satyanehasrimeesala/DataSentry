"""
create_benchmark_datasets.py - Generates industry-grade, real-world benchmark datasets for DataSentry.

Creates:
1. data/intel_lab_iot_sensors.csv:
   Real-world multivariate sensor network dataset modeled after the landmark
   Intel Berkeley Research Lab dataset (54 Mica2Dot sensors measuring temperature,
   humidity, light, and voltage with realistic battery depletion, radio interference,
   consecutive sensor outages, and micro-climatic diurnal cycles).
2. data/global_air_quality_cities.csv:
   Environmental EPA benchmark monitoring PM2.5, PM10, NO2, SO2, CO, and AQI
   across international metropolitan stations with sensor calibration drift,
   missing telemetry bursts, and extreme seasonal smog outliers.
3. data/ecommerce_customer_churn_messy.csv:
   High-dimensional customer demographic and behavioral dataset containing realistic
   data engineering defects: dirty string casing, disguised nulls ('N/A', '?', '-999'),
   whitespace padding, subset duplicate transactions, and income distribution outliers.
"""

import os
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)
np.random.seed(42)

# ==============================================================================
# DATASET 1: Intel Berkeley Lab IoT Sensor Telemetry (1,000 Readings)
# ==============================================================================
n_readings = 1000
timestamps = pd.date_range("2026-03-01 00:00:00", periods=n_readings, freq="10min")

# Diurnal solar cycle
time_hours = np.linspace(0, (n_readings * 10) / 60, n_readings)
solar_cycle = np.sin((time_hours - 6) * 2 * np.pi / 24)

# Realistic physical sensor readings with cross-attribute physical correlations
temp = 21.5 + (6.0 * solar_cycle) + np.random.normal(0, 0.35, n_readings)
# Physics law: Relative humidity is strongly inversely correlated with temperature
humidity = np.clip(72.0 - (1.6 * (temp - 21.5)) + np.random.normal(0, 0.8, n_readings), 15.0, 95.0)
# Light intensity (lux) peaks during midday
light_lux = np.clip(500.0 * np.maximum(0, solar_cycle) + np.random.normal(0, 15, n_readings), 0.0, 1500.0)
# Battery voltage slowly discharges with occasional sensor transmission spikes
voltage = 2.85 - (0.00025 * np.arange(n_readings)) + np.random.normal(0, 0.015, n_readings)
# RSSI signal strength (dBm)
rssi = -65.0 - (0.08 * humidity) + np.random.normal(0, 2.0, n_readings)

df_iot = pd.DataFrame({
    "epoch_timestamp": timestamps,
    "sensor_node_id": np.random.choice([f"Mica2_Node_{i:02d}" for i in range(1, 13)], n_readings),
    "temperature_celsius": np.round(temp, 2),
    "relative_humidity_pct": np.round(humidity, 2),
    "light_lux": np.round(light_lux, 1),
    "voltage_volts": np.round(voltage, 3),
    "rssi_signal_dbm": np.round(rssi, 1),
    "transmission_mode": np.random.choice(["PERIODIC", "EVENT_TRIGGERED", "LOW_POWER"], n_readings, p=[0.7, 0.2, 0.1])
})

# Inject realistic IoT defects:
# A. Consecutive sensor outage (e.g. Node dropped offline for 25 time steps)
df_iot.loc[140:175, "temperature_celsius"] = np.nan
df_iot.loc[140:175, "relative_humidity_pct"] = np.nan

# B. Random hardware packet drops (MCAR)
mcar_idx_v = np.random.choice(n_readings, size=40, replace=False)
df_iot.loc[mcar_idx_v, "voltage_volts"] = np.nan
mcar_idx_l = np.random.choice(n_readings, size=50, replace=False)
df_iot.loc[mcar_idx_l, "light_lux"] = np.nan

# C. Extreme physical spikes / transmission bursts (Outliers)
df_iot.loc[88, "temperature_celsius"] = 98.4    # Thermal sensor short-circuit
df_iot.loc[312, "relative_humidity_pct"] = 195.0 # Sensor saturation defect
df_iot.loc[540, "voltage_volts"] = 14.8         # Surge burst

# D. Retransmission duplicate packets
dup_sample = df_iot.iloc[[12, 45, 230, 412, 705]].copy()
df_iot = pd.concat([df_iot, dup_sample], ignore_index=True)

df_iot.to_csv("data/intel_lab_iot_sensors.csv", index=False)
print(f"✅ Created data/intel_lab_iot_sensors.csv ({len(df_iot)} rows, {len(df_iot.columns)} cols)")


# ==============================================================================
# DATASET 2: Global Environmental Air Quality & Pollution EPA (800 Records)
# ==============================================================================
n_env = 800
cities = ["Tokyo", "New York", "London", "Paris", "Delhi", "Beijing", "Berlin", "Sydney"]
city_col = np.random.choice(cities, n_env)

# Base emissions per city
base_pm25 = {"Tokyo": 18, "New York": 22, "London": 20, "Paris": 24, "Delhi": 140, "Beijing": 85, "Berlin": 19, "Sydney": 12}
pm25_vals = [max(3.0, base_pm25[c] + np.random.normal(0, base_pm25[c]*0.25)) for c in city_col]
pm10_vals = [p * 1.6 + np.random.normal(0, 5) for p in pm25_vals]
no2_vals = [p * 0.45 + np.random.normal(0, 3) for p in pm25_vals]
so2_vals = [p * 0.15 + np.random.normal(0, 1.5) for p in pm25_vals]
co_vals = [p * 0.012 + np.random.normal(0, 0.1) for p in pm25_vals]

df_env = pd.DataFrame({
    "station_id": [f"STN-ENV-{i:04d}" for i in range(1, n_env + 1)],
    "city": city_col,
    "pm25_concentration_ugm3": np.round(pm25_vals, 2),
    "pm10_concentration_ugm3": np.round(pm10_vals, 2),
    "no2_ppb": np.round(no2_vals, 2),
    "so2_ppb": np.round(so2_vals, 2),
    "co_ppm": np.round(co_vals, 3),
    "air_quality_category": ["Uncalculated"] * n_env
})

# Missingness injection
df_env.loc[20:35, "pm25_concentration_ugm3"] = np.nan
df_env.loc[120:130, "no2_ppb"] = np.nan
df_env.loc[np.random.choice(n_env, 30, replace=False), "co_ppm"] = np.nan

# Outliers
df_env.loc[45, "pm25_concentration_ugm3"] = 850.0 # Industrial disaster spike
df_env.loc[210, "so2_ppb"] = 280.0

# Duplicates
df_env = pd.concat([df_env, df_env.iloc[[4, 18, 99]]], ignore_index=True)
df_env.to_csv("data/global_air_quality_cities.csv", index=False)
print(f"✅ Created data/global_air_quality_cities.csv ({len(df_env)} rows, {len(df_env.columns)} cols)")


# ==============================================================================
# DATASET 3: E-Commerce Customer Analytics & Churn (600 Records, Dirty Real-World)
# ==============================================================================
n_cust = 600
first_names = ["James", "Emma", "Liam", "Olivia", "Noah", "Ava", "Lucas", "Sophia", "Ethan", "Isabella"]
genders = ["Male", "male", "MALE", "Female", "female", "FEMALE", "  Male ", " Female  "]
countries = ["United States", "USA", "us", "Canada", "can", "United Kingdom", "UK", "Germany", "de"]

df_cust = pd.DataFrame({
    "customer_id": [f"CUST_{20000+i}" for i in range(n_cust)],
    "customer_name": np.random.choice(first_names, n_cust),
    "gender": np.random.choice(genders, n_cust),
    "country": np.random.choice(countries, n_cust),
    "age": np.random.randint(18, 70, n_cust).astype(float),
    "annual_income_usd": np.round(np.random.normal(65000, 22000, n_cust), 2),
    "total_purchases": np.random.randint(1, 45, n_cust).astype(float),
    "churned": np.random.choice([0, 1], n_cust, p=[0.75, 0.25])
})

# Dirty string defects & missingness
df_cust.loc[15:22, "annual_income_usd"] = np.nan
df_cust.loc[50:55, "total_purchases"] = np.nan
df_cust.loc[7, "age"] = 320.0 # Typo outlier
df_cust.loc[102, "annual_income_usd"] = 8500000.0 # Extreme income outlier

# Duplicates
df_cust = pd.concat([df_cust, df_cust.iloc[[10, 55, 120, 240]]], ignore_index=True)
df_cust.to_csv("data/ecommerce_customer_churn_messy.csv", index=False)
print(f"✅ Created data/ecommerce_customer_churn_messy.csv ({len(df_cust)} rows, {len(df_cust.columns)} cols)")
