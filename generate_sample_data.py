import os
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)
np.random.seed(42)

# 1. IoT Sensor Network Dataset (for GARDA+, Outliers, and Imputation)
n = 120
timestamps = pd.date_range("2026-01-01 00:00", periods=n, freq="15min")
temp_base = 24.0 + 5.0 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 0.4, n)
humidity = 70.0 - 1.2 * temp_base + np.random.normal(0, 1.0, n)
pressure = 1013.25 - 0.05 * temp_base + np.random.normal(0, 0.2, n)
air_quality = 45.0 + 0.8 * temp_base + np.random.normal(0, 1.5, n)

df_iot = pd.DataFrame({
    "timestamp": timestamps,
    "temperature_c": np.round(temp_base, 2),
    "relative_humidity_pct": np.round(humidity, 2),
    "atmospheric_pressure_hpa": np.round(pressure, 2),
    "pm25_air_quality": np.round(air_quality, 2),
    "sensor_status": ["ACTIVE"] * n
})

# Inject missing values (MCAR and Outage gaps)
df_iot.loc[10:14, "temperature_c"] = np.nan
df_iot.loc[30:33, "relative_humidity_pct"] = np.nan
df_iot.loc[65, "pm25_air_quality"] = np.nan
df_iot.loc[80:82, "atmospheric_pressure_hpa"] = np.nan

# Inject extreme outliers
df_iot.loc[18, "temperature_c"] = 98.5
df_iot.loc[52, "pm25_air_quality"] = 490.0

# Inject subtle duplicate rows
df_iot = pd.concat([df_iot, df_iot.iloc[[5, 22]]], ignore_index=True)

df_iot.to_csv("data/iot_sensor_network_sample.csv", index=False)
print("Created data/iot_sensor_network_sample.csv")

# 2. Healthcare Patient Records (Categorical & Inconsistencies)
n_patients = 100
genders = np.random.choice(["Male", "male", "MALE", "Female", "female", "FEMALE", "  Male ", " Female "], n_patients)
blood_types = np.random.choice(["A+", "B+", "O+", "AB+", "O-", "a+", "b+"], n_patients)
ages = np.random.randint(18, 85, n_patients).astype(float)
systolic = 120 + (ages * 0.4) + np.random.normal(0, 8, n_patients)
cholesterol = 190 + (ages * 0.5) + np.random.normal(0, 15, n_patients)

# Inject defects
ages[5] = 250.0 # Outlier
systolic[12] = np.nan # Missing
cholesterol[25] = np.nan

df_health = pd.DataFrame({
    "patient_id": [f"PID-{1000+i}" for i in range(n_patients)],
    "age": ages,
    "gender": genders,
    "blood_type": blood_types,
    "systolic_bp": np.round(systolic, 1),
    "cholesterol_mg_dl": np.round(cholesterol, 1),
    "smoking_status": np.random.choice(["YES", "No", "never", "Former", "  yes"], n_patients)
})

# Duplicates
df_health = pd.concat([df_health, df_health.iloc[[2, 17, 44]]], ignore_index=True)
df_health.to_csv("data/healthcare_patient_sample.csv", index=False)
print("Created data/healthcare_patient_sample.csv")
