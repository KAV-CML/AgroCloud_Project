"""
Cloud Machine Learning - Core Team Assignment Prototype Dashboard
System: Dynamic AgroCloud REST Streaming & Multi-Output Inference Engine
Authors: Team Members 1, 2, 3, and 4
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import time
import os

# Set browser layout configurations for an enterprise dashboard style
st.set_page_config(page_title="AgroCloud Engine", layout="wide")

st.title("🌾 AgroCloud: Real-Time Multi-Output Crop & Fertilizer Recommendation Platform")
st.markdown("""
### **Master's Cloud Intelligence Prototype**
This terminal orchestrates live meteorological streams from the **Open-Meteo API** alongside 4 individual machine learning architectures to predict dual crop type and soil amendment configurations.
""")

# -------------------------------------------------------------------------
# 1. ARTIFACT INITIALIZATION & SAFELOADING
# -------------------------------------------------------------------------
@st.cache_resource
def load_ml_artifacts():
    artifacts = {}
    try:
        artifacts['scaler'] = joblib.load('scaler.pkl')
        artifacts['encoder'] = joblib.load('label_encoder.pkl')
        
        # Safe load each group member's specific serialized engine weights
        if os.path.exists('logistic_regression_model.pkl'):
            artifacts['Logistic Regression (M1)'] = joblib.load('logistic_regression_model.pkl')
        if os.path.exists('knn_model.pkl'):
            artifacts['K-Nearest Neighbors (M2)'] = joblib.load('knn_model.pkl')
        if os.path.exists('random_forest_model.pkl'):
            artifacts['Random Forest (M3)'] = joblib.load('random_forest_model.pkl')
        if os.path.exists('lightgbm_model.pkl'):
            artifacts['lightgbm_model.pkl'] = joblib.load('lightgbm_model.pkl')
    except Exception as e:
        st.error(f"Error initializing system binaries: {e}")
    return artifacts

artifacts = load_ml_artifacts()

# -------------------------------------------------------------------------
# 2. SIDEBAR TELEMETRY CONFIGURATION
# -------------------------------------------------------------------------
st.sidebar.header("🗺️ Geospatial & Soil Telemetry")

# Geospatial Parameters (Defaults to Dublin coordinates)
lat = st.sidebar.number_input("Target Latitude", value=53.3498, format="%.4f")
lon = st.sidebar.number_input("Target Longitude", value=-6.2603, format="%.4f")

st.sidebar.subheader("Chemical Soil Profile Input")
N = st.sidebar.slider("Nitrogen (N) Content (mg/kg)", 0, 150, 90)
P = st.sidebar.slider("Phosphorus (P) Content (mg/kg)", 0, 150, 42)
K = st.sidebar.slider("Potassium (K) Content (mg/kg)", 0, 250, 43)
ph = st.sidebar.slider("Soil Substrate pH Level", 3.5, 10.0, 6.5, step=0.1)
rainfall = st.sidebar.slider("Expected Cumulative Rainfall (mm)", 20.0, 300.0, 202.9)

# -------------------------------------------------------------------------
# 3. LIVE REST API FETCH & AGGREGATION FEATURE ENGINEERING
# -------------------------------------------------------------------------
st.subheader("📡 Live Weather API Ingestion Layer")

@st.cache_data(ttl=3600)  # Cache for 1 hour to prevent redundant API thrashing
def fetch_live_weather_features(latitude, longitude):
    endpoint = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,soil_temperature_0_to_7cm",
        "forecast_days": 14
    }
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            hourly_data = data.get("hourly", {})
            
            # Feature engineering raw hourly forecast time-series into spatial aggregates
            avg_temp = np.mean(hourly_data.get("temperature_2m", [25.0]))
            avg_humidity = np.mean(hourly_data.get("relative_humidity_2m", [80.0]))
            avg_soil_temp = np.mean(hourly_data.get("soil_temperature_0_to_7cm", [20.0]))
            
            return {"temperature": avg_temp, "humidity": avg_humidity, "soil_temp": avg_soil_temp, "status": "Success"}
    except Exception as e:
        return {"status": f"Fallback Mode: {str(e)}", "temperature": 25.0, "humidity": 80.0, "soil_temp": 20.0}

with st.spinner("Extracting real-time weather matrices from Open-Meteo REST node..."):
    weather_features = fetch_live_weather_features(lat, lon)

if weather_features["status"] == "Success":
    st.success(f"Telemetry pipeline fully synchronized for location coordinates ({lat}, {lon})")
else:
    st.warning(f"Using default fallback vectors. Status: {weather_features['status']}")

# Display live metrics cards
w_col1, w_col2, w_col3 = st.columns(3)
w_col1.metric("Engineered Temp Vector (Mean)", f"{weather_features['temperature']:.2f} °C")
w_col2.metric("Engineered Humidity Vector (Mean)", f"{weather_features['humidity']:.2f} %")
w_col3.metric("Live Soil Substrate Layer Temp", f"{weather_features['soil_temp']:.2f} °C")

# -------------------------------------------------------------------------
# 4. PARALLEL MODEL INFERENCE & CLOUD THROUGHPUT TIMING
# -------------------------------------------------------------------------
st.subheader("⚙️ High-Throughput Edge Multi-Model Inference Engine")

# Build data observation matrix mirroring our features
input_data = pd.DataFrame([{
    'N': N, 'P': P, 'K': K,
    'temperature': weather_features['temperature'],
    'humidity': weather_features['humidity'],
    'ph': ph,
    'rainfall': rainfall
}])

# Process features through our original standard scaler model pipeline
input_scaled = artifacts['scaler'].transform(input_data)

latency_logs = {}
predictions = {}

# Compute model outputs simultaneously to benchmark real-time trade-offs
for name, model in artifacts.items():
    if name in ['scaler', 'encoder']:
        continue
        
    t_start = time.perf_counter()
    encoded_pred = model.predict(input_scaled)
    t_end = time.perf_counter()
    
    # Calculate inference time in milliseconds
    latency_ms = (t_end - t_start) * 1000
    latency_logs[name] = latency_ms
    
    # Decode compound class output string back into discrete targets
    decoded_string = artifacts['encoder'].inverse_transform(encoded_pred)[0]
    crop, fertilizer = decoded_string.split(" || ")
    predictions[name] = {"Crop": crop, "Fertilizer": fertilizer}

# Display individual outputs matching team report subsections
if predictions:
    pred_cols = st.columns(len(predictions))
    for idx, (model_name, results) in enumerate(predictions.items()):
        with pred_cols[idx]:
            st.info(f"**{model_name}**")
            st.metric("Recommended Crop", results["Crop"])
            st.metric("Assigned Fertilizer", results["Fertilizer"])
            st.caption(f"Compute Latency: `{latency_logs[model_name]:.4f} ms`")
else:
    st.error("No team artifacts located. Please compile individual member scripts first.")

# -------------------------------------------------------------------------
# 5. LATENCY SYSTEM BENCHMARK ANALYSIS
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Empirical Latency & Performance Architecture Comparison")

chart_data = pd.DataFrame({
    'Model Architecture': list(latency_logs.keys()),
    'Inference Latency (Milliseconds)': list(latency_logs.values())
})

col_left, col_right = st.columns([2, 1])
with col_left:
    st.bar_chart(data=chart_data, x='Model Architecture', y='Inference Latency (Milliseconds)')
with col_right:
    st.markdown("**Core Analysis for Technical Report:**")
    st.markdown("""
    * **Baseline (Logistic Regression):** Exceptionally low latency. High spatial efficiency, fitting for highly-scalable deployments.
    * **Non-Parametric (KNN):** Low latency on small datasets, but computational overhead shifts entirely to inference as testing vectors increase.
    * **Tree-Ensembles (Random Forest vs LightGBM):** Random Forest trades computing latency for structural depth. LightGBM demonstrates optimal modern execution, yielding extremely rapid throughput speeds.
    """)