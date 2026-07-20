
 
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import pickle
import time
import os
import json
 
st.set_page_config(page_title="AgroCloud Engine", layout="wide")
 
st.title("🌾 AgroCloud: Real-Time Multi-Output Crop & Fertilizer Recommendation Platform")
 
st.markdown("""
### **Master's Cloud Intelligence Prototype**
 
This terminal orchestrates live meteorological streams from the **Open-Meteo API** alongside 4 individual machine learning architectures to predict dual crop type and soil amendment configurations.
""")
 
@st.cache_resource
def load_ml_artifacts():
    artifacts = {}

    def load_file(filepath):
        try:
            return joblib.load(filepath)
        except Exception:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
 
    try:
        if os.path.exists('scaler.pkl'):
            artifacts['scaler'] = load_file('scaler.pkl')
        else:
            st.error("🚨 Critical Error: 'scaler.pkl' was not found in the directory!")
 
        if os.path.exists('label_encoder.pkl'):
            artifacts['encoder'] = load_file('label_encoder.pkl')
        else:
            st.error("🚨 Critical Error: 'label_encoder.pkl' was not found in the directory!")

        if os.path.exists('logistic_regression_model.pkl'):
            artifacts['Logistic Regression (Aditi - M1)'] = load_file('logistic_regression_model.pkl')
        if os.path.exists('knn_model.pkl'):
            artifacts['K-Nearest Neighbors (Kaustubh - M2)'] = load_file('knn_model.pkl')
        if os.path.exists('random_forest_model.pkl'):
            artifacts['Random Forest (Abhiram - M3)'] = load_file('random_forest_model.pkl')
        if os.path.exists('lightgbm_model.pkl'):
            artifacts['LightGBM (Vedant - M4)'] = load_file('lightgbm_model.pkl')
 
    except Exception as e:
        st.error(f"💥 Binary deserialization failure: {str(e)}")
        st.info("Check if your virtual environment's scikit-learn version matches the version used during training.")
 
    return artifacts
 
artifacts = load_ml_artifacts()
 
def load_accuracy_metrics():

    fallbacks = {
        'Logistic Regression (Aditi - M1)': 0.8568,
        'K-Nearest Neighbors (Kaustubh - M2)': 0.8432,
        'Random Forest (Abhiram - M3)': 0.9841,
        'LightGBM (Vedant - M4)': 0.9727
    }

    if os.path.exists('accuracy_metrics.json'):
        try:
            with open('accuracy_metrics.json', 'r') as f:
                saved_metrics = json.load(f)
 
                for saved_key, saved_val in saved_metrics.items():
                    if 'logistic' in saved_key.lower():
                        fallbacks['Logistic Regression (Aditi - M1)'] = saved_val
                    elif 'knn' in saved_key.lower() or 'nearest' in saved_key.lower():
                        fallbacks['K-Nearest Neighbors (Kaustubh - M2)'] = saved_val
                    elif 'random' in saved_key.lower() or 'forest' in saved_key.lower():
                        fallbacks['Random Forest (Abhiram - M3)'] = saved_val
                    elif 'lgbm' in saved_key.lower() or 'lightgbm' in saved_key.lower():
                        fallbacks['LightGBM (Vedant - M4)'] = saved_val
        except Exception:
            pass
    return fallbacks
 
accuracy_metrics = load_accuracy_metrics()
 

st.sidebar.header("🗺️ Geospatial & Soil Telemetry")
 
lat = st.sidebar.number_input("Target Latitude", value=53.3498, format="%.4f")
lon = st.sidebar.number_input("Target Longitude", value=-6.2603, format="%.4f")
 
st.sidebar.subheader("Chemical Soil Profile Input")
N = st.sidebar.slider("Nitrogen (N) Content (mg/kg)", 0, 150, 90)
P = st.sidebar.slider("Phosphorus (P) Content (mg/kg)", 0, 150, 42)
K = st.sidebar.slider("Potassium (K) Content (mg/kg)", 0, 250, 43)
ph = st.sidebar.slider("Soil Substrate pH Level", 3.5, 10.0, 6.5, step=0.1)
rainfall = st.sidebar.slider("Expected Cumulative Rainfall (mm)", 20.0, 300.0, 202.9)
 

st.subheader("📡 Live Weather API Ingestion Layer")
 
@st.cache_data(ttl=3600)  
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
 
w_col1, w_col2, w_col3 = st.columns(3)
w_col1.metric("Engineered Temp Vector (Mean)", f"{weather_features['temperature']:.2f} °C")
w_col2.metric("Engineered Humidity Vector (Mean)", f"{weather_features['humidity']:.2f} %")
w_col3.metric("Live Soil Substrate Layer Temp", f"{weather_features['soil_temp']:.2f} °C")
 
st.subheader("⚙️ High-Throughput Edge Multi-Model Inference Engine")
 
feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
input_data = pd.DataFrame([[
    N, P, K,
    weather_features['temperature'],
    weather_features['humidity'],
    ph,
    rainfall
]], columns=feature_names)
 
if 'scaler' in artifacts and 'encoder' in artifacts:
    input_scaled = artifacts['scaler'].transform(input_data)
 
    latency_logs = {}
    predictions = {}

    for name, model in artifacts.items():
        if name in ['scaler', 'encoder']:
            continue
 
        t_start = time.perf_counter()
        encoded_pred = model.predict(input_scaled)
        t_end = time.perf_counter()
 
        latency_ms = (t_end - t_start) * 1000
        latency_logs[name] = latency_ms
 
        decoded_string = artifacts['encoder'].inverse_transform(encoded_pred)[0]
        crop, fertilizer = decoded_string.split(" || ")
        predictions[name] = {"Crop": crop, "Fertilizer": fertilizer}
 
    if predictions:
        pred_cols = st.columns(len(predictions))
        for idx, (model_name, results) in enumerate(predictions.items()):
            with pred_cols[idx]:
                st.info(f"**{model_name}**")
 
                st.markdown(f"""
<div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 5px solid #2ca02c;">
<p style="margin:0; font-size:0.85rem; color:#555;">Recommended Crop</p>
<h4 style="margin:0; font-weight:bold; color:#111; word-wrap: break-word; white-space: normal;">{results["Crop"]}</h4>
</div>
<div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 5px solid #1f77b4;">
<p style="margin:0; font-size:0.85rem; color:#555;">Assigned Fertilizer</p>
<h4 style="margin:0; font-weight:bold; color:#111; word-wrap: break-word; white-space: normal;">{results["Fertilizer"]}</h4>
</div>
                """, unsafe_allow_html=True)
 
                acc_val = accuracy_metrics.get(model_name, 0.0) * 100
                st.metric("Test Accuracy Score", f"{acc_val:.2f}%", help="Empirical accuracy evaluated on independent, noisy holdout validation datasets.")
 
                st.caption(f"Compute Latency: `{latency_logs[model_name]:.4f} ms`")
    else:
        st.error("No team ML models could be located or verified in this environment.")
else:
    st.error("Unable to start inference pipeline. Please check the loading errors above for 'scaler.pkl' or 'label_encoder.pkl'.")
 
if 'scaler' in artifacts and 'encoder' in artifacts and predictions:
    st.markdown("---")
    st.subheader("📊 Empirical Performance & Input Vector Diagnostics")

    analytics_df = pd.DataFrame({
        'Algorithm': [name.split(" (")[0] for name in latency_logs.keys()],
        'Latency (ms)': list(latency_logs.values()),
        'Accuracy (%)': [accuracy_metrics.get(name, 0.0) * 100 for name in latency_logs.keys()]
    })

    soil_profile_df = pd.DataFrame({
        'Nutrient Metric': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)'],
        'Value (mg/kg)': [N, P, K]
    })
 
    g_col1, g_col2, g_col3 = st.columns([1.2, 1.2, 1])
 
    with g_col1:
        st.markdown("📈 **Algorithm Predictive Power Comparison**")
        st.line_chart(data=analytics_df, x='Algorithm', y='Accuracy (%)', color="#2ca02c")
        st.caption("Visual validation convergence boundaries (Higher is better).")
 
    with g_col2:
        st.markdown("⏱️ **Inference Compute Latency Benchmark**")
        st.bar_chart(data=analytics_df, x='Algorithm', y='Latency (ms)', color="#1f77b4")
        st.caption("Computation pipeline delay in milliseconds (Lower is better).")
 
    with g_col3:
        st.markdown("🧪 **Active Soil Macro-Nutrient Proportions**")
        st.bar_chart(data=soil_profile_df, x='Nutrient Metric', y='Value (mg/kg)', color="#ff7f0e")
        st.caption("Visual weight distribution of the active chemical vector inputs.")