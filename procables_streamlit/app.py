# app.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

MODEL_PATH = "model_final.h5"
SCALER_PATH = "scaler.joblib"

st.set_page_config(page_title="PROCABLES Predictor", layout="wide")
st.title("🔬 PROCABLES: Promoter / Strong-Weak Prediction")

@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error("❌ Model or scaler not found. Please train the model first using train.py")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()

st.sidebar.header("Input Options")
option = st.sidebar.radio("Choose Input Method", ["Manual Input", "Upload CSV"])

def make_prediction(data):
    X_scaled = scaler.transform(data)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    probs = model.predict(X_scaled).ravel()
    preds = (probs > 0.5).astype(int)
    return preds, probs

if option == "Manual Input":
    st.write("### Enter feature values manually:")
    num_features = st.number_input("Number of features", min_value=1, value=10, step=1)
    values = []
    cols = st.columns(5)
    for i in range(num_features):
        col = cols[i % 5]
        val = col.number_input(f"f{i+1}", value=0.0, step=0.01)
        values.append(val)
    if st.button("Predict"):
        arr = np.array([values])
        preds, probs = make_prediction(arr)
        st.success(f"Prediction: {'Positive (1)' if preds[0]==1 else 'Negative (0)'}")
        st.info(f"Probability: {probs[0]:.4f}")

elif option == "Upload CSV":
    st.write("### Upload a CSV file containing only feature columns:")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        if st.button("Predict All"):
            preds, probs = make_prediction(df.values)
            result = df.copy()
            result["Prediction"] = preds
            result["Probability"] = probs
            st.success("✅ Predictions complete!")
            st.dataframe(result.head())
            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "predictions.csv", "text/csv")
