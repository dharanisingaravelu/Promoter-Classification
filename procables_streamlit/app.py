import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------------------------------------------
# ⚙️ Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="PROCABLES: Promoter / Strong-Weak Prediction",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 PROCABLES: Promoter / Strong-Weak Prediction")
st.markdown("---")

# -------------------------------------------------------
# 🧩 Load model and scaler
# -------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = load_model("model_final.h5", compile=False)
        scaler = joblib.load("scaler.joblib")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model or scaler: {e}")
        return None, None
model, scaler = load_artifacts()

if model is None or scaler is None:
    st.warning("⚠️ Model or scaler not found. Please ensure `model_final.h5` and `scaler.joblib` are in the same folder as this app.")
    st.stop()

st.success("✅ Model and scaler loaded successfully!")

# -------------------------------------------------------
# 📂 File Upload Section
# -------------------------------------------------------
st.header("📁 Upload Your Input CSV File")
uploaded_file = st.file_uploader("Upload your feature CSV (with or without 'class' column)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data:")
        st.dataframe(data.head())

        if 'class' in data.columns:
            X = data.drop(columns=['class']).values
        else:
            X = data.values

        # Apply the same scaling as used during training
        X_scaled = scaler.transform(X)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        # -------------------------------------------------------
        # 🔮 Make Predictions
        # -------------------------------------------------------
        preds = model.predict(X_scaled)
        preds_binary = (preds > 0.5).astype(int)

        # Attach predictions to dataframe
        data['Predicted_Class'] = preds_binary
        data['Predicted_Probability'] = preds

        st.markdown("### ✅ Prediction Results:")
        st.dataframe(data.head())

        # -------------------------------------------------------
        # 📥 Download Results
        # -------------------------------------------------------
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"❌ Error processing the uploaded file: {e}")
else:
    st.info("👆 Please upload a CSV file to begin prediction.")
