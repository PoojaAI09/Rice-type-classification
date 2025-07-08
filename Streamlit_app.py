import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Rice Classifier", page_icon="🌾")
st.title("🌾 Rice Type Classifier (with Built-in Data)")

# Load model
try:
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Load dataset directly
try:
    df = pd.read_csv("riceClassification.csv")
    st.subheader("📄 Sample Data from riceClassification.csv")
    st.dataframe(df.head())

    # Drop unnecessary columns
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)

    X = df.drop('Class', axis=1)

    # Predict
    predictions = model.predict(X)

    # Display results
    st.subheader("🔍 Predicted Rice Types")
    st.write(predictions)

except Exception as e:
    st.error(f"❌ Failed to load or process dataset: {e}")
