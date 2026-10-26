
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌱")

st.title("🌱 Crop Yield Prediction")
st.write("Predict crop yield using a trained Random Forest model.")

MODEL_PATH = Path(__file__).resolve().parent / "random_forest_crop_yield.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Get the available country and crop names from the saved model
preprocessor = model.named_steps["preprocessor"]
categorical_encoder = preprocessor.named_transformers_["categorical"]
countries = sorted(categorical_encoder.categories_[0].tolist())
crops = sorted(categorical_encoder.categories_[1].tolist())

st.subheader("Enter Crop and Weather Details")

with st.form("prediction_form"):
    country = st.selectbox("Country", countries)
    crop = st.selectbox("Crop", crops)
    year = st.number_input("Year", min_value=1990, max_value=2035, value=2013)
    rainfall = st.number_input(
        "Average annual rainfall (mm)", min_value=0.0, value=1000.0
    )
    pesticides = st.number_input(
        "Pesticide usage (tonnes)", min_value=0.0, value=1000.0
    )
    temperature = st.number_input(
        "Average temperature (°C)", value=20.0
    )
    submitted = st.form_submit_button("Predict Crop Yield")

if submitted:
    input_data = pd.DataFrame([{
        "Area": country,
        "Item": crop,
        "Year": year,
        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides,
        "avg_temp": temperature
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted crop yield: {prediction:,.2f} hg/ha")
    st.caption(
        "This is an estimate from the trained model, not a guarantee "
        "of actual crop yield. Predictions for years outside the training "
        "period may be less reliable."
    )
