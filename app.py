
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌱",
    layout="wide",
)

MODEL_PATH = Path(__file__).resolve().parent / "random_forest_crop_yield.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
preprocessor = model.named_steps["preprocessor"]
regressor = model.named_steps["regressor"]

# ---------------- Country and Crop Options ----------------
categorical_transformer = next(
    transformer
    for transformer in preprocessor.named_transformers_.values()
    if hasattr(transformer, "categories_")
)

area_options = sorted(list(categorical_transformer.categories_[0]))
crop_options = sorted(list(categorical_transformer.categories_[1]))

# ---------------- Custom Styling ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f3f8f1;
    }

    .main-title {
        color: #176b37;
        font-size: 38px;
        font-weight: 800;
        text-align: center;
    }

    .subtitle {
        color: #53665a;
        text-align: center;
        font-size: 17px;
    }

    .section-heading {
        color: #7b2cbf;
        font-size: 24px;
        font-weight: 800;
        padding: 8px 0;
        margin-top: 10px;
    }

    .input-heading {
        color: #176b37;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .result-card {
        background: white;
        border-left: 6px solid #218838;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 3px 12px rgba(0,0,0,0.07);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Website Heading ----------------
st.markdown(
    '<div class="main-title">🌱 Crop Yield Prediction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Machine learning for smarter agricultural planning'
    '</div>',
    unsafe_allow_html=True,
)

st.write("")
st.info("Enter the details below to predict the estimated crop yield.")

# ---------------- Input Form ----------------
with st.form("crop_prediction_form"):

    st.markdown(
        '<div class="section-heading">🌾 Crop and Location</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="input-heading">🌍 Country / Area</div>',
            unsafe_allow_html=True,
        )
        area = st.selectbox(
            "Country",
            options=[""] + area_options,
            format_func=lambda x: "Select country" if x == "" else x,
            label_visibility="collapsed",
        )

    with col2:
        st.markdown(
            '<div class="input-heading">🌾 Crop Name</div>',
            unsafe_allow_html=True,
        )
        item = st.selectbox(
            "Crop",
            options=[""] + crop_options,
            format_func=lambda x: "Select crop name" if x == "" else x,
            label_visibility="collapsed",
        )

    st.divider()

    st.markdown(
        '<div class="section-heading">🌦️ Year and Weather Details</div>',
        unsafe_allow_html=True,
    )

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(
            '<div class="input-heading">📅 Year of Cultivation</div>',
            unsafe_allow_html=True,
        )
        year_text = st.text_input(
            "Year",
            placeholder="Enter year (e.g. 2013)",
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="input-heading">🌧️ Annual Rainfall (mm)</div>',
            unsafe_allow_html=True,
        )
        rainfall_text = st.text_input(
            "Rainfall",
            placeholder="Enter annual rainfall in mm",
            label_visibility="collapsed",
        )

    with col4:
        st.markdown(
            '<div class="input-heading">🧪 Pesticides Used (tonnes)</div>',
            unsafe_allow_html=True,
        )
        pesticides_text = st.text_input(
            "Pesticides",
            placeholder="Enter pesticide quantity in tonnes",
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="input-heading">🌡️ Average Temperature (°C)</div>',
            unsafe_allow_html=True,
        )
        temperature_text = st.text_input(
            "Temperature",
            placeholder="Enter average temperature in °C",
            label_visibility="collapsed",
        )

    submitted = st.form_submit_button(
        "🌱 Predict Crop Yield",
        use_container_width=True,
        type="primary",
    )

# ---------------- Prediction ----------------
if submitted:

    # Validate empty inputs
    if not area or not item:
        st.error("Please select a country and a crop.")

    elif not all([
        year_text.strip(),
        rainfall_text.strip(),
        pesticides_text.strip(),
        temperature_text.strip(),
    ]):
        st.error("Please fill in all four year and weather input boxes.")

    else:
        try:
            year_value = int(year_text.strip())
            rainfall = float(rainfall_text.strip())
            pesticides = float(pesticides_text.strip())
            avg_temp = float(temperature_text.strip())

            if not 1990 <= year_value <= 2035:
                st.error("Year must be between 1990 and 2035.")

            elif rainfall < 0 or pesticides < 0:
                st.error("Rainfall and pesticide quantity cannot be negative.")

            elif not -10 <= avg_temp <= 60:
                st.error("Temperature must be between -10°C and 60°C.")

            else:
                input_data = pd.DataFrame([{
                    "Area": area,
                    "Item": item,
                    "Year": year_value,
                    "average_rain_fall_mm_per_year": rainfall,
                    "pesticides_tonnes": pesticides,
                    "avg_temp": avg_temp,
                }])

                prediction = float(model.predict(input_data)[0])

                st.divider()
                st.subheader("📊 Prediction Result")

                st.markdown(
                    f"""
                    <div class="result-card">
                        <p style="color:#53665a;font-size:18px;">
                            Estimated Yield for {item}
                        </p>
                        <h1 style="color:#176b37;">
                            {prediction:,.2f}
                        </h1>
                        <p style="color:#53665a;">
                            hg/ha (hectograms per hectare)
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    "This is an estimate, not a guarantee of actual yield. "
                    "Predictions for years outside the training period "
                    "may be less reliable."
                )

                # ---------------- Model Performance ----------------
                st.divider()
                st.subheader("📈 Model Performance")

                st.write("Previously measured Random Forest test results:")

                metric1, metric2, metric3 = st.columns(3)

                metric1.metric("R² Score", "0.9558")
                metric2.metric("Mean Absolute Error (MAE)", "10,289.22")
                metric3.metric("Root Mean Squared Error (RMSE)", "20,253.93")

                st.caption(
                    "Error metrics are measured in hg/ha. "
                    "These are previously measured test results."
                )

                # ---------------- SHAP Explainability ----------------
                st.divider()
                st.subheader("🔍 Why Did the Model Make This Prediction?")

                try:
                    import shap
                    import matplotlib.pyplot as plt

                    with st.spinner("Calculating SHAP explanations..."):
                        transformed_input = preprocessor.transform(input_data)

                        if hasattr(transformed_input, "toarray"):
                            transformed_input = transformed_input.toarray()

                        explainer = shap.TreeExplainer(regressor)
                        shap_values = explainer.shap_values(transformed_input)

                        if isinstance(shap_values, list):
                            contributions = shap_values[0][0]
                        else:
                            contributions = shap_values[0]

                        if getattr(contributions, "ndim", 1) > 1:
                            contributions = contributions[:, 0]

                        feature_names = preprocessor.get_feature_names_out()

                        importance = pd.DataFrame({
                            "Feature": feature_names,
                            "SHAP value": contributions,
                        })

                        importance["Magnitude"] = (
                            importance["SHAP value"].abs()
                        )

                        top_features = importance.nlargest(
                            10, "Magnitude"
                        ).sort_values("SHAP value")

                        fig, ax = plt.subplots(figsize=(10, 6))

                        bar_colors = [
                            "#218838" if value >= 0 else "#d9534f"
                            for value in top_features["SHAP value"]
                        ]

                        ax.barh(
                            top_features["Feature"],
                            top_features["SHAP value"],
                            color=bar_colors,
                        )

                        ax.axvline(0, color="black", linewidth=0.8)
                        ax.set_xlabel("SHAP value (hg/ha contribution)")
                        ax.set_title(
                            "Top 10 Factors Influencing This Prediction"
                        )

                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    st.markdown(
                        """
                        **How to understand the chart**

                        - **Green bars:** features pushing the prediction higher.
                        - **Red bars:** features pushing the prediction lower.
                        - **Longer bars:** larger contributions relative to
                          the model's baseline prediction.

                        SHAP explains model behavior; it does not prove
                        that a feature causes a real-world change in yield.
                        """
                    )

                except Exception as error:
                    st.warning(
                        "The prediction succeeded, but the SHAP chart "
                        "could not be generated."
                    )
                    st.code(str(error))

        except ValueError:
            st.error(
                "Please enter valid numbers in the year and weather boxes."
            )

# ---------------- Footer ----------------
st.divider()

st.caption(
    "Crop Yield Prediction | Random Forest | Explainable AI with SHAP"
)
