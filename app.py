import streamlit as st
import pandas as pd
import joblib


# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="SCC AI Predictor",
    page_icon="🦷",
    layout="wide"
)


# ==============================
# LOAD MODEL FILES
# ==============================

model = joblib.load("SCC_best_model.pkl")
feature_names = joblib.load("feature_names.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# ==============================
# HEADER
# ==============================

st.title("🦷 SCC AI PREDICTOR TOOL")

st.subheader(
    "Artificial Intelligence Risk Prediction for Oral Squamous Cell Carcinoma Risk"
)

st.write(
    """
This machine learning tool predicts oral cancer diagnosis patterns 
using clinical and lifestyle-related features.

**For research and educational purposes only. 
It is not a replacement for professional clinical diagnosis.**
"""
)


st.divider()


# ==============================
# SIDEBAR
# ==============================

with st.sidebar:

    st.header("About the Model")

    st.write(
        """
        This AI model was developed using machine learning 
        and clinical risk factors associated with Oral Squamous 
        Cell Carcinoma.
        
        Model:
        Logistic Regression
        
        Dataset:
        SCC Clinical Dataset
        """
    )


# ==============================
# INPUT FORM
# ==============================


st.header("Patient Information")


col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )


    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )


    tobacco = st.selectbox(
        "Tobacco Use",
        ["Yes", "No"]
    )


    alcohol = st.selectbox(
        "Alcohol Consumption",
        ["Yes", "No"]
    )


    hpv = st.selectbox(
        "HPV Infection",
        ["Yes", "No"]
    )


    betel = st.selectbox(
        "Betel Quid Use",
        ["No", "Yes"]
    )


    sun = st.selectbox(
        "Chronic Sun Exposure",
        ["No", "Yes"]
    )


    hygiene = st.selectbox(
        "Poor Oral Hygiene",
        ["Yes", "No"]
    )


with col2:

    diet = st.selectbox(
        "Diet (Fruits & Vegetables Intake)",
        ["Low", "Moderate", "High"]
    )


    family_history = st.selectbox(
        "Family History of Cancer",
        ["No", "Yes"]
    )


    immune = st.selectbox(
        "Compromised Immune System",
        ["No", "Yes"]
    )


    lesions = st.selectbox(
        "Oral Lesions",
        ["No", "Yes"]
    )


    bleeding = st.selectbox(
        "Unexplained Bleeding",
        ["No", "Yes"]
    )


    swallowing = st.selectbox(
        "Difficulty Swallowing",
        ["No", "Yes"]
    )


    patches = st.selectbox(
        "White or Red Patches in Mouth",
        ["No", "Yes"]
    )


    tumor_size = st.number_input(
        "Tumor Size (cm)",
        min_value=0.0,
        max_value=20.0,
        value=2.0
    )

        # ==============================
# PREDICTION SECTION
# ==============================


st.divider()

if st.button("🔍 Predict Oral Cancer Risk"):

    # Create input dataframe using ORIGINAL column names
    patient_data = pd.DataFrame({

        "Age": [age],

        "Gender": [gender],

        "Tobacco Use": [tobacco],

        "Alcohol Consumption": [alcohol],

        "HPV Infection": [hpv],

        "Betel Quid Use": [betel],

        "Chronic Sun Exposure": [sun],

        "Poor Oral Hygiene": [hygiene],

        "Diet (Fruits & Vegetables Intake)": [diet],

        "Family History of Cancer": [family_history],

        "Compromised Immune System": [immune],

        "Oral Lesions": [lesions],

        "Unexplained Bleeding": [bleeding],

        "Difficulty Swallowing": [swallowing],

        "White or Red Patches in Mouth": [patches],

        "Tumor Size (cm)": [tumor_size]

    })


    # Apply same encoding used during training
    patient_encoded = pd.get_dummies(patient_data)


    # Match training features exactly
    patient_encoded = patient_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )


    # Prediction
    prediction = model.predict(patient_encoded)


    # Convert numerical prediction back to label
    diagnosis = label_encoder.inverse_transform(prediction)[0]


    # Probability (if available)
    probability = None

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(patient_encoded)[0]



    # ==============================
    # DISPLAY RESULT
    # ==============================


    st.subheader("AI Prediction Result")


    if diagnosis == "Yes":

        st.error(
            "⚠️ Prediction: Oral Cancer Risk Detected"
        )

    else:

        st.success(
            "✅ Prediction: Lower Oral Cancer Risk Pattern"
        )


    if probability is not None:

        confidence = max(probability) * 100

        st.info(
            f"Model confidence: {confidence:.1f}%"
        )


    st.warning(
        """
        Important:
        
        This AI prediction is intended for research,
        education, and demonstration purposes.
        It does not replace clinical examination,
        biopsy, imaging, or specialist assessment.
        """
    )


# ==============================
# FOOTER
# ==============================

st.divider()

st.caption(
    "Developed for Oral Pathology AI Research | SCC Machine Learning Predictor"
)
