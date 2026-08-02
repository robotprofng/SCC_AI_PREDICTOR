import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SCC AI Predictor",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# CANCER AI THEME
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #160B0F;
        color: #F8F5F2;
    }


    html, body, [class*="css"] {
        color: #F8F5F2;
        font-family: Arial, sans-serif;
    }


    h1 {
        color: #E8B4B8 !important;
        font-size: 42px !important;
        font-weight: 800;
    }


    h2, h3 {
        color: #F4C7C3 !important;
    }


    p {
        color: #F8F5F2 !important;
    }


    section[data-testid="stSidebar"] {

        background-color: #240E14;

    }


    section[data-testid="stSidebar"] * {

        color: #F8F5F2 !important;

    }


    div[data-baseweb="select"] > div {

        background-color: #2B151C;
        color: white;

    }


    input {

        background-color: #2B151C !important;
        color: white !important;

    }


    .stButton > button {

        background-color: #8B1E3F;
        color: white;
        width: 100%;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;

    }


    .stButton > button:hover {

        background-color: #B8325A;

    }


    hr {

        border-color: #8B1E3F;

    }

    </style>

    """,

    unsafe_allow_html=True
)



# ============================================================
# LOAD MODEL FILES
# ============================================================

model = joblib.load("SCC_best_model.pkl")

feature_names = joblib.load("feature_names.pkl")

label_encoder = joblib.load("label_encoder.pkl")



# ============================================================
# HEADER
# ============================================================

st.title("🧬 SCC AI Predictor")

st.subheader(
    "Machine Learning-Based Oral Cancer Risk Prediction"
)


st.write(
    """
    This artificial intelligence tool estimates oral cancer diagnosis
    patterns using clinical and lifestyle-related factors.

    **Developed for research and educational purposes.**

    ⚠️ This tool does not replace clinical examination,
    biopsy, imaging, or specialist diagnosis.
    """
)


st.divider()



# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧬 About the AI Model")

    st.write(
        """
        **Model:**
        Logistic Regression


        **Application:**
        Oral Squamous Cell Carcinoma (SCC)


        **Input Variables:**
        Clinical and lifestyle risk factors


        **Purpose:**
        Research and AI education
        """
    )



# ============================================================
# PATIENT INPUTS
# ============================================================


st.header("Patient Clinical Information")


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



# ============================================================
# PREDICTION
# ============================================================


st.divider()


if st.button("🧬 Generate AI Risk Assessment"):


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


    patient_encoded = pd.get_dummies(patient_data)


    patient_encoded = patient_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )


    prediction = model.predict(patient_encoded)


    diagnosis = label_encoder.inverse_transform(prediction)[0]



    st.header("AI Assessment Result")



    if diagnosis == "Yes":

        st.error(
            "⚠️ Model Prediction: Higher Oral Cancer Risk Pattern Detected"
        )


    else:

        st.success(
            "✅ Model Prediction: Lower Oral Cancer Risk Pattern"
        )



    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(patient_encoded)[0]

        confidence = max(probability) * 100


        st.info(
            f"Prediction confidence: {confidence:.1f}%"
        )



    st.warning(
        """
        Clinical Reminder:

        AI predictions should be interpreted alongside
        clinical examination, patient history, imaging,
        and histopathological confirmation.
        """
    )



# ============================================================
# FOOTER
# ============================================================


st.divider()


st.caption(
    "🧬 SCC AI Predictor | Oral Pathology Artificial Intelligence Research"
)
