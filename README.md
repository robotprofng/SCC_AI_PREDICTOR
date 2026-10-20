# Oral SCC AI Predictor

**Applying machine learning to oral cancer risk prediction through an interactive healthcare AI application.**

[Live Demo](https://scc-ai-predictor.streamlit.app/) · [Source Code](https://github.com/robotprofng/SCC_AI_PREDICTOR)

## About the Project

The Oral SCC AI Predictor is a healthcare AI project developed by Dr. Akinshipo and Dr. Irene Udebuana to explore the application of machine learning to oral squamous cell carcinoma (SCC).

I worked on translating a trained Logistic Regression model into an interactive web application that accepts clinical and lifestyle-related inputs and returns a model-generated prediction.

The project brings together dentistry, data science and software development to explore how predictive modelling can be applied to oral healthcare research.

## Key Features

- **Clinical and lifestyle inputs:** Collects 16 patient-related variables through an interactive interface.
- **Machine learning predictions:** Uses a trained Logistic Regression model to generate predictions from submitted inputs.
- **Automated data preparation:** Structures input data and prepares features for the model.
- **Interactive web interface:** Makes the prediction workflow accessible through a browser.
- **Prediction output:** Displays the model's predicted class and associated confidence score.

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- Joblib

## Run Locally

Clone the repository:

```bash
git clone https://github.com/robotprofng/SCC_AI_PREDICTOR.git
cd SCC_AI_PREDICTOR
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies and run the application:

```bash
pip install -r requirements.txt
streamlit run app.py
```

For Windows, activate the environment with `.venv\Scripts\activate` in Command Prompt or `.venv\Scripts\Activate.ps1` in PowerShell.

## Clinical Disclaimer

This project is intended for research and educational exploration. Its predictions have not been established as clinically reliable and must not be used to diagnose or rule out oral cancer, determine treatment, or replace assessment by a qualified healthcare professional.

A model confidence score is not equivalent to diagnostic accuracy or a confirmed probability of disease.

## Contributors

- Dr. Akinshipo
- Dr. Irene Udebuana

---

**Healthcare AI · Predictive Modelling · Machine Learning · Application Development**
# SCC_AI_PREDICTOR
