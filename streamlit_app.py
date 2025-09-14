import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- DATA AND MODEL LOADING ---

# Cache the data loading to avoid reloading on every interaction.
@st.cache_data
def load_data():
    """Loads the heart disease dataset from a CSV file."""
    try:
        df = pd.read_csv('heart.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'heart.csv' not found. Please make sure the file is in the same directory as app.py.")
        return None

# Cache the model training to avoid retraining on every interaction.
@st.cache_resource
def train_model(df):
    """Trains a Logistic Regression model on the provided dataframe."""
    X = df.drop('target', axis=1)
    y = df['target']
    # We train the model on the full dataset for the live app
    model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
    model.fit(X, y)
    return model

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# --- Main Application Logic ---
data = load_data()
if data is not None:
    model = train_model(data)

    # --- USER INTERFACE ---
    st.title("Heart Disease Prediction")
    st.markdown("Enter the patient's details below to predict the likelihood of heart disease.")

    # Create columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=100, value=52, help="The patient's age in years.")
        sex = st.selectbox('Sex', options=[(1, 'Male'), (0, 'Female')], format_func=lambda x: x[1], help="The patient's gender.")[0]
        cp = st.selectbox('Chest Pain Type (CP)', options=[
            (0, 'Typical Angina'), (1, 'Atypical Angina'),
            (2, 'Non-anginal Pain'), (3, 'Asymptomatic')
        ], format_func=lambda x: x[1], help="The type of chest pain experienced by the patient.")[0]
        trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=125, help="The patient's resting blood pressure in mm Hg. Normal is < 120 mmHg.")
        chol = st.number_input('Serum Cholesterol (chol)', min_value=100, max_value=600, value=212, help="The patient's cholesterol measurement in mg/dL. Desirable is < 200 mg/dL.")
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[(1, 'True'), (0, 'False')], format_func=lambda x: x[1], help="Indicates if the patient's fasting blood sugar is greater than 120 mg/dL.")[0]

    with col2:
        restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', options=[
            (0, 'Normal'), (1, 'ST-T wave abnormality'), (2, 'Probable or definite left ventricular hypertrophy')
        ], format_func=lambda x: x[1], help="Results of the resting electrocardiogram.")[0]
        thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=168, help="The highest heart rate achieved during a stress test. Often estimated as 220 minus your age.")
        exang = st.selectbox('Exercise Induced Angina (exang)', options=[(1, 'Yes'), (0, 'No')], format_func=lambda x: x[1], help="Whether the patient experienced angina (chest pain) during exercise.")[0]
        oldpeak = st.number_input('ST depression induced by exercise (oldpeak)', min_value=0.0, max_value=6.2, value=1.0, step=0.1, help="Measures the depression of the ST segment on an ECG during exercise relative to rest. Normal is typically 0.")
        slope = st.selectbox('Slope of the peak exercise ST segment (slope)', options=[
            (0, 'Upsloping'), (1, 'Flat'), (2, 'Downsloping')
        ], format_func=lambda x: x[1], help="The slope of the ST segment during the peak of the exercise test.")[0]
        ca = st.selectbox('Number of major vessels colored by fluoroscopy (ca)', options=[0, 1, 2, 3, 4], help="The number of major coronary arteries (0-4) that are seen to be narrowed during a fluoroscopy.")
        thal = st.selectbox('Thalassemia (thal)', options=[
            (0, 'Normal'), (1, 'Fixed defect'), (2, 'Reversible defect'), (3, 'Unknown')
        ], format_func=lambda x: x[1], help="A blood disorder called Thalassemia. This shows the result from a specific test.")[0]

    # --- PREDICTION LOGIC ---
    if st.button('**Predict**', use_container_width=True):
        # Prepare the feature array in the correct order for the model
        features = np.array([[
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]])

        # Make a prediction
        prediction = model.predict(features)

        st.write("---")
        st.subheader("Prediction Result")

        # Display the result with appropriate messaging
        if prediction[0] == 1:
            st.error('**High probability of Heart Disease**', icon="💔")
            st.warning("This is a prediction based on the provided data and does not constitute medical advice. Please consult a healthcare professional.", icon="⚠️")
        else:
            st.success('**Low probability of Heart Disease**', icon="💖")
            st.info("This is a prediction based on the provided data. Continue to maintain a healthy lifestyle and consult a doctor for regular check-ups.", icon="ℹ️")

