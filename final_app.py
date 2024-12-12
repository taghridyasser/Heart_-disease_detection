#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import pandas as pd

# Load the trained model pipeline
loaded_model = joblib.load(open('best_xgb_model_pipeline.pkl', 'rb'))
# Function for prediction
def heart_disease_detection(input_data):
    # Make the prediction
    prediction = loaded_model.predict(input_data)

    if prediction[0] == 0:
        return 'The person does not have heart disease'
    else:
        return 'The person has heart disease'

# Main Streamlit app
def main():
    # Title for the user interface
    st.title('Heart Disease Detection Web App')

    # Getting input data from the user
    age = st.number_input('Age', min_value=0, max_value=120)
    sex = st.selectbox('Sex', ['male', 'female'])
    chest_pain_type = st.selectbox(
        'Chest Pain Type',
        ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic']
    )
    resting_bp_s = st.number_input('Resting Blood Pressure', min_value=60, max_value=300, step=1)
    cholesterol = st.number_input('Cholesterol', min_value=100, max_value=500, step=1)
    fasting_blood_sugar = st.radio("Fasting Blood Sugar", ["true", "false"])
    resting_ecg = st.selectbox(
        'Resting Electrocardiographic Results',
        ['normal', 'having ST-T wave abnormality', 'left ventricular hypertrophy ']
        
    )
    max_heart_rate = st.number_input('Maximum Heart Rate Achieved', min_value=40, max_value=300, step=1)
    exercise_angina = st.radio("Exercise Induced Angina", ["yes", "no"])
    oldpeak = st.number_input('ST depression induced by exercise relative to rest', min_value=0.0, step=0.1)
    st_slope = st.selectbox(
        'Slope of the peak exercise ST segment',
        ['upsloping', 'flat', 'downsloping']
    )

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "chest pain type": [chest_pain_type],
        "resting bp s": [resting_bp_s],
        "cholesterol": [cholesterol],
        "fasting blood sugar": [fasting_blood_sugar],
        "resting ecg": [resting_ecg],
        "max heart rate": [max_heart_rate],
        "exercise angina": [exercise_angina],
        "oldpeak": [oldpeak],
        "ST slope": [st_slope]
    })

    # Ensure all categories are present in the input data
    input_data['sex'] = input_data['sex'].astype('category')
    input_data['chest pain type'] = input_data['chest pain type'].astype('category')
    input_data['fasting blood sugar'] = input_data['fasting blood sugar'].astype('category')
    input_data['resting ecg'] = input_data['resting ecg'].astype('category')
    input_data['exercise angina'] = input_data['exercise angina'].astype('category')
    input_data['ST slope'] = input_data['ST slope'].astype('category')

    # Prediction button
    if st.button('Predict'):
        result = heart_disease_detection(input_data)
        st.success(result)

if __name__ == '__main__':
    main()

