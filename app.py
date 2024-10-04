import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model and scaler
@st.cache_resource
def load_model():
    with open('logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('standard_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

model, scaler = load_model()

# Define the app
st.title('Student Performance Prediction App')

st.write("""
This app predicts whether a student will pass or fail based on various factors.
Please fill in the following information:
""")

# Create input fields
age = st.slider('Age', 15, 22, 17)
gender = st.selectbox('Gender', ['Male', 'Female'])
department = st.selectbox('Department', ['Science', 'Arts', 'Commerce'])
boarding = st.selectbox('Boarding or Day Student', ['Boarding', 'Day'])
attendance = st.slider('Attendance Score', 0, 100, 85)
computer_hours = st.slider('Computer Hours per Week', 0, 20, 5)
lesson_hours = st.slider('Extra Lesson Hours per Week', 0, 15, 5)
mental_health = st.slider('Mental Health Score', 0, 100, 70)
guardian_income = st.number_input('Guardian Income (NGN)', min_value=0, max_value=1000000, value=200000)
guardian_education = st.selectbox('Guardian Education Level', ['No Formal Education', 'Primary', 'Secondary', 'Tertiary'])
guardian_relationship = st.selectbox('Guardian Relationship', ['Both Parents', 'Single Mom', 'Single Dad', 'Grandparent', 'Other Relative'])
teacher_education = st.selectbox('Minimum Teacher Education', ['Diploma', 'Bachelor\'s', 'Master\'s'])
teacher_experience = st.slider('Average Teacher Years of Experience', 1, 20, 5)

# Create a dictionary with user inputs
user_data = {
    'Age': age,
    'Gender': gender,
    'Department': department,
    'Boarding_or_Day': boarding,
    'Attendance_Score': attendance,
    'Computer_Hours': computer_hours,
    'Lesson_Hours': lesson_hours,
    'Mental_Health_Score': mental_health,
    'Guardian_Income': guardian_income,
    'Guardian_Education': guardian_education,
    'Guardian_Relationship': guardian_relationship,
    'Min_Teacher_Education': teacher_education,
    'Avg_Teacher_Years_of_Experience': teacher_experience
}

# Convert user input to DataFrame
input_df = pd.DataFrame([user_data])

# Perform one-hot encoding
categorical_cols = ['Gender', 'Department', 'Boarding_or_Day', 'Guardian_Education', 'Guardian_Relationship', 'Min_Teacher_Education']
input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

# Ensure all columns from training are present
for col in model.feature_names_in_:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder columns to match training data
input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

# Scale continuous features
cont_cols = ['Age', 'Attendance_Score', 'Computer_Hours', 'Lesson_Hours', 'Mental_Health_Score', 'Guardian_Income', 'Avg_Teacher_Years_of_Experience']
input_encoded[cont_cols] = scaler.transform(input_encoded[cont_cols])

# Make prediction
if st.button('Predict Performance'):
    prediction = model.predict(input_encoded)
    probability = model.predict_proba(input_encoded)

    st.subheader('Prediction Result:')
    if prediction[0] == 'Pass':
        st.success(f"The student is likely to PASS with a probability of {probability[0][1]:.2f}")
    else:
        st.error(f"The student is likely to FAIL with a probability of {probability[0][0]:.2f}")

    st.write('Note: This prediction is based on the provided information and should be used as a guide only.')

# Add information about the model
st.sidebar.header('About the Model')
st.sidebar.write("""
This model uses Logistic Regression to predict student performance.
It was trained on synthetic data representing various factors that may influence a student's academic success.
The model's accuracy on the test set was approximately 85%.
""")

# Add feature importance
feature_importance = pd.DataFrame({
    'feature': model.feature_names_in_,
    'importance': abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

st.sidebar.subheader('Top 10 Most Important Features:')
st.sidebar.table(feature_importance)