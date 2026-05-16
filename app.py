import streamlit as st

import joblib


# Load Model
model = joblib.load("model.pkl")


# Title
st.title("Student Performance Prediction System")


# Inputs
hours = st.number_input("Study Hours")

attendance = st.number_input("Attendance Percentage")

previous_marks = st.number_input("Previous Marks")


# Prediction Button
if st.button("Predict Marks"):

    prediction = model.predict([
        [hours, attendance, previous_marks]
    ])

    st.success(
        f"Predicted Marks: {prediction[0]:.2f}"
    )