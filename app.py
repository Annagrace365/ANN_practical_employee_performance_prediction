
import streamlit as st
import numpy as np
import tensorflow as tf


# Page title
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="📊"
)


# App heading
st.title("📊 Employee Performance Predictor")

st.write(
    "Enter Training Hours and Attendance to predict employee performance."
)


# Attendance threshold
attendance_threshold = 50


# Load trained ANN model
model = tf.keras.models.load_model(
    "employee_performance_ann.keras"
)


# User inputs
training_hours = st.number_input(
    "Training Hours",
    min_value=0.0,
    max_value=30.0,
    value=8.0,
    step=1.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)


# Display attendance requirement
st.info(
    f"Minimum Attendance Required: {attendance_threshold}%"
)


# Prediction button
if st.button("Predict Performance"):

    # Check attendance threshold first
    if attendance < attendance_threshold:

        st.warning(
            f"Minimum attendance required is "
            f"{attendance_threshold}%. "
            "Performance prediction cannot be considered."
        )

    else:

        # Prepare input
        input_data = np.array([
            [training_hours, attendance]
        ])


        # ANN prediction
        probability = model.predict(
            input_data,
            verbose=0
        )[0][0]


        # Convert probability into result
        if probability >= 0.5:
            result = "Good"
        else:
            result = "Needs Improvement"


        # Display result
        st.subheader("Prediction Result")


        if result == "Good":
            st.success("Performance: GOOD")
        else:
            st.warning("Performance: NEEDS IMPROVEMENT")


        # Display probability
        st.write(
            "Good Probability:",
            round(float(probability) * 100, 2),
            "%"
        )


