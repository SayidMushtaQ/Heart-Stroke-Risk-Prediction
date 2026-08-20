import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# LOAD MODEL, SCALER AND EXPECTED COLUMNS
# --------------------------------------------------

model = joblib.load("Logistic_regression.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("❤️ Heart Disease Prediction")

st.markdown(
    "Enter the patient's information below to predict the risk of heart disease."
)


# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------

age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)


sex = st.selectbox(
    "Sex",
    ["M", "F"]
)


chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)


resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=250,
    value=120
)


cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=100,
    max_value=600,
    value=200
)


fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)


resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)


max_hr = st.slider(
    "Maximum Heart Rate",
    min_value=60,
    max_value=220,
    value=150
)


exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["Y", "N"]
)


oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    min_value=0.0,
    max_value=6.0,
    value=1.0,
    step=0.1
)


st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button("🔮 Predict", use_container_width=True):

    # ----------------------------------------------
    # CREATE RAW INPUT
    # ----------------------------------------------

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        "Sex_" + sex: 1,

        "ChestPainType_" + chest_pain: 1,

        "RestingECG_" + resting_ecg: 1,

        "ExerciseAngina_" + exercise_angina: 1,

        "ST_Slope_" + st_slope: 1
    }


    # ----------------------------------------------
    # CONVERT INPUT TO DATAFRAME
    # ----------------------------------------------

    input_df = pd.DataFrame([raw_input])


    # ----------------------------------------------
    # ADD MISSING COLUMNS
    # ----------------------------------------------

    for col in expected_columns:

        if col not in input_df.columns:
            input_df[col] = 0


    # ----------------------------------------------
    # REMOVE EXTRA COLUMNS
    # AND KEEP EXACT MODEL COLUMN ORDER
    # ----------------------------------------------

    input_df = input_df[expected_columns]


    # ----------------------------------------------
    # SCALE INPUT
    # ----------------------------------------------

    input_scaled = scaler.transform(input_df)


    # ----------------------------------------------
    # MAKE PREDICTION
    # ----------------------------------------------

    prediction = model.predict(input_scaled)[0]


    # ----------------------------------------------
    # GET PROBABILITY
    # ----------------------------------------------

    probability = model.predict_proba(input_scaled)[0][1]


    # ----------------------------------------------
    # DISPLAY RESULT
    # ----------------------------------------------

    if prediction == 1:

        st.error("⚠️ High Risk of Heart Disease")

        st.write(
            f"Estimated probability: **{probability * 100:.2f}%**"
        )

    else:

        st.success("✅ Low Risk of Heart Disease")

        st.write(
            f"Estimated probability: **{probability * 100:.2f}%**"
        )


    # ----------------------------------------------
    # SHOW INPUT DATA
    # ----------------------------------------------

    with st.expander("View Input Data"):

        st.dataframe(input_df)