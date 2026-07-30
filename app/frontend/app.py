"""Streamlit frontend.

All predictions are obtained over HTTP from the FastAPI service; this module
never imports joblib or loads the model artifact.
"""

import os

import requests
import streamlit as st

# FastAPI address
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)


st.set_page_config(
    page_title="Riyadh Air Quality",
    page_icon="🌫️",
    layout="centered",
)


st.title("Riyadh Air Quality Risk")

st.write(
    "Enter the current air-quality and weather values "
    "to predict whether PM2.5 pollution will be high "
    "in the next hour."
)


# Check whether FastAPI is available
try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )
    health_response.raise_for_status()

    health_data = health_response.json()

    if health_data["model_loaded"]:
        st.success("API connected and model loaded.")
    else:
        st.warning("API connected, but the model is not loaded.")

except requests.RequestException:
    st.error(
        "FastAPI is unavailable. "
        "Make sure the API is running on port 8000."
    )


with st.form("prediction_form"):

    st.subheader("Current air-quality measurements")

    pm2_5 = st.number_input(
        "Current PM2.5",
        min_value=0.0,
        max_value=1000.0,
        value=30.0,
    )

    pm10 = st.number_input(
        "Current PM10",
        min_value=0.0,
        max_value=1500.0,
        value=65.0,
    )

    st.subheader("Current weather measurements")

    temperature_2m = st.number_input(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=65.0,
        value=34.0,
    )

    relative_humidity_2m = st.number_input(
        "Relative humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
    )

    wind_speed_10m = st.number_input(
        "Wind speed",
        min_value=0.0,
        max_value=200.0,
        value=12.0,
    )

    st.subheader("Time information")

    hour = st.slider(
        "Hour of the day",
        min_value=0,
        max_value=23,
        value=14,
    )

    day_of_week = st.selectbox(
        "Day of the week",
        options=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
        ],
        format_func=lambda day: [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ][day],
    )

    st.subheader("Historical PM2.5 measurements")

    pm2_5_lag_1 = st.number_input(
        "PM2.5 one hour ago",
        min_value=0.0,
        max_value=1000.0,
        value=29.0,
    )

    pm2_5_lag_3 = st.number_input(
        "PM2.5 three hours ago",
        min_value=0.0,
        max_value=1000.0,
        value=27.0,
    )

    pm2_5_rolling_mean_6 = st.number_input(
        "Average PM2.5 during the previous six hours",
        min_value=0.0,
        max_value=1000.0,
        value=28.0,
    )

    submitted = st.form_submit_button(
        "Predict next-hour risk"
    )


if submitted:

    payload = {
        "pm2_5": pm2_5,
        "pm10": pm10,
        "temperature_2m": temperature_2m,
        "relative_humidity_2m": relative_humidity_2m,
        "wind_speed_10m": wind_speed_10m,
        "hour": hour,
        "day_of_week": day_of_week,
        "pm2_5_lag_1": pm2_5_lag_1,
        "pm2_5_lag_3": pm2_5_lag_3,
        "pm2_5_rolling_mean_6": pm2_5_rolling_mean_6,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10,
        )

        response.raise_for_status()
        result = response.json()

        st.divider()
        st.subheader("Prediction result")

        st.metric(
            label="High-pollution probability",
            value=f"{result['probability']:.1%}",
        )

        if result["prediction"] == 1:
            st.error(
                "High PM2.5 pollution risk is predicted "
                "for the next hour."
            )
        else:
            st.success(
                "Normal PM2.5 pollution risk is predicted "
                "for the next hour."
            )

        st.caption(
            f"Request ID: {result['request_id']}"
        )

    except requests.RequestException as error:
        st.error(
            f"Prediction request failed: {error}"
        )