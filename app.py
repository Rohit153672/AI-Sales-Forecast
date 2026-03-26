import streamlit as st
import requests

st.title("📦 Supply Chain Forecasting System")

st.write("Enter input values:")

data = {
    "lag_1": st.number_input("lag_1", value=100),
    "lag_2": st.number_input("lag_2", value=110),
    "lag_3": st.number_input("lag_3", value=105),
    "lag_4": st.number_input("lag_4", value=120),
    "lag_12": st.number_input("lag_12", value=130),
    "rolling_4": st.number_input("rolling_4", value=110),
    "price": st.number_input("price", value=20),
    "promo": st.number_input("promo", value=1),
    "week": st.number_input("week", value=12),
    "month": st.number_input("month", value=3),
}

if st.button("Predict"):
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"data": data}
    )

    result = response.json()

    st.subheader("📊 Forecast")
    st.write(result["forecast"])

    st.subheader("🧠 Key Drivers")
    st.write(result["drivers"])

    st.subheader("📦 Inventory Recommendation")
    st.write(result["inventory"])