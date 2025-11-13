import streamlit as st
import requests

st.title("AI Sales Forecasting (Walmart Dataset)")

st.write("Enter weekly features to get prediction + SHAP explanations.")

with st.form("form"):
    store_id = st.number_input("Store ID", 1)
    sku = st.number_input("SKU (Dept No.)", 1)
    price = st.number_input("Price Proxy (MarkDown1)", 0.0)
    promo = st.selectbox("Promo", [0, 1])

    lag_1 = st.number_input("lag_1", 0.0)
    lag_2 = st.number_input("lag_2", 0.0)
    lag_3 = st.number_input("lag_3", 0.0)
    lag_4 = st.number_input("lag_4", 0.0)
    lag_12 = st.number_input("lag_12", 0.0)

    rolling_4 = st.number_input("rolling_4 mean", 0.0)
    week = st.number_input("week", 1)
    month = st.number_input("month", 1)

    submit = st.form_submit_button("Predict")

if submit:
    payload = {
        "data": {
            "store_id": store_id,
            "sku": sku,
            "price": price,
            "promo": promo,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_4": lag_4,
            "lag_12": lag_12,
            "rolling_4": rolling_4,
            "week": week,
            "month": month
        }
    }

    res = requests.post("http://localhost:8000/predict", json=payload)
    st.write(res.json())
