import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

from datetime import date

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Supply Chain Forecast",
    page_icon="📦",
    layout="wide",
)

st.sidebar.header("Forecast Configuration")

store_id = st.sidebar.number_input(
    "Store ID",
    min_value=1,
    max_value=45,
    value=1,
)

dept_id = st.sidebar.number_input(
    "Department ID",
    min_value=1,
    value=1,
)

forecast_date = st.sidebar.date_input(
    "Forecast Date",
    value=date(2012, 9, 28),
)

forecast_button = st.sidebar.button(
    "🚀 Forecast"
)

def get_explanation(payload):

    response = requests.post(
        f"{API_URL}/explain",
        json=payload,
    )

    response.raise_for_status()

    return response.json()

def get_forecast(payload):

    response = requests.post(
        f"{API_URL}/forecast",
        json=payload,
    )

    response.raise_for_status()

    return response.json()

def get_inventory(payload):

    response = requests.post(
        f"{API_URL}/inventory",
        json=payload,
    )

    response.raise_for_status()

    return response.json()

def get_history(payload):

    response = requests.post(
        f"{API_URL}/history",
        json=payload,
    )

    response.raise_for_status()

    return response.json()

st.title("📦 AI Supply Chain Forecasting")

st.markdown(
    """
Demand Forecasting & Inventory Intelligence using
LightGBM, FastAPI and SHAP Explainability.
"""
)

st.divider()

if forecast_button:

    payload = {
        "store_id": int(store_id),
        "dept_id": int(dept_id),
        "forecast_date": str(forecast_date),
    }

    try:

        # ============================================
        # Call APIs
        # ============================================

        forecast_response = get_forecast(payload)
        inventory_response = get_inventory(payload)
        history_response = get_history(payload)

        forecast = forecast_response["forecast"]
        inventory = inventory_response["inventory"]

        history_df = pd.DataFrame({
            "Date": history_response["dates"],
            "Sales": history_response["sales"]
        })

        history_df["Date"] = pd.to_datetime(history_df["Date"])

        # ============================================
        # KPI CARDS
        # ============================================

        st.subheader("📊 Forecast Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Forecast Demand",
            f"{forecast:,.0f} units"
        )

        col2.metric(
            "Safety Stock",
            f"{inventory['safety_stock']:,.0f}"
        )

        col3.metric(
            "Reorder Point",
            f"{inventory['reorder_point']:,.0f}"
        )

        st.divider()

        # ============================================
        # Inventory Status
        # ============================================

        st.subheader("📦 Inventory Status")

        if inventory["reorder_point"] > forecast:

            st.success(
                "🟢 Inventory level is healthy."
            )

        else:

            st.warning(
                "🟡 Inventory approaching reorder threshold."
            )

        st.divider()

        # ============================================
        # Historical Sales Chart
        # ============================================

        st.subheader("📈 Historical Weekly Sales")

        fig = go.Figure()

        # Historical line
        fig.add_trace(
            go.Scatter(
                x=history_df["Date"],
                y=history_df["Sales"],
                mode="lines+markers",
                name="Historical Sales",
                line=dict(width=3),
            )
        )

        # Forecast connection
        last_date = history_df["Date"].iloc[-1]
        last_sales = history_df["Sales"].iloc[-1]

        fig.add_trace(
            go.Scatter(
                x=[last_date, pd.to_datetime(forecast_date)],
                y=[last_sales, forecast],
                mode="lines",
                line=dict(color="red", dash="dash"),
                showlegend=False,
            )
        )

        # Forecast point
        fig.add_trace(
            go.Scatter(
                x=[pd.to_datetime(forecast_date)],
                y=[forecast],
                mode="markers",
                marker=dict(
                    color="red",
                    size=14,
                    symbol="star",
                ),
                name="Forecast",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Week",
            yaxis_title="Weekly Sales",
            legend_title="Legend",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ============================================
        # SHAP Explainability
        # ============================================

        explanation = get_explanation(payload)

        st.divider()

        st.subheader("🔍 Why This Forecast?")

        feature_names = {
        "lag_52": "Sales 52 Weeks Ago",
        "lag_26": "Sales 26 Weeks Ago",
        "lag_12": "Sales 12 Weeks Ago",
        "lag_4": "Sales 4 Weeks Ago",
        "lag_1": "Last Week Sales",
        "rolling_mean_4": "4-Week Average",
        "rolling_mean_12": "12-Week Average",
        "markdown_1": "Markdown 1",
        "markdown_2": "Markdown 2",
        "markdown_3": "Markdown 3",
        }

        drivers = pd.DataFrame(explanation["drivers"])
        drivers["feature"] = drivers["feature"].replace(feature_names)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=drivers["impact"],
                y=drivers["feature"],
                orientation="h",
                text=drivers["impact"].round(1),
                textposition="outside",
            )
        )

        fig.update_layout(
            title="Top SHAP Drivers",
            template="plotly_dark",
            height=350,
            yaxis=dict(autorange="reversed"),
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:

        st.error(str(e))