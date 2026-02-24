"""Streamlit frontend: fire map + PM2.5 forecast + drift report."""

from __future__ import annotations

import sys
from pathlib import Path

# Make src importable when running via streamlit
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wildfire Smoke Advisory",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

CITIES = {
    "Toronto": (43.65, -79.38),
    "Vancouver": (49.25, -123.12),
    "Calgary": (51.05, -114.07),
    "Edmonton": (53.55, -113.50),
    "Montreal": (45.50, -73.57),
    "Los Angeles": (34.05, -118.24),
    "Seattle": (47.61, -122.33),
    "Portland": (45.52, -122.68),
    "San Francisco": (37.77, -122.42),
    "Denver": (39.74, -104.98),
    "New York": (40.71, -74.01),
}

RISK_COLORS = {
    "Low": "#2ecc71",
    "Moderate": "#f39c12",
    "High": "#e67e22",
    "Hazardous": "#e74c3c",
}


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔥 Smoke Advisory")
    st.caption("24-hour PM2.5 forecast powered by NASA FIRMS + OpenAQ + Open-Meteo")

    location_mode = st.radio("Location input", ["Select city", "Custom lat/lon"])

    if location_mode == "Select city":
        city_name = st.selectbox("City", list(CITIES.keys()), index=0)
        lat, lon = CITIES[city_name]
    else:
        lat = st.number_input("Latitude", value=43.65, format="%.4f", min_value=-90.0, max_value=90.0)
        lon = st.number_input("Longitude", value=-79.38, format="%.4f", min_value=-180.0, max_value=180.0)
        city_name = f"({lat:.2f}, {lon:.2f})"

    st.divider()
    run_forecast = st.button("Run Forecast", type="primary", use_container_width=True)
    st.caption("⚠️ Approximates smoke transport — not a replacement for NOAA HRRR-Smoke")


# ─── Helper functions ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def call_predict_api(lat: float, lon: float) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE}/predict", json={"lat": lat, "lon": lon}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"API unavailable — using synthetic forecast. ({e})")
        return _synthetic_forecast(lat, lon)


def _synthetic_forecast(lat: float, lon: float) -> dict:
    """Fallback when API is not running."""
    rng = np.random.default_rng(int(abs(lat * lon * 100)) % 2**31)
    pm25 = float(rng.uniform(5, 50))
    from src.features.score import pm25_to_advisory, confidence_interval

    adv = pm25_to_advisory(pm25)
    lo, hi = confidence_interval(pm25)
    return {
        "advisory_score": adv["advisory_score"],
        "risk_tier": adv["risk_tier"],
        "pm25_estimate": round(pm25, 2),
        "confidence_lower": lo,
        "confidence_upper": hi,
        "model_version": "fallback",
        "color": adv["color"],
    }


@st.cache_data(ttl=600)
def load_fires() -> pd.DataFrame:
    try:
        from src.ingestion.firms import fetch_firms
        return fetch_firms(days=3)
    except Exception:
        return pd.DataFrame(columns=["latitude", "longitude", "frp", "confidence"])


def build_historical_pm25(lat: float, lon: float, forecast: dict) -> pd.DataFrame:
    """Generate synthetic historical PM2.5 + forecast for the chart."""
    rng = np.random.default_rng(int(abs(lat * 1000 + lon * 1000)) % 2**31)
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    hours = [now - timedelta(hours=h) for h in range(47, -1, -1)]
    base = max(0, forecast["pm25_estimate"] - rng.uniform(5, 15))
    pm25_hist = [max(0, base + rng.normal(0, 4)) for _ in hours]
    return pd.DataFrame({"datetime": hours, "pm25": pm25_hist})


# ─── Main layout ──────────────────────────────────────────────────────────────
tab_forecast, tab_map, tab_drift = st.tabs(["📊 Forecast", "🗺️ Fire Map", "📈 Drift Report"])

# ── Tab 1: Forecast ───────────────────────────────────────────────────────────
with tab_forecast:
    st.header(f"PM2.5 Forecast — {city_name}")

    if run_forecast or "forecast" not in st.session_state:
        with st.spinner("Fetching forecast…"):
            st.session_state["forecast"] = call_predict_api(lat, lon)
            st.session_state["forecast_city"] = city_name
            st.session_state["forecast_lat"] = lat
            st.session_state["forecast_lon"] = lon

    forecast = st.session_state.get("forecast")

    if forecast:
        score = forecast["advisory_score"]
        tier = forecast["risk_tier"]
        pm25 = forecast["pm25_estimate"]
        ci_lo = forecast["confidence_lower"]
        ci_hi = forecast["confidence_upper"]
        color = RISK_COLORS.get(tier, "#888")

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Advisory Score", f"{score} / 100")
        col2.metric("Risk Tier", tier)
        col3.metric("PM2.5 (24h ahead)", f"{pm25:.1f} μg/m³")
        col4.metric("Confidence ±", f"±{(ci_hi - pm25):.1f} μg/m³")

        # Risk badge
        st.markdown(
            f"""
            <div style="background:{color};padding:16px;border-radius:12px;
                        text-align:center;margin:12px 0;">
              <h2 style="color:white;margin:0;">{tier} Risk</h2>
              <p style="color:white;margin:4px 0;">Advisory Score: {score} / 100</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Historical PM2.5 + forecast chart
        hist = build_historical_pm25(
            st.session_state["forecast_lat"],
            st.session_state["forecast_lon"],
            forecast,
        )
        forecast_time = hist["datetime"].iloc[-1] + timedelta(hours=24)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hist["datetime"],
                y=hist["pm25"],
                name="Historical PM2.5",
                line=dict(color="#3498db", width=2),
            )
        )
        # Confidence band
        fig.add_trace(
            go.Scatter(
                x=[forecast_time, forecast_time],
                y=[ci_lo, ci_hi],
                mode="markers",
                marker=dict(size=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[forecast_time],
                y=[pm25],
                mode="markers",
                name="24h Forecast",
                marker=dict(size=14, color=color, symbol="diamond"),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[ci_hi - pm25],
                    arrayminus=[pm25 - ci_lo],
                    color=color,
                ),
            )
        )
        # EPA guideline lines
        for threshold, label in [(12, "Good/Moderate"), (35.4, "Moderate/High"), (55.4, "High/Hazardous")]:
            fig.add_hline(y=threshold, line_dash="dot", line_color="gray", opacity=0.5,
                          annotation_text=label, annotation_position="right")

        fig.update_layout(
            title="Historical PM2.5 + 24h Forecast",
            xaxis_title="Time (UTC)",
            yaxis_title="PM2.5 (μg/m³)",
            template="plotly_white",
            height=400,
            legend=dict(orientation="h", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Model version: {forecast['model_version']} | Data: NASA FIRMS + OpenAQ + Open-Meteo")


# ── Tab 2: Fire Map ────────────────────────────────────────────────────────────
with tab_map:
    st.header("Active Fire Map (Past 3 Days)")

    try:
        import folium
        from streamlit_folium import st_folium

        with st.spinner("Loading fire data…"):
            fires = load_fires()

        m = folium.Map(location=[lat, lon], zoom_start=5, tiles="CartoDB positron")

        # Add sensor location
        folium.Marker(
            location=[lat, lon],
            popup=f"Sensor: {city_name}",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)

        # Add fire markers
        if not fires.empty:
            for _, fire in fires.head(500).iterrows():
                # Size by FRP, color by confidence
                radius = min(15, max(3, float(fire.get("frp", 10)) / 20))
                confidence = str(fire.get("confidence", "nominal")).lower()
                fire_color = {"high": "#e74c3c", "nominal": "#f39c12", "low": "#f1c40f"}.get(confidence, "#e67e22")

                folium.CircleMarker(
                    location=[fire["latitude"], fire["longitude"]],
                    radius=radius,
                    color=fire_color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"FRP: {fire.get('frp', 'N/A'):.1f} MW | Confidence: {confidence}",
                ).add_to(m)

            st.caption(f"Showing {min(500, len(fires))} of {len(fires)} fire detections. Circle size ∝ FRP. Color: red=high, orange=nominal, yellow=low confidence.")
        else:
            st.info("No fire data available. Check FIRMS_API_KEY in .env")

        st_folium(m, width=None, height=500)

    except ImportError:
        st.error("Install folium and streamlit-folium: `pip install folium streamlit-folium`")


# ── Tab 3: Drift Report ────────────────────────────────────────────────────────
with tab_drift:
    st.header("Feature Drift Monitoring")
    st.caption("Evidently report comparing training baseline vs. recent inference data")

    drift_path = Path("data/reports/drift_latest.html")
    if drift_path.exists():
        html = drift_path.read_text(encoding="utf-8")
        st.components.v1.html(html, height=700, scrolling=True)
    else:
        st.info(
            "No drift report found. Generate one by running:\n\n"
            "```bash\npython -m src.monitoring.drift\n```\n\n"
            "Or access it via the API: `GET /drift-report`"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Generate Drift Report Now"):
                with st.spinner("Generating Evidently report…"):
                    try:
                        from src.monitoring.drift import generate_drift_report
                        path = generate_drift_report()
                        st.success(f"Report saved to {path}. Reload the page to view.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
