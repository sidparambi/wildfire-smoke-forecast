# Wildfire Smoke Advisory Forecaster

End-to-end MLOps system that ingests real-time NASA FIRMS satellite fire detections, OpenAQ air quality readings, and Open-Meteo wind data to predict PM2.5 concentrations 24 hours ahead — then maps them to a **0–100 Smoke Advisory Score**.

> **Disclaimer:** Approximates smoke transport using simplified atmospheric physics — not a replacement for NOAA HRRR-Smoke or Environment Canada forecasts.

---

## Architecture

```
NASA FIRMS  ──┐
OpenAQ      ──┼──► Feature Engineering ──► MLflow Experiment Tracking
Open-Meteo  ──┘         │                        │
                         │                        ▼
                    fire_pressure_score    Model Registry
                    lag features           (Ridge/RF/XGBoost/LGBM)
                         │                        │
                         └────────────────────────┘
                                          │
                                    FastAPI /predict
                                          │
                              ┌───────────┴───────────┐
                         Streamlit               Evidently
                         Dashboard              Drift Report
```

## Tech Stack

| Layer | Tool |
|---|---|
| Data ingestion | requests + pandas |
| Feature engineering | pandas + numpy |
| Experiment tracking | **MLflow** |
| Model registry | MLflow Model Registry |
| Models | Ridge / Random Forest / **XGBoost** / LightGBM |
| Data versioning | **DVC** |
| Orchestration | **Prefect** |
| Serving | **FastAPI** |
| Monitoring | **Evidently AI** |
| Frontend | **Streamlit** + Folium + Plotly |
| Containerisation | Docker Compose |
| CI/CD | **GitHub Actions** |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/wildfire-smoke-forecast
cd wildfire-smoke-forecast
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your FIRMS_API_KEY (free at https://firms.modaps.eosdis.nasa.gov/api/)
```

### 3. Run the full stack with Docker Compose

```bash
docker-compose up --build
```

Services:
- **FastAPI** → http://localhost:8000
- **MLflow UI** → http://localhost:5000
- **Prefect** → http://localhost:4200 (if configured)

### 4. Run locally (without Docker)

```bash
# Start MLflow
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &

# Fetch data + engineer features + train models
python -m src.ingestion.firms          # → data/raw/firms.csv
python -m src.ingestion.openaq         # → data/raw/openaq.csv
python -m src.ingestion.weather        # → data/raw/weather.csv
python -m src.training.train           # Logs 4 models to MLflow
python -m src.training.select          # Promotes best to Production registry

# Start API
uvicorn src.serving.app:app --reload --port 8000

# Start Streamlit
streamlit run app/streamlit_app.py
```

---

## API Reference

### `POST /predict`

```json
// Request
{"lat": 43.65, "lon": -79.38}

// Response
{
  "advisory_score": 42,
  "risk_tier": "Moderate",
  "pm25_estimate": 22.5,
  "confidence_lower": 14.5,
  "confidence_upper": 30.5,
  "model_version": "3",
  "color": "yellow"
}
```

### `GET /health`

Returns model load status and version.

### `GET /drift-report`

Serves the latest Evidently HTML drift report.

---

## Feature Engineering

**`fire_pressure_score`** — The core feature:
1. Find all FIRMS fire detections within 500 km in the past 48 h
2. Compare each fire's bearing to the wind direction
3. Fires upwind of the sensor (smoke blows toward sensor) score higher
4. Weight by fire radiative power (FRP) and inverse distance²
5. Sum → scalar score per sensor location per hour

**Additional features:** current PM2.5, wind speed, wind direction, humidity, hour of day, day of year, 24h PM2.5 lag, 48h PM2.5 lag

**Target:** PM2.5 reading 24 hours later

---

## Risk Tiers

| PM2.5 (μg/m³) | Advisory Score | Tier | Color |
|---|---|---|---|
| 0 – 12 | 0 – 20 | Low | 🟢 Green |
| 12 – 35 | 20 – 50 | Moderate | 🟡 Yellow |
| 35 – 55 | 50 – 75 | High | 🟠 Orange |
| 55+ | 75 – 100 | Hazardous | 🔴 Red |

---

## MLOps Pipeline

### Experiment Tracking (MLflow)

Each model training run logs:
- Parameters: model type, feature list, train/val split size
- Metrics: RMSE, MAE, R²
- Artifacts: model pickle, feature importance plot

```bash
# View all runs
mlflow ui --port 5000
```

### Model Promotion (MLflow Registry)

`src/training/select.py` compares validation RMSE across all runs and promotes the best to `Production` if it beats the current production model.

```
smoke-forecaster/
  v1 → Archived
  v2 → Archived
  v3 → Production  ← current best
```

### Orchestration (Prefect)

```bash
python flows/pipeline.py
```

Runs: `fetch_data → engineer_features → train → promote → drift_report`

### Monitoring (Evidently)

```bash
python -m src.monitoring.drift
# → data/reports/drift_latest.html
```

Compares training feature distributions against recent inference data. Accessible at `GET /drift-report`.

---

## Tests

```bash
pytest tests/ -v
```

Covers:
- `test_features.py`: haversine distance, bearing calculation, fire pressure score, PM2.5 → advisory score mapping, feature table construction
- `test_api.py`: all FastAPI endpoints, input validation, response schema

---

## CI/CD (GitHub Actions)

| Workflow | Trigger | Steps |
|---|---|---|
| `ci.yml` | Push / PR | ruff lint + pytest |
| `nightly.yml` | Daily 06:00 UTC | Fetch data + run inference + commit `latest.json` |

---

## Data Sources

| Source | Data | Auth |
|---|---|---|
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | Active fire detections (lat, lon, FRP) | Free API key |
| [OpenAQ](https://openaq.org/) | PM2.5 ground readings | None (public) |
| [Open-Meteo](https://open-meteo.com/) | Wind speed, direction, humidity | None (free) |

---

## Resume Bullet

> Built end-to-end wildfire smoke advisory system ingesting real-time NASA FIRMS satellite fire detections and global air quality sensors; compared Ridge/RF/XGBoost/LightGBM models via MLflow experiment tracking with automated production registry promotion; served advisory score predictions via FastAPI with Evidently drift monitoring, Prefect retraining pipeline, and GitHub Actions CI/CD; deployed on Streamlit Community Cloud.
