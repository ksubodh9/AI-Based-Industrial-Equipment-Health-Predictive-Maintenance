# Industrial Equipment Monitoring and Predictive Maintenance Platform

This project is an end-to-end AI system for industrial equipment health monitoring. It includes:

- ML preprocessing and feature engineering pipeline for sensor data
- Fault classification and maintenance prediction models
- FastAPI backend serving predictions, equipment history, and model health
- Streamlit dashboard for interactive monitoring and explainability
- Docker-friendly architecture for future deployment

## Project Structure

- `ml/pipeline.py` - feature engineering, preprocessing pipeline, model save/load helpers
- `ml/train.py` - training script that builds, evaluates, and exports best-performing models
- `backend/app.py` - FastAPI backend exposing prediction and equipment history APIs
- `frontend/streamlit_app.py` - Streamlit dashboard for live monitoring, health status, and monitoring visuals
- `petrochemical_maintenance.csv` - dataset used for training and monitoring
- `requirements.txt` - Python dependencies

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the models:

```bash
python ml/train.py
```

3. Start the backend:

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

4. Start the Streamlit dashboard:

```bash
streamlit run frontend/streamlit_app.py
```

## API Endpoints

- `GET /health` - service status
- `GET /equipment/{equipment_id}` - historical sensor data for selected equipment
- `POST /predict` - prediction endpoint for fault and maintenance with confidence and explainability
- `GET /model/health` - model health metrics and feature importance

## Notes

- The platform supports dynamic equipment IDs and returns both fault and maintenance risk scores.
- The dashboard visualizes temperature, vibration, live sensor values, and model monitoring information.
- A future Docker deployment can containerize the backend and dashboard with `uvicorn` and `streamlit`.
