import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.pipeline import (
    load_dataset,
    engineer_features,
    load_models,
    predict_fault,
    predict_maintenance,
    explain_prediction,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / 'petrochemical_maintenance.csv'

app = FastAPI(
    title='Industrial Equipment Monitoring API',
    description='Backend service for predictive maintenance and fault classification',
    version='1.0.0',
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class SensorPayload(BaseModel):
    timestamp: Optional[str]
    equipment_id: str
    vibration_x: float
    vibration_y: float
    vibration_z: float
    temperature_c: float
    current_a: float
    rpm: float
    pressure_bar: float
    wavelet_feature_1: float
    wavelet_feature_2: float
    wavelet_feature_3: float
    wavelet_feature_4: float
    wavelet_feature_5: float

class PredictionResponse(BaseModel):
    equipment_id: str
    predicted_fault: str
    fault_confidence: float
    predicted_maintenance: bool
    maintenance_confidence: float
    health_status: str
    recommended_action: str
    fault_probabilities: List[float]
    maintenance_probabilities: List[float]
    explanation: Optional[List[dict]] = None

class EquipmentHistoryResponse(BaseModel):
    equipment_id: str
    history: List[dict]
    latest: dict

class ModelHealthResponse(BaseModel):
    fault_accuracy: Optional[float]
    maintenance_accuracy: Optional[float]
    latest_prediction_confidence: Optional[float]
    feature_importance: Optional[List[dict]] = None
    prediction_distribution: Optional[dict] = None

raw_data = load_dataset(str(DATA_PATH))
fault_pipeline, maintenance_pipeline, label_encoder, metrics = load_models()


def get_health_status(fault_label: str, fault_conf: float, maintenance_required: bool, maintenance_conf: float) -> str:
    if maintenance_required or fault_label != 'no_fault':
        if fault_conf > 0.75 or maintenance_conf > 0.75:
            return 'Critical'
        return 'Warning'
    return 'Healthy'


def get_recommended_action(status: str) -> str:
    if status == 'Critical':
        return 'Inspect equipment immediately and schedule maintenance.'
    if status == 'Warning':
        return 'Monitor closely and prepare a service window.'
    return 'Continue monitoring under normal operation.'


@app.get('/health', response_model=dict)
def health_check():
    return {'status': 'ok', 'loaded_equipment': int(raw_data['equipment_id'].nunique())}


@app.get('/equipment/{equipment_id}', response_model=EquipmentHistoryResponse)
def get_equipment_history(equipment_id: str):
    subset = raw_data[raw_data['equipment_id'] == equipment_id].sort_values('timestamp')
    if subset.empty:
        raise HTTPException(status_code=404, detail='Equipment ID not found')
    history = subset.tail(200).to_dict(orient='records')
    latest = history[-1]
    return EquipmentHistoryResponse(equipment_id=equipment_id, history=history, latest=latest)


@app.post('/predict', response_model=PredictionResponse)
def predict(payload: SensorPayload):
    data = payload.dict()
    row = pd.DataFrame([data])
    fault_label, fault_conf, fault_probs = predict_fault(fault_pipeline, label_encoder, row)
    maintenance_required, maintenance_conf, maintenance_probs = predict_maintenance(maintenance_pipeline, row)
    status = get_health_status(fault_label, fault_conf, maintenance_required, maintenance_conf)
    explanation = explain_prediction(fault_pipeline, row, top_n=5)
    return PredictionResponse(
        equipment_id=payload.equipment_id,
        predicted_fault=fault_label,
        fault_confidence=round(fault_conf, 4),
        predicted_maintenance=maintenance_required,
        maintenance_confidence=round(maintenance_conf, 4),
        health_status=status,
        recommended_action=get_recommended_action(status),
        fault_probabilities=[round(float(x), 4) for x in fault_probs],
        maintenance_probabilities=[round(float(x), 4) for x in maintenance_probs],
        explanation=explanation,
    )


@app.get('/model/health', response_model=ModelHealthResponse)
def model_health():
    fault_accuracy = None
    maintenance_accuracy = None
    if metrics is not None:
        try:
            fault_accuracy = metrics['fault_report']['accuracy']
        except Exception:
            fault_accuracy = None
        try:
            maintenance_accuracy = metrics['maintenance_report']['accuracy']
        except Exception:
            maintenance_accuracy = None

    importance = []
    try:
        model = fault_pipeline.named_steps['model']
        if hasattr(model, 'feature_importances_'):
            names = list(fault_pipeline.named_steps['preprocessor'].get_feature_names_out())
            values = model.feature_importances_
            importance = sorted(
                [{'feature': n, 'importance': float(v)} for n, v in zip(names, values)],
                key=lambda x: x['importance'],
                reverse=True,
            )[:10]
    except Exception:
        importance = []

    distribution = {
        'fault_types': raw_data['fault_type'].value_counts(normalize=True).to_dict(),
        'maintenance': raw_data['maintenance_required'].value_counts(normalize=True).to_dict(),
    }
    latest_confidence = None
    return ModelHealthResponse(
        fault_accuracy=round(fault_accuracy, 4) if fault_accuracy is not None else None,
        maintenance_accuracy=round(maintenance_accuracy, 4) if maintenance_accuracy is not None else None,
        latest_prediction_confidence=latest_confidence,
        feature_importance=importance,
        prediction_distribution=distribution,
    )
