import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
try:
    API_URL = st.secrets.get('API_URL')
except Exception:
    API_URL = None
API_URL = API_URL or os.getenv('API_URL', 'http://localhost:8000')
DATA_PATH = BASE_DIR / 'petrochemical_maintenance.csv'

# Ensure project root is on sys.path so `from ml import ...` works in different runtimes
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

@st.cache_data
def load_local_history():
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    return df


def local_predict_from_series(series: pd.Series) -> dict:
    """Load local models and run prediction using ml.pipeline helpers.
    Returns a dict matching the backend /predict response shape.
    """
    try:
        # import lazily to avoid requiring ml at Streamlit import time
        from ml.pipeline import load_models, predict_fault, predict_maintenance, explain_prediction
    except Exception as e:
        raise RuntimeError(f'Local models not available: {e}')

    fault_pipeline, maintenance_pipeline, label_encoder, metrics = load_models()
    row = pd.DataFrame([series.to_dict()])
    fault_label, fault_conf, fault_probs = predict_fault(fault_pipeline, label_encoder, row)
    maintenance_required, maintenance_conf, maintenance_probs = predict_maintenance(maintenance_pipeline, row)
    explanation = explain_prediction(fault_pipeline, row, top_n=5)
    status = 'Critical' if maintenance_required or (fault_label != 'no_fault' and fault_conf > 0.75) else ('Warning' if maintenance_required or fault_label != 'no_fault' else 'Healthy')
    recommended = 'Inspect equipment immediately and schedule maintenance.' if status == 'Critical' else ('Monitor closely and prepare a service window.' if status == 'Warning' else 'Continue monitoring under normal operation.')
    return {
        'equipment_id': series['equipment_id'],
        'predicted_fault': fault_label,
        'fault_confidence': round(fault_conf, 4),
        'predicted_maintenance': maintenance_required,
        'maintenance_confidence': round(maintenance_conf, 4),
        'health_status': status,
        'recommended_action': recommended,
        'fault_probabilities': [round(float(x), 4) for x in fault_probs],
        'maintenance_probabilities': [round(float(x), 4) for x in maintenance_probs],
        'explanation': explanation,
    }


def compute_health_label(fault: str, maintenance: bool, fault_conf: float, maintenance_conf: float):
    if maintenance or fault != 'no_fault':
        if max(fault_conf, maintenance_conf) >= 0.75:
            return 'Critical', '🔴 Critical'
        return 'Warning', '🟠 Warning'
    return 'Healthy', '🟢 Healthy'


def main():
    st.set_page_config(page_title='Industrial Equipment Monitoring', layout='wide')
    st.title('Industrial Equipment Health & Predictive Maintenance Dashboard')

    df = load_local_history()
    equipment_ids = sorted(df['equipment_id'].unique())
    equipment_id = st.sidebar.selectbox('Select Equipment ID', equipment_ids)
    st.sidebar.markdown('### Model monitoring')
    st.sidebar.markdown('Connects to backend at ' + API_URL)

    if equipment_id is None:
        st.warning('Select an equipment ID to begin.')
        return

    history_url = f'{API_URL}/equipment/{equipment_id}'
    predict_url = f'{API_URL}/predict'
    health_url = f'{API_URL}/model/health'

    # Try to fetch history from remote backend, otherwise use local CSV
    history_data = None
    try:
        history_resp = requests.get(history_url, timeout=6)
        history_resp.raise_for_status()
        history_data = history_resp.json()
    except Exception as err:
        st.warning(f'Unable to fetch equipment history from backend ({err}). Using local dataset fallback.')
        subset = df[df['equipment_id'] == equipment_id].sort_values('timestamp')
        if subset.empty:
            st.error(f'No history available for equipment {equipment_id}')
            return
        history = subset.tail(200).to_dict(orient='records')
        latest_local = history[-1]
        history_data = {
            'equipment_id': equipment_id,
            'history': history,
            'latest': latest_local,
        }

    history_df = pd.DataFrame(history_data['history'])
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    latest = history_df.iloc[-1]

    st.subheader(f'Current status for {equipment_id}')
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_temp = px.line(history_df, x='timestamp', y='temperature_c', markers=True, title='Temperature over time')
        st.plotly_chart(fig_temp, use_container_width=True)
        history_df['vibration_magnitude'] = np.sqrt(history_df['vibration_x']**2 + history_df['vibration_y']**2 + history_df['vibration_z']**2)
        fig_vib = px.line(history_df, x='timestamp', y='vibration_magnitude', markers=True, title='Vibration magnitude over time')
        st.plotly_chart(fig_vib, use_container_width=True)
    with col2:
        st.metric('Temperature (°C)', f'{latest.temperature_c:.2f}')
        st.metric('RPM', f'{latest.rpm:.0f}')
        st.metric('Pressure (bar)', f'{latest.pressure_bar:.3f}')
        st.metric('Current (A)', f'{latest.current_a:.3f}')
        st.metric('Latest fault', latest.fault_type)
        st.metric('Maintenance required', 'Yes' if latest.maintenance_required else 'No')

    payload = {
        'timestamp': latest.timestamp.isoformat(),
        'equipment_id': latest.equipment_id,
        'vibration_x': float(latest.vibration_x),
        'vibration_y': float(latest.vibration_y),
        'vibration_z': float(latest.vibration_z),
        'temperature_c': float(latest.temperature_c),
        'current_a': float(latest.current_a),
        'rpm': float(latest.rpm),
        'pressure_bar': float(latest.pressure_bar),
        'wavelet_feature_1': float(latest.wavelet_feature_1),
        'wavelet_feature_2': float(latest.wavelet_feature_2),
        'wavelet_feature_3': float(latest.wavelet_feature_3),
        'wavelet_feature_4': float(latest.wavelet_feature_4),
        'wavelet_feature_5': float(latest.wavelet_feature_5),
    }

    # Try remote backend first, fall back to local model if unavailable
    prediction = None
    try:
        predict_resp = requests.post(predict_url, json=payload, timeout=6)
        predict_resp.raise_for_status()
        prediction = predict_resp.json()
    except Exception as err:
        st.warning(f'Prediction request failed (backend unreachable). Trying local model: {err}')
        try:
            prediction = local_predict_from_series(latest)
        except Exception as e:
            st.error(f'Local prediction also failed: {e}')
            return

    status_text, status_badge = compute_health_label(
        prediction['predicted_fault'], prediction['predicted_maintenance'], prediction['fault_confidence'], prediction['maintenance_confidence']
    )

    st.markdown('## Live Prediction & Equipment Health')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Predicted Fault', prediction['predicted_fault'])
        st.metric('Fault confidence', f"{prediction['fault_confidence']:.2%}")
    with col2:
        st.metric('Maintenance alert', 'Yes' if prediction['predicted_maintenance'] else 'No')
        st.metric('Maintenance confidence', f"{prediction['maintenance_confidence']:.2%}")
    with col3:
        st.markdown(f'### {status_badge}')
        st.write(prediction['recommended_action'])

    st.markdown('### Model explainability')
    if prediction.get('explanation'):
        explanation_df = pd.DataFrame(prediction['explanation'])
        st.bar_chart(explanation_df.set_index('feature')['impact'])
    else:
        st.info('SHAP explainability not available for this model in the current environment.')

    st.markdown('### Model health monitoring')
    # Try to fetch model health from remote backend, otherwise use local metrics
    health = None
    try:
        health_resp = requests.get(health_url, timeout=6)
        health_resp.raise_for_status()
        health = health_resp.json()
    except Exception as err:
        st.warning(f'Unable to fetch model health from backend ({err}). Using local metrics fallback.')
        try:
            from joblib import load
            metrics = load(BASE_DIR / 'ml' / 'model_store' / 'metrics.joblib')
            fault_acc = None
            maint_acc = None
            try:
                fault_acc = metrics.get('fault_report', {}).get('accuracy')
            except Exception:
                fault_acc = None
            try:
                maint_acc = metrics.get('maintenance_report', {}).get('accuracy')
            except Exception:
                maint_acc = None
            feature_importance = None
            try:
                from ml.pipeline import load_models
                fault_pipeline, _, _, _ = load_models()
                model = fault_pipeline.named_steps['model']
                if hasattr(model, 'feature_importances_'):
                    names = list(fault_pipeline.named_steps['preprocessor'].get_feature_names_out())
                    values = model.feature_importances_
                    feature_importance = sorted(
                        [{'feature': n, 'importance': float(v)} for n, v in zip(names, values)],
                        key=lambda x: x['importance'],
                        reverse=True,
                    )[:10]
            except Exception:
                feature_importance = None
            distribution = {
                'fault_types': df['fault_type'].value_counts(normalize=True).to_dict(),
                'maintenance': df['maintenance_required'].value_counts(normalize=True).to_dict(),
            }
            health = {
                'fault_accuracy': fault_acc,
                'maintenance_accuracy': maint_acc,
                'latest_prediction_confidence': None,
                'feature_importance': feature_importance,
                'prediction_distribution': distribution,
            }
        except Exception as e:
            st.error(f'Local model health fallback failed: {e}')
            return

    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        st.metric('Fault model accuracy', f"{health.get('fault_accuracy', 0):.2%}")
        st.metric('Maintenance model accuracy', f"{health.get('maintenance_accuracy', 0):.2%}")
        st.metric('Latest prediction confidence', f"{prediction['fault_confidence']:.2%}")
    with metrics_col2:
        if health.get('feature_importance'):
            importance_df = pd.DataFrame(health['feature_importance'])
            fig_imp = px.bar(importance_df, x='importance', y='feature', orientation='h', title='Top feature importances')
            st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('### Prediction distribution')
    dist = health.get('prediction_distribution', {})
    if dist:
        st.write('Fault type distribution:')
        st.write(pd.Series(dist.get('fault_types', {})).rename('share'))
        st.write('Maintenance distribution:')
        st.write(pd.Series(dist.get('maintenance', {})).rename('share'))

    st.markdown('### Anomaly simulation')
    vibration_mean = history_df['vibration_magnitude'].mean()
    vibration_latest = history_df['vibration_magnitude'].iloc[-1]
    anomaly_score = abs(vibration_latest - vibration_mean) / (history_df['vibration_magnitude'].std() + 1e-6)
    st.write(f'Vibration anomaly score: {anomaly_score:.2f}')
    if anomaly_score > 2.0:
        st.warning('Anomaly indicator: current vibration is significantly different from historical behavior.')
    else:
        st.success('Anomaly indicator: behavior remains within expected bounds.')

    st.markdown('---')
    st.write('Data refreshes when the selected equipment changes.')


if __name__ == '__main__':
    main()
