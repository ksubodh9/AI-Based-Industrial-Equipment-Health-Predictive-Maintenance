import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from joblib import dump, load

RAW_FEATURES = [
    'timestamp',
    'equipment_id',
    'vibration_x',
    'vibration_y',
    'vibration_z',
    'temperature_c',
    'current_a',
    'rpm',
    'pressure_bar',
    'wavelet_feature_1',
    'wavelet_feature_2',
    'wavelet_feature_3',
    'wavelet_feature_4',
    'wavelet_feature_5',
]

DERIVED_FEATURES = [
    'vibration_magnitude',
    'vibration_mean',
    'vibration_std',
    'temp_pressure_ratio',
    'power_index',
    'hour',
    'day_of_week',
]

MODEL_FEATURES = [
    'equipment_id',
    'vibration_x',
    'vibration_y',
    'vibration_z',
    'vibration_magnitude',
    'vibration_mean',
    'vibration_std',
    'temperature_c',
    'current_a',
    'rpm',
    'pressure_bar',
    'temp_pressure_ratio',
    'power_index',
    'hour',
    'day_of_week',
    'wavelet_feature_1',
    'wavelet_feature_2',
    'wavelet_feature_3',
    'wavelet_feature_4',
    'wavelet_feature_5',
]

CATEGORY_FEATURES = ['equipment_id']
NUMERIC_FEATURES = [c for c in MODEL_FEATURES if c not in CATEGORY_FEATURES]

MODEL_DIR = Path(__file__).resolve().parents[1] / 'ml' / 'model_store'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour.fillna(0).astype(int)
        df['day_of_week'] = df['timestamp'].dt.dayofweek.fillna(0).astype(int)
    else:
        df['hour'] = 0
        df['day_of_week'] = 0

    df['vibration_magnitude'] = np.sqrt(
        df['vibration_x'] ** 2 + df['vibration_y'] ** 2 + df['vibration_z'] ** 2
    )
    df['vibration_mean'] = df[['vibration_x', 'vibration_y', 'vibration_z']].mean(axis=1)
    df['vibration_std'] = df[['vibration_x', 'vibration_y', 'vibration_z']].std(axis=1).fillna(0)
    df['temp_pressure_ratio'] = df['temperature_c'] / (df['pressure_bar'].replace(0, 1e-3))
    df['power_index'] = df['current_a'] * df['rpm']

    return df


def build_preprocessing_pipeline() -> ColumnTransformer:
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, CATEGORY_FEATURES),
            ('num', numeric_transformer, NUMERIC_FEATURES),
        ],
        remainder='drop',
        sparse_threshold=0,
    )
    return preprocessor


def create_pipeline(model) -> Pipeline:
    return Pipeline(steps=[
        ('preprocessor', build_preprocessing_pipeline()),
        ('model', model),
    ])


def prepare_training_data(df: pd.DataFrame):
    df = engineer_features(df)
    X = df[MODEL_FEATURES]
    y_fault = df['fault_type'].astype(str)
    y_maint = df['maintenance_required'].astype(int)
    label_encoder = LabelEncoder()
    y_fault_encoded = label_encoder.fit_transform(y_fault)
    return X, y_fault_encoded, y_maint, label_encoder


def save_model_artifacts(fault_pipeline, maintenance_pipeline, label_encoder, metrics: dict):
    dump(fault_pipeline, MODEL_DIR / 'fault_pipeline.joblib')
    dump(maintenance_pipeline, MODEL_DIR / 'maintenance_pipeline.joblib')
    dump(label_encoder, MODEL_DIR / 'fault_label_encoder.joblib')
    dump(metrics, MODEL_DIR / 'metrics.joblib')


def load_models():
    fault_pipeline = load(MODEL_DIR / 'fault_pipeline.joblib')
    maintenance_pipeline = load(MODEL_DIR / 'maintenance_pipeline.joblib')
    label_encoder = load(MODEL_DIR / 'fault_label_encoder.joblib')
    metrics = load(MODEL_DIR / 'metrics.joblib')
    return fault_pipeline, maintenance_pipeline, label_encoder, metrics


def predict_fault(fault_pipeline, label_encoder, row: pd.DataFrame):
    row = engineer_features(row)
    proba = fault_pipeline.predict_proba(row[MODEL_FEATURES])[0]
    pred = np.argmax(proba)
    return label_encoder.inverse_transform([pred])[0], float(proba[pred]), proba.tolist()


def predict_maintenance(maintenance_pipeline, row: pd.DataFrame):
    row = engineer_features(row)
    proba = maintenance_pipeline.predict_proba(row[MODEL_FEATURES])[0]
    pred = np.argmax(proba)
    return bool(pred), float(proba[pred]), proba.tolist()


def explain_prediction(fault_pipeline, row: pd.DataFrame, top_n: int = 5):
    try:
        import shap
    except ImportError:
        return []

    row = engineer_features(row)
    preprocessor = fault_pipeline.named_steps['preprocessor']
    model = fault_pipeline.named_steps['model']
    X_transformed = preprocessor.transform(row[MODEL_FEATURES])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    try:
        if isinstance(shap_values, list):
            shap_values = shap_values[np.argmax([np.abs(s).sum() for s in shap_values])]
        shap_arr = np.asarray(shap_values)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[np.argmax(np.abs(shap_arr).sum(axis=(1, 2)))]
        if shap_arr.ndim == 2 and shap_arr.shape[0] == 1:
            shap_arr = shap_arr[0]
        shap_arr = shap_arr.reshape(-1)

        feature_names = []
        cat_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(CATEGORY_FEATURES)
        feature_names.extend(cat_names.tolist())
        feature_names.extend(NUMERIC_FEATURES)
        contributions = dict(zip(feature_names, shap_arr))
        ranked = sorted(contributions.items(), key=lambda x: abs(float(x[1])), reverse=True)[:top_n]
        return [{'feature': name, 'impact': float(value)} for name, value in ranked]
    except Exception:
        return []
