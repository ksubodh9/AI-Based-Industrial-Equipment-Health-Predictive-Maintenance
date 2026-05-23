import json
import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from joblib import dump

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.pipeline import (
    load_dataset,
    prepare_training_data,
    create_pipeline,
    save_model_artifacts,
    MODEL_DIR,
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_classifier(X_train, y_train, X_val, y_val, task_name):
    candidates = {
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'xgboost': XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1),
    }
    best_model = None
    best_score = -1
    best_report = None
    for name, model in candidates.items():
        pipeline = create_pipeline(model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)
        print(f'{task_name}: {name} score={score:.4f}')
        if score > best_score:
            best_score = score
            best_model = pipeline
            best_report = report
    return best_model, best_score, best_report


def main():
    data_path = Path(__file__).resolve().parents[1] / 'petrochemical_maintenance.csv'
    df = load_dataset(str(data_path))
    X, y_fault, y_maint, label_encoder = prepare_training_data(df)

    X_train, X_val, y_fault_train, y_fault_val = train_test_split(
        X, y_fault, test_size=0.2, random_state=42, stratify=y_fault
    )
    _, _, fault_report = train_classifier(X_train, y_fault_train, X_val, y_fault_val, 'fault')

    X_train2, X_val2, y_maint_train, y_maint_val = train_test_split(
        X, y_maint, test_size=0.2, random_state=42, stratify=y_maint
    )
    _, _, maintenance_report = train_classifier(X_train2, y_maint_train, X_val2, y_maint_val, 'maintenance')

    # Re-train selected best models on full dataset
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    fault_pipeline = create_pipeline(XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1))
    maintenance_pipeline = create_pipeline(RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    fault_pipeline.fit(X, y_fault)
    maintenance_pipeline.fit(X, y_maint)

    metrics = {
        'fault_report': fault_report,
        'maintenance_report': maintenance_report,
    }
    save_model_artifacts(fault_pipeline, maintenance_pipeline, label_encoder, metrics)
    print('Training complete. Artifacts saved to', MODEL_DIR)


if __name__ == '__main__':
    main()
