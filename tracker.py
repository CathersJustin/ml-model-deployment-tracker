import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

REGISTRY_PATH = Path('model_registry.json')
PREDICTIONS_LOG = Path('data/predictions_log.csv')


# Registry handling

def load_registry() -> List[Dict[str, Any]]:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return []


def save_registry(registry: List[Dict[str, Any]]):
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)


def get_model(name: str, version: str) -> Dict[str, Any]:
    registry = load_registry()
    for model in registry:
        if model['name'] == name and model['version'] == version:
            return model
    return {}


# Prediction logging

def init_prediction_log():
    if not PREDICTIONS_LOG.exists():
        df = pd.DataFrame(columns=[
            'timestamp', 'model_name', 'model_version',
            'input_summary', 'prediction', 'true_label', 'confidence'
        ])
        df.to_csv(PREDICTIONS_LOG, index=False)


def log_prediction(model_name: str, model_version: str, input_summary: str, prediction: str, true_label: str, confidence: float):
    init_prediction_log()
    df = pd.read_csv(PREDICTIONS_LOG)
    df = df.append({
        'timestamp': datetime.utcnow().isoformat(),
        'model_name': model_name,
        'model_version': model_version,
        'input_summary': input_summary,
        'prediction': prediction,
        'true_label': true_label,
        'confidence': confidence
    }, ignore_index=True)
    df.to_csv(PREDICTIONS_LOG, index=False)


def load_predictions() -> pd.DataFrame:
    init_prediction_log()
    return pd.read_csv(PREDICTIONS_LOG)


# Retraining

def add_model_version(name: str, algorithm: str, metrics: Dict[str, Any]):
    registry = load_registry()
    versions = [m for m in registry if m['name'] == name]
    new_version = str(len(versions) + 1)
    registry.append({
        'name': name,
        'version': new_version,
        'algorithm': algorithm,
        'training_date': datetime.utcnow().strftime('%Y-%m-%d'),
        'accuracy': metrics.get('accuracy'),
        'status': 'deployed'
    })
    save_registry(registry)
    return new_version