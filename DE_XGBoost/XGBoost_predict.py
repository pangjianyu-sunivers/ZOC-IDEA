import pandas as pd
from xgboost import XGBClassifier
import pickle
import re


def load_model():
    """Load the trained model and the ordered feature-name list."""
    model = XGBClassifier()
    model.load_model('/data/DE_XGBoost/xgb_model.json')

    with open('/data/DE_XGBoost/feature_names.pkl', 'rb') as f:
        features = pickle.load(f)

    return model, features


def _normalize_numeric_feature(frame, column_name):
    if column_name not in frame.columns:
        return

    series = frame[column_name]
    if pd.api.types.is_numeric_dtype(series):
        return

    normalized = pd.to_numeric(
        series.astype(str).str.extract(r'([0-9]+(?:\.[0-9]+)?)')[0],
        errors='coerce',
    )
    frame[column_name] = normalized


def _normalize_pid_column(frame):
    if 'video_name' in frame.columns:
        extracted = frame['video_name'].astype(str).str.extract(r"pid_([^_]+)")[0]
        frame['pid'] = extracted.where(extracted.notna(), frame.get('pid'))

    if 'pid' in frame.columns:
        frame['pid'] = frame['pid'].apply(lambda value: '' if pd.isna(value) else str(value).strip())


def predict_new_data(file_path):
    """Run the trained XGBoost classifier on a new batch of feature rows."""
    new_df = pd.read_excel(file_path + '/blinking_results.xlsx')

    model, required_features = load_model()

    missing_features = set(required_features) - set(new_df.columns)
    if missing_features:
        raise ValueError(f'Missing required features: {missing_features}')

    _normalize_numeric_feature(new_df, 'OSDI')
    _normalize_pid_column(new_df)

    X_new = new_df[required_features].copy()
    for feature_name in required_features:
        if not pd.api.types.is_numeric_dtype(X_new[feature_name]):
            X_new[feature_name] = pd.to_numeric(X_new[feature_name], errors='coerce')

    preds = model.predict(X_new)
    preds_proba = model.predict_proba(X_new)[:, 1]

    new_df['predicted_dryeye_label'] = preds
    new_df['dryeye_prob'] = preds_proba
    new_df.to_excel(file_path + '/predict_results.xlsx', index=False)
    print(f'Prediction complete; results saved to: {file_path}/predict_results.xlsx')

    pid = pd.Series(new_df['pid'])
    preds = pd.Series(preds)
    preds_proba = pd.Series(preds_proba)

    return pid, preds, preds_proba
