"""
model.py
========
Ensemble model training and evaluation for Insurance Fraud Detection.
"""

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


RANDOM_STATE = 42


def train_test_stratified_split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size,
                            random_state=RANDOM_STATE, stratify=y)


def tune_xgboost(X_train, y_train, n_trials=50):
    """Optuna Bayesian search for XGBoost hyperparameters."""
    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())

    def objective(trial):
        params = {
            'n_estimators'    : trial.suggest_int('n_estimators', 200, 800),
            'max_depth'       : trial.suggest_int('max_depth', 4, 10),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma'           : trial.suggest_float('gamma', 0, 1),
            'reg_alpha'       : trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda'      : trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'scale_pos_weight': scale_pos,
            'random_state'    : RANDOM_STATE,
            'n_jobs'          : -1,
            'eval_metric'     : 'auc',
            'use_label_encoder': False,
        }
        model = xgb.XGBClassifier(**params)
        cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def tune_lightgbm(X_train, y_train, n_trials=50):
    """Optuna Bayesian search for LightGBM hyperparameters."""
    def objective(trial):
        params = {
            'n_estimators'     : trial.suggest_int('n_estimators', 200, 800),
            'max_depth'        : trial.suggest_int('max_depth', 4, 12),
            'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves'       : trial.suggest_int('num_leaves', 20, 150),
            'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'class_weight'     : 'balanced',
            'random_state'     : RANDOM_STATE,
            'n_jobs'           : -1,
            'verbosity'        : -1,
        }
        model  = lgb.LGBMClassifier(**params)
        cv     = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def find_best_threshold(model, X_val, y_val):
    """Find threshold that maximises F1 on validation set."""
    y_proba = model.predict_proba(X_val)[:, 1]
    best_f1, best_t = 0, 0.5
    for t in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_val, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(best_t, 2), round(best_f1, 4)


def evaluate_model(name, model, X_val, y_val, threshold=0.5):
    """Returns a dict of evaluation metrics for a trained model."""
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    return {
        'Model'    : name,
        'AUC-ROC'  : round(roc_auc_score(y_val, y_proba), 4),
        'Avg Prec' : round(average_precision_score(y_val, y_proba), 4),
        'F1'       : round(f1_score(y_val, y_pred), 4),
        'Precision': round(precision_score(y_val, y_pred), 4),
        'Recall'   : round(recall_score(y_val, y_pred), 4),
        'Accuracy' : round(accuracy_score(y_val, y_pred), 4),
        'Threshold': threshold,
    }


def build_ensemble(xgb_model, lgbm_model, rf_model, X_val, y_val, weights=(3, 3, 1)):
    """
    Builds soft-voting ensemble from three trained models.
    Returns ensemble probabilities and best threshold.
    """
    p_xgb  = xgb_model.predict_proba(X_val)[:, 1]
    p_lgbm = lgbm_model.predict_proba(X_val)[:, 1]
    p_rf   = rf_model.predict_proba(X_val)[:, 1]
    p_ens  = (weights[0]*p_xgb + weights[1]*p_lgbm + weights[2]*p_rf) / sum(weights)

    best_f1, best_t = 0, 0.5
    for t in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_val, (p_ens >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    return p_ens, round(best_t, 2)


def save_model(model, path):
    joblib.dump(model, path)
    print(f"  ✅ Saved: {path}")


def load_model(path):
    return joblib.load(path)