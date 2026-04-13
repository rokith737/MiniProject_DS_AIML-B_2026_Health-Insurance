"""
preprocessing.py
================
Feature engineering pipeline for Insurance Fraud Detection.
Called by M1 and M3 Colab notebooks.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ── Date / duration cleaning ──────────────────────────────────────────────────

def clean_beneficiary(df):
    """
    Cleans beneficiary (patient demographics) DataFrame.
    - Computes Age from DOB
    - Creates is_deceased flag
    - Remaps chronic condition columns to 0/1
    - Computes total_chronic_conditions score
    """
    df = df.copy()
    for col in ['DOB', 'DOD']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    ref_date = pd.Timestamp('2009-12-01')
    df['Age'] = ((ref_date - df['DOB']).dt.days / 365.25).round(1).clip(0, 120)
    df['is_deceased'] = df['DOD'].notna().astype(int)

    chronic_cols = [
        'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
        'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
        'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
        'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
        'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
        'ChronicCond_stroke',
    ]
    for col in [c for c in chronic_cols if c in df.columns]:
        df[col] = df[col].astype(str).map({'1': 1, '2': 0}).fillna(0).astype(int)

    available = [c for c in chronic_cols if c in df.columns]
    df['total_chronic_conditions'] = df[available].sum(axis=1)

    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype(str).map({'1': 0, '2': 1}).fillna(0).astype(int)

    reimb_cols = [c for c in df.columns if 'Reimbursement' in c or 'Deductible' in c]
    df[reimb_cols] = df[reimb_cols].fillna(0)
    df.drop(columns=['DOB', 'DOD'], inplace=True, errors='ignore')
    return df


def clean_inpatient(df):
    """
    Cleans inpatient (hospital admission) claims.
    - Computes hospital_stay_days and claim_duration_days
    - Fills missing physician codes, diagnosis codes, monetary amounts
    """
    df = df.copy()
    for col in ['AdmissionDt', 'DischargeDt', 'ClaimStartDt', 'ClaimEndDt']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if {'AdmissionDt', 'DischargeDt'}.issubset(df.columns):
        df['hospital_stay_days'] = (df['DischargeDt'] - df['AdmissionDt']).dt.days.clip(0, 365).fillna(0)

    if {'ClaimStartDt', 'ClaimEndDt'}.issubset(df.columns):
        df['claim_duration_days'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days.clip(0, 365).fillna(0)

    for col in ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', 'UNKNOWN')

    diag_cols = [c for c in df.columns if 'ClmDiagnosisCode' in c or 'ClmProcedureCode' in c]
    for col in diag_cols:
        df[col] = df[col].astype(str).replace('nan', 'MISSING')

    money_cols = [c for c in df.columns if any(k in c for k in ['Amt', 'Reimbursement', 'Deductible'])]
    df[money_cols] = df[money_cols].fillna(0)
    df.drop(columns=['AdmissionDt', 'DischargeDt', 'ClaimStartDt', 'ClaimEndDt'],
            inplace=True, errors='ignore')
    return df


def clean_outpatient(df):
    """Cleans outpatient (non-admitted) claims."""
    df = df.copy()
    for col in ['ClaimStartDt', 'ClaimEndDt']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if {'ClaimStartDt', 'ClaimEndDt'}.issubset(df.columns):
        df['claim_duration_days'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days.clip(0, 365).fillna(0)

    for col in ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', 'UNKNOWN')

    diag_cols = [c for c in df.columns if 'ClmDiagnosisCode' in c]
    for col in diag_cols:
        df[col] = df[col].astype(str).replace('nan', 'MISSING')

    money_cols = [c for c in df.columns if any(k in c for k in ['Amt', 'Reimbursement', 'Deductible'])]
    df[money_cols] = df[money_cols].fillna(0)
    df.drop(columns=['ClaimStartDt', 'ClaimEndDt'], inplace=True, errors='ignore')
    return df


# ── Provider-level aggregation ────────────────────────────────────────────────

def aggregate_to_provider(inp_df, out_df, bene_df):
    """
    Merges inpatient + outpatient claims with beneficiary data,
    then aggregates to one row per Provider.
    Returns provider-level DataFrame with 30+ aggregate features.
    """
    bene_cols = ['BeneID', 'Age', 'is_deceased', 'total_chronic_conditions',
                 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
                 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt']
    bene_cols = [c for c in bene_cols if c in bene_df.columns]

    inp_merged = inp_df.merge(bene_df[bene_cols], on='BeneID', how='left')
    out_merged = out_df.merge(bene_df[bene_cols], on='BeneID', how='left')

    inp_agg = inp_merged.groupby('Provider').agg(
        ip_claim_count             = ('ClaimID', 'count'),
        ip_unique_patients         = ('BeneID', 'nunique'),
        ip_total_reimbursement     = ('InscClaimAmtReimbursed', 'sum'),
        ip_avg_reimbursement       = ('InscClaimAmtReimbursed', 'mean'),
        ip_max_reimbursement       = ('InscClaimAmtReimbursed', 'max'),
        ip_total_deductible        = ('DeductibleAmtPaid', 'sum'),
        ip_avg_deductible          = ('DeductibleAmtPaid', 'mean'),
        ip_avg_stay_days           = ('hospital_stay_days', 'mean'),
        ip_max_stay_days           = ('hospital_stay_days', 'max'),
        ip_total_stay_days         = ('hospital_stay_days', 'sum'),
        ip_avg_patient_age         = ('Age', 'mean'),
        ip_deceased_patient_count  = ('is_deceased', 'sum'),
        ip_avg_chronic_cond        = ('total_chronic_conditions', 'mean'),
        ip_max_chronic_cond        = ('total_chronic_conditions', 'max'),
        ip_unique_attending_phys   = ('AttendingPhysician', 'nunique'),
        ip_unique_operating_phys   = ('OperatingPhysician', 'nunique'),
    ).reset_index()

    out_agg = out_merged.groupby('Provider').agg(
        op_claim_count             = ('ClaimID', 'count'),
        op_unique_patients         = ('BeneID', 'nunique'),
        op_total_reimbursement     = ('InscClaimAmtReimbursed', 'sum'),
        op_avg_reimbursement       = ('InscClaimAmtReimbursed', 'mean'),
        op_max_reimbursement       = ('InscClaimAmtReimbursed', 'max'),
        op_total_deductible        = ('DeductibleAmtPaid', 'sum'),
        op_avg_patient_age         = ('Age', 'mean'),
        op_avg_chronic_cond        = ('total_chronic_conditions', 'mean'),
        op_unique_attending_phys   = ('AttendingPhysician', 'nunique'),
        op_avg_claim_duration      = ('claim_duration_days', 'mean'),
    ).reset_index()

    provider_df = inp_agg.merge(out_agg, on='Provider', how='outer').fillna(0)

    eps = 1e-9
    total_reimb   = provider_df['ip_total_reimbursement'] + provider_df['op_total_reimbursement']
    total_patients = provider_df['ip_unique_patients'] + provider_df['op_unique_patients'] + 1
    total_deductible = provider_df['ip_total_deductible'] + provider_df['op_total_deductible'] + 1

    provider_df['ip_op_claim_ratio']     = (provider_df['ip_claim_count'] / (provider_df['op_claim_count'] + 1)).round(4)
    provider_df['avg_reimb_per_patient'] = (total_reimb / total_patients).round(2)
    provider_df['total_unique_patients'] = provider_df['ip_unique_patients'] + provider_df['op_unique_patients']
    provider_df['deductible_reimb_ratio']= (total_reimb / total_deductible).round(4)

    return provider_df


# ── Feature engineering ───────────────────────────────────────────────────────

def add_ratio_features(df):
    """Adds 8 ratio features capturing fraudulent billing relationships."""
    df = df.copy()
    eps = 1e-9

    df['feat_reimb_per_ip_claim']       = (df['ip_total_reimbursement'] / (df['ip_claim_count'] + eps)).round(2)
    df['feat_reimb_per_unique_patient'] = (df['ip_total_reimbursement'] / (df['ip_unique_patients'] + eps)).round(2)
    df['feat_reimb_per_physician']      = (df['ip_total_reimbursement'] / (df['ip_unique_attending_phys'] + eps)).round(2)
    df['feat_ip_op_patient_ratio']      = (df['ip_unique_patients'] / (df['op_unique_patients'] + eps)).round(4)
    df['feat_deductible_ratio']         = (df['ip_total_deductible'] / (df['ip_total_reimbursement'] + eps)).round(4)
    df['feat_claims_per_patient']       = (df['ip_claim_count'] / (df['ip_unique_patients'] + eps)).round(4)
    df['feat_op_physician_reuse_rate']  = (df['ip_claim_count'] / (df['ip_unique_operating_phys'] + eps)).round(4)
    df['feat_stay_days_per_claim']      = (df['ip_total_stay_days'] / (df['ip_claim_count'] + eps)).round(4)
    return df


def add_risk_scores(df):
    """Adds 6 composite risk scores (0–1 percentile rank)."""
    df = df.copy()

    def pct(s): return s.rank(pct=True).fillna(0)

    df['risk_financial']          = (pct(df.get('ip_avg_reimbursement', pd.Series(dtype=float))) +
                                     pct(df.get('ip_total_reimbursement', pd.Series(dtype=float)))).div(2)
    df['risk_volume']             = (pct(df.get('ip_claim_count', pd.Series(dtype=float))) +
                                     pct(df.get('total_unique_patients', pd.Series(dtype=float)))).div(2)
    df['risk_medical_complexity'] = pct(df.get('ip_avg_chronic_cond', pd.Series(dtype=float)))
    df['risk_physician_pattern']  = pct(df.get('ip_unique_attending_phys', pd.Series(dtype=float)))
    df['risk_stay_duration']      = pct(df.get('ip_avg_stay_days', pd.Series(dtype=float)))

    risk_cols = ['risk_financial', 'risk_volume', 'risk_medical_complexity',
                 'risk_physician_pattern', 'risk_stay_duration']
    weights   = [2.0, 1.5, 1.0, 1.2, 1.3]
    df['risk_composite_score'] = sum(df[c] * w for c, w in zip(risk_cols, weights)) / sum(weights)
    return df


def apply_smote(X, y, random_state=42):
    """Applies SMOTE to balance class distribution. Only call on training data."""
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_sm, y_sm = smote.fit_resample(X, y)
    print(f"  Before SMOTE: {dict(y.value_counts())}")
    print(f"  After  SMOTE: {dict(pd.Series(y_sm).value_counts())}")
    return X_sm, y_sm