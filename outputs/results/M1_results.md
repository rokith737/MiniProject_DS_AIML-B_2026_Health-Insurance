# M1 — Data Merging & Preprocessing Results

## Dataset Loading Summary

| File | Rows | Columns | RAM |
|---|---|---|---|
| Train-1542865627584.csv | 5,410 | 2 | 0.5 MB |
| Train_Beneficiarydata-1542865627584.csv | 138,556 | 25 | 38.5 MB |
| Train_Inpatientdata-1542865627584.csv | 40,474 | 30 | — |
| Train_Outpatientdata-1542865627584.csv | 517,737 | 27 | — |
| Test-1542969243754.csv | 1,353 | 1 | 0.1 MB |
| Test_Beneficiarydata-1542969243754.csv | 63,968 | 25 | 17.9 MB |
| **Total claim rows** | **~760,000+** | | **~123 MB** |

## Missing Value Report (Before Cleaning)

### train_bene
| Column | Missing Count | Missing % |
|---|---|---|
| DOD (Date of Death) | 137,135 | 98.97% |

> Note: 98.97% missing DOD means 98.97% of patients are alive — this is expected, not a data error.

### train_inp (Inpatient)
| Column | Missing % | Action |
|---|---|---|
| ClmProcedureCode_6 | 100.00% | Fill with 'MISSING' |
| ClmProcedureCode_5 | 99.98% | Fill with 'MISSING' |
| ClmProcedureCode_4 | 99.71% | Fill with 'MISSING' |
| ClmDiagnosisCode_10 | 90.30% | Fill with 'MISSING' |
| OtherPhysician | 88.41% | Fill with 'UNKNOWN' |
| ClmProcedureCode_2 | 86.52% | Fill with 'MISSING' |
| OperatingPhysician | ~12% | Fill with 'UNKNOWN' |

## Cleaning Results

### Beneficiary Data
```
Train bene shape: (138,556, 26)
Test  bene shape: (63,968, 26)
New columns: Age, is_deceased, total_chronic_conditions

Age statistics:
  mean = 73.59 years  |  std = 12.73
  min  = 26 years     |  max = 100.9 years
  50th percentile = 74.3 years

total_chronic_conditions:
  mean = 3.74  |  std = 2.35
  range = 0 to 11
```

### Inpatient Data
```
Train inpatient shape: (40,474, 28)
New columns: hospital_stay_days, claim_duration_days

hospital_stay_days:
  mean = 5.67 days  |  std = 5.64
  median = 4 days   |  max = 35 days
```

## Provider-Level Aggregation

```
Train provider dataset: 5,410 providers × 33 features
Test  provider dataset: 1,353 providers × 33 features
```

### Merge steps completed:
1. Inpatient claims + Beneficiary → by BeneID
2. Outpatient claims + Beneficiary → by BeneID
3. Inpatient aggregates (per Provider)
4. Outpatient aggregates (per Provider)
5. Merge both → one row per Provider
6. Derived ratio features added

## Fraud Label Distribution

```
Fraud labels attached to 5,410 providers:
  Non-fraud (0) : 4,904 providers  (90.6%)
  Fraud     (1) :   506 providers  ( 9.4%)
```

> Severe class imbalance confirmed — 9.4% fraud rate → SMOTE required

## SMOTE Balancing

```
Before SMOTE:  {0: 4904, 1: 506}    →  9.4% fraud
After  SMOTE:  {0: 4904, 1: 4904}   →  50.0% fraud
Original size : 5,410 providers
SMOTE size    : 9,808 samples
```

## Final Feature List (32 features from M1)

```
01. ip_claim_count              09. ip_avg_stay_days
02. ip_unique_patients          10. ip_max_stay_days
03. ip_total_reimbursement      11. ip_total_stay_days
04. ip_avg_reimbursement        12. ip_avg_patient_age
05. ip_max_reimbursement        13. ip_deceased_patient_count
06. ip_total_deductible         14. ip_avg_chronic_cond
07. ip_avg_deductible           15. ip_max_chronic_cond
08. ip_unique_attending_phys    16. ip_unique_operating_phys
... + outpatient equivalents + ratio features
```

## Output Files Saved to Google Drive

| File | Description |
|---|---|
| train_provider_merged.parquet | 5,410 providers × 34 cols (with label) |
| test_provider_merged.parquet | 1,353 providers × 34 cols |
| train_smote_balanced.parquet | 9,808 rows after SMOTE |
| test_final.parquet | Test set for inference |
| feature_columns.json | List of 32 feature names |

## Remaining Null Values After Cleaning: **0**
