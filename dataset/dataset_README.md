# Dataset — CMS Medicare Healthcare Provider Fraud Detection

## Source
- **Name:** Healthcare Provider Fraud Detection Analysis
- **Platform:** Kaggle
- **URL:** https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
- **Contributor:** Rohit Anand Gupta (Kaggle username: rohitrox)
- **License:** Open — publicly available for research use

---

## How to Download

1. Create a free Kaggle account at https://www.kaggle.com
2. Go to the dataset link above
3. Click **Download (All)** — downloads as a zip file (~123 MB)
4. Extract the zip — you will get exactly 8 CSV files
5. Place all 8 CSV files inside `dataset/raw_data/`

---

## File Descriptions

### Train Files (used for model training)

| File | Rows | Columns | Size | Description |
|---|---|---|---|---|
| `Train-1542865627584.csv` | 5,410 | 2 | 0.5 MB | Provider fraud labels — `Provider`, `PotentialFraud` (Yes/No) |
| `Train_Beneficiarydata-1542865627584.csv` | 138,556 | 25 | 38.5 MB | Patient demographics — DOB, Gender, Race, State, Chronic conditions |
| `Train_Inpatientdata-1542865627584.csv` | 40,474 | 30 | ~15 MB | Hospital admission claims — Diagnosis codes, Physician IDs, Amounts |
| `Train_Outpatientdata-1542865627584.csv` | 517,737 | 27 | ~60 MB | OPD visit claims — ClaimStartDt, Reimbursement, Physician codes |

### Test Files (used for inference only — no fraud labels)

| File | Rows | Columns | Description |
|---|---|---|---|
| `Test-1542969243754.csv` | 1,353 | 1 | Provider IDs only — no labels |
| `Test_Beneficiarydata-1542969243754.csv` | 63,968 | 25 | Test patient demographics |
| `Test_Inpatientdata-1542969243754.csv` | — | 30 | Test inpatient claims |
| `Test_Outpatientdata-1542969243754.csv` | — | 27 | Test outpatient claims |

---

## Key Columns Explained

### Train.csv
| Column | Type | Description |
|---|---|---|
| `Provider` | String | Unique provider ID (e.g., PRV55000) — primary key |
| `PotentialFraud` | String | `Yes` = fraudulent, `No` = legitimate |

### Beneficiary Data
| Column | Description |
|---|---|
| `BeneID` | Unique patient ID — foreign key linking to claim files |
| `DOB` | Date of birth (used to compute Age) |
| `DOD` | Date of death — 98.97% missing (most patients alive) |
| `Gender` | 1=Male, 2=Female |
| `ChronicCond_*` | 11 columns: 1=has condition, 2=no condition |
| `IPAnnualReimbursementAmt` | Annual inpatient reimbursement total |
| `OPAnnualReimbursementAmt` | Annual outpatient reimbursement total |
| `State`, `County` | Geographic location |

### Inpatient Data
| Column | Description |
|---|---|
| `ClaimID` | Unique claim identifier |
| `BeneID` | Links to beneficiary |
| `Provider` | Links to fraud label |
| `AdmissionDt` | Hospital admission date |
| `DischargeDt` | Hospital discharge date |
| `AttendingPhysician` | Primary physician ID |
| `OperatingPhysician` | Surgeon ID |
| `InscClaimAmtReimbursed` | Claim reimbursement amount |
| `DeductibleAmtPaid` | Deductible amount |
| `ClmDiagnosisCode_1..10` | ICD-10 diagnosis codes |
| `ClmProcedureCode_1..6` | CPT procedure codes |

### Outpatient Data
Same structure as Inpatient but without `AdmissionDt`/`DischargeDt`.

---

## Dataset Statistics (Verified from M1 notebook run)

```
Total claim rows (train):  700,767
  └── Inpatient  :  40,474 rows
  └── Outpatient : 517,737 rows
  └── Beneficiary: 138,556 unique patients

Provider-level (after aggregation):
  └── Train providers: 5,410
  └── Test  providers: 1,353

Fraud distribution (train):
  └── Non-fraud : 4,904 providers  (90.6%)
  └── Fraud     :   506 providers  ( 9.4%)

Missing values after cleaning: 0
```

---

## Important Notes

1. **Raw CSVs are NOT committed to this repository** — files are too large for GitHub (>100 MB total).
   Download them from Kaggle and place in `dataset/raw_data/`.

2. **Processed parquet files** are saved to `dataset/processed_data/` after running
   `notebooks/preprocessing.ipynb` in Google Colab.

3. **Test labels are not available** — this dataset is structured like a Kaggle competition.
   The test set has only Provider IDs, no `PotentialFraud` column.
   For model evaluation, an 80/20 split of the training data is used.

4. **Class imbalance** — Only 9.4% of providers are fraudulent. SMOTE is applied in
   `notebooks/preprocessing.ipynb` to balance this to 50:50 for training.

---

## Citation

```
Gupta, R. A. (2019). Healthcare Provider Fraud Detection Analysis.
Kaggle Dataset. https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
```

Based on publicly available CMS (Centers for Medicare & Medicaid Services) Medicare data:
- Part B: Physician and Other Supplier Utilization and Payment Data
- LEIE: List of Excluded Individuals and Entities (fraud labels)
