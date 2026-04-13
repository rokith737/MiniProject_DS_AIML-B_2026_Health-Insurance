# Outputs

This folder contains all generated outputs from the project notebooks and model training.

## Folder Structure

```
outputs/
├── graphs/          ← All plots and visualizations (PNG files)
│   ├── M1_preprocessing_overview.png    (from M1 notebook)
│   ├── M1_smote_balancing.png           (from M1 notebook)
│   ├── plot1_class_distribution.png     (from M2 EDA notebook)
│   ├── plot2_financial_signals.png      (from M2 EDA notebook)
│   ├── plot3_patient_volume.png         (from M2 EDA notebook)
│   ├── plot4_physician_stay.png         (from M2 EDA notebook)
│   ├── plot5_top_features_boxplot.png   (from M2 EDA notebook)
│   ├── plot6_correlation_heatmap.png    (from M2 EDA notebook)
│   ├── plot7_outlier_analysis.png       (from M2 EDA notebook)
│   ├── plot8_fraud_rate_by_quantile.png (from M2 EDA notebook)
│   ├── plot_M3_feature_importance_MI.png (from M3 notebook)
│   ├── plot_M4_roc_pr_curves.png        (from M4 training)
│   ├── plot_M4_confusion_matrices.png   (from M4 training)
│   ├── plot_M4_xgb_feature_importance.png (from M4 training)
│   ├── plot_M5_shap_summary.png         (from M5 XAI)
│   ├── plot_M5_shap_bar.png             (from M5 XAI)
│   └── plot_M5_shap_dependence.png      (from M5 XAI)
│
└── results/         ← Metrics, JSON results, text summaries
    ├── M1_results.md                    ← Real M1 output (this file)
    ├── model_results.md                 ← Fill in after M4 training
    └── model_comparison.json            ← Auto-saved by M4 notebook
```

## How to Generate All Outputs

Run the Colab notebooks in order:

| Step | Notebook | Outputs Generated |
|---|---|---|
| 1 | `notebooks/preprocessing.ipynb` | M1 plots, parquet files |
| 2 | `notebooks/data_understanding.ipynb` | EDA plots (plot1–plot8) |
| 3 | `notebooks/visualization.ipynb` | Additional visualizations |
| 4 | M4 (Colab) | ROC curve, confusion matrix, feature importance |
| 5 | M5 (Colab) | SHAP summary, waterfall, dependence plots |

## Plots Already Generated (from M1 run)

- `M1_preprocessing_overview.png` — 6-panel overview: class distribution, reimbursement gap, 
  patient counts, chronic conditions, claim count distribution, null check
- `M1_smote_balancing.png` — Before/after SMOTE class balance

> Download these from Google Drive:
> `/content/drive/MyDrive/insurance_fraud/processed/M1_preprocessing_overview.png`
> `/content/drive/MyDrive/insurance_fraud/processed/M1_smote_balancing.png`
