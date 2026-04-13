"""
analysis.py
===========
SHAP explainability and investigator report generation
for Insurance Fraud Detection System.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from datetime import datetime


# ── SHAP helpers ──────────────────────────────────────────────────────────────

def compute_shap_values(model, X_val, feature_cols):
    """
    Computes exact SHAP values using TreeExplainer.
    Returns shap.Explanation object.
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X_val)
    return explainer, shap_vals


def get_top_fraud_reasons(shap_vals, feat_df, feature_cols, pop_means, top_n=5):
    """
    Returns top N plain-English fraud reasons for a single provider.
    Uses positive SHAP values (features pushing toward fraud).
    """
    shap_series = pd.Series(shap_vals.values[0], index=feature_cols)
    top         = shap_series[shap_series > 0].sort_values(ascending=False).head(top_n)

    TEMPLATES = {
        'ip_avg_reimbursement'          : 'Avg reimbursement is {ratio:.1f}x the population average',
        'ip_total_reimbursement'        : 'Total billing is {ratio:.1f}x higher than a typical provider',
        'ip_avg_stay_days'              : 'Hospital stay of {val:.1f} days is {ratio:.1f}x the average',
        'ip_claim_count'                : 'Filed {val:.0f} claims — {ratio:.1f}x more than average',
        'total_unique_patients'         : 'Served {val:.0f} unique patients — {ratio:.1f}x typical',
        'ip_avg_chronic_cond'           : '{val:.2f} avg chronic conditions — possible diagnosis stuffing',
        'ip_unique_attending_phys'      : 'Used {val:.0f} unique physicians — {ratio:.1f}x more than average',
        'risk_composite_score'          : 'Composite fraud risk score in the high-risk tier',
        'risk_financial'                : 'Financial anomaly score is unusually high',
        'feat_reimb_per_unique_patient' : 'Extracts ${val:,.0f} per unique patient — very high',
        'feat_claims_per_patient'       : '{val:.2f} claims per patient — possible duplicate billing',
        'interact_volume_x_reimb'       : 'High volume × high reimbursement — billing mill pattern',
        'interact_stay_x_reimb'         : 'Long stay × high claim — overbilling signal',
        'feat_total_flags_triggered'    : 'Triggered {val:.0f}/8 extreme outlier anomaly flags',
    }

    reasons = []
    for feat, sv_val in top.items():
        val   = float(feat_df[feat].iloc[0])
        avg   = pop_means.get(feat, 1e-9)
        ratio = val / max(avg, 1e-9)
        tmpl  = TEMPLATES.get(feat,
                    '{feat} = {val:.3f} ({ratio:.1f}x avg)'.replace('{feat}', feat))
        try:
            text = tmpl.format(val=val, avg=avg, ratio=ratio)
        except Exception:
            text = feat.replace('_', ' ').title()

        reasons.append({
            'feature': feat,
            'shap'   : round(float(sv_val), 4),
            'text'   : text,
        })
    return reasons


def plot_shap_summary(shap_vals, X_val, feature_cols, save_path=None):
    """Saves SHAP summary beeswarm plot."""
    fig, ax = plt.subplots(figsize=(12, 9))
    shap.summary_plot(shap_vals.values, X_val, feature_names=feature_cols,
                      max_display=20, show=False)
    plt.title('SHAP Summary — Feature Importance', fontsize=13)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved: {save_path}")
    plt.close()


def plot_shap_waterfall(shap_vals, provider_idx, save_path=None):
    """Saves SHAP waterfall plot for a single provider."""
    fig, ax = plt.subplots(figsize=(12, 7))
    shap.waterfall_plot(shap_vals[provider_idx], max_display=15, show=False)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved: {save_path}")
    plt.close()


# ── Risk tier ─────────────────────────────────────────────────────────────────

def get_risk_tier(score, threshold):
    if   score >= 0.85:            return 'CRITICAL', 'Immediate investigation — block claim'
    elif score >= threshold:       return 'HIGH',     'Flag for detailed manual review'
    elif score >= threshold * 0.7: return 'MEDIUM',   'Process with additional verification'
    else:                          return 'LOW',       'Approve — within normal parameters'


# ── Investigator PDF report ───────────────────────────────────────────────────

def generate_investigator_report(results_list, output_path, metadata):
    """
    Generates a multi-provider fraud investigation PDF report.
    results_list: list of dicts with keys fraud_score, risk_tier, reasons, provider_idx
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        print("reportlab not installed — skipping PDF generation.")
        return

    doc    = SimpleDocTemplate(output_path, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    ts = ParagraphStyle('T', parent=styles['Title'], fontSize=18,
                        textColor=colors.HexColor('#1DB8C4'), alignment=TA_CENTER)
    ss = ParagraphStyle('S', parent=styles['Normal'], fontSize=10,
                        textColor=colors.HexColor('#888888'), alignment=TA_CENTER)
    hs = ParagraphStyle('H', parent=styles['Heading2'], fontSize=13,
                        textColor=colors.HexColor('#1A3A6B'))
    bs = ParagraphStyle('B', parent=styles['Normal'], fontSize=10)

    story.append(Paragraph('FRAUD INVESTIGATION REPORT', ts))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')} | "
        f"Model AUC: {metadata.get('val_auc_roc', 'N/A')}", ss))
    story.append(HRFlowable(width='100%', thickness=1.5,
                             color=colors.HexColor('#1DB8C4')))
    story.append(Spacer(1, 0.5*cm))

    for rank, res in enumerate(results_list, 1):
        story.append(Paragraph(
            f"Provider #{rank} — Score: {res['fraud_score']:.4f} | "
            f"Tier: {res['risk_tier']}", hs))
        story.append(Paragraph(f"Action: {res['action']}", bs))
        story.append(Spacer(1, 0.2*cm))
        for i, r in enumerate(res.get('reasons', [])[:5], 1):
            story.append(Paragraph(
                f"{i}. {r['text']}  [SHAP: +{r['shap']:.4f}]", bs))
        story.append(Spacer(1, 0.5*cm))

    doc.build(story)
    print(f"  ✅ Report saved: {output_path}")