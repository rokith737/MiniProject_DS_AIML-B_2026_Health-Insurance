"""
Insurance Fraud Detection System — Flask Backend
=================================================
Folder layout expected:
  flask_app/
    app.py                        ← this file
    models/
      model_xgboost.pkl
      model_lightgbm.pkl
      model_random_forest.pkl
      final_feature_columns.json
      model_metadata.json
      ensemble_config.json
      X_val.parquet
    templates/
      index.html
    uploads/                      (auto-created)

Poppler (needed only for PDF upload):
  Extract to  C:\\poppler
  The folder  C:\\poppler\\Library\\bin  must contain pdftoppm.exe
  If your extraction gave a different path, update POPPLER_PATH below.

Run:   python app.py
Open:  http://localhost:5000
"""

import os
import sys
import json
import re
import warnings
import traceback

import numpy as np
import pandas as pd
import joblib
import shap

from flask import Flask, render_template, request, jsonify
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")

# ── Poppler path (Windows) ────────────────────────────────────────────────────
POPPLER_PATH = r"C:\poppler\Library\bin"
# Change this if pdftoppm.exe is somewhere else, e.g.:
#   r"C:\poppler\bin"
#   r"C:\Program Files\poppler\Library\bin"

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

# ── Load models ───────────────────────────────────────────────────────────────
print("=" * 55)
print("  Loading models…")

xgb_model  = joblib.load(os.path.join(MODEL_DIR, "model_xgboost.pkl"));       print("  ✅ XGBoost")
lgbm_model = joblib.load(os.path.join(MODEL_DIR, "model_lightgbm.pkl"));      print("  ✅ LightGBM")
rf_model   = joblib.load(os.path.join(MODEL_DIR, "model_random_forest.pkl")); print("  ✅ Random Forest")

with open(os.path.join(MODEL_DIR, "final_feature_columns.json")) as fh:
    FEATURE_COLS = json.load(fh)
print(f"  ✅ Features: {len(FEATURE_COLS)}")

with open(os.path.join(MODEL_DIR, "model_metadata.json")) as fh:
    METADATA = json.load(fh)
print(f"  ✅ Metadata  AUC={METADATA['val_auc_roc']}  F1={METADATA['val_f1']}")

with open(os.path.join(MODEL_DIR, "ensemble_config.json")) as fh:
    ENS_CFG = json.load(fh)

X_val_raw = pd.read_parquet(os.path.join(MODEL_DIR, "X_val.parquet"))
X_val_raw = X_val_raw[[c for c in FEATURE_COLS if c in X_val_raw.columns]]
POP_MEANS = X_val_raw.mean().to_dict()
POP_STDS  = X_val_raw.std().to_dict()
print("  ✅ Population stats loaded")

explainer = shap.TreeExplainer(xgb_model)
print("  ✅ SHAP explainer ready")

THRESHOLD = METADATA["best_threshold"]
WEIGHTS   = ENS_CFG["weights"]          # [3, 3, 1]
ocr_reader = None                        # lazy-loaded on first /scan call

print(f"\n  🚀 Ready  threshold={THRESHOLD}  AUC={METADATA['val_auc_roc']}")
print("=" * 55)


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def build_feature_df(input_dict):
    """Convert 6 user inputs → full 57-feature vector expected by the model."""
    row = {col: float(POP_MEANS.get(col, 0.0)) for col in FEATURE_COLS}
    eps = 1e-9

    def iv(key, default=None):
        v = input_dict.get(key, default if default is not None else POP_MEANS.get(key, 0))
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(POP_MEANS.get(key, 0))

    reimb   = iv("ip_avg_reimbursement");     reimb   = max(reimb,   0)
    stays   = iv("ip_avg_stay_days");         stays   = max(stays,   0)
    claims  = max(iv("ip_claim_count"), 1)
    pats    = max(iv("total_unique_patients"), 1)
    chronic = iv("ip_avg_chronic_cond");      chronic = max(chronic,  0)
    phys    = max(iv("ip_unique_attending_phys"), 1)

    # ── Direct ──────────────────────────────────────────────────────────────
    _set(row, "ip_avg_reimbursement",     reimb)
    _set(row, "ip_avg_stay_days",         stays)
    _set(row, "ip_claim_count",           claims)
    _set(row, "total_unique_patients",    pats)
    _set(row, "ip_avg_chronic_cond",      chronic)
    _set(row, "ip_unique_attending_phys", phys)

    # ── Derived base ─────────────────────────────────────────────────────────
    _set(row, "ip_total_reimbursement",    reimb * claims)
    _set(row, "ip_total_stay_days",        stays * claims)
    _set(row, "ip_max_stay_days",          stays * 1.5)
    _set(row, "ip_avg_patient_age",        iv("ip_avg_patient_age", 68))
    _set(row, "op_claim_count",            claims * 0.6)
    _set(row, "op_unique_patients",        pats   * 0.7)
    _set(row, "op_avg_reimbursement",      reimb  * 0.3)
    _set(row, "ip_unique_operating_phys",  max(1, phys * 0.6))
    _set(row, "ip_max_chronic_cond",       min(11, chronic + 1))
    _set(row, "ip_total_deductible",       reimb * 0.08)
    _set(row, "ip_avg_deductible",         reimb * 0.08)
    _set(row, "op_avg_patient_age",        iv("ip_avg_patient_age", 68))
    _set(row, "ip_avg_annual_ip_reimb",    reimb * 12)
    _set(row, "ip_avg_annual_op_reimb",    reimb * 0.3 * 12)
    _set(row, "op_avg_chronic_cond",       chronic)
    _set(row, "op_total_deductible",       reimb * 0.06)
    _set(row, "op_avg_claim_duration",     stays * 0.4)
    _set(row, "op_unique_attending_phys",  max(1, phys * 0.7))
    _set(row, "op_total_reimbursement",    reimb * 0.3 * claims)
    _set(row, "op_max_reimbursement",      reimb * 0.5)
    _set(row, "ip_deceased_patient_count", max(0, pats * 0.02))
    _set(row, "op_avg_patient_age",        68)

    # ── Ratio features ───────────────────────────────────────────────────────
    _set(row, "feat_reimb_per_ip_claim",       reimb)
    _set(row, "feat_reimb_per_unique_patient", (reimb * claims) / pats)
    _set(row, "feat_reimb_per_physician",      (reimb * claims) / phys)
    _set(row, "feat_claims_per_patient",       claims / pats)
    _set(row, "feat_op_physician_reuse_rate",  claims / phys)
    _set(row, "feat_stay_days_per_claim",      stays)
    _set(row, "feat_deductible_ratio",         0.08)
    _set(row, "feat_ip_op_patient_ratio",      1.4)
    _set(row, "avg_reimb_per_patient",         (reimb * claims) / pats)
    _set(row, "deductible_reimb_ratio",        0.08)

    flags = sum([
        reimb   > POP_MEANS.get("ip_avg_reimbursement",    0) * 2,
        stays   > POP_MEANS.get("ip_avg_stay_days",        0) * 2,
        claims  > POP_MEANS.get("ip_claim_count",          0) * 2,
        pats    > POP_MEANS.get("total_unique_patients",   0) * 2,
        chronic > 5,
        phys    > POP_MEANS.get("ip_unique_attending_phys",0) * 2,
    ])
    _set(row, "feat_total_flags_triggered", flags)

    # ── Risk scores (0–1) ────────────────────────────────────────────────────
    def pct(val, key, mult=3):
        return min(val / max(POP_MEANS.get(key, 1) * mult, eps), 1.0)

    rp = pct(reimb,   "ip_avg_reimbursement")
    sp = pct(stays,   "ip_avg_stay_days")
    cp = min(chronic / 11, 1.0)
    vp = pct(claims,  "ip_claim_count")
    pp = pct(phys,    "ip_unique_attending_phys")
    composite = (2*rp + 1.5*vp + cp + pp + 1.3*sp) / 6.8

    _set(row, "risk_financial",          rp)
    _set(row, "risk_stay_duration",      sp)
    _set(row, "risk_medical_complexity", cp)
    _set(row, "risk_volume",             vp)
    _set(row, "risk_physician_pattern",  pp)
    _set(row, "risk_composite_score",    composite)

    # ── Interaction features ─────────────────────────────────────────────────
    _set(row, "interact_volume_x_reimb",              np.log1p(claims) * np.log1p(reimb))
    _set(row, "interact_patients_x_chronic",          np.log1p(pats)   * chronic)
    _set(row, "interact_stay_x_reimb",                stays * np.log1p(reimb))
    _set(row, "interact_physicians_x_patients",       np.log1p(phys) * np.log1p(pats))
    _set(row, "interact_financial_x_physician",       rp * pp)
    _set(row, "interact_volume_x_stay",               vp * sp)
    _set(row, "interact_composite_squared",           composite ** 2)
    _set(row, "interact_low_deductible_x_high_reimb", min(12 * np.log1p(reimb * claims), 1e6))

    # ── Z-score features ─────────────────────────────────────────────────────
    for col in FEATURE_COLS:
        if col.startswith("zscore_"):
            base = col[len("zscore_"):]
            if base in row:
                mean = POP_MEANS.get(base, 0)
                std  = max(POP_STDS.get(base, 1), eps)
                row[col] = (row[base] - mean) / std

    # ── Bin features ─────────────────────────────────────────────────────────
    for col in FEATURE_COLS:
        if col.startswith("bin_"):
            base = col[len("bin_"):]
            if base in row:
                avg = POP_MEANS.get(base, 0)
                row[col] = min(int((row[base] / max(avg * 0.5, eps)) * 1.5), 4)

    # ── Anomaly flag features ────────────────────────────────────────────────
    for col in FEATURE_COLS:
        if col.startswith("flag_") and col != "feat_total_flags_triggered":
            if "very_low" in col:
                row[col] = 1 if row.get("feat_deductible_ratio", 1) < 0.05 else 0
            else:
                base     = col.replace("flag_extreme_", "").replace("flag_", "")
                base_val = row.get(base, 0)
                thresh   = POP_MEANS.get(base, 0) * 2
                row[col] = 1 if base_val > thresh else 0

    df = pd.DataFrame([row])[FEATURE_COLS]
    return df.fillna(0).replace([np.inf, -np.inf], 0)


def _set(row, key, val):
    """Set row[key] only if key exists in FEATURE_COLS."""
    if key in row:
        row[key] = val


# ═════════════════════════════════════════════════════════════════════════════
#  PREDICTION & EXPLANATION
# ═════════════════════════════════════════════════════════════════════════════

def ensemble_predict(feat_df):
    p_xgb  = float(xgb_model.predict_proba(feat_df)[:, 1][0])
    p_lgbm = float(lgbm_model.predict_proba(feat_df)[:, 1][0])
    p_rf   = float(rf_model.predict_proba(feat_df)[:, 1][0])
    w      = WEIGHTS
    p_ens  = (w[0]*p_xgb + w[1]*p_lgbm + w[2]*p_rf) / sum(w)
    return round(p_ens, 4), round(p_xgb, 4), round(p_lgbm, 4), round(p_rf, 4)


def get_shap_reasons(feat_df, input_vals):
    sv     = explainer(feat_df)
    shap_s = pd.Series(sv.values[0], index=FEATURE_COLS)
    top5   = shap_s[shap_s > 0].sort_values(ascending=False).head(5)

    def iv(k):
        try: return float(input_vals.get(k, POP_MEANS.get(k, 0)))
        except: return 0.0

    def rx(k, v):
        return v / max(POP_MEANS.get(k, 1), 1e-9)

    reimb   = iv("ip_avg_reimbursement")
    stays   = iv("ip_avg_stay_days")
    claims  = max(iv("ip_claim_count"), 1)
    pats    = max(iv("total_unique_patients"), 1)
    chronic = iv("ip_avg_chronic_cond")
    phys    = iv("ip_unique_attending_phys")

    TMPLS = {
        "ip_avg_reimbursement"          : f"Avg reimbursement ${reimb:,.0f} is {rx('ip_avg_reimbursement',reimb):.1f}x the population average",
        "ip_total_reimbursement"        : f"Total billing ${reimb*claims:,.0f} is {rx('ip_avg_reimbursement',reimb):.1f}x higher than a typical provider",
        "ip_avg_stay_days"              : f"Hospital stay of {stays:.1f} days is {rx('ip_avg_stay_days',stays):.1f}x the average ({POP_MEANS.get('ip_avg_stay_days',0):.1f} days)",
        "ip_claim_count"                : f"Filed {claims:.0f} inpatient claims — {rx('ip_claim_count',claims):.1f}x more than the average provider",
        "total_unique_patients"         : f"Served {pats:.0f} unique patients — {rx('total_unique_patients',pats):.1f}x a typical provider",
        "ip_avg_chronic_cond"           : f"Avg {chronic:.2f} chronic conditions per patient — possible diagnosis stuffing",
        "ip_unique_attending_phys"      : f"Used {phys:.0f} unique physicians — {rx('ip_unique_attending_phys',phys):.1f}x more than average",
        "risk_composite_score"          : "Overall composite fraud risk score is in the high-risk tier",
        "risk_financial"                : "Financial anomaly score is unusually high across billing metrics",
        "risk_volume"                   : "Claim volume risk score is in the top tier",
        "risk_stay_duration"            : "Hospital stay duration is statistically anomalous",
        "risk_physician_pattern"        : "Physician usage pattern is statistically anomalous",
        "feat_reimb_per_unique_patient" : f"Extracts ${reimb*claims/pats:,.0f} per patient — very high per-patient billing",
        "feat_claims_per_patient"       : f"Files {claims/pats:.2f} claims per patient — possible duplicate billing",
        "feat_op_physician_reuse_rate"  : f"Same physicians reused across {claims/phys:.1f} claims on average",
        "feat_total_flags_triggered"    : "Provider triggered multiple extreme-outlier anomaly flags simultaneously",
        "interact_volume_x_reimb"       : "High claim volume combined with high reimbursement — billing mill pattern",
        "interact_stay_x_reimb"         : "Long hospital stay combined with high claim amount — overbilling signal",
        "interact_patients_x_chronic"   : "Large patient base with inflated chronic condition scores — diagnosis stuffing at scale",
        "interact_composite_squared"    : "Composite risk score is extremely elevated — multiple fraud signals amplifying each other",
    }

    reasons = []
    for feat, sv_val in top5.items():
        reasons.append({
            "feature": feat,
            "shap"   : round(float(sv_val), 4),
            "text"   : TMPLS.get(feat, feat.replace("_", " ").title()),
        })
    return reasons


def get_risk_tier(score):
    if   score >= 0.85:              return "CRITICAL", "#E84C4C", "Immediate investigation required — block claim"
    elif score >= THRESHOLD:         return "HIGH",     "#FF8C42", "Flag for detailed manual review before approval"
    elif score >= THRESHOLD * 0.7:   return "MEDIUM",   "#F5C842", "Process with additional verification steps"
    else:                            return "LOW",       "#22C97A", "Approve — claim is within normal parameters"


# ═════════════════════════════════════════════════════════════════════════════
#  OCR FIELD PARSER
# ═════════════════════════════════════════════════════════════════════════════

OCR_FIELD_PATTERNS = {
    "ip_avg_reimbursement" : [
        r"(?:claim\s*amount|total\s*claim|amount\s*claimed|reimburs)[^\d]*\$?\s*([\d,]+)",
        r"\$\s*([\d,]{4,})",
    ],
    "ip_avg_stay_days"     : [
        r"([\d]+)\s*days?\s*(?:stay|admit|hospital|duration)",
        r"(?:stay|duration|length)[^\d]*([\d]+)",
    ],
    "ip_avg_chronic_cond"  : [
        r"([\d]+)\s*(?:diagnos|condition|chronic)",
    ],
    "ip_avg_patient_age"   : [
        r"(?:age|aged?)[^\d]*([\d]{2,3})",
    ],
    "ip_claim_count"       : [
        r"(?:claim\s*count|number\s*of\s*claims?|claims?\s*filed)[^\d]*([\d]+)",
    ],
    "total_unique_patients": [
        r"(?:patient|beneficiary)\s*count[^\d]*([\d]+)",
        r"([\d]+)\s*(?:unique\s*)?patients?",
    ],
}


def parse_ocr_text(raw_text):
    fields = {}
    for field, patterns in OCR_FIELD_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, raw_text, re.IGNORECASE)
            if m:
                try:
                    fields[field] = float(re.sub(r"[,\s]", "", m.group(1)))
                    break
                except ValueError:
                    pass
    return fields


# ═════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template(
        "index.html",
        model_auc=METADATA["val_auc_roc"],
        threshold=THRESHOLD,
        model_f1=METADATA["val_f1"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data    = request.get_json(force=True, silent=True) or {}
        feat_df = build_feature_df(data)
        score, xgb_s, lgbm_s, rf_s = ensemble_predict(feat_df)
        tier, color, action = get_risk_tier(score)
        reasons = get_shap_reasons(feat_df, data)

        return jsonify({
            "success"        : True,
            "fraud_score"    : score,
            "fraud_predicted": score >= THRESHOLD,
            "risk_tier"      : tier,
            "tier_color"     : color,
            "action"         : action,
            "xgb_score"      : xgb_s,
            "lgbm_score"     : lgbm_s,
            "rf_score"       : rf_s,
            "threshold"      : THRESHOLD,
            "reasons"        : reasons,
            "predicted_at"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    except Exception as exc:
        app.logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/scan", methods=["POST"])
def scan_document():
    """
    OCR scan endpoint.
    Always returns JSON — never an HTML error page.
    """
    global ocr_reader

    # ── Guard: no file ───────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded. Please attach a file."}), 400

    f   = request.files["file"]
    ext = Path(f.filename).suffix.lower() if f.filename else ""

    if ext not in {".pdf", ".jpg", ".jpeg", ".png", ".bmp"}:
        return jsonify({
            "success": False,
            "error"  : f'Unsupported format "{ext}". Upload a PDF, JPG, or PNG.'
        }), 400

    # ── Save temp file ───────────────────────────────────────────
    safe_name = secure_filename(f.filename) or "upload" + ext
    tmp_path  = os.path.join(UPLOAD_FOLDER, safe_name)
    try:
        f.save(tmp_path)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Could not save file: {exc}"}), 500

    try:
        # ── Load EasyOCR on first call ───────────────────────────
        if ocr_reader is None:
            try:
                import easyocr
                print("Loading EasyOCR (first call — downloading model if needed)…")
                ocr_reader = easyocr.Reader(["en"], gpu=False)
                print("✅ EasyOCR ready.")
            except ImportError:
                _cleanup(tmp_path)
                return jsonify({
                    "success": False,
                    "error"  : "EasyOCR is not installed. Run:  pip install easyocr"
                }), 500

        # ── Convert to PIL image ─────────────────────────────────
        from PIL import Image

        if ext == ".pdf":
            try:
                from pdf2image import convert_from_path
                poppler_ok = os.path.isdir(POPPLER_PATH)
                pages = (
                    convert_from_path(tmp_path, dpi=200, poppler_path=POPPLER_PATH)
                    if poppler_ok
                    else convert_from_path(tmp_path, dpi=200)
                )
                pil_img = pages[0]
            except Exception as pdf_exc:
                _cleanup(tmp_path)
                return jsonify({
                    "success": False,
                    "error"  : (
                        f"PDF conversion failed: {pdf_exc}. "
                        "Make sure Poppler is installed and POPPLER_PATH in app.py is correct. "
                        "Or upload a JPG/PNG screenshot of the claim instead — that always works."
                    )
                }), 500
        else:
            try:
                pil_img = Image.open(tmp_path).convert("RGB")
            except Exception as img_exc:
                _cleanup(tmp_path)
                return jsonify({"success": False, "error": f"Cannot open image: {img_exc}"}), 500

        # ── Run OCR ──────────────────────────────────────────────
        try:
            ocr_results = ocr_reader.readtext(np.array(pil_img))
            raw_text    = " ".join([r[1] for r in ocr_results])
        except Exception as ocr_exc:
            _cleanup(tmp_path)
            return jsonify({"success": False, "error": f"OCR failed: {ocr_exc}"}), 500

        # ── Parse fields ─────────────────────────────────────────
        fields  = parse_ocr_text(raw_text)

        # ── Predict ──────────────────────────────────────────────
        feat_df = build_feature_df(fields)
        score, xgb_s, lgbm_s, rf_s = ensemble_predict(feat_df)
        tier, color, action = get_risk_tier(score)
        reasons = get_shap_reasons(feat_df, fields)

        _cleanup(tmp_path)

        return jsonify({
            "success"         : True,
            "fraud_score"     : score,
            "fraud_predicted" : score >= THRESHOLD,
            "risk_tier"       : tier,
            "tier_color"      : color,
            "action"          : action,
            "xgb_score"       : xgb_s,
            "lgbm_score"      : lgbm_s,
            "rf_score"        : rf_s,
            "extracted_fields": {k: str(round(v, 2)) for k, v in fields.items()},
            "ocr_regions"     : len(ocr_results),
            "reasons"         : reasons,
            "predicted_at"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    except Exception as exc:
        # ── Catch-all — ALWAYS return JSON, never HTML ───────────
        _cleanup(tmp_path)
        app.logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error"  : f"Unexpected error: {exc}",
            "detail" : traceback.format_exc(),
        }), 500


@app.route("/health")
def health():
    return jsonify({
        "status"       : "running",
        "model_auc"    : METADATA["val_auc_roc"],
        "model_f1"     : METADATA["val_f1"],
        "threshold"    : THRESHOLD,
        "features"     : len(FEATURE_COLS),
        "poppler_found": os.path.isdir(POPPLER_PATH),
        "poppler_path" : POPPLER_PATH,
        "timestamp"    : datetime.now().isoformat(),
    })


# ── Utility ───────────────────────────────────────────────────────────────────
def _cleanup(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ── Error handlers — always return JSON, never HTML ──────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"success": False, "error": "Method not allowed"}), 405

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"success": False, "error": "File too large (max 16 MB)"}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n  Browser:       http://localhost:5000")
    print("  Health check:  http://localhost:5000/health\n")
    app.run(debug=True, host="0.0.0.0", port=5000)