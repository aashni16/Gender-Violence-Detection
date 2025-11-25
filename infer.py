#!/usr/bin/env python3
"""
Robust inference CLI for Gender-Violence & Offensive detection.

Usage examples:
  python infer.py --text "She was assaulted at the party" --output demo_single_out.txt
  python infer.py --input sample_texts.csv --output demo_predictions.csv

Behavior:
- Attempts to load a TF-IDF vectorizer at models/tfidf_vectorizer.joblib
  and an sklearn model at models/baseline_model.joblib.
- If model files are missing or loading fails, falls back to a rule-based predictor
  (simple keyword heuristics) so CI/demo runs without trained artifacts.
"""
import argparse
import os
import sys
import re
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

VIOLENCE_KEYWORDS = {
    "rape","assault","abuse","molest","hit","stab","attack","rape","sexual assault",
    "beaten","beating","raped","molested"
}
GENDER_KEYWORDS = {"woman","women","man","men","girl","boy","female","male","she","he","her","him"}
OFFENSIVE_KEYWORDS = {
    "fuck","shit","bitch","asshole","bastard","slut","whore","cunt","dick","idiot","stupid"
}

def simple_preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def rule_predict_one(text: str):
    t = simple_preprocess(text)
    toks = set(t.split())
    has_violence = any(k in t for k in VIOLENCE_KEYWORDS)
    has_gender = len(toks.intersection(GENDER_KEYWORDS)) > 0
    is_gv = has_violence and has_gender
    is_off = any(k in t for k in OFFENSIVE_KEYWORDS) or ("offensive" in t)
    return {
        "pred_gv": bool(is_gv),
        "pred_off": bool(is_off),
        "method": "rule"
    }

def try_load_model():
    vec_path = os.path.join("models", "tfidf_vectorizer.joblib")
    model_path = os.path.join("models", "baseline_model.joblib")
    if joblib is None:
        return None, None
    if os.path.exists(vec_path) and os.path.exists(model_path):
        try:
            vec = joblib.load(vec_path)
            model = joblib.load(model_path)
            return vec, model
        except Exception as e:
            print("Warning: failed to load model artifacts:", e, file=sys.stderr)
            return None, None
    return None, None

def predict_with_model(texts, vec, model):
    X = vec.transform(texts)
    try:
        preds = model.predict(X)
    except Exception as e:
        # If it's multioutput / label-structured, attempt row-wise predict
        print("Model predict failed:", e, file=sys.stderr)
        preds = []
        for i in range(X.shape[0]):
            try:
                preds.append(model.predict(X[i]))
            except Exception:
                preds.append(None)
    # Normalize preds into a dict per text.
    rows = []
    for i, p in enumerate(preds):
        row = {}
        if isinstance(p, (list, tuple)):
            # assume [gv_pred, off_pred] or similar
            if len(p) >= 2:
                row["pred_gv"] = bool(p[0])
                row["pred_off"] = bool(p[1])
            elif len(p) == 1:
                row["pred_gv"] = bool(p[0])
                row["pred_off"] = False
        elif isinstance(p, (int, float, str)):
            # single label: try to interpret
            row["pred_gv"] = bool(p)
            row["pred_off"] = False
        elif p is None:
            row["pred_gv"] = False
            row["pred_off"] = False
        else:
            # fallback
            row["pred_gv"] = False
            row["pred_off"] = False
        row["method"] = "model"
        rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser(description="Inference for Gender-Violence project")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Single input text for inference")
    group.add_argument("--input", type=str, help="CSV input path (expects column 'text' or first column)")
    ap.add_argument("--output", type=str, default="predictions.csv", help="Output CSV or TXT path")
    args = ap.parse_args()

    input_texts = []
    if args.text:
        input_texts = [args.text]
    else:
        if not os.path.exists(args.input):
            print("ERROR: input file not found:", args.input, file=sys.stderr)
            sys.exit(2)
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        # find text column
        if "text" in df.columns:
            input_texts = df["text"].astype(str).tolist()
        else:
            # use first column
            input_texts = df.iloc[:,0].astype(str).tolist()

    vec, model = try_load_model()
    outputs = []
    if vec is not None and model is not None:
        try:
            model_rows = predict_with_model(input_texts, vec, model)
            for txt, r in zip(input_texts, model_rows):
                out = {"text": txt, **r}
                outputs.append(out)
        except Exception as e:
            print("Model path present but prediction failed. Falling back to rules. Error:", e, file=sys.stderr)
            for txt in input_texts:
                rr = rule_predict_one(txt)
                outputs.append({"text": txt, **rr})
    else:
        # fallback to rule-based
        for txt in input_texts:
            rr = rule_predict_one(txt)
            outputs.append({"text": txt, **rr})

    # Save results
    out_path = args.output
    if out_path.lower().endswith(".csv"):
        out_df = pd.DataFrame(outputs)
        out_df.to_csv(out_path, index=False)
        print(f"Wrote CSV predictions to: {out_path}")
    else:
        # write text file
        with open(out_path, "w", encoding="utf-8") as f:
            for o in outputs:
                f.write(f"TEXT: {o['text']}\nPRED_GV: {o['pred_gv']}\nPRED_OFF: {o['pred_off']}\nMETHOD: {o['method']}\n---\n")
        print(f"Wrote text predictions to: {out_path}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
