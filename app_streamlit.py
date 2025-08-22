import io
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---- Classic (TF-IDF + SVM) paths ----
CLASSIC_MODELS = {
    "GV (Gender-Violence, SVM)": "models/gv_clf.joblib",
    "OFF (General Offensive, SVM)": "models/offense_clf.joblib",
}

# ---- Transformers (HF) paths ----
HF_MODELS = {
    "DistilBERT (EN-OFF)": "models/distilbert_off",
    # "HateBERT (EN-OFF)": "models/hatebert_off",  # enable if you train it later
}


@st.cache_resource
def load_classic(path):
    return load(path)

@st.cache_resource
def load_hf(path):
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(path)
    mdl.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    return tok, mdl, device

def classic_predict_single(text, model):
    if not text or not text.strip(): return None
    pred = int(model.predict([text])[0])
    # decision score if available
    if hasattr(model, "decision_function"):
        score = float(model.decision_function([text])[0])
        return pred, score
    return pred, None

def classic_predict_batch(df, text_col, model):
    X = df[text_col].fillna("")
    preds = model.predict(X).astype(int)
    score = None
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
    out = df.copy()
    if score is not None:
        out["p_score"] = score
    out["label"] = preds
    out["risk"] = np.where((out.get("p_score", 0) > 0.0), "High Risk", "Low Risk")
    return out

def hf_predict_single(text, tok, mdl, device):
    if not text or not text.strip(): return None
    inputs = tok([text], truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = mdl(**inputs).logits
    prob1 = torch.softmax(logits, dim=-1)[0,1].item()
    return int(prob1 >= 0.5), prob1

def hf_predict_batch(df, text_col, tok, mdl, device):
    X = df[text_col].fillna("").tolist()
    probs, preds = [], []
    for i in range(0, len(X), 64):
        batch = X[i:i+64]
        inputs = tok(batch, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = mdl(**inputs).logits
        p = torch.softmax(logits, dim=-1)[:,1].cpu().numpy()
        probs.extend(p.tolist())
        preds.extend((p >= 0.5).astype(int).tolist())
    out = df.copy()
    out["prob_1"] = probs
    out["label"] = preds
    out["risk"] = np.where(out["prob_1"] >= 0.5, "High Risk", "Low Risk")
    return out

# ---------------- UI ----------------
st.set_page_config(page_title="VioLens — Abuse Detector", layout="wide")
st.title("VioLens — Abuse Detection")

family = st.selectbox("Choose model family", ["TF-IDF (classic)", "Transformers (HF)"])

if family == "TF-IDF (classic)":
    mname = st.selectbox("Classic model", list(CLASSIC_MODELS.keys()))
    model = load_classic(CLASSIC_MODELS[mname])

    tab1, tab2 = st.tabs(["Text (single)", "Batch CSV"])
    with tab1:
        txt = st.text_area("Enter text", height=150)
        if st.button("Analyze text (classic)"):
            res = classic_predict_single(txt, model)
            if res is None:
                st.warning("Please enter text.")
            else:
                pred, score = res
                tag = "🟥 High Risk (1)" if pred==1 else "🟩 Low Risk (0)"
                st.markdown(f"**Prediction:** {tag}")
                if score is not None:
                    st.markdown(f"**Decision score:** {score:.4f}  (>0 ⇒ class 1)")

    with tab2:
        up = st.file_uploader("Upload CSV (must have a 'text' column)", type=["csv"])
        text_col = st.text_input("Text column", value="text")
        if st.button("Run batch (classic)") and up is not None:
            df = pd.read_csv(up)
            if text_col not in df.columns:
                st.error(f"Column '{text_col}' not found.")
            else:
                out = classic_predict_batch(df, text_col, model)
                st.dataframe(out.head(20), use_container_width=True)
                buf = io.StringIO(); out.to_csv(buf, index=False)
                st.download_button("Download predictions", buf.getvalue(), "predictions_classic.csv", "text/csv")

else:
    hname = st.selectbox("Transformer model", list(HF_MODELS.keys()))
    tok, mdl, device = load_hf(HF_MODELS[hname])

    tab1, tab2 = st.tabs(["Text (single)", "Batch CSV"])
    with tab1:
        txt = st.text_area("Enter text", height=150)
        if st.button("Analyze text (HF)"):
            res = hf_predict_single(txt, tok, mdl, device)
            if res is None:
                st.warning("Please enter text.")
            else:
                pred, prob1 = res
                tag = "🟥 High Risk (1)" if pred==1 else "🟩 Low Risk (0)"
                st.markdown(f"**Prediction:** {tag}")
                st.markdown(f"**Probability (class 1):** {prob1:.3f}")

    with tab2:
        up = st.file_uploader("Upload CSV (must have a 'text' column)", type=["csv"])
        text_col = st.text_input("Text column", value="text")
        if st.button("Run batch (HF)") and up is not None:
            df = pd.read_csv(up)
            if text_col not in df.columns:
                st.error(f"Column '{text_col}' not found.")
            else:
                out = hf_predict_batch(df, text_col, tok, mdl, device)
                st.dataframe(out.head(20), use_container_width=True)
                buf = io.StringIO(); out.to_csv(buf, index=False)
                st.download_button("Download predictions", buf.getvalue(), "predictions_hf.csv", "text/csv")
