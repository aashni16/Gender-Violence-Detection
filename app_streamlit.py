import io
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

MODEL_PATHS = {
    "Gender-Violence (GV)": "models/gv_clf.joblib",
    "General Offensive (OFF)": "models/offense_clf.joblib",
}

@st.cache_resource
def load_model(path): return load(path)

def predict_single(text, model):
    if not text or not text.strip(): return None
    pred = int(model.predict([text])[0])
    score = float(model.decision_function([text])[0])
    return pred, score

def predict_batch(df, text_col, model):
    if text_col not in df.columns: raise ValueError(f"Column '{text_col}' not found.")
    X = df[text_col].fillna("")
    preds = model.predict(X).astype(int)
    scores = model.decision_function(X)
    out = df.copy()
    out["p_score"] = scores
    out["label"] = preds
    out["risk"] = np.where(out["p_score"] > 0.0, "High Risk", "Low Risk")
    return out

st.set_page_config(page_title="VioLens — Abuse Risk Detector", layout="wide")
st.title("VioLens — Abuse Risk Detector")
model_name = st.selectbox("Choose model", list(MODEL_PATHS.keys()))
model = load_model(MODEL_PATHS[model_name])
st.caption("Prediction: 1 = related/abusive, 0 = not.")

tab1, tab2 = st.tabs(["Text (single)", "Batch CSV"])

with tab1:
    st.subheader("Single text")
    txt = st.text_area("Enter text", height=150)
    if st.button("Analyze text"):
        res = predict_single(txt, model)
        if res is None:
            st.warning("Please enter text.")
        else:
            pred, score = res
            tag = "🟥 High Risk (1)" if pred == 1 else "🟩 Low Risk (0)"
            st.markdown(f"**Prediction:** {tag}")
            st.markdown(f"**Decision score:** {score:.4f}  (>0 ⇒ class 1)")

with tab2:
    st.subheader("Batch (CSV with a `text` column)")
    up = st.file_uploader("Upload CSV", type=["csv"])
    text_col = st.text_input("Text column", value="text")
    if st.button("Run batch") and up is not None:
        df = pd.read_csv(up)
        out = predict_batch(df, text_col, model)
        st.success(f"Rows: {len(out)}")
        st.dataframe(out.head(20), use_container_width=True)
        buf = io.StringIO(); out.to_csv(buf, index=False)
        st.download_button("Download predictions", buf.getvalue(), "predictions.csv", "text/csv")
