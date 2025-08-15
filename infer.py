import argparse, os, sys
import pandas as pd
from joblib import load

MODELS = {
    "gv": "models/gv_clf.joblib",
    "offense": "models/offense_clf.joblib"
}

def load_model(kind: str):
    path = MODELS[kind]
    if not os.path.exists(path):
        print(f"Model '{kind}' not found at {path}. Train first.")
        sys.exit(1)
    return load(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["gv","offense"], required=True, help="gv or offense")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str)
    g.add_argument("--csv", type=str)
    p.add_argument("--text_col", default="text")
    args = p.parse_args()

    model = load_model(args.model)

    if args.text:
        print(int(model.predict([args.text])[0]))
    else:
        df = pd.read_csv(args.csv)
        if args.text_col not in df.columns:
            print(f"Column '{args.text_col}' missing in CSV."); sys.exit(1)
        df["pred_label"] = model.predict(df[args.text_col].fillna("")).astype(int)
        out = args.csv.replace(".csv", f"_{args.model}_pred.csv")
        df.to_csv(out, index=False)
        print("Saved ->", out)

if __name__ == "__main__":
    main()
