# Gender Violence & Offensive Language Detection

This project detects:
- **Gender-Violence Victimization Patterns (GV)** — text mentioning gender and violence/abuse.
- **General Offensive Language (OFF)** — general insults, hate speech, or abusive terms.

## How It Works
- Uses the Hate Speech and Offensive Language dataset by Davidson et al.
- GV labels are generated using custom keyword rules.
- OFF labels come from the dataset’s original annotations.
- Models are trained using TF-IDF + Linear SVM.
- Includes a Streamlit app for interactive testing.

# Gender-Violence & Offensive Language Detection

## Quick summary
This project detects:
- **GV** — gender-violence mentions (rule-derived labels),
- **OFF** — general offensive language (from Davidson dataset labels).

Repo contains: data preprocessing (`prepare_data.py`), baseline training (`train.py`), transformer training (`train_transformer.py`), inference (`infer.py`), and a Streamlit demo (`app_streamlit.py`). A small `sample_texts.csv` is included for demoing inference.

---

## Run options

### Option A — Run a quick demo using GitHub Actions (no local install)
1. Add the workflow file at `.github/workflows/demo.yml` (see repo instructions).
2. Go to the **Actions** tab in your GitHub repo, pick **CI - Demo Run** and click **Run workflow**.
3. After it finishes, open the workflow run and download the artifact named `demo-output`.

This runs an inference on the included `sample_texts.csv` (or single sample text) and uploads `predictions.csv`.

### Option B — Run locally (recommended for training)
```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows PowerShell
# .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# quick inference
python infer.py --text "She was assaulted at the party"
# or from CSV
python infer.py --input sample_texts.csv --output output_preds.csv
