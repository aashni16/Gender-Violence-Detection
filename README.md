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


