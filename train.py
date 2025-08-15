import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from joblib import dump

os.makedirs("models", exist_ok=True)

def train_and_save(label_col: str, out_name: str):
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    X_train, y_train = train["text"].fillna(""), train[label_col]
    X_test,  y_test  = test["text"].fillna(""),  test[label_col]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2),
                                  min_df=3, max_df=0.9, sublinear_tf=True)),
        ("clf", LinearSVC(class_weight="balanced", C=1.0, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    print(f"\n=== {label_col} ===")
    print(classification_report(y_test, pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    # AUC via decision_function
    try:
        scores = pipe.decision_function(X_test)
        if scores.ndim > 1: scores = scores[:,1]
        print("ROC-AUC:", round(roc_auc_score(y_test, scores), 4))
    except Exception:
        pass

    path = f"models/{out_name}.joblib"
    dump(pipe, path)
    print("Saved ->", path)

def main():
    train_and_save("gv_label", "gv_clf")         # gender-violence
    train_and_save("offense_label", "offense_clf")  # general offensive

if __name__ == "__main__":
    main()
