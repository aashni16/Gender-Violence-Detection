import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import weaklabel_gender_violence

URL = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
os.makedirs("data", exist_ok=True)

def main():
    df = pd.read_csv(URL)
    df = df.rename(columns={"tweet": "text"})
    df = df[["text", "class"]].dropna()

    # --- Labels ---
    # GV: weak rule label
    df["gv_label"] = df["text"].apply(weaklabel_gender_violence)
    # OFF: from dataset (0=hate, 1=offensive, 2=neither) -> abusive/offensive binary
    df["offense_label"] = df["class"].apply(lambda c: 1 if c in (0, 1) else 0)

    # Split once; keep both labels
    train_df, test_df = train_test_split(
        df[["text", "gv_label", "offense_label"]],
        test_size=0.2, random_state=42, stratify=df["offense_label"]
    )

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("Saved -> data/train.csv", train_df.shape,
          "| gv_pos:", round(train_df["gv_label"].mean(),4),
          "| off_pos:", round(train_df["offense_label"].mean(),4))
    print("Saved -> data/test.csv ", test_df.shape,
          "| gv_pos:", round(test_df["gv_label"].mean(),4),
          "| off_pos:", round(test_df["offense_label"].mean(),4))

if __name__ == "__main__":
    main()
