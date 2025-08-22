import pandas as pd
tr = pd.read_csv("data/train.csv")[["text","offense_label"]].rename(columns={"offense_label":"label"})
te = pd.read_csv("data/test.csv")[["text","offense_label"]].rename(columns={"offense_label":"label"})
out = pd.concat([tr, te], ignore_index=True)
out.to_csv("data/off_en.csv", index=False)
print("Wrote data/off_en.csv", out.shape)
