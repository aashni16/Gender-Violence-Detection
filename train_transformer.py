import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch


def load_csv(path: str, seed: int = 42, train_frac: float = 1.0):
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    # optional stratified subsample for speed
    if 0 < train_frac < 1.0:
        keep, _ = train_test_split(
            df,
            test_size=1.0 - train_frac,
            random_state=seed,
            stratify=df["label"],
        )
    else:
        keep = df

    # stratified train/val split
    tr, va = train_test_split(
        keep, test_size=0.1, random_state=seed, stratify=keep["label"]
    )
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def tokenize_fn(tokenizer, max_len: int):
    def _tok(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        # keep labels so model gets loss
        enc["labels"] = batch["label"]
        return enc
    return _tok


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits[:, 1] > logits[:, 0]).astype(int)
    return {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="e.g. distilbert-base-uncased, GroNLP/hateBERT, google/muril-base-cased")
    ap.add_argument("--data_csv", required=True, help="CSV with columns: text,label (0/1)")
    ap.add_argument("--out_dir", required=True, help="Output directory to save model/tokenizer")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=64)         # shorter seq = faster on CPU
    ap.add_argument("--train_frac", type=float, default=0.3)   # subsample for speed
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()

    tr_df, va_df = load_csv(args.data_csv, seed=args.seed, train_frac=args.train_frac)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    tr_ds = Dataset.from_pandas(tr_df[["text", "label"]], preserve_index=False)
    va_ds = Dataset.from_pandas(va_df[["text", "label"]], preserve_index=False)

    tr_ds = tr_ds.map(tokenize_fn(tok, args.max_len), batched=True, remove_columns=["text", "label"])
    va_ds = va_ds.map(tokenize_fn(tok, args.max_len), batched=True, remove_columns=["text", "label"])

    targs = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        report_to="none",
        dataloader_num_workers=0,   # Windows-friendly
        no_cuda=not use_cuda,       # force CPU if no GPU
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=mdl,
        args=targs,
        train_dataset=tr_ds,
        eval_dataset=va_ds,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Saved:", args.out_dir)


if __name__ == "__main__":
    main()
