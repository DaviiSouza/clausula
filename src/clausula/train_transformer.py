"""Fine-tune a transformer for multi-label unfair clause detection.

Usage:
    python -m clausula.train_transformer --model distilroberta-base
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from clausula.data import load_splits
from clausula.labels import LABELS
from clausula.metrics import per_label, summary, tune_thresholds

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def to_dataset(df, tokenizer, max_length):
    ds = Dataset.from_dict({"text": df["text"].tolist(), "labels": df[LABELS].values.astype("float32").tolist()})
    return ds.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
                  batched=True, remove_columns=["text"])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return summary(labels.astype(int), sigmoid(logits) >= 0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None, help="output folder name (default: model name)")
    parser.add_argument("--no-save-model", action="store_true", help="keep only metrics and predictions (for seed runs)")
    args = parser.parse_args()

    set_seed(args.seed)
    run_name = args.run_name or args.model.split("/")[-1]
    out_dir = MODELS_DIR / run_name

    splits = load_splits()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = {name: to_dataset(df, tokenizer, args.max_length) for name, df in splits.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        warmup_steps=0.1,  # float < 1 = fraction of total steps (transformers >= 5)
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Per-category thresholds on validation, then a single evaluation on test.
    val_probs = sigmoid(trainer.predict(data["validation"]).predictions)
    thresholds = tune_thresholds(splits["validation"][LABELS].values, val_probs)

    start = time.perf_counter()
    test_probs = sigmoid(trainer.predict(data["test"]).predictions)
    ms_per_sentence = (time.perf_counter() - start) / len(data["test"]) * 1000

    y_test = splits["test"][LABELS].values
    test_pred = test_probs >= thresholds
    results = {
        "model": run_name,
        **{k: round(float(v), 3) for k, v in summary(y_test, test_pred).items()},
        "ms_per_sentence": round(ms_per_sentence, 3),
        "thresholds": dict(zip(LABELS, thresholds.round(2).tolist(), strict=True)),
        "per_label": per_label(y_test, test_pred).round(3).to_dict(orient="index"),
        "args": vars(args),
    }

    if not args.no_save_model:
        trainer.save_model(str(out_dir))
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    np.save(out_dir / "test_probs.npy", test_probs)
    np.save(out_dir / "thresholds.npy", thresholds)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ("model", "macro_f1", "micro_f1", "unfair_f1", "ms_per_sentence")}, indent=2))


if __name__ == "__main__":
    main()
