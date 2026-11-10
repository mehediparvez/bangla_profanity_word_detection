import os
import json
import time
import dataclasses
import glob
import shutil
import torch
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .dataset import BanglaDataset
from .model import get_tokenizer, load_model


def compute_metrics_for_trainer(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }


def _warmup_kwargs(ratio):
    # transformers 5 removed warmup_ratio; warmup_steps now accepts a float ratio instead
    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    if "warmup_ratio" in fields:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": ratio}


def _has_saved_model(model_dir):
    weights = ("model.safetensors", "pytorch_model.bin")
    return os.path.exists(os.path.join(model_dir, "config.json")) and any(
        os.path.exists(os.path.join(model_dir, w)) for w in weights
    )


def _sync_to_disk():
    # Ask the OS to write out buffered files; on Colab this nudges Google Drive to upload sooner
    if hasattr(os, "sync"):
        os.sync()


def split_validation(texts, labels, val_size=0.1, seed=42):
    """Hold out a stratified validation split from the training data.

    Early stopping and best-model selection use this split, so the test split
    is only ever used for the final scores.
    """
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=val_size, random_state=seed, stratify=labels
    )
    return train_texts, val_texts, train_labels, val_labels


def train_model(
    model_name,
    train_texts,
    test_texts,
    train_labels,
    test_labels,
    config,
    output_dir="results",
    run_id=1,
    seed=42,
    split_seed=42,
    val_size=0.1,
    resume=True,
    keep_model=True,
):
    run_output_dir = os.path.join(output_dir, f"run_{run_id}")
    best_model_dir = os.path.join(run_output_dir, "best_model")
    metrics_path = os.path.join(run_output_dir, "metrics.json")

    # Skip runs that already finished, so an interrupted session (e.g. Colab) can resume.
    # A run counts as finished only if its saved model is also there when one is expected;
    # otherwise (e.g. lost in a Drive sync) it is trained again.
    if resume and os.path.exists(metrics_path):
        if not keep_model or _has_saved_model(best_model_dir):
            print(f"Found finished run at {run_output_dir}, skipping training")
            with open(metrics_path) as f:
                return json.load(f), best_model_dir
        print(f"{run_output_dir} has metrics but no saved model; training it again")

    tokenizer = get_tokenizer(model_name)
    model = load_model(model_name, config.get("num_labels", 2))

    fit_texts, val_texts, fit_labels, val_labels = split_validation(
        train_texts, train_labels, val_size=val_size, seed=split_seed
    )
    max_length = config["max_length"]
    train_dataset = BanglaDataset(fit_texts, fit_labels, tokenizer, max_length)
    val_dataset = BanglaDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = BanglaDataset(test_texts, test_labels, tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        learning_rate=config["learning_rate"],
        **_warmup_kwargs(config.get("warmup_ratio", 0.1)),
        weight_decay=config.get("weight_decay", 0.01),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_for_trainer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    # Final scores are on the held-out test split; "eval_" prefix keeps downstream keys stable
    metrics = trainer.evaluate(test_dataset, metric_key_prefix="eval")
    metrics["train_time_seconds"] = train_time
    metrics["seed"] = seed

    # Checkpoints hold optimizer state (~3 GB each for XLM-R); only the final model is worth keeping
    for checkpoint_dir in glob.glob(os.path.join(run_output_dir, "checkpoint-*")):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    if keep_model:
        trainer.save_model(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
    # metrics.json is written last: it marks the run as finished for the resume check above
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    _sync_to_disk()

    return metrics, best_model_dir


def train_with_multiple_runs(
    model_name,
    train_texts,
    test_texts,
    train_labels,
    test_labels,
    config,
    output_dir="results",
    num_runs=3,
    seed=42,
    val_size=0.1,
):
    all_metrics = []
    for run_id in range(1, num_runs + 1):
        print(f"\n{'='*50}")
        print(f"Run {run_id}/{num_runs}")
        print(f"{'='*50}")
        metrics, model_dir = train_model(
            model_name,
            train_texts,
            test_texts,
            train_labels,
            test_labels,
            config,
            output_dir=os.path.join(output_dir, model_name.split("/")[-1]),
            run_id=run_id,
            # A different seed per run, so the mean/std across runs is meaningful
            seed=seed + run_id,
            split_seed=seed,
            val_size=val_size,
            # Evaluation and the demo load run_1; other runs only contribute metrics
            keep_model=(run_id == 1),
        )
        all_metrics.append(metrics)
        print(f"Run {run_id} - Accuracy: {metrics['eval_accuracy']:.4f}, "
              f"F1: {metrics['eval_f1']:.4f}")

    return all_metrics


def train_learning_curve(
    model_name,
    train_texts,
    test_texts,
    train_labels,
    test_labels,
    config,
    fractions=(0.25, 0.5, 1.0),
    output_dir="results",
    seed=42,
    val_size=0.1,
):
    results = []
    n = len(train_texts)
    for frac in fractions:
        k = int(n * frac)
        print(f"\nTraining on {frac*100:.0f}% of data ({k} samples)...")
        metrics, _ = train_model(
            model_name,
            train_texts[:k],
            test_texts,
            train_labels[:k],
            test_labels,
            config,
            output_dir=os.path.join(
                output_dir, model_name.split("/")[-1], f"frac_{frac}"
            ),
            seed=seed,
            split_seed=seed,
            val_size=val_size,
            keep_model=False,
        )
        metrics["fraction"] = frac
        metrics["num_samples"] = k
        results.append(metrics)
    return results
