"""
Command-line entry point for the Bangla Profanity Detection pipeline.

Usage:
    python main.py --step all          # Run full pipeline
    python main.py --step train        # Train models only
    python main.py --step evaluate     # Evaluate trained models
    python main.py --step baseline     # better_profanity library baseline
    python main.py --step llm          # LLM zero-shot baseline (needs API key)
    python main.py --step visualize    # Generate plots from saved results
    python main.py --step demo         # Launch Gradio demo

Every step uses the same stratified train/test split of data.train_path,
so all approaches are scored on identical test samples.
"""

import argparse
import os
import time

from src.utils import load_config, save_metrics, ensure_dirs
from src.dataset import load_dataset
from src.train import train_with_multiple_runs, train_learning_curve
from src.evaluate import evaluate_model, aggregate_runs
from src.model import load_trained_model
from src.llm_baseline import run_llm_baseline, run_better_profanity_baseline
from src.results import build_evaluation, save_evaluation, collect_results
from src.visualize import generate_all_plots


def load_split(config):
    data_cfg = config["data"]
    return load_dataset(
        data_cfg["train_path"],
        text_column=data_cfg["text_column"],
        label_column=data_cfg["label_column"],
        test_size=data_cfg["test_size"],
        random_seed=data_cfg["random_seed"],
    )


def step_train(config):
    train_texts, test_texts, train_labels, test_labels = load_split(config)
    print(f"Loaded {len(train_texts)} train, {len(test_texts)} test samples")

    results_dir = config["paths"]["results"]
    num_runs = config["training"]["num_runs"]
    seed = config["data"]["random_seed"]
    val_size = config["training"].get("val_size", 0.1)

    for key in ["banglabert", "xlmr"]:
        model_cfg = config["models"][key]
        model_name = model_cfg["name"]
        print(f"\n{'='*60}")
        print(f"Training {model_name} ({num_runs} runs)")
        print(f"{'='*60}")

        all_metrics = train_with_multiple_runs(
            model_name,
            train_texts, test_texts,
            train_labels, test_labels,
            model_cfg,
            output_dir=results_dir,
            num_runs=num_runs,
            seed=seed,
            val_size=val_size,
        )
        summary = aggregate_runs(all_metrics)
        save_metrics(summary, os.path.join(results_dir, key, "summary.json"))
        save_metrics(all_metrics, os.path.join(results_dir, key, "all_runs.json"))
        print(f"\n{key} summary: F1={summary['eval_f1']['mean']:.4f} "
              f"(+/- {summary['eval_f1']['std']:.4f})")

        print(f"\nTraining learning curve for {model_name}...")
        lc_results = train_learning_curve(
            model_name,
            train_texts, test_texts,
            train_labels, test_labels,
            model_cfg,
            output_dir=results_dir,
            seed=seed,
            val_size=val_size,
        )
        save_metrics(lc_results, os.path.join(results_dir, key, "learning_curve.json"))


def step_evaluate(config):
    _, test_texts, _, test_labels = load_split(config)

    results_dir = config["paths"]["results"]
    all_results = {}

    for key in ["banglabert", "xlmr"]:
        model_cfg = config["models"][key]
        best_model_dir = os.path.join(results_dir, model_cfg["name"].split("/")[-1], "run_1", "best_model")

        if not os.path.exists(best_model_dir):
            print(f"Skipping {key}: no trained model at {best_model_dir}")
            continue

        print(f"\nEvaluating {key}...")
        model, tokenizer = load_trained_model(best_model_dir)
        import torch
        if torch.cuda.is_available():
            model = model.cuda()

        start_time = time.time()
        _, preds = evaluate_model(
            model, tokenizer, test_texts, test_labels, max_length=model_cfg["max_length"]
        )
        evaluation = build_evaluation(test_labels, preds, time.time() - start_time)
        save_evaluation(evaluation, results_dir, key)
        all_results[key] = evaluation
        metrics = evaluation["metrics"]
        print(f"{key}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    return all_results


def step_baseline(config):
    _, test_texts, _, test_labels = load_split(config)
    print(f"\nRunning better_profanity baseline on {len(test_texts)} test samples...")

    start_time = time.time()
    preds = run_better_profanity_baseline(test_texts, test_labels)
    evaluation = build_evaluation(test_labels, preds, time.time() - start_time)
    save_evaluation(evaluation, config["paths"]["results"], "better_profanity")

    metrics = evaluation["metrics"]
    print(f"better_profanity: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    return evaluation


def _make_llm_client(config):
    """Return (provider, client, model_name, llm_cfg), or None when no API key is set."""
    if os.environ.get("GROQ_API_KEY"):
        from groq import Groq

        llm_cfg = config["llm"]["groq"]
        return "groq", Groq(), llm_cfg["model"], llm_cfg

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        import google.generativeai as genai

        llm_cfg = config["llm"]["gemini"]
        genai.configure(api_key=gemini_key)
        return "gemini", genai.GenerativeModel(llm_cfg["model"]), llm_cfg["model"], llm_cfg

    return None


def step_llm(config):
    llm = _make_llm_client(config)
    if llm is None:
        print("Skipping LLM baseline: set GROQ_API_KEY or GEMINI_API_KEY to run it.")
        return None
    provider, client, model_name, llm_cfg = llm

    _, test_texts, _, test_labels = load_split(config)
    print(f"\nRunning {provider} ({model_name}) zero-shot on {len(test_texts)} test samples...")

    start_time = time.time()
    output = run_llm_baseline(
        test_texts,
        test_labels,
        client,
        model_name=model_name,
        batch_size=llm_cfg.get("batch_size", 25),
        delay=llm_cfg.get("delay_between_batches", 2),
        max_retries=llm_cfg.get("max_retries", 3),
        provider=provider,
        reasoning_effort=llm_cfg.get("reasoning_effort"),
        cache_path=os.path.join(config["paths"]["results"], "llm", "predictions_cache.json"),
    )
    print(f"Finished in {time.time() - start_time:.1f}s "
          f"({output['newly_classified']} comments sent to the API)")
    # Excludes cached comments, so the reported speed stays honest on a resumed run
    inference_time = output["inference_time"]

    if not output["valid_predictions"]:
        print("LLM baseline produced no usable predictions; nothing saved.")
        return None

    # Failed batches are excluded from scoring rather than counted as non-toxic
    evaluation = build_evaluation(
        output["valid_labels"],
        output["valid_predictions"],
        inference_time,
        extra={
            "provider": provider,
            "model": model_name,
            "total": output["total"],
            "successful": output["successful"],
            "failed_indices": output["failed_indices"],
        },
    )
    save_evaluation(evaluation, config["paths"]["results"], "llm")

    metrics = evaluation["metrics"]
    print(f"llm: {output['successful']}/{output['total']} classified, "
          f"Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    return evaluation


def step_visualize(config):
    plots_dir = config["paths"]["plots"]
    ensure_dirs(plots_dir)

    comparison_results, curves, cm_dict, error_counts = collect_results(
        config["paths"]["results"]
    )
    if comparison_results:
        generate_all_plots(comparison_results, curves, cm_dict, error_counts, plots_dir)
    else:
        print("No results found. Run training/evaluation or a baseline step first.")


def step_demo():
    from app.demo import demo
    demo.launch(server_name="0.0.0.0", server_port=7860)


def main():
    parser = argparse.ArgumentParser(description="Bangla Profanity Detection Pipeline")
    parser.add_argument(
        "--step",
        choices=["train", "evaluate", "baseline", "llm", "visualize", "demo", "all"],
        required=True,
        help="Pipeline step to run",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config["paths"]["results"], config["paths"]["plots"])

    if args.step in ("train", "all"):
        step_train(config)

    if args.step in ("evaluate", "all"):
        step_evaluate(config)

    if args.step in ("baseline", "all"):
        step_baseline(config)

    if args.step in ("llm", "all"):
        step_llm(config)

    if args.step in ("visualize", "all"):
        step_visualize(config)

    if args.step == "demo":
        step_demo()


if __name__ == "__main__":
    main()
