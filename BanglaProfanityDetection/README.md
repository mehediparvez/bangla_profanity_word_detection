# Bangla Profanity & Toxic Comment Detection

A machine learning project for detecting profanity and toxic comments in Bangla text using transformer-based models (BanglaBERT, XLM-RoBERTa), LLM baselines, and a library baseline.

## Project Overview

This project compares multiple approaches for Bangla toxic comment classification:

| Approach | Model | Type |
|----------|-------|------|
| Library Baseline | `better_profanity` | Rule-based word list |
| Fine-tuned Transformer | BanglaBERT (`sagorsarker/bangla-bert-base`) | Fine-tuned classifier |
| Fine-tuned Transformer | XLM-RoBERTa (`xlm-roberta-base`) | Fine-tuned classifier |
| LLM Zero-shot | Gemini Flash-Lite / Groq | Prompt-based |

## Project Structure

```
BanglaProfanityDetection/
├── README.md
├── requirements.txt
├── main.py                      # CLI entry point for full pipeline
├── Makefile                     # Automation shortcuts
├── configs/
│   └── config.yaml              # Hyperparameters and paths
├── data/
│   └── README.md                # Dataset download instructions
├── src/
│   ├── __init__.py
│   ├── dataset.py               # Dataset loading and preprocessing
│   ├── model.py                 # Model definitions
│   ├── train.py                 # Training loop with multi-run support
│   ├── evaluate.py              # Evaluation metrics and error analysis
│   ├── predict.py               # Inference utilities
│   ├── llm_baseline.py          # Groq/Gemini LLM zero-shot baseline
│   ├── visualize.py             # Publication-ready chart generation
│   ├── results.py               # Shared results format (save/load for all models)
│   └── utils.py                 # Helper functions
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_evaluate.py
│   ├── test_utils.py
│   ├── test_results.py
│   └── test_llm_baseline.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_better_profanity.ipynb
│   ├── 03_finetune_banglabert.ipynb
│   ├── 04_finetune_xlmr.ipynb
│   ├── 05_llm_baseline.ipynb
│   └── 06_evaluation_and_visualization.ipynb
├── app/
│   └── demo.py                  # Gradio demo application
├── results/                     # Saved metrics and CSVs
└── plots/                       # Generated charts and figures
```

## Setup

### 1. Install Dependencies
```bash
cd BanglaProfanityDetection
pip install -r requirements.txt
```

### 2. Download Dataset
See `data/README.md` for where to download a dataset, then convert it to `data/train.csv`:
```bash
python data/prepare_data.py path/to/download.csv                         # show its columns
python data/prepare_data.py path/to/download.csv --text-col sentence --label-col hate
```

### 3. Configure
Edit `configs/config.yaml` to set paths and hyperparameters.
API keys are read from the environment: `GROQ_API_KEY` (preferred) or `GEMINI_API_KEY`.

All approaches are evaluated on the same stratified test split of `data/train.csv`
(`test_size` / `random_seed` in the config), so their scores are directly comparable.

## Running the Project

### Step 1: Data Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Step 2: Library Baseline
```bash
jupyter notebook notebooks/02_baseline_better_profanity.ipynb
```

### Step 3: Fine-tune Models (run on Colab/Kaggle with GPU; see "Training on Google Colab")
```bash
jupyter notebook notebooks/03_finetune_banglabert.ipynb
jupyter notebook notebooks/04_finetune_xlmr.ipynb
```

### Step 4: LLM Baseline
```bash
jupyter notebook notebooks/05_llm_baseline.ipynb
```

### Step 5: Evaluation & Visualization
```bash
jupyter notebook notebooks/06_evaluation_and_visualization.ipynb
```
Notebook 06 loads whatever notebooks 02-05 saved under `results/` and plots it;
models that have not been run yet are left out.

### Step 6: Demo App
```bash
cd app
python demo.py
```

### Alternative: Run Full Pipeline via CLI
```bash
python main.py --step all        # Train + Evaluate + Baselines + Visualize
python main.py --step train      # Fine-tune BanglaBERT and XLM-R (3 runs + learning curve)
python main.py --step evaluate   # Evaluate saved fine-tuned models
python main.py --step baseline   # better_profanity library baseline
python main.py --step llm        # LLM zero-shot baseline (skipped without an API key)
python main.py --step visualize  # Generate plots from everything in results/
python main.py --step demo       # Launch Gradio demo
```

### Results Layout
Notebooks and the CLI write the same files, one directory per model
(`better_profanity`, `banglabert`, `xlmr`, `llm`):

| File | Contents | Written for |
|------|----------|-------------|
| `results/<model>/evaluation.json` | metrics, inference speed, confusion matrix, FP/FN counts | all models |
| `results/<model>/summary.json` | mean / std over the training runs | fine-tuned models |
| `results/<model>/learning_curve.json` | metrics at 25% / 50% / 100% of training data | fine-tuned models |

### Run Tests
```bash
make test
# or
cd .. && python -m pytest BanglaProfanityDetection/tests/ -v
```

## Training on Google Colab

Every notebook starts with a setup cell that only runs on Colab. It mounts Google Drive,
clones this branch into `MyDrive/Set_A`, `cd`s into `notebooks/` and installs dependencies.
Results and models are written to Drive, so they survive disconnects.

1. Push this branch to GitHub (the setup cell clones `bangal_profanity`).
2. Open a notebook in Colab: File > Open notebook > GitHub, then search `mehediparvez/Set_A` and pick the `bangal_profanity` branch.
3. Runtime > Change runtime type > **T4 GPU**.
4. Run the setup cell once. Then upload `train.csv` into
   `MyDrive/Set_A/BanglaProfanityDetection/data/` (Drive web UI or the Colab file browser).
5. Run the notebooks in order: 01, 02, 03 (BanglaBERT), 04 (XLM-R), 05 (LLM, needs `GROQ_API_KEY`
   set via Colab Secrets or `os.environ`), 06 (plots).
6. If a session disconnects, re-open the notebook and run it again. Finished runs
   (`results/<model>/run_*/metrics.json`) are skipped automatically.
7. Download `results/` and `plots/` from Drive, or copy `results/bangla-bert-base/run_1/best_model`
   locally to run the demo.

### Expected time (free T4, ~30k comments, default config)
| Job | Time |
|-----|------|
| better_profanity baseline | < 1 min |
| BanglaBERT: 3 runs + learning curve (25/50/100%) | ~2-3 h |
| XLM-R: 3 runs + learning curve | ~3-4 h |
| LLM baseline (Groq free tier) | ~15-30 min |
| Evaluation + plots | ~10 min |

That's roughly 5-7 GPU hours in total. Free Colab sessions disconnect, so plan for 2-3 sessions
(e.g. one per model). Early stopping usually shortens this. Setting `training.num_runs: 1` roughly
halves it, but then there are no error bars.

Disk: only `run_1/best_model` is kept per model (~0.7 GB BanglaBERT, ~1.1 GB XLM-R). Epoch
checkpoints and the models from the other runs are deleted, so this fits the free 15 GB Drive.

### Training on a local 4 GB GPU
It's possible but slow (roughly 15-25 h for everything). In `configs/config.yaml` set
`max_length: 128`, `batch_size: 8` (BanglaBERT) / `4` (XLM-R),
`gradient_accumulation_steps: 2` / `4`, and if XLM-R still runs out of memory, `gradient_checkpointing: true`.

## Visualizations Generated

- Model comparison bar chart (accuracy, F1, precision, recall)
- Accuracy vs. inference speed scatter plot
- Learning curves (25%, 50%, 100% training data)
- Confusion matrices for each model
- Error type distribution bar chart

## Free Tier Compute Strategy

| Need | Platform | Limits |
|------|----------|--------|
| GPU for fine-tuning | Kaggle (primary), Colab (backup) | ~30 h/week, 12-hour sessions |
| LLM inference | Groq free tier | ~1,000 req/day, 200K tokens/day |
| LLM inference (alt) | Gemini Flash-Lite | ~500 req/day on free tier |

## License

MIT
