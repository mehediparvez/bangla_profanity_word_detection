# Dataset

Every step needs a single labelled file, `data/train.csv`, with two columns:

| Column | Description |
|--------|-------------|
| `text` | The Bangla comment text |
| `label` | 0 = non-toxic, 1 = toxic |

```csv
text,label
"এটি একটি সুন্দর মন্তব্য",0
"তুই একটা বোকা",1
```

Every notebook and `main.py` step splits it with the same stratified `test_size` / `random_seed`
from `configs/config.yaml`, so all models are evaluated on exactly the same held-out test samples.
Training also holds out `training.val_size` of the training split for early stopping.

## Where to get the data

| Dataset | Size | Labels | Notes |
|---------|------|--------|-------|
| **Bengali Hate Speech Dataset** (Kaggle: `naurosahmed/bengali-hate-speech-dataset`) | ~30k | binary hate / not hate | Recommended starting point; social-media comments across sports, politics, religion, entertainment, etc. |
| **Multi-labeled Bengali Toxic Comments** (Kaggle, search the name) | ~16k | vulgar, hate, religious, threat, troll, insult flags | Closest to "profanity"; toxic = any flag set |
| **BD-SHS** (Romim et al., LREC 2022, GitHub `naurosromim/hate-speech-dataset-for-Bengali-social-media`) | ~50k | hate / not hate (+ target, type) | Largest; training takes proportionally longer |
| Bengali Hate Speech (GitHub `rezacsedu/Bengali-Hate-Speech-Dataset`) | ~8k | multi-class hate categories | Smaller; map hate classes to 1 |

Check each dataset's license and cite the paper in your report.

## Converting a download to train.csv

```bash
# 1. Look at the columns of the downloaded file
python data/prepare_data.py data/raw/downloaded.csv

# 2a. Binary label column
python data/prepare_data.py data/raw/downloaded.csv --text-col sentence --label-col hate

# 2b. Multi-label flags: toxic if any flag is 1
python data/prepare_data.py data/raw/downloaded.csv --text-col text \
    --any-of vulgar hate religious threat troll insult

# 2c. Text labels: list the values that mean toxic
python data/prepare_data.py data/raw/downloaded.csv --text-col comment --label-col class \
    --toxic-values abusive offensive
```

Several files (e.g. separate train/test downloads) can be passed at once and are concatenated.
The script strips empty rows and duplicate texts and prints the class balance.

```
data/
├── README.md
├── prepare_data.py
├── train.csv          (generated; git-ignored)
└── raw/               (original downloads; optional)
```
