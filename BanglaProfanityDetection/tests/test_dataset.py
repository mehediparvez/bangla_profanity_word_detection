import os
import tempfile
import pytest
import pandas as pd

from src.dataset import load_dataset, BanglaDataset


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "text": [
            "এটি একটি ভালো মন্তব্য",
            "তুই একটা গাধা",
            "আজকের আবহাওয়া সুন্দর",
            "বোকা কোথাকার",
            "বাংলাদেশ সুন্দর দেশ",
            "শালা বদমাশ",
            "ধন্যবাদ সবাইকে",
            "চোর কোথাকার",
            "আমি খুশি",
            "মূর্খ লোক",
        ],
        "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_dataset_split(sample_csv):
    train_texts, test_texts, train_labels, test_labels = load_dataset(
        sample_csv, test_size=0.2, random_seed=42
    )
    assert len(train_texts) == 8
    assert len(test_texts) == 2
    assert len(train_labels) == 8
    assert len(test_labels) == 2
    assert all(l in (0, 1) for l in train_labels + test_labels)


def test_load_dataset_no_split(sample_csv):
    texts, labels = load_dataset(sample_csv, split=False)
    assert len(texts) == 10
    assert len(labels) == 10


def test_bangla_dataset(sample_csv):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    texts = ["এটি পরীক্ষা", "আরেকটি পরীক্ষা"]
    labels = [0, 1]
    ds = BanglaDataset(texts, labels, tokenizer, max_length=32)
    assert len(ds) == 2

    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["input_ids"].shape[0] == 32
