import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class BanglaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_dataset(
    path,
    text_column="text",
    label_column="label",
    test_size=0.2,
    random_seed=42,
    split=True,
):
    df = pd.read_csv(path)
    df = df[[text_column, label_column]].dropna()
    df[text_column] = df[text_column].astype(str).str.strip()
    df[label_column] = df[label_column].astype(int)

    if not split:
        return df[text_column].tolist(), df[label_column].tolist()

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df[text_column].tolist(),
        df[label_column].tolist(),
        test_size=test_size,
        random_state=random_seed,
        stratify=df[label_column].tolist(),
    )
    return train_texts, test_texts, train_labels, test_labels


def create_dataloaders(
    train_texts, test_texts, train_labels, test_labels, tokenizer, config
):
    train_dataset = BanglaDataset(
        train_texts, train_labels, tokenizer, config["max_length"]
    )
    test_dataset = BanglaDataset(
        test_texts, test_labels, tokenizer, config["max_length"]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )
    return train_loader, test_loader
