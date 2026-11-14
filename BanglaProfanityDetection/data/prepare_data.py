"""
Convert a downloaded Bangla toxic/hate-speech dataset into data/train.csv (text,label).

Usage:
    # Single binary label column (e.g. Kaggle "Bengali Hate Speech Dataset": sentence, hate)
    python data/prepare_data.py raw/bengali_hate.csv --text-col sentence --label-col hate

    # Multi-label flags, toxic if any flag is set (e.g. "Multi-labeled Bengali Toxic Comments")
    python data/prepare_data.py raw/toxic.csv --text-col text \
        --any-of vulgar hate religious threat troll insult

    # String labels: list which values count as toxic
    python data/prepare_data.py raw/abuse.csv --text-col comment --label-col class \
        --toxic-values abusive offensive

Several input files (e.g. separate train/test downloads) are concatenated.
Run without column options to print the columns of each file.
"""

import argparse
import os
import sys

import pandas as pd


def read_any(path):
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8")


def to_binary(df, args):
    if args.any_of:
        flags = df[args.any_of].fillna(0).astype(int)
        return (flags.sum(axis=1) > 0).astype(int)

    labels = df[args.label_col]
    if args.toxic_values:
        toxic = {v.lower() for v in args.toxic_values}
        return labels.astype(str).str.strip().str.lower().isin(toxic).astype(int)

    labels = pd.to_numeric(labels, errors="coerce")
    if not labels.dropna().isin([0, 1]).all():
        sys.exit(f"Column '{args.label_col}' is not 0/1; pass --toxic-values to map it.")
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="Downloaded CSV/XLSX file(s)")
    parser.add_argument("--text-col", help="Column holding the comment text")
    parser.add_argument("--label-col", help="Column holding the label")
    parser.add_argument("--any-of", nargs="+", help="Multi-label flag columns; toxic if any is 1")
    parser.add_argument("--toxic-values", nargs="+", help="Label values that mean toxic")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "train.csv"))
    args = parser.parse_args()
    # Windows consoles default to cp1252, which cannot print Bangla
    sys.stdout.reconfigure(encoding="utf-8")

    frames = [read_any(p) for p in args.inputs]

    if not args.text_col or not (args.label_col or args.any_of):
        for path, df in zip(args.inputs, frames):
            print(f"{path}: {len(df)} rows, columns = {list(df.columns)}")
            print(df.head(3).to_string(), "\n")
        print("Pass --text-col and --label-col (or --any-of) to convert.")
        return

    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({"text": df[args.text_col].astype(str).str.strip(), "label": to_binary(df, args)})
    out = out[out["text"].str.len() > 0].dropna()
    out["label"] = out["label"].astype(int)
    out = out.drop_duplicates(subset="text")

    out.to_csv(args.output, index=False, encoding="utf-8")
    counts = out["label"].value_counts().to_dict()
    print(f"Wrote {len(out)} rows to {args.output} "
          f"(non-toxic={counts.get(0, 0)}, toxic={counts.get(1, 0)})")


if __name__ == "__main__":
    main()
