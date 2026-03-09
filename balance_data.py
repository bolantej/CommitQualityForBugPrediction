import os
import csv

import pandas as pd


INPUT_PATH = os.path.join("data", "filtered_commits.csv")
OUTPUT_PATH = os.path.join("data", "balanced_commits.csv")


def balance_dataset():
    """Downsample the majority class to match the minority class."""
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found. Run filter_data.py first.")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} commits from {INPUT_PATH}")

    counts = df["label"].value_counts()
    print(f"\nBefore balancing:")
    print(f"  Label 1 (buggy): {counts.get(1, 0)}")
    print(f"  Label 0 (clean): {counts.get(0, 0)}")

    minority_count = counts.min()
    majority_label = counts.idxmax()
    minority_label = counts.idxmin()

    df_minority = df[df["label"] == minority_label]
    df_majority = df[df["label"] == majority_label].sample(
        n=minority_count, random_state=42
    )

    df_balanced = pd.concat([df_minority, df_majority], ignore_index=True)

    balanced_counts = df_balanced["label"].value_counts()
    print(f"\nAfter balancing:")
    print(f"  Label 1 (buggy): {balanced_counts.get(1, 0)}")
    print(f"  Label 0 (clean): {balanced_counts.get(0, 0)}")
    print(f"  Total: {len(df_balanced)}")

    df_balanced.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    balance_dataset()
