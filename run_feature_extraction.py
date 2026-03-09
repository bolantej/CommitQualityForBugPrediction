import os
import csv
import time

import joblib
import pandas as pd
import scipy.sparse

from feature_extraction import extract_commit_features

INPUT_PATH = os.path.join("data", "balanced_commits.csv")
OUTPUT_FEATURES_PATH = os.path.join("data", "features.csv")
OUTPUT_TFRF_PATH = os.path.join("data", "tfrf_matrix.npz")
OUTPUT_VECTORIZER_PATH = os.path.join("data", "tfrf_vectorizer.joblib")

NUMERIC_FEATURES = [
    "title_length",
    "body_length",
    "commit_hashes_title",
    "commit_hashes_body",
    "title_capitalized",
    "title_imperative",
    "ext_refs_title",
    "ext_refs_body",
    "filenames_title",
    "filenames_body",
]


def run_feature_extraction():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found. Run balance_data.py first.")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} commits from {INPUT_PATH}")

    counts = df["label"].value_counts()
    print(f"  Label 1 (buggy): {counts.get(1, 0)}")
    print(f"  Label 0 (clean): {counts.get(0, 0)}")

    print("\nExtracting features...")
    start = time.time()
    result_df, _tfrf_df, tfrf_vectorizer, tfrf_matrix = extract_commit_features(
        df, return_vectorizer=True
    )
    elapsed = time.time() - start
    print(f"Feature extraction completed in {elapsed:.1f}s")

    # Save numeric features + label
    feature_cols = ["commit_id"] + NUMERIC_FEATURES + ["label"]
    result_df[feature_cols].to_csv(
        OUTPUT_FEATURES_PATH, index=False, quoting=csv.QUOTE_ALL
    )
    print(f"\nSaved numeric features to {OUTPUT_FEATURES_PATH}")

    # Save TF-RF sparse matrix (use raw sparse matrix directly, not the DataFrame)
    tfrf_sparse = scipy.sparse.csr_matrix(tfrf_matrix)
    scipy.sparse.save_npz(OUTPUT_TFRF_PATH, tfrf_sparse)
    print(f"Saved TF-RF matrix {tfrf_sparse.shape} to {OUTPUT_TFRF_PATH}")

    # Save fitted vectorizer
    joblib.dump(tfrf_vectorizer, OUTPUT_VECTORIZER_PATH)
    print(f"Saved TF-RF vectorizer to {OUTPUT_VECTORIZER_PATH}")

    # Summary
    print(f"\nSummary:")
    print(f"  Rows: {len(result_df)}")
    print(f"  Numeric features: {len(NUMERIC_FEATURES)}")
    print(f"  Vocabulary size: {len(tfrf_vectorizer.get_feature_names_out())}")
    print(f"  TF-RF matrix shape: {tfrf_sparse.shape}")


if __name__ == "__main__":
    run_feature_extraction()
