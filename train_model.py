import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import load
import sys
import os

FEATURES_PATH = os.path.join("data", "features.csv")
TFRF_MATRIX_PATH = os.path.join("data", "tfrf_matrix.npz")
VECTORIZER_PATH = os.path.join("data", "tfrf_vectorizer.joblib")

def prepare_data():
    if not os.path.exists(FEATURES_PATH):
        print(f"Error: {FEATURES_PATH} not found. Run run_feature_extraction.py first.")
        return
    features = pd.read_csv(FEATURES_PATH)
    if not os.path.exists(TFRF_MATRIX_PATH):
        print(f"Error: {TFRF_MATRIX_PATH} not found. Run run_feature_extraction.py first.")
        return
    matrix = sp.load_npz(TFRF_MATRIX_PATH)

    vocab = None
    if not os.path.exists(VECTORIZER_PATH):
        print(f"Warning: {VECTORIZER_PATH} not found — TF-RF features will be shown by index.")
    else:
        vectorizer = load(VECTORIZER_PATH)
        vocab = vectorizer.get_feature_names_out()
        print(f"Loaded vectorizer vocabulary: {len(vocab)} tokens")

    X = features.drop(columns=['commit_id', 'label'])
    y = features['label']
    combined_X, _ = combine_features(X, matrix)
    
    csv_feature_names = X.columns.tolist()
    return combined_X, y, csv_feature_names, vocab

def combine_features(X_csv, X_tfrf):
    if X_csv.shape[0] != X_tfrf.shape[0]:
        print(
            f"Error: CSV has {X_csv.shape[0]} rows but TF-RF matrix has {X_tfrf.shape[0]} rows."
        )
        sys.exit(1)

    # Scale dense features then convert to sparse so hstack works
    scaler = StandardScaler()
    X_csv_scaled = scaler.fit_transform(X_csv)
    X_csv_sparse = sp.csr_matrix(X_csv_scaled)

    X_combined = sp.hstack([X_csv_sparse, X_tfrf.tocsr()], format='csr')
    print(
        f"Combined feature matrix: {X_combined.shape[0]} rows, {X_combined.shape[1]} features, {X_csv.shape[1]} CSV + {X_tfrf.shape[1]} TF-RF)"
    )
    return X_combined, scaler


def train_and_evaluate(X, y, csv_feature_names, tfrf_vocab,
                       test_size=0.2, random_state=1, C=1.0, max_iter=2000):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LinearSVC(C=C, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + "=" * 55)
    print("LINEAR SVM — EVALUATION RESULTS")
    print("=" * 55)
    print(f"\nDataset size  : {X.shape[0]} samples")
    print(f"Total features: {X.shape[1]}  ({len(csv_feature_names)} CSV + {len(tfrf_vocab)} TF-RF)")
    print(f"Train / Test  : {X_train.shape[0]} / {X_test.shape[0]}")
    print(f"Classes       : {sorted(y.unique().tolist())}")
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nCSV Feature Coefficients:")
    if len(model.classes_) == 2:
        coef = model.coef_[0]
    else:
        coef = np.mean(model.coef_, axis=0)

    n_csv = len(csv_feature_names)
    csv_coef = np.abs(coef[:n_csv])
    importance = pd.Series(csv_coef, index=csv_feature_names).sort_values(ascending=False)
    for feat, val in importance.items():
        print(f"  {feat:<35} {val:.4f}")

    print("\nTop 20 TF-RF Positive Feature Coefficients:")
    tfrf_coef = coef[n_csv:]
    top_pos_idx = np.argsort(tfrf_coef)[::-1][:20]
    for idx in top_pos_idx:
        label = tfrf_vocab[idx] if tfrf_vocab is not None else f"token [{idx:>6}]"
        print(f"  {label:<35} {tfrf_coef[idx]:.4f}")

    # Top TF-RF token weights
    print("\nTop 20 TF-RF Negative Feature Coefficients:")
    top_neg_idx = np.argsort(tfrf_coef)[:20]
    for idx in top_neg_idx:
        label = tfrf_vocab[idx] if tfrf_vocab is not None else f"token [{idx:>6}]"
        print(f"  {label:<35} {tfrf_coef[idx]:.4f}")

    return model


def main():
    X, y, csv_feature_names, tfrf_vocab = prepare_data()
    train_and_evaluate(X, y, csv_feature_names, tfrf_vocab)


if __name__ == "__main__":
    main()