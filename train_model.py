import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from joblib import load
import sys
import os
from sklearn.preprocessing import MinMaxScaler

FEATURES_PATH = os.path.join("data", "features.csv")
TEST_FEATURES_PATH = os.path.join("data", "test_features.csv")
TEST_COMMITS_PATH = os.path.join("data", "test_balanced_commits.csv")
TFRF_MATRIX_PATH = os.path.join("data", "tfrf_matrix.npz")
VECTORIZER_PATH = os.path.join("data", "tfrf_vectorizer.joblib")

def prepare_data():
    combined_test_X = None
    test_y = None
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
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    y = features['label']
    combined_X, scaler = combine_features(X_scaled, matrix)

    if not os.path.exists(TEST_FEATURES_PATH):
        print(f"Warning: {TEST_FEATURES_PATH} not found — Cannot compute against a second file")
    else:
        test_commits = pd.read_csv(TEST_COMMITS_PATH)
        test_y = test_commits['label']
        test_matrix = vectorizer.transform(test_commits['summary'].fillna(""))
        test_features = pd.read_csv(TEST_FEATURES_PATH).drop(columns=['commit_id', 'label'])
        scaler = MinMaxScaler(feature_range=(0, 1))
        weighted_test_features = scaler.fit_transform(test_features)

        combined_test_X, _ = combine_features(weighted_test_features, test_matrix, scaler=scaler)



    
    csv_feature_names = X.columns.tolist()
    return combined_X, y, csv_feature_names, vocab, combined_test_X, test_y

def combine_features(X_csv, X_tfrf, scaler=None):
    if X_csv.shape[0] != X_tfrf.shape[0]:
        print(
            f"Error: CSV has {X_csv.shape[0]} rows but TF-RF matrix has {X_tfrf.shape[0]} rows."
        )
        sys.exit(1)

    # Scale dense features then convert to sparse so hstack works
    if scaler is None:
        scaler = StandardScaler()
    X_csv_scaled = scaler.fit_transform(X_csv)
    X_csv_sparse = sp.csr_matrix(X_csv_scaled)

    X_combined = sp.hstack([X_csv_sparse, X_tfrf.tocsr()], format='csr')
    print(
        f"Combined feature matrix: {X_combined.shape[0]} rows, {X_combined.shape[1]} features, {X_csv.shape[1]} CSV + {X_tfrf.shape[1]} TF-RF)"
    )
    return X_combined, scaler

#Used for individually testing and training a model on itself
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

#train a model on one dataset, test on the other
def train_and_test_and_evaluate(X_test, y_test, X_train, y_train, csv_feature_names, tfrf_vocab,
                       test_size=0.2, random_state=1, C=1.0, max_iter=2000):


    model = LinearSVC(C=C, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + "=" * 55)
    print("LINEAR SVM — EVALUATION RESULTS")
    print("=" * 55)
    print(f"\nDataset size  : {X_train.shape[0]} samples")
    print(f"Total features: {X_train.shape[1]}  ({len(csv_feature_names)} CSV + {len(tfrf_vocab)} TF-RF)")
    print(f"Train / Test  : {X_train.shape[0]} / {X_test.shape[0]}")
    print(f"Classes       : {sorted(y_train.unique().tolist())}")
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

#train multiple models for different c values 
def train_for_c_and_evaluate(X, y, csv_feature_names, tfrf_vocab,
                       test_size=0.2, random_state=1, C=1.0,
                       max_iter=2000, n_splits=10):
    """
    Train and evaluate LinearSVC models for one or more C values using
    k-fold cross-validation, then do a final hold-out evaluation.

    Parameters
    ----------
    C : float or list of float
        Regularisation strength(s) to try.
    n_splits : int
        Number of folds for stratified k-fold CV.
    """
    C_values = [C] if isinstance(C, (int, float)) else list(C)
    multi = len(C_values) > 1

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # ── Cross-validation sweep ────────────────────────────────────────────────
    cv_results = []
    for c_val in C_values:
        fold_acc, fold_f1 = [], []
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train_full, y_train_full), 1):
            X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
            y_tr, y_val = y_train_full.iloc[tr_idx], y_train_full.iloc[val_idx]

            m = LinearSVC(C=c_val, max_iter=max_iter, random_state=random_state)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_val)

            fold_acc.append(accuracy_score(y_val, preds))
            fold_f1.append(f1_score(y_val, preds, average="macro", zero_division=0))

        cv_results.append({
            "C": c_val,
            "cv_acc_mean": np.mean(fold_acc),
            "cv_acc_std":  np.std(fold_acc),
            "cv_f1_mean":  np.mean(fold_f1),
            "cv_f1_std":   np.std(fold_f1),
        })

    # ── CV comparison table ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("LINEAR SVM — CROSS-VALIDATION COMPARISON")
    print("=" * 65)
    print(f"{'C':>10}  {'CV Acc':>10}  {'± Acc':>8}  {'CV F1':>10}  {'± F1':>8}")
    print("-" * 65)
    best = max(cv_results, key=lambda r: r["cv_f1_mean"])
    for r in cv_results:
        marker = "  ◀ best" if r["C"] == best["C"] else ""
        print(f"{r['C']:>10.4g}  {r['cv_acc_mean']:>10.4f}  "
              f"{r['cv_acc_std']:>8.4f}  {r['cv_f1_mean']:>10.4f}  "
              f"{r['cv_f1_std']:>8.4f}{marker}")
    print("=" * 65)

    # ── Final model: best C on full train set ─────────────────────────────────
    best_C = best["C"]
    print(f"\nTraining final model with best C={best_C} on full training set …")

    model = LinearSVC(C=best_C, max_iter=max_iter, random_state=random_state)
    model.fit(X_train_full, y_train_full)
    y_pred = model.predict(X_test)

    print("\n" + "=" * 65)
    print(f"LINEAR SVM — HOLD-OUT EVALUATION  (C={best_C})")
    print("=" * 65)
    print(f"\nDataset size  : {X.shape[0]} samples")
    print(f"Total features: {X.shape[1]}  "
          f"({len(csv_feature_names)} CSV + {len(tfrf_vocab)} TF-RF)")
    print(f"Train / Test  : {X_train_full.shape[0]} / {X_test.shape[0]}")
    print(f"Classes       : {sorted(y.unique().tolist())}")
    print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── Feature coefficients ──────────────────────────────────────────────────
    coef = model.coef_[0] if len(model.classes_) == 2 else np.mean(model.coef_, axis=0)
    n_csv = len(csv_feature_names)

    print("\nCSV Feature Coefficients (|coef|):")
    csv_coef = np.abs(coef[:n_csv])
    importance = pd.Series(csv_coef, index=csv_feature_names).sort_values(ascending=False)
    for feat, val in importance.items():
        print(f"  {feat:<35} {val:.4f}")

    tfrf_coef = coef[n_csv:]

    print("\nTop 20 TF-RF Positive Feature Coefficients:")
    for idx in np.argsort(tfrf_coef)[::-1][:20]:
        label = tfrf_vocab[idx] if tfrf_vocab is not None else f"token [{idx:>6}]"
        print(f"  {label:<35} {tfrf_coef[idx]:.4f}")

    print("\nTop 20 TF-RF Negative Feature Coefficients:")
    for idx in np.argsort(tfrf_coef)[:20]:
        label = tfrf_vocab[idx] if tfrf_vocab is not None else f"token [{idx:>6}]"
        print(f"  {label:<35} {tfrf_coef[idx]:.4f}")

    return model, cv_results

def main():
    X, y, csv_feature_names, tfrf_vocab, X_test, y_test = prepare_data()
    #train_for_c_and_evaluate(X, y, csv_feature_names, tfrf_vocab, C=[.05, .075, .1, .125, 1.5, .175, .2], max_iter=4000)
    #train_and_test_and_evaluate(X_test, y_test, X, y, csv_feature_names, tfrf_vocab, C=.1, max_iter=4000)
    train_and_evaluate(X, y, csv_feature_names, tfrf_vocab, C=.1, max_iter=4000)


if __name__ == "__main__":
    main()