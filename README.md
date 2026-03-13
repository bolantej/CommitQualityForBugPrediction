# Commit Quality for Bug Prediction

## Prerequisites

- Python 3.11+
- Git
- Local copy of the Bug Hunter dataset - https://data.mendeley.com/datasets/8tx7kjbkg4/2

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
Next, download and extract the BugHunter dataset from mendeley data. The collect_data.py script is looking for a file called "file.csv" in the root of the project. Our results were done using the full/all/file.csv data.

## Usage

### Step 1: Collect raw commit data

```bash
python collect_data.py
```

Downloads git repository for each project in the BugHunters dataset, extracts bug-introducing commit messages from the git log, and labels the commits using the data in file.csv.

### Step 2: Filter commits

```bash
python filter_data.py
```

Removes generic messages, bot commits, and merge commits. Outputs `data/filtered_commits.csv`.

### Step 3: Balance dataset

```bash
python balance_data.py
```

Downsamples the majority class (buggy) to match the minority class (clean), producing a balanced dataset. Outputs `data/balanced_commits.csv`.

### Step 4: Feature extraction

```bash
python run_feature_extraction.py
```

Extracts 10 numeric features (title/body length, capitalization, imperative mood, external references, filenames, commit hashes) and a TF-RF sparse matrix from commit messages. Outputs `data/features.csv`, `data/tfrf_matrix.npz`, and `data/tfrf_vectorizer.joblib`.

### Step 5: Model training (future)

```bash
python train_model.py
```

Trains a linear SVM on the combined numeric features and TF-RF matrix. Prints an accuracy table, confusion matrix, and some of the coefficient weights


## Data

- **Source**: [BugHunter Dataset](https://data.mendeley.com/datasets/8tx7kjbkg4/2)
- **`data/raw_commits.csv`**: All collected commits with columns: `commit_id, summary, body, author, parents, label`
- **`data/filtered_commits.csv`**: Cleaned commits after removing noise
- **`data/balanced_commits.csv`**: Balanced dataset with equal buggy/clean counts
- Label `1` = bug-introducing commit, `0` = clean commit
