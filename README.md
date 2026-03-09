# Commit Quality for Bug Prediction

## Prerequisites

- Python 3.11+
- Mercurial (`brew install mercurial`) OR <insert windows command>
- A local clone of mozilla-central (`hg clone https://hg.mozilla.org/mozilla-central ./mozilla-central`)

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Step 1: Collect raw commit data

```bash
python collect_data.py --hg-repo ./mozilla-central
```

Downloads Mozilla's regressors-regressions dataset, extracts bug-introducing commit messages from the local hg repo, and samples an equal number of non-buggy commits. Outputs `data/raw_commits.csv`.

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

### Step 4: Feature extraction (future)

TODO

### Step 5: Model training (future)

TODO


## Data

- **Source**: [Mozilla regressors-regressions dataset](https://github.com/mozilla/regressors-regressions-dataset)
- **`data/raw_commits.csv`**: All collected commits with columns: `commit_id, summary, body, author, parents, label`
- **`data/filtered_commits.csv`**: Cleaned commits after removing noise
- **`data/balanced_commits.csv`**: Balanced dataset with equal buggy/clean counts
- Label `1` = bug-introducing commit, `0` = clean commit
