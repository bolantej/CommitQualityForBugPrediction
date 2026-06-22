# Commit Quality for Bug Prediction

## Project Summary
This was completed as part of a class research project at Oregon State University

Modern software projects accumulate commits faster than reviewers can inspect them, and while LLM-based code review tools can catch defects in pull requests, running them on every change is expensive. This project investigates a cheaper alternative: whether the quality of a commit message alone can predict whether that commit introduces a bug, with no analysis of the underlying code.

The premise is that a clear, well-structured commit message tends to reflect a more careful change, while a vague one (for example, "fixed some bugs") offers weaker assurance. We test this on commits drawn from two sources, Mozilla Firefox and the BugHunter dataset of 14 Java projects, labeling each commit as bug-inducing or clean from known bug-fix history. Every message is converted into a feature vector that pairs ten structural quality signals (ex. summary length, capitalization, imperative mood, file and external references, etc.) with its lexical content (term frequency vector), weighted using Term Frequency-Relevance Frequency (TF-RF) rather than the more common TF-IDF, since TF-RF is designed for labeled data. A linear Support Vector Machine is then trained on these features to classify each commit. The intended application is a tiered review pipeline, in which the classifier serves as a fast, near zero-cost first pass that flags suspicious commits for escalation to more expensive static analysis or LLM-based review, rather than subjecting every commit to deep inspection.

Our results are conditionally positive. Within a single project the commit message carries a measurable signal: the Mozilla model reached an F1 score of 0.89, and the BugHunters models achieved 0.62 and 0.65 on two separate models. Performance degraded sharply under cross-project evaluation, falling to around or just under 0.50 when a model trained on one dataset was tested on the other. Inspection of the feature weights explains why: the TF-RF lexical features consistently dominated the structural ones, and much of what the models learned amounted to project-specific vocabulary (reviewer names, component tags, subsystem keywords) rather than transferable markers of commit quality. We conclude that the approach is best suited as a repository-specific tool retrained per project, not as a general-purpose bug detector.

| Trained on | Tested on | F1 |
|---|---|---|
| Mozilla | Mozilla | 0.89 |
| BugHunter (unigrams + bigrams) | BugHunter | 0.65 |
| BugHunter (unigrams) | BugHunter | 0.62 |
| Mozilla | BugHunter | 0.50 |
| BugHunter (unigrams + bigrams) | Mozilla | 0.47 |
| BugHunter (unigrams) | Mozilla | 0.46 |

The [full paper](paper/paper.pdf) covers the feature definitions, filtering pipeline, and per-label results.
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

- **Source**: [Mozilla regressors-regressions dataset](https://github.com/mozilla/regressors-regressions-dataset)
- **`data/raw_commits.csv`**: All collected commits with columns: `commit_id, summary, body, author, parents, label`
- **`data/filtered_commits.csv`**: Cleaned commits after removing noise
- **`data/balanced_commits.csv`**: Balanced dataset with equal buggy/clean counts
- Label `1` = bug-introducing commit, `0` = clean commit
