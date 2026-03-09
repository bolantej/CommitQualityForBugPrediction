import argparse
import csv
import json
import os
import random
import re
import urllib.request

import hglib
import pandas as pd


DATASET_URL = (
    "https://raw.githubusercontent.com/mozilla/regressors-regressions-dataset/main/dataset.csv"
)
DATA_DIR = "data"
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "raw_commits.csv")
CHECKPOINT_PATH = os.path.join(DATA_DIR, ".collect_checkpoint.json")


def download_dataset():
    """Download the regressors-regressions dataset if not already cached."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATASET_PATH):
        print(f"Dataset already cached at {DATASET_PATH}")
        return
    print(f"Downloading dataset from {DATASET_URL} ...")
    urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
    print(f"Saved to {DATASET_PATH}")


def parse_buggy_hashes():
    """Parse unique bug-introducing commit hashes from the dataset."""
    df = pd.read_csv(DATASET_PATH)
    hashes = set()
    for _, row in df.iterrows():
        if str(row.get("NO_BUG", "")).strip().upper() == "TRUE":
            continue
        raw = str(row.get("BUG_COMMITS_MERCURIAL", ""))
        if not raw or raw == "nan":
            continue
        for h in re.split(r"[,\s]+", raw):
            h = h.strip()
            if h:
                hashes.add(h)
    print(f"Found {len(hashes)} unique bug-introducing commit hashes")
    return hashes


def extract_commit_info(client, rev):
    """Extract commit message parts and metadata from a mercurial revision."""
    try:
        logs = client.log(revrange=rev, follow=False)
        if not logs:
            return None
        entry = logs[0]
        # hglib's log() uses {desc|firstline}, so fetch full description separately
        rev_bytes = rev.encode() if isinstance(rev, str) else rev
        full_desc = client.rawcommand(
            [b"log", b"-r", rev_bytes, b"--template", b"{desc}"]
        ).decode("utf-8", errors="replace")
        parts = full_desc.split("\n", 1)
        summary = parts[0]
        body = parts[1] if len(parts) > 1 else ""
        author = entry.author.decode("utf-8", errors="replace") if isinstance(entry.author, bytes) else entry.author
        node = entry.node.decode("utf-8", errors="replace") if isinstance(entry.node, bytes) else entry.node
        # Get parents
        parents_list = client.log(revrange=f"parents({rev})", follow=False)
        parent_nodes = []
        for p in parents_list:
            pn = p.node.decode("utf-8", errors="replace") if isinstance(p.node, bytes) else p.node
            parent_nodes.append(pn)
        parents_str = ",".join(parent_nodes)
        date = entry.date.isoformat() if entry.date else ""
        return {
            "commit_id": node,
            "summary": summary,
            "body": body,
            "author": author,
            "date": date,
            "parents": parents_str,
        }
    except Exception as e:
        print(f"  Warning: could not extract rev {rev}: {e}")
        return None


def load_checkpoint():
    """Load checkpoint data if it exists."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return None


def save_checkpoint(data):
    """Save checkpoint data."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f)


def collect_commits(hg_repo, buggy_hashes, seed):
    """Collect buggy and non-buggy commit data from the hg repo."""
    checkpoint = load_checkpoint()
    buggy_commits = checkpoint.get("buggy_commits", []) if checkpoint else []
    nonbuggy_commits = checkpoint.get("nonbuggy_commits", []) if checkpoint else []
    processed_buggy = set(c["commit_id"] for c in buggy_commits)
    phase = checkpoint.get("phase", "buggy") if checkpoint else "buggy"

    print(f"Opening hg repo at {hg_repo} ...")
    client = hglib.open(hg_repo)

    if phase == "buggy":
        remaining = [h for h in buggy_hashes if h not in processed_buggy]
        print(f"Extracting {len(remaining)} buggy commits ({len(processed_buggy)} already done) ...")
        for i, h in enumerate(remaining):
            info = extract_commit_info(client, h)
            if info:
                info["label"] = 1
                buggy_commits.append(info)
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(remaining)} buggy commits")
                save_checkpoint({"buggy_commits": buggy_commits, "nonbuggy_commits": [], "phase": "buggy"})
        print(f"Collected {len(buggy_commits)} valid buggy commits")
        save_checkpoint({"buggy_commits": buggy_commits, "nonbuggy_commits": [], "phase": "nonbuggy"})
        phase = "nonbuggy"

    if phase == "nonbuggy":
        n_needed = len(buggy_commits)
        if len(nonbuggy_commits) >= n_needed:
            print(f"Already have {len(nonbuggy_commits)} non-buggy commits")
        else:
            buggy_ids = set(c["commit_id"] for c in buggy_commits)
            # Also include the original hashes (which may be short prefixes)
            buggy_ids.update(buggy_hashes)

            print("Getting all commit hashes from repo (this may take a while) ...")
            all_logs = client.log(revrange="0:tip", follow=False)
            all_nodes = []
            for entry in all_logs:
                node = entry.node.decode("utf-8", errors="replace") if isinstance(entry.node, bytes) else entry.node
                if node not in buggy_ids:
                    all_nodes.append(node)
            print(f"Pool of {len(all_nodes)} non-buggy candidates")

            already_done = set(c["commit_id"] for c in nonbuggy_commits)
            rng = random.Random(seed)
            sample_size = n_needed - len(nonbuggy_commits)
            if sample_size > len(all_nodes):
                print(f"Warning: only {len(all_nodes)} non-buggy candidates available, need {sample_size}")
                sample_size = len(all_nodes)
            sampled = rng.sample(all_nodes, sample_size)

            print(f"Extracting {len(sampled)} non-buggy commits ...")
            for i, node in enumerate(sampled):
                if node in already_done:
                    continue
                info = extract_commit_info(client, node)
                if info:
                    info["label"] = 0
                    nonbuggy_commits.append(info)
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(sampled)} non-buggy commits")
                    save_checkpoint({"buggy_commits": buggy_commits, "nonbuggy_commits": nonbuggy_commits, "phase": "nonbuggy"})
            print(f"Collected {len(nonbuggy_commits)} non-buggy commits")

    client.close()

    all_commits = buggy_commits + nonbuggy_commits
    df = pd.DataFrame(all_commits, columns=["commit_id", "summary", "body", "author", "date", "parents", "label"])
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved {len(df)} commits to {OUTPUT_PATH}")

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    return df


def main():
    parser = argparse.ArgumentParser(description="Collect commit data for bug prediction")
    parser.add_argument("--hg-repo", required=True, help="Path to local mozilla-central hg clone")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    download_dataset()
    buggy_hashes = parse_buggy_hashes()
    collect_commits(args.hg_repo, buggy_hashes, args.seed)


if __name__ == "__main__":
    main()
