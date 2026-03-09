import os
import re
import csv

import pandas as pd


INPUT_PATH = os.path.join("data", "raw_commits.csv")
OUTPUT_PATH = os.path.join("data", "filtered_commits.csv")
REJECTED_PATH = os.path.join("data", "rejected_commits.csv")

# Generic/default commit message patterns
GENERIC_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"^initial commit$", re.IGNORECASE),
    re.compile(r"^update readme", re.IGNORECASE),
    re.compile(r"^squash", re.IGNORECASE),
]

# Known bot author patterns
BOT_NAMES = [
    "ffxbld",
    "dependabot",
    "renovate",
    "greenkeeper",
    "bors",
]

BOT_EMAIL_PATTERNS = [
    re.compile(r"\bbot\b", re.IGNORECASE),
]


def is_generic_message(summary):
    """Check if a commit message matches generic/default patterns."""
    if not isinstance(summary, str):
        return True
    for pattern in GENERIC_PATTERNS:
        if pattern.search(summary):
            return True
    return False


def is_bot_author(author):
    """Check if the author looks like a bot."""
    if not isinstance(author, str):
        return False
    author_lower = author.lower()
    for bot_name in BOT_NAMES:
        if bot_name in author_lower:
            return True
    for pattern in BOT_EMAIL_PATTERNS:
        if pattern.search(author):
            return True
    return False


def is_merge_commit(summary, parents):
    """Check if a commit is a merge commit."""
    if isinstance(summary, str):
        if re.match(r"^[Mm]erge\b", summary):
            return True
        if re.search(r"\bmerge\b.+\binto\b", summary, re.IGNORECASE):
            return True
    if isinstance(parents, str) and len(parents.split(",")) >= 2:
        return True
    return False


def filter_commits():
    """Apply all filters to raw_commits.csv and save filtered_commits.csv."""
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found. Run collect_data.py first.")
        return

    df = pd.read_csv(INPUT_PATH)
    initial_count = len(df)
    print(f"Loaded {initial_count} commits from {INPUT_PATH}")

    rejected_dfs = []

    # Filter 1: Generic messages
    mask_generic = df["summary"].apply(is_generic_message)
    removed_generic = mask_generic.sum()
    if removed_generic > 0:
        rejected = df[mask_generic].copy()
        rejected["reason"] = "generic_message"
        rejected_dfs.append(rejected)
    df = df[~mask_generic]
    print(f"Removed {removed_generic} generic/default messages")

    # Filter 2: Bot commits
    mask_bot = df["author"].apply(is_bot_author)
    removed_bot = mask_bot.sum()
    if removed_bot > 0:
        rejected = df[mask_bot].copy()
        rejected["reason"] = "bot_author"
        rejected_dfs.append(rejected)
    df = df[~mask_bot]
    print(f"Removed {removed_bot} bot commits")

    # Filter 3: Merge commits
    mask_merge = df.apply(lambda row: is_merge_commit(row["summary"], row["parents"]), axis=1)
    removed_merge = mask_merge.sum()
    if removed_merge > 0:
        rejected = df[mask_merge].copy()
        rejected["reason"] = "merge_commit"
        rejected_dfs.append(rejected)
    df = df[~mask_merge]
    print(f"Removed {removed_merge} merge commits")

    final_count = len(df)
    total_removed = initial_count - final_count
    print(f"\nTotal removed: {total_removed}")
    print(f"Remaining: {final_count} commits ({final_count // 2 if final_count > 0 else 0} per class approx)")
    print(f"  Label 1 (buggy): {(df['label'] == 1).sum()}")
    print(f"  Label 0 (clean): {(df['label'] == 0).sum()}")

    df.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nSaved to {OUTPUT_PATH}")

    if rejected_dfs:
        rejected_df = pd.concat(rejected_dfs, ignore_index=True)
        rejected_df.to_csv(REJECTED_PATH, index=False, quoting=csv.QUOTE_ALL)
        print(f"Saved {len(rejected_df)} rejected commits to {REJECTED_PATH}")


if __name__ == "__main__":
    filter_commits()
