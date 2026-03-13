import csv
import os
import subprocess
from pathlib import Path

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["Project", "Hash", "Number of Bugs"])


def run_git_command(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def clone_repo(repo_url: str, target_dir: Path) -> None:
    if target_dir.exists():
        # Update existing clone/mirror
        run_git_command(["fetch", "--all", "--tags"], cwd=target_dir)
    else:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        run_git_command(
            ["clone", "--mirror", repo_url, str(target_dir.name)], cwd=target_dir.parent
        )


def collect_commit_data(project: dict, base_dir: str = "git_repos") -> list[dict]:
    repo_dir = Path(base_dir) / f"{project['project_name']}"
    print(repo_dir)
    clone_repo(project["project_url"], repo_dir)

    # iterate over all commits in the history
    out = run_git_command(["rev-list", "--all", "--before=2020-06-01"], repo_dir)
    commit_ids = [line for line in out.splitlines() if line]

    rows = []
    for index, commit_id in enumerate(commit_ids, start=1):
        try:
            print(
                f"Gathering metadata for repo {project['project_name']} commit {index}/{len(commit_ids)}"
            )
            show_output = run_git_command(
                ["show", "-s", "--format=%H%n%an%n%ae%n%cI%n%P%n%s%n%b", commit_id],
                repo_dir,
            )
            (
                full_hash,
                author_name,
                author_email,
                author_date,
                parents,
                summary,
                *body_lines,
            ) = show_output.split("\n")

            body = ""
            if body_lines:
                body = " ".join(body_lines)

            parents_str = ""
            if parents:
                parents_str = ",".join(parents.split("\n"))

            rows.append(
                {
                    "project": project["project_name"],
                    "commit_id": full_hash,
                    "summary": summary,
                    "body": body,
                    "author": author_name,
                    "date": author_date,
                    "parents": parents_str,
                }
            )
        except:
            continue
    return rows


def main():
    project_urls = [
        {
            "project_name": "Android-Universal-Image-Loader",
            "project_url": "https://github.com/nostra13/Android-Universal-Image-Loader",
        },
        {
            "project_name": "BroadleafCommerce",
            "project_url": "https://github.com/BroadleafCommerce/BroadleafCommerce",
        },
        {
            "project_name": "MapDB",
            "project_url": "https://github.com/jankotek/mapdb",
        },
        {
            "project_name": "antlr4",
            "project_url": "https://github.com/antlr/antlr4",
        },
        {
            "project_name": "ceylon-ide-eclipse",
            "project_url": "https://github.com/eclipse-archived/ceylon-ide-eclipse",
        },
        {
            "project_name": "elasticsearch",
            "project_url": "https://github.com/elastic/elasticsearch",
        },
        {
            "project_name": "hazelcast",
            "project_url": "https://github.com/hazelcast/hazelcast",
        },
        {
            "project_name": "junit",
            "project_url": "https://github.com/junit-team/junit4",
        },
        {
            "project_name": "mcMMO",
            "project_url": "https://github.com/mcMMO-Dev/mcMMO",
        },
        {
            "project_name": "neo4j",
            "project_url": "https://github.com/neo4j/neo4j",
        },
        {
            "project_name": "netty",
            "project_url": "https://github.com/netty/netty",
        },
        {
            "project_name": "orientdb",
            "project_url": "https://github.com/orientechnologies/orientdb",
        },
        {
            "project_name": "oryx",
            "project_url": "https://github.com/ossrs/oryx",
        },
        {
            "project_name": "titan",
            "project_url": "https://github.com/thinkaurelius/titan",
        },
    ]

    df = load_csv("file.csv")
    buggy = df[df["Number of Bugs"] > 0]

    output_path = "data/raw_commits.csv"
    output_exists = os.path.exists(output_path)

    rows = pd.DataFrame()

    bug_set: set[str] = set(buggy["Hash"])

    if not output_exists:
        r = []
        for proj in project_urls:
            data = collect_commit_data(proj, "git_repos")
            r.extend(data)
            break

        rows = pd.DataFrame(
            r, columns=["commit_id", "summary", "body", "author", "date", "parents"]
        )
        rows["label"] = rows["commit_id"].isin(bug_set).astype(int)
        rows.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    else:
        rows = pd.read_csv(
            output_path,
            usecols=[
                "commit_id",
                "summary",
                "body",
                "author",
                "date",
                "parents",
                "label",
            ],
        )


if __name__ == "__main__":
    main()
