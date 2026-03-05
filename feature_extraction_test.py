import unittest
import pandas as pd
import numpy as np
from feature_extraction import (
    length,
    is_capitalized,
    batch_imperative_mood,
    count_external_refs,
    count_filenames,
    count_commit_hashes,
    extract_commit_features,
)

#LLM Generated Tests

def make_sample_df():
    return pd.DataFrame({
        "summary": [
            "Fix null pointer exception in UserService.java",  # imperative, capitalized, filename
            "add support for PROJ-42 login flow",              # imperative (lowercase), JIRA ref
            "update README.md with installation steps",        # imperative, filename
            "wip",                                             # not imperative, not capitalized
            "Revert abc1234 breaking change from #99",         # commit hash in title, issue ref
        ],
        "body": [
            "Resolves #201. See https://github.com/org/repo/issues/201 for details.",
            "Implements PROJ-42 as requested. Related to abc1234def5678.",
            "",
            "still working on it",
            "This reverts commit abc1234def56789012345678. Fixes regression introduced in #99.",
        ],
        "label": [1, 0, 0, 1, 1],
    })


# ─────────────────────────────────────────────
# Feature 1: Title Length
# ─────────────────────────────────────────────

class TestTitleLength(unittest.TestCase):
    def test_normal_string(self):
        self.assertEqual(length("Fix the bug"), 11)

    def test_empty_string(self):
        self.assertEqual(length(""), 0)

    def test_whitespace_only(self):
        self.assertEqual(length("   "), 0)

    def test_leading_trailing_whitespace_stripped(self):
        self.assertEqual(length("  Fix bug  "), len("Fix bug"))

    def test_none(self):
        self.assertEqual(length(None), 0)

    def test_nan(self):
        self.assertEqual(length(np.nan), 0)


# ─────────────────────────────────────────────
# Feature 2: Capitalization
# ─────────────────────────────────────────────

class TestIsCapitalized(unittest.TestCase):
    def test_capitalized(self):
        self.assertEqual(is_capitalized("Fix the bug"), 1)

    def test_lowercase(self):
        self.assertEqual(is_capitalized("fix the bug"), 0)

    def test_empty_string(self):
        self.assertEqual(is_capitalized(""), 0)

    def test_none(self):
        self.assertEqual(is_capitalized(None), 0)

    def test_starts_with_number(self):
        self.assertEqual(is_capitalized("123 fix bug"), 0)

    def test_single_uppercase_char(self):
        self.assertEqual(is_capitalized("F"), 1)


# ─────────────────────────────────────────────
# Feature 3: Imperative Mood
# ─────────────────────────────────────────────

class TestImperativeMood(unittest.TestCase):
    def _run(self, texts):
        return batch_imperative_mood(pd.Series(texts)).tolist()

    def test_imperative_capitalized(self):
        self.assertEqual(self._run(["Fix the null pointer exception"])[0], 1)

    def test_imperative_lowercase(self):
        self.assertEqual(self._run(["add support for login"])[0], 1)

    def test_past_tense_not_imperative(self):
        self.assertEqual(self._run(["Fixed the null pointer exception"])[0], 0)

    def test_third_person_not_imperative(self):
        self.assertEqual(self._run(["Fixes the null pointer exception"])[0], 0)

    def test_gerund_not_imperative(self):
        self.assertEqual(self._run(["Fixing the null pointer exception"])[0], 0)

    def test_non_verb_start(self):
        self.assertEqual(self._run(["wip"])[0], 0)

    def test_empty_string(self):
        self.assertEqual(self._run([""])[0], 0)

    def test_batch_processes_multiple(self):
        results = self._run([
            "Fix the bug",      # imperative
            "Fixed the bug",    # past tense
            "Add new feature",  # imperative
            "wip",              # not a verb
        ])
        self.assertEqual(results[0], 1)
        self.assertEqual(results[1], 0)
        self.assertEqual(results[2], 1)
        self.assertEqual(results[3], 0)


# ─────────────────────────────────────────────
# Features 4 & 5: External References
# ─────────────────────────────────────────────

class TestExternalRefs(unittest.TestCase):
    def test_github_issue(self):
        self.assertEqual(count_external_refs("Fixes #123"), 1)

    def test_jira_ticket(self):
        self.assertEqual(count_external_refs("Resolves PROJ-42"), 1)

    def test_url(self):
        self.assertEqual(count_external_refs("See https://github.com/org/repo/issues/1"), 1)

    def test_multiple_refs(self):
        self.assertEqual(count_external_refs("Fixes #1, see https://example.com and PROJ-99"), 3)

    def test_no_refs(self):
        self.assertEqual(count_external_refs("Fix the null pointer exception"), 0)

    def test_empty_string(self):
        self.assertEqual(count_external_refs(""), 0)

    def test_none(self):
        self.assertEqual(count_external_refs(None), 0)


# ─────────────────────────────────────────────
# Features 6 & 7: Filename References
# ─────────────────────────────────────────────

class TestFilenames(unittest.TestCase):
    def test_python_file(self):
        self.assertEqual(count_filenames("Fix bug in utils.py"), 1)

    def test_markdown_file(self):
        self.assertEqual(count_filenames("Update README.md"), 1)

    def test_java_file(self):
        self.assertEqual(count_filenames("Fix null pointer in UserService.java"), 1)

    def test_multiple_files(self):
        self.assertEqual(count_filenames("Rename foo.py to bar.py"), 2)

    def test_no_filenames(self):
        self.assertEqual(count_filenames("Fix the null pointer exception"), 0)

    def test_empty_string(self):
        self.assertEqual(count_filenames(""), 0)

    def test_none(self):
        self.assertEqual(count_filenames(None), 0)


# ─────────────────────────────────────────────
# Feature 8: Body Length
# ─────────────────────────────────────────────

class TestBodyLength(unittest.TestCase):
    def test_normal_body(self):
        self.assertEqual(body_length("This fixes the issue"), 20)

    def test_empty_body(self):
        self.assertEqual(body_length(""), 0)

    def test_whitespace_only(self):
        self.assertEqual(body_length("   "), 0)

    def test_none(self):
        self.assertEqual(body_length(None), 0)

    def test_nan(self):
        self.assertEqual(body_length(np.nan), 0)


# ─────────────────────────────────────────────
# Features 9 & 10: Commit Hashes
# ─────────────────────────────────────────────

class TestCommitHashes(unittest.TestCase):
    def test_short_hash(self):
        self.assertEqual(count_commit_hashes("Revert abc1234 from main"), 1)

    def test_full_hash(self):
        self.assertEqual(count_commit_hashes("Revert abc1234def56789012345678"), 1)

    def test_multiple_hashes(self):
        self.assertEqual(count_commit_hashes("Cherry-pick abc1234 and def56789"), 2)

    def test_no_hashes(self):
        self.assertEqual(count_commit_hashes("Fix the null pointer exception"), 0)

    def test_too_short_not_matched(self):
        # 6 hex chars should NOT match (minimum is 7)
        self.assertEqual(count_commit_hashes("abc123"), 0)

    def test_empty_string(self):
        self.assertEqual(count_commit_hashes(""), 0)

    def test_none(self):
        self.assertEqual(count_commit_hashes(None), 0)


# ─────────────────────────────────────────────
# Feature 11: TF-RF (integration level)
# ─────────────────────────────────────────────

class TestTFRF(unittest.TestCase):
    def setUp(self):
        self.df = make_sample_df()

    def test_tfrf_returns_dataframe(self):
        _, tfrf_df = extract_commit_features(self.df)
        self.assertIsInstance(tfrf_df, pd.DataFrame)

    def test_tfrf_row_count_matches(self):
        _, tfrf_df = extract_commit_features(self.df)
        self.assertEqual(len(tfrf_df), len(self.df))

    def test_tfrf_index_matches_input(self):
        _, tfrf_df = extract_commit_features(self.df)
        self.assertEqual(list(tfrf_df.index), list(self.df.index))

    def test_tfrf_has_columns(self):
        _, tfrf_df = extract_commit_features(self.df)
        self.assertGreater(len(tfrf_df.columns), 0)


# ─────────────────────────────────────────────
# Full Pipeline: extract_commit_features
# ─────────────────────────────────────────────

class TestExtractCommitFeatures(unittest.TestCase):
    def setUp(self):
        self.df = make_sample_df()
        self.result, self.tfrf_df = extract_commit_features(self.df)

    def test_output_has_all_feat_columns(self):
        expected_cols = [
            "title_length",
            "title_capitalized",
            "title_imperative",
            "ext_refs_title",
            "ext_refs_body",
            "filenames_title",
            "filenames_body",
            "body_length",
            "commit_hashes_title",
            "commit_hashes_body",
        ]
        for col in expected_cols:
            self.assertIn(col, self.result.columns, msg=f"Missing column: {col}")

    def test_original_columns_preserved(self):
        for col in ["summary", "body", "label"]:
            self.assertIn(col, self.result.columns)

    def test_row_count_unchanged(self):
        self.assertEqual(len(self.result), len(self.df))

    def test_index_preserved(self):
        self.assertEqual(list(self.result.index), list(self.df.index))

    def test_specific_row_values(self):
        # Row 0: "Fix null pointer exception in UserService.java"
        self.assertEqual(self.result.loc[0, "title_capitalized"], 1)
        self.assertEqual(self.result.loc[0, "title_imperative"], 1)
        self.assertEqual(self.result.loc[0, "filenames_title"], 1)   # UserService.java

        # Row 1: "add support for PROJ-42 login flow"
        self.assertEqual(self.result.loc[1, "title_capitalized"], 0) # lowercase
        self.assertEqual(self.result.loc[1, "ext_refs_title"], 1)    # PROJ-42

        # Row 3: "wip"
        self.assertEqual(self.result.loc[3, "title_imperative"], 0)
        self.assertEqual(self.result.loc[3, "title_capitalized"], 0)

        # Row 4: "Revert abc1234 breaking change from #99"
        self.assertEqual(self.result.loc[4, "commit_hashes_title"], 1) # abc1234
        self.assertEqual(self.result.loc[4, "ext_refs_title"], 1)      # #99

    def test_handles_missing_body(self):
        df = pd.DataFrame({
            "summary": ["Fix bug", "Add feature"],
            "body": [None, np.nan],
            "label": [1, 0],
        })
        result, _ = extract_commit_features(df)
        self.assertEqual(result["body_length"].tolist(), [0, 0])
        self.assertEqual(result["ext_refs_body"].tolist(), [0, 0])

    def test_handles_missing_summary(self):
        df = pd.DataFrame({
            "summary": [None, "Fix bug"],  # one None alongside a valid row so TF-RF has variance
            "body": ["some body", "another body"],
            "label": [1, 0],
        })
        result, _ = extract_commit_features(df)
        self.assertEqual(result.loc[0, "title_length"], 0)
        self.assertEqual(result.loc[0, "title_capitalized"], 0)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)