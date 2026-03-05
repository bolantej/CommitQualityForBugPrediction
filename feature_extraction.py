import re
import pandas as pd
import spacy

from tfrf import TFRFVectorizer

_nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

RE_EXTERNAL_REF = re.compile(
    r'(#\d+)'                                          # GitHub/GitLab issue or PR (#123)
    r'|(https?://\S+)'                                 # URLs
    r'|([A-Z][A-Z0-9_]+-\d+)',                        # JIRA-style tickets (PROJ-123)
    re.IGNORECASE
)

# Commit hashes: 7–40 hex characters (short or full SHA)
RE_COMMIT_HASH = re.compile(r'\b[0-9a-f]{7,40}\b', re.IGNORECASE)

# Filename references: words containing a dot-extension (e.g. foo.py, README.md)
RE_FILENAME = re.compile(
    r'\b[\w\-/\\]+\.\w{1,10}\b'
)

def batch_imperative_mood(summaries: pd.Series, batch_size: int = 1000) -> pd.Series: 
    texts = summaries.fillna("").tolist()
    results = []
    for doc in _nlp.pipe(texts, batch_size=batch_size):
        imperative = 0
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                imperative = int(token.morph.get("VerbForm") == ["Inf"])
                break  # Only the root verb matters
        results.append(imperative)
    return pd.Series(results, index=summaries.index)

def length(string: str) -> int:
    return len(string.strip()) if isinstance(string, str) else 0

def is_capitalized(summary: str) -> int:
    s = summary.strip() if isinstance(summary, str) else ""
    return int(len(s) > 0 and s[0].isupper())

def count_external_refs(text: str) -> int:
    """Count external references (#123, URLs, JIRA tickets) in a string."""
    if not isinstance(text, str):
        return 0
    return len(RE_EXTERNAL_REF.findall(text))

def count_filenames(text: str) -> int:
    """Count filename-like tokens (e.g. foo.py, README.md) in a string."""
    if not isinstance(text, str):
        return 0
    return len(RE_FILENAME.findall(text))


def count_commit_hashes(text: str) -> int:
    """Count commit hash references (7–40 hex chars) in a string."""
    if not isinstance(text, str):
        return 0
    return len(RE_COMMIT_HASH.findall(text))


def extract_commit_features(df: pd.DataFrame):
    result = df.copy()
    summary = result['summary'].fillna("")
    body    = result['body'].fillna("")
    labels  = result['label']

    result["title_length"]      = summary.apply(length)
    result["body_length"]       = body.apply(length)
    
    result["commit_hashes_title"] = summary.apply(count_commit_hashes)
    result["commit_hashes_body"]  = body.apply(count_commit_hashes)

    result["title_capitalized"] = summary.apply(is_capitalized)
    result["title_imperative"]    = batch_imperative_mood(summary)

    result["ext_refs_title"]    = summary.apply(count_external_refs)
    result["ext_refs_body"]     = body.apply(count_external_refs)

    result["filenames_title"]   = summary.apply(count_filenames)
    result["filenames_body"]    = body.apply(count_filenames)



    tfrf_vectorizer = TFRFVectorizer()
    tfrf_matrix = tfrf_vectorizer.fit_transform(summary, labels) #could be changed from summary to body (or used on a combined summary/body)

    #vocab      = tfrf_vectorizer.get_feature_names_out()
    tfrf_df    = pd.DataFrame.sparse.from_spmatrix(tfrf_matrix, index=result.index)

    #result = pd.concat([result, tfrf_df], axis=1) #could return one dataframe insteda of two 

    return result, tfrf_df