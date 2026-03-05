import numpy as np
import scipy.sparse as sp

from sklearn.utils.fixes import _IS_32BIT
from sklearn.utils.validation import validate_data
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

#Taken from the sklearn.feature_extract.text.py file
def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.issparse(X) and X.format == "csr":
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(X.indptr)

class TFRFTransformer(TfidfTransformer):
    def __init__(self, *, use_rf=True, norm="l2", use_idf=False, smooth_idf=True, sublinear_tf=False):
        super().__init__(norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
        self.use_rf = use_rf
    def fit(self, X, y):
        X = validate_data(
            self, X, accept_sparse=("csr", "csc"), accept_large_sparse=not _IS_32BIT
        )
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in (np.float64, np.float32) else np.float64

        #if using tf-rf
        if self.use_rf:
            pos_mask = y == 1
            neg_mask = y == 0
            X_pos = X[pos_mask]
            X_neg = X[neg_mask]

            a = _document_frequency(X_pos)
            c = _document_frequency(X_neg)

            c_safe = np.maximum(1, c)
            self.rf_ = np.log2(2 + (a / c_safe))
            self.idf_ = self.rf_

        return self



class TFRFVectorizer(TfidfVectorizer):
    def fit(self, raw_documents, y):
        self._tfidf = TFRFTransformer(            
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf)
        X = super(TfidfVectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X, y)
        return self

    def fit_transform(self, raw_documents, y):
        self._tfidf = TFRFTransformer(            
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf)
        X = super(TfidfVectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X, y)
        return self._tfidf.transform(X, copy=False)