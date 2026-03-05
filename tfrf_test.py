import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from tfrf import TFRFTransformer
from tfrf import TFRFVectorizer

#LLM Generated Test


# Basic corpus with clear positive/negative separation
corpus = [
    "the cat sat on the mat",      # positive
    "the cat ate the rat",          # positive
    "the dog ran in the park",      # negative
    "the dog chased the ball",      # negative
]
y = np.array([1, 1, 0, 0])

def get_vectorizer_and_matrix():
    cv = CountVectorizer()
    X = cv.fit_transform(corpus)
    return cv, X

def test_rf_weights_are_correct_shape():
    cv, X = get_vectorizer_and_matrix()
    transformer = TFRFTransformer()
    transformer.fit(X, y)
    assert transformer.rf_.shape == (X.shape[1],), "RF weights should have one value per feature"

def test_rf_and_idf_are_same_object():
    cv, X = get_vectorizer_and_matrix()
    transformer = TFRFTransformer()
    transformer.fit(X, y)
    np.testing.assert_array_equal(transformer.rf_, transformer.idf_)

def test_positive_class_terms_get_higher_weights():
    """Terms exclusive to positive class should outweigh terms exclusive to negative class."""
    cv, X = get_vectorizer_and_matrix()
    vocab = cv.vocabulary_
    transformer = TFRFTransformer()
    transformer.fit(X, y)

    cat_weight = transformer.rf_[vocab["cat"]]   # only in positive docs
    dog_weight = transformer.rf_[vocab["dog"]]   # only in negative docs

    assert cat_weight > dog_weight, "Positive-class terms should have higher RF weight"

def test_shared_terms_get_low_weights():
    """Terms appearing in both classes (like 'the') should get low weight."""
    cv, X = get_vectorizer_and_matrix()
    vocab = cv.vocabulary_
    transformer = TFRFTransformer()
    transformer.fit(X, y)

    the_weight = transformer.rf_[vocab["the"]]
    cat_weight = transformer.rf_[vocab["cat"]]

    assert the_weight < cat_weight, "Shared terms should have lower weight than class-specific terms"

def test_minimum_weight_is_one():
    """log2(2 + 0) = 1, so no term should have weight below 1."""
    cv, X = get_vectorizer_and_matrix()
    transformer = TFRFTransformer()
    transformer.fit(X, y)
    assert np.all(transformer.rf_ >= 1.0), "No RF weight should be below 1"

def test_transform_output_shape():
    cv, X = get_vectorizer_and_matrix()
    transformer = TFRFTransformer()
    transformer.fit(X, y)
    result = transformer.transform(X)
    assert result.shape == X.shape, "Transform output shape should match input shape"

def test_use_rf_false_falls_back_to_idf():
    """When use_rf=False, behavior should match base TfidfTransformer."""
    from sklearn.feature_extraction.text import TfidfTransformer
    cv, X = get_vectorizer_and_matrix()

    rf_transformer = TFRFTransformer(use_rf=False, use_idf=True)
    rf_transformer.fit(X, y)

    idf_transformer = TfidfTransformer(use_idf=True)
    idf_transformer.fit(X)

    np.testing.assert_array_almost_equal(rf_transformer.idf_, idf_transformer.idf_)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])