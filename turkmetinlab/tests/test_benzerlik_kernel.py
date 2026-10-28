"""Frozen pre-refactor product oracle; no corpus/model/network downloads."""
import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tmlab.benzerlik import kosinus_matrisi, en_yakin_komsular


def old_product(X):
    result = np.asarray((X @ X.T).toarray(), dtype=np.float64)
    np.clip(result, 0, 1, out=result)
    return result


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("chunk", [1, 2, 9])
def test_library_product_preserves_old_scores_and_tied_neighbors(dtype, chunk):
    # Turkish characters, duplicate text, disjoint text and missing/empty document vector.
    X = TfidfVectorizer(dtype=dtype).fit_transform([
        "İstanbul ışık çığ şeker", "İstanbul ışık çığ şeker",
        "ışık öğretmen öğrenci", "çelik korozyon", ""
    ])
    expected = old_product(X)
    np.fill_diagonal(expected, 1.0)  # Preserve the existing zero-vector convention.
    actual = kosinus_matrisi(X)
    tolerance = 2e-7 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
    assert actual.dtype == np.float64
    for k in (0, 1, 3, 20):
        count = max(0, min(k, X.shape[0] - 1))
        scores = old_product(X)
        np.fill_diagonal(scores, -1)
        indices = np.array([np.lexsort((np.arange(len(scores)), -row))[:count]
                            for row in scores])
        found, values = en_yakin_komsular(X, k=k, parca=chunk)
        np.testing.assert_array_equal(found, indices)
        np.testing.assert_allclose(values, np.take_along_axis(scores, indices, 1),
                                   atol=tolerance, rtol=0)


def test_l2_tolerance_does_not_introduce_renormalization():
    X = sparse.csr_matrix([[1 - 4e-7, 0.0], [1.0, 0.0]])
    expected = old_product(X)
    np.fill_diagonal(expected, 1.0)
    np.testing.assert_allclose(kosinus_matrisi(X), expected, atol=1e-12, rtol=0)
    assert kosinus_matrisi(X)[0, 1] < 1.0


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0), (1, 2)])
def test_empty_axes_and_single_zero_row_preserve_contract(shape):
    X = sparse.csr_matrix(shape, dtype=float)
    expected = old_product(X)
    np.fill_diagonal(expected, 1)
    np.testing.assert_array_equal(kosinus_matrisi(X), expected)
    found, scores = en_yakin_komsular(X, k=10, parca=1)
    count = max(0, min(10, shape[0] - 1))
    assert found.shape == scores.shape == (shape[0], count)
    for i in range(shape[0]):
        assert found[i].tolist() == [j for j in range(shape[0]) if j != i][:count]
    np.testing.assert_array_equal(scores, np.zeros_like(scores))


@pytest.mark.parametrize("values", [[[2.0, 0.0]], [[np.nan, 0.0]], [[np.inf, 0.0]]])
def test_invalid_or_missing_numeric_input_remains_rejected(values):
    with pytest.raises(ValueError, match="L2"):
        kosinus_matrisi(sparse.csr_matrix(values))
