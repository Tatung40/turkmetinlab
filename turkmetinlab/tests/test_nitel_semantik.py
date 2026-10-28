import numpy as np
import pytest
from tmlab.nitel_semantik import semantic_merge_candidates

def test_close_vectors_become_suggestions():
    vectors = np.array([[1,0,0],[.99,.1,0],[0,1,0]], dtype=np.float32)
    result = semantic_merge_candidates(["Zaman tasarrufu","İş yükü azalması","Maliyet"], vectors)
    assert len(result) == 1
    assert result[0]["researcher_approval_required"]
    assert result[0]["code_a"] == "Zaman tasarrufu"

def test_finite_and_dimension_checks():
    with pytest.raises(ValueError, match="embedding"):
        semantic_merge_candidates(["a","b"], np.array([[1,0],[np.nan,1]]))
    with pytest.raises(ValueError, match="embedding"):
        semantic_merge_candidates(["a","b"], np.ones((3,3)))

def test_no_model_download_for_trivial_input():
    assert semantic_merge_candidates(["Tek"], np.empty((0,2))) == []
