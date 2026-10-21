import numpy as np
import pytest
from tmlab.bertopic_adapter import validate
def test_rejects_mismatched_embedding_count():
    with pytest.raises(ValueError):validate(["x"]*20,np.ones((19,384)))
def test_allows_precomputed_vectors():
    documents,vectors=validate(["örnek metin"]*20,np.ones((20,384)),n_topics=4)
    assert len(documents)==20 and vectors.shape==(20,384)
def test_rejects_missing_values():
    d=np.ones((20,384));d[0,0]=np.nan
    with pytest.raises(ValueError):validate(["metin"]*20,d,n_topics=4)
