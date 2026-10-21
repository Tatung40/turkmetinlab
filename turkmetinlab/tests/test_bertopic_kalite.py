import numpy as np
import pytest
from tmlab.bertopic_kalite import _npmi, _diversity, compare

def test_metric_boundaries():
    assert _diversity([["a","b"],["a","c"]]) == pytest.approx(0.75)
    assert _npmi([["a","b"]],["a b","a b","c"]) == pytest.approx(1.0)
    assert _npmi([["aa","cc"]],["aa","cc","aa"]) == pytest.approx(-1.0)
    assert _npmi([["eğitim","öğretim"]],["Eğitim, öğretim.","Eğitim öğretim","araştırma"]) == pytest.approx(1.0)

def test_validation_precedes_heavy_import():
    with pytest.raises(ValueError, match="10\\+"):
        compare(["çok kısa"]*9,np.zeros((9,4)),k=2)
    with pytest.raises(ValueError, match="10\\\\+"):
        compare(["eğitim araştırması"]*501,np.zeros((501,4)),k=2)
    with pytest.raises(ValueError, match="Embedding"):
        compare(["eğitim bilimsel veri"]*12,np.zeros((11,4)),k=2)

def test_real_bertopic_with_synthetic_embeddings():
    pytest.importorskip("bertopic")
    samples=["iklim kuraklık yağış baraj tarım"]*12 + ["öğrenci öğretmen eğitim sınav sınıf"]*12 + ["doktor hastane tedavi sağlık hasta"]*12
    rng=np.random.default_rng(42)
    emb=np.concatenate([rng.normal(0,.005,(12,16))+np.eye(16)[i] for i in range(3)])
    result=compare(samples,emb,k=3)
    assert result["n"]==36
    assert set(result["models"])=={"BERTopic-E5-KMeans","NMF","LDA"}
    for model in result["models"].values():
        assert sum(model["topic_counts"].values())==36
        assert 0 <= model["diversity"] <= 1
