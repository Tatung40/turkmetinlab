import pytest
from tmlab.konu_ornekleri import topic_examples

def test_topic_examples_traceable_and_no_hallucinated_text():
    texts=["İklim krizi büyüyor", "İklim krizi büyüyor", "Sağlıkta yeni çalışmalar"]
    table=topic_examples(texts,[0,0,1])
    assert len(table)==2
    assert set(table["Kaynak metin alıntısı"]).issubset(set(texts))
    assert table["Belge sırası"].tolist()==[1,3]

def test_topic_examples_reject_misalignment():
    with pytest.raises(ValueError):
        topic_examples(["x"],[0,1])
