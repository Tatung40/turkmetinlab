"""Tests for experimental opt-in matcher; production preprocessing is unchanged."""
from tmlab.akademik_terimler import load_lexicon, match_terms

def test_explicit_opt_in():
    d=load_lexicon()
    assert d["varsayilan_etkin"] is False
    assert d["uzman_onayi"] is False
    assert len(d["terimler"]) >= 90

def test_phrase_boundary_and_offsets():
    s="DUYGUSAL EMEK, örgütsel bağlılık ve emekli bireyler."
    out=match_terms(s)
    assert [x["kanonik"] for x in out][:2] == ["duygusal emek", "örgütsel bağlılık"]
    assert all(s[x["baslangic"]:x["bitis"]] == x["metin"] for x in out)

def test_longest_match_and_variants():
    s="Örgütsel vatandaşlık davranışı; Cronbach alpha ve öz yeterlilik."
    out=match_terms(s)
    assert any(x["kanonik"]=="örgütsel vatandaşlık davranışı" for x in out)
    assert any(x["kanonik"]=="cronbach alfa" for x in out)
    assert any(x["kanonik"]=="öz yeterlik" for x in out)
