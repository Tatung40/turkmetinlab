from tmlab.nitel_ucretsiz import suggest_merges, merge_approved_codes
import pytest

def test_similar_labels_suggested_without_changing_data():
    names=["Zaman tasarrufu","Zaman Tasarrufu","Ekonomik sorunlar"]
    pairs=suggest_merges(names,threshold=.75)
    assert any({x["code_a"],x["code_b"]}=={"Zaman tasarrufu","Zaman Tasarrufu"} for x in pairs)
    assert all(x["review_required"] for x in pairs)

def test_distinct_codes_not_claimed_equivalent():
    assert suggest_merges(["Öğretmen desteği","Teknoloji maliyeti"],threshold=.90)==[]

def test_approved_merge_preserves_evidence():
    original=[{"participant_id":"T1","code":"Zaman Tasarrufu","quote":"Daha hızlı","start":0,"end":10,
               "category":"Verim","theme":"Faydalar"}]
    merged=merge_approved_codes(original,{"Zaman Tasarrufu":"Zaman tasarrufu"})
    assert merged[0]["code"]=="Zaman tasarrufu"
    assert merged[0]["quote"]==original[0]["quote"]
    assert original[0]["code"]=="Zaman Tasarrufu"

def test_limit():
    with pytest.raises(ValueError):
        suggest_merges(["a","b"],threshold=.2)
