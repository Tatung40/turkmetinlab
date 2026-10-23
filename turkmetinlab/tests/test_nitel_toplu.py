import json
from tmlab.nitel_toplu import batch_analyze, regroup_once

def test_batch_proposals_are_separate_from_approval():
    src=[("T01","Ders hazırlama sürem azaldı."),("T02","Yanlış bilgiler var.")]
    def fake(prompt):
        if "Ders hazırlama" in prompt:
            return json.dumps({"codes":[{"quote":"sürem azaldı","code":"Zaman tasarrufu",
                "category":"Verim","theme":"Faydalar"}]},ensure_ascii=False)
        return '{"codes":[{"quote":"Yanlış bilgiler","code":"Güvensizlik", "category":"Risk","theme":"Riskler"}]}'
    result=batch_analyze(src,fake)
    assert result["processed"]==2
    assert len(result["proposals"])==2
    assert result["requires_researcher_approval"]
    assert not result["errors"]
    assert result["hierarchy"]["denominator"]==2

def test_malformed_output_is_reported_not_copied():
    out=batch_analyze([("T01","Güzel metin.")],lambda prompt:"not json")
    assert out["processed"]==1 and out["proposals"]==[]
    assert out["errors"][0]["participant_id"]=="T01"

def test_grouping_requires_complete_code_coverage():
    data=batch_analyze([("T01","İşler hızlandı.")],
        lambda p:'{"codes":[{"quote":"hızlandı","code":"Hız","category":"İş","theme":"Fayda"}]}')
    def model(prompt):
        assert "İşler hızlandı" not in prompt
        return '{"mapping":[{"code":"Hız","category":"Verimlilik","theme":"Katkılar"}]}'
    grouped=regroup_once(data,model)
    assert grouped["proposals"][0]["theme"]=="Katkılar"
    assert data["proposals"][0]["theme"]=="Fayda"
