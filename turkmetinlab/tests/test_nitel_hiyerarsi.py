import json
import pytest
from tmlab.nitel_hiyerarsi import build_hierarchy, prepare_hierarchy_prompt, validate_hierarchy_response, apply_hierarchy

RECORDS=[
 {"participant_id":"T01","quote":"A","code":"Zaman tasarrufu","category":"Verim","theme":"Yarar","start":0,"end":1},
 {"participant_id":"T01","quote":"B","code":"Zaman tasarrufu","category":"Verim","theme":"Yarar","start":2,"end":3},
 {"participant_id":"T02","quote":"C","code":"Hatalı bilgi","category":"Güvenilirlik","theme":"Riskler","start":0,"end":1},
]

def test_hierarchy_counts_participants_not_quotes():
    result=build_hierarchy(RECORDS,["T01","T02","T03"])
    assert result["denominator"]==3
    row=next(r for r in result["mapping"] if r["code"]=="Zaman tasarrufu")
    assert row["n_evidence"]==2
    assert row["n_participants"]==1
    assert row["percentage"]==33.33
    assert len(result["themes"])==2

def test_single_low_token_prompt_without_private_quotes():
    p=prepare_hierarchy_prompt(RECORDS)
    assert "Zaman tasarrufu" in p
    assert '"quote"' not in p and '"participant_id"' not in p

def test_hierarchy_llm_must_cover_existing_codes_once():
    data={"mapping":[{"code":"Zaman tasarrufu","category":"Öğretim verimliliği","theme":"Faydalar"},
                     {"code":"Hatalı bilgi","category":"Güven","theme":"Riskler"}]}
    mapping=validate_hierarchy_response(json.dumps(data,ensure_ascii=False),
                                        ["Zaman tasarrufu","Hatalı bilgi"])
    edited=apply_hierarchy(RECORDS,mapping)
    assert edited[0]["quote"]=="A" and edited[0]["theme"]=="Faydalar"
    assert edited[0]["start"]==0
    with pytest.raises(ValueError):
        validate_hierarchy_response(json.dumps({"mapping":data["mapping"][:1]}),
                                    ["Zaman tasarrufu","Hatalı bilgi"])
    with pytest.raises(ValueError):
        validate_hierarchy_response(json.dumps({"mapping":data["mapping"]*2}),
                                    ["Zaman tasarrufu","Hatalı bilgi"])

def test_conflict_is_flagged_not_silently_resolved():
    changed=[dict(RECORDS[0],theme="Başka tema"), RECORDS[1]]
    result=build_hierarchy(changed,["T01"])
    assert result["conflicting_codes"]==["Zaman tasarrufu"]


def test_hierarchy_changes_lists_only_differences_and_never_mutates():
    from tmlab.nitel_hiyerarsi import hierarchy_changes
    rows = [{"participant_id": "K1", "quote": "İş yükü çok ağır", "code": "İş yükü", "category": "Zorluk", "theme": "Çalışma"},
            {"participant_id": "K2", "quote": "Öğrenciler ilgili", "code": "İlgi", "category": "Motivasyon", "theme": "Öğrenme"}]
    before = [dict(r) for r in rows]
    mapping = {"İş yükü": ("Zorluk", "Çalışma"), "İlgi": ("Katılım", "Öğrenme")}
    assert hierarchy_changes(rows, mapping) == [{"code": "İlgi", "current_category": "Motivasyon",
        "current_theme": "Öğrenme", "proposed_category": "Katılım", "proposed_theme": "Öğrenme"}]
    assert rows == before
    import pytest
    with pytest.raises(ValueError):
        hierarchy_changes(rows, {"İş yükü": ("Zorluk", "Çalışma")})
