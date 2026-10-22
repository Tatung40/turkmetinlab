import csv
from pathlib import Path
from scripts.nitel_gercek_model_pilot import run

def test_real_model_pilot_runner_with_mock_response():
    f=Path(__file__).parent/"fixtures"/"nitel_20_gorusme.csv"
    with f.open(encoding="utf-8-sig",newline="") as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==20
    def complete(prompt,*,hierarchy=False):
        if hierarchy:
            import json
            codes=["Pilot kodu"]
            return json.dumps({"mapping":[{"code":x,"category":"Pilot kategori","theme":"Pilot tema"} for x in codes]})
        source=prompt.split("KAYNAK:\n",1)[1]
        import json
        return json.dumps({"codes":[{"quote":source,"code":"Pilot kodu",
                         "category":"Ön kategori","theme":"Ön tema"}]},ensure_ascii=False)
    result=run([(r["katilimci_id"],r["metin"]) for r in rows],complete)
    assert result["processed"]==20
    assert len(result["proposals"])==20
    assert result["frequencies"]["themes"][0]["n_participants"]==20
    assert not result["errors"]
    assert result["quality"]["needs_researcher_review"]
