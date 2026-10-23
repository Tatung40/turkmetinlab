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
    assert result["succeeded"]==20
    assert result["failed"]==0
    assert len(result["proposals"])==20
    assert result["frequencies"]["themes"][0]["n_participants"]==20
    assert not result["errors"]
    assert result["quality"]["needs_researcher_review"]

# These are simulated adapter responses, never real Qwen inference.
import json
from urllib.error import URLError
import pytest
from scripts import nitel_gercek_model_pilot as pilot

def response(quote):
    return json.dumps({"codes":[{"quote":quote,"code":"Kod","category":"Kategori","theme":"Tema"}]})

@pytest.mark.parametrize("raw,expected", [
    ('{"codes":[]}', (1, 0)),
    ('not json', (0, 1)),
    ('{"codes":[{},{}]}', (0, 1)),
    (response("uydurma kanıt"), (0, 1)),
])
def test_record_validation_counts(raw, expected):
    result = run([("T01", "Kaynak")], lambda *a, **kw: raw)
    assert (result["succeeded"], result["failed"]) == expected
    assert result["processed"] == result["succeeded"] + result["failed"] == 1

def test_mixed_call_failure_and_success():
    calls = []
    def complete(prompt, *, hierarchy=False):
        calls.append(hierarchy)
        if hierarchy:
            return json.dumps({"mapping":[{"code":"Kod","category":"Kategori","theme":"Tema"}]})
        if prompt.endswith("Hatalı"):
            raise URLError("simulated offline")
        return response("Kaynak")
    result = run([("T01","Hatalı"), ("T02","Kaynak")], complete)
    assert (result["processed"], result["succeeded"], result["failed"]) == (2,1,1)
    assert result["errors"] == [{"participant_id":"T01","reason":"URLError"}]
    assert len(result["proposals"]) == 1
    assert calls == [False, False, True]
    # Keep the original participant denominator; do not silently exclude failures.
    assert result["frequencies"]["codes"][0]["percentage"] == 50

def test_partial_validation_and_hierarchy_error_are_separate():
    def complete(prompt, *, hierarchy=False):
        if hierarchy:
            raise ValueError("simulated invalid hierarchy")
        return json.dumps({"codes":[json.loads(response("Kaynak"))["codes"][0], {}, {}]})
    result = run([("T01","Kaynak")], complete)
    assert (result["processed"], result["succeeded"], result["failed"]) == (1,0,1)
    assert len(result["proposals"]) == 1
    assert len(result["errors"]) == 3
    assert result["errors"][-1]["participant_id"] == "_hierarchy"

@pytest.mark.parametrize("fails", [False, True])
def test_cli_persists_counts_and_returns_failure(tmp_path, monkeypatch, capsys, fails):
    source = tmp_path / "input.csv"
    source.write_text("katilimci_id,metin\nT01,Kaynak\n", encoding="utf-8")
    output = tmp_path / "output.csv"
    def complete(*args, **kwargs):
        if fails:
            raise URLError("simulated offline")
        return '{"codes":[]}'
    monkeypatch.setattr(pilot, "local_completion", complete)
    monkeypatch.setattr("sys.argv", ["pilot", "--model", "qwen3:4b",
        "--input", str(source), "--output", str(output), "--limit", "1"])
    assert pilot.main() == int(fails)
    console = json.loads(capsys.readouterr().out)
    saved = json.loads(output.with_suffix(".report.json").read_text(encoding="utf-8"))
    for report in (console, saved):
        assert (report["processed"], report["succeeded"], report["failed"]) == (1, int(not fails), int(fails))

@pytest.mark.parametrize("first", ['{"codes":[]}', 'not json', response("invented")])
def test_first_record_gate_stops_without_retries(first):
    calls = []
    def complete(*a, **kw):
        calls.append(kw)
        return first
    result = run([("T01", "Kaynak"), ("T02", "Başka")], complete, verify_first=True)
    assert len(calls) == 1
    assert result["processed"] == 1
    assert result["errors"][-1]["reason"] == "FirstRecordNotVerified"

def test_first_record_gate_reuses_first_result():
    calls = []
    def complete(prompt, *, hierarchy=False):
        calls.append(hierarchy)
        if hierarchy:
            return json.dumps({"mapping":[{"code":"Kod","category":"Kategori","theme":"Tema"}]})
        return response(prompt.split("KAYNAK:\n")[1])
    result = run([("T01","Kaynak"), ("T02","Başka")], complete, verify_first=True)
    assert calls == [False, False, True]
    assert (result["processed"], result["succeeded"], result["failed"]) == (2,2,0)
    assert len(result["proposals"]) == 2

def test_hierarchy_transport_error_preserves_outputs():
    from tmlab.nitel_local_model import LocalModelError
    def complete(prompt, *, hierarchy=False):
        if hierarchy: raise LocalModelError("Yerel modele erişilemedi.")
        return response("Kaynak")
    result = run([("T01","Kaynak")], complete)
    assert result["succeeded"] == 1 and result["failed"] == 0
    assert result["errors"] == [{"participant_id":"_hierarchy","reason":"LocalModelError",
        "message":"Yerel modele erişilemedi."}]
    assert len(result["proposals"]) == 1
