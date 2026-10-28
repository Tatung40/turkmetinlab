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

@pytest.mark.parametrize("verify_first", [False, True])
@pytest.mark.parametrize("records", [
    [("T01","Aynı metin"), ("T01","Aynı metin")],
    [("T01","İlk görüş"), ("T01","Farklı görüş")],
    [("T01","İlk"), ("T02","İkinci"), ("T01","Üçüncü")],
    [("T01","İlk"), ("T02","İkinci"), ("T02","Üçüncü")],
    [(" T01 ","İlk"), ("T01","İkinci")],
])
def test_duplicate_ids_rejected_before_any_model_call(records, verify_first):
    original = list(records)
    def forbidden(*a, **kw):
        pytest.fail("Invalid batch must not call the model")
    with pytest.raises(ValueError, match="benzersiz"):
        run(records, forbidden, verify_first=verify_first)
    assert records == original  # Neither identical nor different texts are dropped.

def test_verified_batch_counts_and_percentages_after_one_failure():
    calls = []
    def complete(prompt, *, hierarchy=False):
        calls.append(hierarchy)
        if hierarchy:
            return json.dumps({"mapping":[{"code":"Kod","category":"Kategori","theme":"Tema"}]})
        source = prompt.split("KAYNAK:\n")[1]
        if source == "Hatalı":
            raise URLError("simulated failure")
        return response(source)
    result = run([("T01","Birinci"), ("T02","Hatalı"), ("T03","Üçüncü")],
                 complete, verify_first=True)
    assert (result["processed"], result["succeeded"], result["failed"]) == (3,2,1)
    assert result["participant_ids"] == ["T01","T02","T03"]
    assert calls == [False,False,False,True]
    assert len(result["proposals"]) == 2
    assert result["errors"] == [{"participant_id":"T02","reason":"URLError"}]
    assert result["quality"]["participants"] == 3
    assert result["quality"]["assignments"] == 2
    for level in ("codes","categories","themes"):
        row = result["frequencies"][level][0]
        assert (row["n_participants"], row["n_evidence"], row["percentage"]) == (2,2,66.67)

def test_duplicate_csv_does_not_overwrite_input_or_existing_outputs(tmp_path, monkeypatch, capsys):
    source = tmp_path / "input.csv"
    data = "katilimci_id,metin\nT01,Birinci\nT01,İkinci\n"
    source.write_text(data, encoding="utf-8")
    output = tmp_path / "output.csv"
    report = output.with_suffix(".report.json")
    output.write_text("previous csv", encoding="utf-8")
    report.write_text("previous report", encoding="utf-8")
    def forbidden(*a, **kw):
        pytest.fail("Duplicate CSV must not call the model")
    monkeypatch.setattr(pilot, "local_completion", forbidden)
    monkeypatch.setattr("sys.argv", ["pilot","--model","qwen3:4b","--limit","20",
        "--input",str(source),"--output",str(output)])
    with pytest.raises(SystemExit) as error:
        pilot.main()
    assert error.value.code == 2
    assert "benzersiz" in capsys.readouterr().err
    assert source.read_text(encoding="utf-8") == data
    assert output.read_text(encoding="utf-8") == "previous csv"
    assert report.read_text(encoding="utf-8") == "previous report"

def test_cli_qwen35_explicit_token_limits(tmp_path, monkeypatch, capsys):
    source = tmp_path / "input.csv"
    source.write_text("katilimci_id,metin\nT01,Kaynak\n",encoding="utf-8")
    output = tmp_path / "result.csv"
    calls = []
    def complete(prompt, endpoint, model, **kwargs):
        calls.append((model, kwargs))
        return '{"codes":[]}'
    monkeypatch.setattr(pilot,"local_completion",complete)
    monkeypatch.setattr("sys.argv",["pilot","--input",str(source),"--output",str(output),
        "--model","qwen3.5:4b","--max-tokens","1536","--hierarchy-max-tokens","2048","--limit","1"])
    assert pilot.main() == 0
    assert calls == [("qwen3.5:4b",{"timeout":180,"max_tokens":1536})]
    report = json.loads(output.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert report["max_tokens"] == 1536 and report["hierarchy_max_tokens"] == 2048
    assert report["status"] == "no_codes"


def test_script_runs_as_documented_file_path():
    """`python scripts/nitel_gercek_model_pilot.py` must import tmlab (no model call: --help only)."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).parents[1]
    done = subprocess.run([sys.executable, "scripts/nitel_gercek_model_pilot.py", "--help"], cwd=root,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
