import json
import pytest
from tmlab.nitel_cekirdek import (
    EvidenceCode, analyze_model_response, document_fingerprint,
    frequencies, parse_response, prepare_prompt, validate_codes,
)


def test_turkish_evidence_and_offsets():
    text = "Öğretmenler zaman kazanıyor. Ancak hatalı bilgiler kaygı yaratıyor."
    candidate = [{"quote": "Ancak hatalı bilgiler kaygı yaratıyor.",
                  "code": "Güven sorunu", "category": "Sınırlılıklar",
                  "theme": "YZ riskleri"}]
    good, bad = validate_codes("T01", text, candidate)
    assert bad == []
    assert text[good[0].start:good[0].end] == candidate[0]["quote"]


def test_hallucinated_evidence_is_rejected():
    source = "Ders hazırlarken zaman kazanıyorum."
    good, bad = validate_codes("T02", source, [
        {"quote": "İşimi kaybettim.", "code": "İş kaybı",
         "category": "Risk", "theme": "Ekonomi"}])
    assert not good
    assert len(bad) == 1


def test_json_only_and_limit():
    with pytest.raises(ValueError):
        parse_response("Bence:\n" + json.dumps({"codes": []}))
    with pytest.raises(ValueError):
        parse_response(json.dumps({"codes": [{}] * 13}))
    assert parse_response('{"codes":[]}') == []


def test_prompt_is_compact_with_codebook():
    prompt = prepare_prompt("Yapay zekâ bazen yanılıyor.", ["Güven", "Fayda"])
    assert "Güven" in prompt
    assert "HARFİYEN" in prompt
    with pytest.raises(ValueError):
        prepare_prompt("x" * 12001)


def test_frequency_is_participant_based():
    rows = [
        EvidenceCode("T01", "A", "Zaman", "Verim", "Katkı", 0, 1),
        EvidenceCode("T01", "B", "Zaman", "Verim", "Katkı", 2, 3),
        EvidenceCode("T02", "C", "Risk", "Sınırlılık", "Tehlike", 0, 1),
    ]
    result = {r["code"]: r for r in frequencies(rows)}
    assert result["Zaman"]["n_participants"] == 1
    assert result["Zaman"]["percent"] == 50.0


def test_end_to_end_no_model_cost():
    source = "Öğretmen yapay zekâ ile zamandan tasarruf sağlıyor."
    payload = json.dumps({"codes": [{"quote": "zamandan tasarruf", "code": "Zaman",
                 "category": "Verim", "theme": "Fayda"}]}, ensure_ascii=False)
    result = analyze_model_response("T01", source, payload)
    assert result["requires_human_review"]
    assert len(result["codes"]) == 1
    assert len(document_fingerprint(source)) == 64

def test_deductive_codebook_blocks_unapproved_codes():
    text = "Ders hazırlamak daha kısa sürüyor."
    payload = '{"codes":[{"quote":"Ders hazırlamak","code":"Rastgele","category":"İş","theme":"Süre"}]}'
    result = analyze_model_response("T04", text, payload, ["Zaman tasarrufu"])
    assert result["codes"] == []
    assert len(result["rejected"]) == 1


def test_frequency_includes_uncoded_participants():
    rows = [EvidenceCode("T01","evet","Fayda","Değer","Yarar",0,4)]
    report = frequencies(rows, ["T01","T02","T03","T04"])
    assert report[0]["n_participants"] == 1
    assert report[0]["percent"] == 25.0
