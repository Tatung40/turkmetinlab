"""SWNetTR++ parser and sentiment evidence tests — no network required."""
from tmlab.duygu import parse_lexicon, analyze, analyze_corpus

def test_parse_semicolon_comma_decimal_and_readme_preamble():
    data = b"README\nWORD;TONE;POLARITY\niyi;0,5;1\nkotu;-0,5;-1\ncok iyi;0,8;1\n"
    d = parse_lexicon(data)
    assert d[("iyi",)] == (0.5, 1)
    assert d[("cok","iyi")] == (0.8, 1)

def test_longest_match_boundaries_and_no_guessing():
    d = parse_lexicon("WORD;TONE;POLARITY\niyi;0,5;1\nçok iyi;0,8;1\nkötü;-0,6;-1\n".encode())
    out = analyze("ÇOK İYİ ama kötüydü; kötü.", d)
    assert out["Eşleşen ifade"] == 2
    assert out["Olumlu eşleşme"] == 1
    assert out["Olumsuz eşleşme"] == 1
    assert out["Eşleşmeler"][0]["Sözcük/ifade"] == "ÇOK İYİ"

def test_unknown_and_empty():
    d = {("iyi",):(0.5,1)}
    assert analyze("bilinmeyen",d)["Kapsama (%)"] == 0
    assert analyze("",d)["Eşleşen ifade"] == 0
    summaries, details = analyze_corpus(["iyi",""],["A","B"],d)
    assert len(summaries) == 2 and len(details) == 1

def test_invalid_data():
    try:
        parse_lexicon(b"not a lexicon")
    except ValueError:
        pass
    else:
        assert False
