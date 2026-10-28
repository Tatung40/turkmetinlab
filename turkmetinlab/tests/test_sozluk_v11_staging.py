"""Staging-only lexical integration smoke checks; no Streamlit import required."""
from pathlib import Path
import ast
from tmlab.akademik_terimler import load_lexicon
from tmlab import terimler as tm

def test_staging_only_opt_in_guards():
    app = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(app.read_text(encoding="utf-8"))
    funcs = {x.name: x for x in tree.body if isinstance(x, ast.FunctionDef)}
    assert {"_etkin_sozluk", "_terim_kenar_cubugu"} <= funcs.keys()
    apptext = app.read_text(encoding="utf-8")
    assert 'if DENEYSEL_ANLAMSAL and ayar.get("deneysel"):' in apptext
    assert '"Akademik Sözlük v1.1 — deneysel terimleri de ekle", False' in apptext

def test_lexicon_compatible_with_existing_matcher():
    d = load_lexicon()
    accepted = []
    for i, x in enumerate(d["terimler"], 1):
        canonical = str(x["kanonik"]).strip()
        term = str(x["terim"]).strip()
        if not canonical or not term or len(canonical.split()) < 2:
            continue
        t = tm.kullanici_terimi(canonical, term if tm.katla(term) != tm.katla(canonical) else "",
                               terim_id=f"DENEYSEL11_{i:03d}", alan="Akademik v1.1 (deneysel)")
        accepted.append(t)
    assert len(accepted) >= 80
    matcher = tm.TerimEslestirici(accepted, False, "birim")
    assert len(matcher) >= 80
