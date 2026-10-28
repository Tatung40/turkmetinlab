"""Simulated local completion; proves the approval gate, not real model inference."""
import json
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ROWS = [{"participant_id": "K1", "quote": "İş yükü çok ağır", "code": "İş yükü", "category": "Zorluk",
         "theme": "Çalışma", "start": 0, "end": 16}]


def _app(monkeypatch, answer):
    import tmlab.nitel_local_model as adapter
    monkeypatch.setattr(adapter, "local_completion", lambda *a, **kw: answer)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "nitel_demo.py"), default_timeout=30)
    app.session_state["nitel_accepted"] = [dict(r) for r in ROWS]
    app.session_state["nitel_hierarchy_prompt"] = "istem"
    return app.run()


def _click(app, label):
    next(b for b in app.button if b.label == label).click().run()
    assert not app.exception


def test_model_hierarchy_waits_for_researcher_approval(monkeypatch):
    app = _app(monkeypatch, json.dumps({"mapping": [{"code": "İş yükü", "category": "Yük", "theme": "Mesleki stres"}]}))
    _click(app, "Yerel modelle kategori ve temaları otomatik öner")
    assert app.session_state["nitel_accepted"] == ROWS  # not applied yet
    assert app.session_state["nitel_hierarchy_proposal"] == {"İş yükü": ("Yük", "Mesleki stres")}
    _click(app, "Önerilen kategori/temaları onayla")
    assert [(r["category"], r["theme"], r["quote"]) for r in app.session_state["nitel_accepted"]] == \
        [("Yük", "Mesleki stres", "İş yükü çok ağır")]


def test_rejected_or_invalid_model_hierarchy_leaves_approved_codes(monkeypatch):
    app = _app(monkeypatch, json.dumps({"mapping": [{"code": "İş yükü", "category": "Yük", "theme": "Stres"}]}))
    _click(app, "Yerel modelle kategori ve temaları otomatik öner")
    _click(app, "Hiyerarşi önerisini reddet")
    assert app.session_state["nitel_accepted"] == ROWS and "nitel_hierarchy_proposal" not in app.session_state
    app = _app(monkeypatch, json.dumps({"mapping": [{"code": "Uydurma", "category": "X", "theme": "Y"}]}))
    _click(app, "Yerel modelle kategori ve temaları otomatik öner")
    assert app.session_state["nitel_accepted"] == ROWS and any("doğrulanamadı" in e.value for e in app.error)
