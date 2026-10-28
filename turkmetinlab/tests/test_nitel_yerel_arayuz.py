"""UI integration uses simulated completion; it does not prove model inference."""
from pathlib import Path
import json
from streamlit.testing.v1 import AppTest

def test_local_batch_settings_and_review(tmp_path, monkeypatch):
    import pandas as pd
    import tmlab.nitel_local_model as adapter
    monkeypatch.setattr("streamlit.file_uploader", lambda *a, **kw: object())
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw:
        pd.DataFrame({"id":["T01"],"text":["Kaynak"]}))
    calls = []
    def complete(prompt, endpoint, model, **kwargs):
        calls.append((endpoint, model, kwargs))
        return json.dumps({"codes":[{"quote":"Kaynak","code":"Kod","category":"Kategori","theme":"Tema"}]})
    monkeypatch.setattr(adapter, "local_completion", complete)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "nitel_demo.py")).run()
    assert not app.exception
    next(b for b in app.button if b.label == "Seçilen kayıtları otomatik kodla").click().run()
    assert not app.exception
    assert calls == [("http://127.0.0.1:11434/v1/chat/completions", "qwen3:4b", {"timeout":180, "max_tokens":1024})]
    outcome = app.session_state["nitel_batch_result"]
    assert (outcome["processed"],outcome["succeeded"],outcome["failed"]) == (1,1,0)
    assert app.session_state["nitel_accepted"] == []
