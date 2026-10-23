import pytest
from tmlab.nitel_local_model import local_completion

@pytest.mark.parametrize("url", [
    "https://api.example.org/v1/chat/completions",
    "http://192.168.1.25:11434/v1/chat/completions",
    "http://example.com/v1/chat/completions",
    "http://127.0.0.1:11434/admin",
    "http://127.0.0.1:11434/v1/chat/completions?private=true",
])
def test_nonlocal_or_wrong_target_rejected(url):
    with pytest.raises(ValueError):
        local_completion("data",url,"test")

def test_loopback_protocol_with_mock(monkeypatch):
    import tmlab.nitel_local_model as mod
    import json
    class Fake:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,size):
            return json.dumps({"choices":[{"message":{"content":'{"codes":[]}'}}]}).encode()
    def mocked(req,timeout):
        body=json.loads(req.data)
        assert body["temperature"]==0
        assert body["max_tokens"]==650
        assert req.full_url == "http://127.0.0.1:11434/v1/chat/completions"
        return Fake()
    monkeypatch.setattr(mod,"urlopen",mocked)
    assert mod.local_completion("hello","http://127.0.0.1:11434/v1/chat/completions","local-model")=='{"codes":[]}'

def test_qwen3_no_think_and_other_models_unchanged(monkeypatch):
    import json
    import tmlab.nitel_local_model as mod
    captured=[]
    class Fake:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,size): return b'{"choices":[{"message":{"content":"{}"}}]}'
    def mocked(req,timeout):
        captured.append(json.loads(req.data)["messages"][0]["content"])
        return Fake()
    monkeypatch.setattr(mod,"urlopen",mocked)
    endpoint="http://127.0.0.1:11434/v1/chat/completions"
    mod.local_completion("Deneme",endpoint,"qwen3:4b")
    mod.local_completion("Deneme",endpoint,"another-model")
    assert captured==["/no_think\nDeneme","Deneme"]

# All transport tests below use mocks; no local or external server is contacted.
import json
from urllib.error import URLError, HTTPError
import tmlab.nitel_local_model as mod

def test_local_transport_has_no_proxy_or_redirect():
    from urllib.request import ProxyHandler
    opener = mod.urlopen.__self__
    assert not any(isinstance(h, ProxyHandler) and h.proxies for h in opener.handlers)
    guard = next(h for h in opener.handlers if isinstance(h, mod._NoRedirect))
    assert guard.redirect_request(None, None, 307, "", {}, "https://example.org") is None

@pytest.mark.parametrize("exc,fragment", [
    (URLError("private server data"), "aynı"),
    (PermissionError("private server data"), "aynı"),
    (TimeoutError(), "süre"),
    (HTTPError("http://127.0.0.1", 404, "", {}, None), "bulunamadı"),
    (HTTPError("http://127.0.0.1", 307, "", {}, None), "yönlendirmesi"),
])
def test_actionable_safe_errors(monkeypatch, exc, fragment):
    def fail(*a, **kw): raise exc
    monkeypatch.setattr(mod, "urlopen", fail)
    with pytest.raises(mod.LocalModelError, match=fragment) as error:
        mod.local_completion("sensitive participant text", "http://127.0.0.1:11434/v1/chat/completions", "qwen3:4b")
    assert "private server data" not in str(error.value)
    assert "sensitive participant text" not in str(error.value)

@pytest.mark.parametrize("payload", [
    b"not json", b"{}", b'{"choices":[]}',
    b'{"choices":[{"message":{"content":""}}]}',
    b'{"choices":[{"finish_reason":"length","message":{"content":"{}"}}]}',
])
def test_invalid_or_truncated_server_response(monkeypatch, payload):
    class Fake:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, size): return payload
    monkeypatch.setattr(mod, "urlopen", lambda *a, **kw: Fake())
    with pytest.raises(mod.LocalModelError):
        mod.local_completion("source", "http://127.0.0.1:11434/v1/chat/completions", "qwen3:4b")

@pytest.mark.parametrize("thinking", [False, True])
def test_qwen35_ollama_supported_control_and_bounded_budget(monkeypatch, thinking):
    captured = []
    class Fake:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def read(self,size):
            return b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}'
    def send(req, **kwargs):
        captured.append(json.loads(req.data))
        return Fake()
    monkeypatch.setattr(mod, "urlopen", send)
    assert mod.local_completion("Kaynak", "http://127.0.0.1:11434/v1/chat/completions",
        "qwen3.5:4b", max_tokens=1024, thinking=thinking) == "{}"
    payload = captured[0]
    assert payload["max_tokens"] == 1024
    assert payload["response_format"] == {"type":"json_object"}
    assert payload["stream"] is False
    assert "think" not in payload and "options" not in payload and "num_predict" not in payload
    assert payload.get("reasoning_effort") == (None if thinking else "none")
    assert "/no_think" not in payload["messages"][0]["content"]
    assert payload["messages"][0]["content"].endswith("Kaynak")
    assert len(captured) == 1

@pytest.mark.parametrize("content,reason", [
    ('{"codes":[]}', "length"),
    ("", "stop"),
    ('{"codes":[', "stop"),
    ('[]', "stop"),
    ('```json\\n{}\\n```', "stop"),
])
def test_qwen35_never_salvages_invalid_or_cut_off_answer(monkeypatch, content, reason):
    class Fake:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def read(self,size):
            return json.dumps({"choices":[{"finish_reason":reason,"message":{
                "content":content,"reasoning_content":"not evidence"}}]}).encode()
    monkeypatch.setattr(mod, "urlopen", lambda *a, **kw: Fake())
    with pytest.raises(mod.LocalModelError):
        mod.local_completion("Source", "http://127.0.0.1:11434/v1/chat/completions", "qwen3.5:4b")

@pytest.mark.parametrize("budget", [0, -1, 127, 4097, 1024.5])
def test_invalid_token_budget_rejected_before_transport(monkeypatch, budget):
    def forbidden(*a, **kw): pytest.fail("No request for invalid token limits")
    monkeypatch.setattr(mod,"urlopen",forbidden)
    with pytest.raises(ValueError):
        mod.local_completion("Source","http://127.0.0.1:11434/v1/chat/completions",
            "qwen3.5:4b",max_tokens=budget)

def test_qwen35_quote_mismatch_remains_failed(monkeypatch):
    from scripts.nitel_gercek_model_pilot import run
    class Fake:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def read(self,size):
            content = json.dumps({"codes":[{"quote":"invented","code":"Kod",
                "category":"Kategori","theme":"Tema"}]})
            return json.dumps({"choices":[{"finish_reason":"stop",
                "message":{"content":content}}]}).encode()
    monkeypatch.setattr(mod,"urlopen",lambda *a, **kw: Fake())
    result = run([("T01","Özgün ifade")], lambda prompt:
        mod.local_completion(prompt,"http://127.0.0.1:11434/v1/chat/completions","qwen3.5:4b"))
    assert (result["processed"],result["succeeded"],result["failed"]) == (1,0,1)
    assert result["proposals"] == []
    assert "birebir" in result["errors"][0]["reason"]
