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
