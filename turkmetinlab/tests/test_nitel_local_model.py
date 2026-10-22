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
