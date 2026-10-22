"""Optional OpenAI-compatible LOCAL model connector for qualitative suggestions.

Only loopback targets are allowed: no cloud API fees or third-party data transfer.
A local Ollama / LM Studio server can expose /v1/chat/completions.
"""
from __future__ import annotations
import ipaddress
import json
from urllib.parse import urlparse
from urllib.request import Request, urlopen

MAX_RESPONSE_BYTES = 65536

def local_completion(prompt: str, endpoint: str, model: str, *, timeout: int = 90, max_tokens: int = 650, thinking: bool = False) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Yerel model adresi yalnızca http://127.0.0.1:PORT/v1/chat/completions olabilir.")
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
    except ValueError as exc:
        raise ValueError("IP adresi açıkça belirtilmiş olmalı.") from exc
    if not address.is_loopback or parsed.port is None or parsed.path != "/v1/chat/completions":
        raise ValueError("Yalnızca yerel loopback model sunucusuna izin verilir.")
    if not model or len(model) > 150 or not (1 <= timeout <= 180) or not (128 <= max_tokens <= 4096):
        raise ValueError("Model adı veya zaman aşımı geçersiz.")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens, "stream": False}).encode("utf-8")
    if len(body) > 60000:
        raise ValueError("Model istemi çok uzun.")
    if thinking is False and 'qwen3' in model.lower():
        # Qwen3 için düşünme metnini kapat; desteklemeyen sunucular bu alanı yok sayabilir.
        payload = json.loads(body)
        payload["messages"][0]["content"] = "/no_think\n" + prompt
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("Model yanıtı boyut sınırını aşıyor.")
    data = json.loads(payload)
    answer = data["choices"][0]["message"]["content"]
    if not isinstance(answer, str):
        raise ValueError("Model yanıtı metin değil.")
    return answer
