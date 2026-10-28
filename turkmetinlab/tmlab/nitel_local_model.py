"""Optional OpenAI-compatible LOCAL model connector for qualitative suggestions.

Only loopback targets are allowed: no cloud API fees or third-party data transfer.
A local Ollama / LM Studio server can expose /v1/chat/completions.
"""
from __future__ import annotations
import ipaddress
import json
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
from urllib.error import HTTPError, URLError


class LocalModelError(RuntimeError):
    """Safe diagnostic without source text or server response bodies."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# Loopback requests stay on this machine, including on proxy-configured PCs.
# Redirects must never forward participant text outside the validated endpoint.
urlopen = build_opener(ProxyHandler({}), _NoRedirect()).open

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
    if not model or len(model) > 150 or not (1 <= timeout <= 900) or type(max_tokens) is not int or not (128 <= max_tokens <= 4096):
        raise ValueError("Model adı veya zaman aşımı geçersiz.")
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": max_tokens, "stream": False,
               "response_format": {"type": "json_object"}}
    if "qwen3.5" in model.lower():
        # Ollama /v1: reasoning_effort=none maps to think=false.
        # Do not send native /api/chat-only think/options fields here.
        if thinking is False:
            payload["reasoning_effort"] = "none"
        payload["messages"][0]["content"] = (
            "Yalnızca istenen şemaya uygun kısa JSON nesnesi döndür. "
            "Açıklama, düşünme metni veya Markdown ekleme. Etiketleri kısa tut; "
            "kanıt alıntısını değiştirme.\n" + prompt)
    elif thinking is False and "qwen3" in model.lower():
        payload["messages"][0]["content"] = "/no_think\n" + prompt
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(body) > 60000:
        raise ValueError("Model istemi çok uzun.")
    req = Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        if exc.code == 404:
            message = "Model veya /v1/chat/completions bulunamadı; Ollama model adını ve sunucu adresini kontrol edin."
        elif 300 <= exc.code < 400:
            message = "Yerel sunucu yönlendirmesi engellendi; doğrudan loopback adresini kullanın."
        else:
            message = f"Yerel model HTTP {exc.code} hatası verdi; sunucu ayarlarını kontrol edin."
        raise LocalModelError(message) from exc
    except (TimeoutError, URLError, OSError) as exc:
        raise LocalModelError(
            "Yerel modele erişilemedi veya süre doldu. Ollama/llama.cpp ve Streamlit aynı "
            "bilgisayarda çalışmalı; model adı, port ve CPU için zaman aşımını kontrol edin. "
            "Streamlit Cloud bilgisayarınızdaki localhost adresine erişemez."
        ) from exc
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("Model yanıtı boyut sınırını aşıyor.")
    try:
        data = json.loads(payload)
        choice = data["choices"][0]
        answer = choice["message"]["content"]
        if choice.get("finish_reason") == "length":
            raise LocalModelError("Model yanıtı token sınırında kesildi; kodlama tamamlanmadı. "
                                  "Ollama sürümünü/düşünme ayarını kontrol edin; gerekirse yanıt token "
                                  "sınırını kontrollü artırıp tek görüşü yeniden deneyin.")
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LocalModelError("Sunucu geçerli chat completions JSON yanıtı döndürmedi.") from exc
    if not isinstance(answer, str) or not answer.strip():
        raise LocalModelError("Model yanıtı boş veya metin değil.")
    try:
        if not isinstance(json.loads(answer), dict):
            raise ValueError("JSON object required")
    except (ValueError, TypeError) as exc:
        raise LocalModelError("Model yanıtı eksik veya geçersiz JSON nesnesi; kodlama kabul edilmedi.") from exc
    return answer
