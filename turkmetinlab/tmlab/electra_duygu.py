"""Experimental Turkish sentiment ELECTRA-small ONNX inference, CPU only.

Source: https://huggingface.co/ayse-solmaz/turkish-sentiment-electra-small
Model card: 13.7M parameters, 3 classes, 64-token training limit.
No user text sent to external services. We download public model artifacts only.
"""
from __future__ import annotations
import json
import urllib.request
import numpy as np
from functools import lru_cache
from scipy.special import softmax

REPO = "ayse-solmaz/turkish-sentiment-electra-small"
REV = "main"
BASE = "https://huggingface.co/" + REPO + "/resolve/" + REV + "/"
MAX_SIZES = {"onnx/model_quantized.onnx": 24_000_000,
             "tokenizer.json": 5_000_000, "config.json": 50_000}
MAX_TOKENS = 64
MAX_DOCUMENTS = 100
DEFAULT_LABELS = ("Negative","Neutral","Positive")
# REV is a moving branch, not a pinned commit: record it with every prediction.
MODEL_ID = REPO + "@" + REV

def _fetch(filename: str, limit: int) -> bytes:
    req = urllib.request.Request(BASE+filename, headers={"User-Agent":"TurkMetinLab-2.5-evaluation"})
    with urllib.request.urlopen(req,timeout=35) as response:
        data = response.read(limit+1)
    if len(data)>limit: raise ValueError("Model dosyası beklenen boyuttan büyük")
    return data

@lru_cache(maxsize=1)
def load_model():
    """Load immutable/public model into process cache; no user data is cached."""
    import onnxruntime as ort
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_str(_fetch("tokenizer.json",MAX_SIZES["tokenizer.json"]).decode("utf-8"))
    metadata = json.loads(_fetch("config.json",MAX_SIZES["config.json"]))
    labels = {int(k):str(v) for k,v in metadata.get("id2label",{}).items()}
    if not labels: labels = dict(enumerate(DEFAULT_LABELS))
    if sorted(labels)!=list(range(len(labels))):
        raise ValueError("Model id2label sırası 0..n-1 değil")
    # Library truncation keeps [CLS]/[SEP]; slicing ids would drop the closing special token.
    tokenizer.enable_truncation(max_length=MAX_TOKENS)
    # Load public model weights entirely in memory; no disk writes.
    model_bytes=_fetch("onnx/model_quantized.onnx",MAX_SIZES["onnx/model_quantized.onnx"])
    try:
        session=ort.InferenceSession(model_bytes,providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ValueError("INT8 ONNX modeli bu ortamda açılamadı") from exc
    return tokenizer,session,labels

def infer(text: str, state, max_length: int = MAX_TOKENS):
    tokenizer,session,labels=state
    if not isinstance(text,str): text=str(text)
    # The training task used short texts; clip rather than making unvalidated long-document claims.
    if (tokenizer.truncation or {}).get("max_length")!=max_length:
        tokenizer.enable_truncation(max_length=max_length)
    encoding=tokenizer.encode(text)
    ids=list(encoding.ids)
    if not ids: raise ValueError("Boş metin")
    feeds={}
    for item in session.get_inputs():
        if item.name=="input_ids": vals=ids
        elif item.name=="attention_mask": vals=[1]*len(ids)
        elif item.name=="token_type_ids": vals=list(encoding.type_ids[:len(ids)])
        else: raise ValueError("Bilinmeyen ONNX girişi: "+item.name)
        feeds[item.name]=np.asarray([vals],dtype=np.int64)
    logits=np.asarray(session.run(None,feeds)[0],dtype=np.float64)[0]
    if len(logits)!=len(labels):
        raise ValueError("Model çıktı sayısı etiket sayısıyla uyuşmuyor")
    scores=softmax(logits)
    ix=int(np.argmax(scores))
    return {"Etiket":labels[ix],
            "En yüksek model skoru":round(float(scores[ix]),4),
            "İşlenen token":len(ids),"Kesildi mi":bool(encoding.overflowing),
            "Model":MODEL_ID,
            **{f"Skor: {labels[i]}":round(float(x),4) for i,x in enumerate(scores)}}

def classify_corpus(texts,names,state):
    if len(texts)>MAX_DOCUMENTS: raise ValueError("Deneysel CPU için en fazla 100 belge")
    return [{"Belge":str(name),**infer(str(text),state)} for name,text in zip(names,texts)]
