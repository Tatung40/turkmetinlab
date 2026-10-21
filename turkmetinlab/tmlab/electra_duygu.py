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

REPO = "ayse-solmaz/turkish-sentiment-electra-small"
REV = "main"
BASE = "https://huggingface.co/" + REPO + "/resolve/" + REV + "/"
MAX_SIZES = {"onnx/model_quantized.onnx": 24_000_000,
             "tokenizer.json": 5_000_000, "config.json": 50_000}
MAX_TOKENS = 64
MAX_DOCUMENTS = 100
DEFAULT_LABELS = ("Negative","Neutral","Positive")

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
    encoding=tokenizer.encode(text)
    ids=list(encoding.ids[:max_length])
    if not ids: raise ValueError("Boş metin")
    feeds={}
    for item in session.get_inputs():
        if item.name=="input_ids": vals=ids
        elif item.name=="attention_mask": vals=[1]*len(ids)
        elif item.name=="token_type_ids": vals=list(encoding.type_ids[:len(ids)])
        else: raise ValueError("Bilinmeyen ONNX girişi: "+item.name)
        feeds[item.name]=np.asarray([vals],dtype=np.int64)
    logits=np.asarray(session.run(None,feeds)[0])[0]
    shifted=logits-np.max(logits)
    scores=np.exp(shifted);scores=scores/scores.sum()
    ix=int(np.argmax(scores))
    return {"Etiket":labels.get(ix,str(ix)),
            "En yüksek model skoru":round(float(scores[ix]),4),
            "İşlenen token":len(ids),"Kesildi mi":len(encoding.ids)>len(ids),
            **{f"Skor: {labels.get(i,str(i))}":round(float(x),4) for i,x in enumerate(scores)}}

def classify_corpus(texts,names,state):
    if len(texts)>MAX_DOCUMENTS: raise ValueError("Deneysel CPU için en fazla 100 belge")
    return [{"Belge":str(name),**infer(str(text),state)} for name,text in zip(names,texts)]
