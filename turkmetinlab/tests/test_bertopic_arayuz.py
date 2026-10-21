"""End-to-end BERTopic UI test in isolated staging CI.

Uses real BERTopic and a deterministic tiny ONNX fixture, not semantic E5 weights.
"""
import os
import io
import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

pytest.importorskip("bertopic")
from anlamsal_model import kucuk_model_olustur

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY=os.path.join(ROOT,"staging","streamlit_app.py")

def test_bertopic_pipeline_in_staging(monkeypatch,tmp_path):
    folder=str(tmp_path/"e5")
    kucuk_model_olustur(folder,azami_token=512)
    monkeypatch.setenv("TML_E5_ONNX_YEREL",folder)
    rows=[]
    vocab=[
      "iklim kuraklık yağmur baraj çevre atmosfer tarım",
      "eğitim sınıf öğretmen matematik öğrenci sınav öğrenme",
      "hastane hekim sağlık hasta tedavi doktor ilaç",
    ]
    for i in range(30):
        rows.append({"kimlik":f"yorum{i+1}", "metin":vocab[i%3]+f" araştırma {i%5}"})
    blob=pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    at=AppTest.from_file(ENTRY,default_timeout=300).run()
    assert not at.exception,[e.value for e in at.exception]
    at.radio(key="kaynak").set_value("Kendi dosyamı yükle").run()
    at.file_uploader[0].set_value(("yorum.csv",blob,"text/csv")).run()
    assert not at.exception,[e.value for e in at.exception]
    assert next((b for b in at.button if b.key=="bertopic_run"),None) is not None
    at.slider(key="bertopic_n").set_value(3).run()
    at.button(key="bertopic_run").click().run()
    assert not at.exception,[e.value for e in at.exception]
    assert any(b.key=="bertopic_topics_csv" for b in at.get("download_button"))
    assert any(b.key=="bertopic_docs_csv" for b in at.get("download_button"))
