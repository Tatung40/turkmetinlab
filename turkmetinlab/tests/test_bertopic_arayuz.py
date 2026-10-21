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
    # Debug the actual topic computation before Streamlit's generic error wrapper.
    from tmlab.anlamsal import E5Kodlayici, BELGE_ONEKI
    from tmlab.bertopic_adapter import run as topic_run
    texts = [entry["metin"] for entry in rows]
    vectors = E5Kodlayici(folder).kodla(texts, BELGE_ONEKI)
    _, labels, info = topic_run(texts, vectors, n_topics=3, min_df=1)
    assert len(labels) == len(texts)
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
