"""TürkMetinLab 2.5 STAGING — deneysel anlamsal analiz sekmesi açık. Canlı uygulama (app.py) değişmeden çalışır.

Streamlit Community Cloud'da ayrı bir uygulama olarak, bu dosya giriş noktası seçilerek dağıtılır. Bu klasördeki
requirements.txt kullanılır. Model kaynağı Streamlit "Secrets" bölümünden okunur (README.md).
"""
import os
import runpy
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KOK not in sys.path:
    sys.path.insert(0, KOK)
os.environ["TML_DENEYSEL_ANLAMSAL"] = "1"

try:  # Streamlit Secrets → ortam değişkenleri (yalnızca model kaynağı ayarları)
    import streamlit as st

    for _ad in ("TML_E5_ONNX_REPO", "TML_E5_ONNX_REV", "TML_E5_ONNX_SHA256", "HF_TOKEN", "TML_E5_BELGE_SINIRI"):
        if not os.environ.get(_ad) and _ad in st.secrets:
            os.environ[_ad] = str(st.secrets[_ad])
except Exception:  # secrets dosyası yoksa: sekme, eksik yapılandırmayı kullanıcıya bildirir
    pass

runpy.run_path(os.path.join(KOK, "app.py"), run_name="__main__")
