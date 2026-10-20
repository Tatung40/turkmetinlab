"""Doğrulanmış E5-small ONNX FP32 dosyalarını kullanıcının Hugging Face deposuna yükler (staging uygulaması için).

Yalnızca iş akışında uyum ve staging testleri geçtikten sonra çalışır. Yüklenen: model_fp32.onnx, tokenizer.json,
onnx_kunye.json, README.md (model kartı: kaynak, MIT lisansı, dönüştürme yöntemi). INT8 yüklenmez.
Çıktı JSON'u, Streamlit "Secrets" için gereken değerleri (depo, sürüm, SHA-256) içerir. Tek deneme.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile

KART = """---
license: mit
base_model: intfloat/multilingual-e5-small
library_name: onnxruntime
language: [tr, multilingual]
---
# multilingual-e5-small — ONNX FP32 (TürkMetinLab 2.5 staging)

[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) modelinin ONNX FP32
dönüşümüdür (MIT lisansı; özgün telif hakkı sahiplerine aittir). Ağırlıklar değiştirilmemiş, nicemlenmemiştir.

- Dönüştürme: `torch.onnx.export` (opset 17), çıktı `last_hidden_state`; havuzlama ve normalizasyon modele gömülü
  değildir (dikkat maskeli ortalama + L2, istemci tarafında).
- Doğrulama: TürkMetinLab fizibilite çalışmasında PyTorch sürümüyle aynı sıralamaları üretti (20 ve 500 belge).
- Kaynak sürüm ve SHA-256 özetleri: `onnx_kunye.json`.
"""


def sha(yol):
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()


def ana():
    a = argparse.ArgumentParser()
    a.add_argument("--onnx-klasor", required=True)
    a.add_argument("--repo", required=True)
    a.add_argument("--gizli", default="false")
    a.add_argument("--cikti", required=True)
    args = a.parse_args()
    from huggingface_hub import HfApi

    kunye = json.load(open(os.path.join(args.onnx_klasor, "onnx_kunye.json"), encoding="utf-8"))
    for ad in ("model_fp32.onnx", "tokenizer.json"):
        if sha(os.path.join(args.onnx_klasor, ad)) != kunye["dosyalar"][ad]["sha256"]:
            raise SystemExit(f"{ad} künyeyle uyuşmuyor; yayın yapılmadı.")
    with tempfile.TemporaryDirectory() as d:
        for ad in ("model_fp32.onnx", "tokenizer.json", "onnx_kunye.json"):
            shutil.copy(os.path.join(args.onnx_klasor, ad), d)
        open(os.path.join(d, "README.md"), "w", encoding="utf-8").write(KART)
        api = HfApi()
        api.create_repo(args.repo, private=args.gizli.lower() == "true", exist_ok=True)
        bilgi = api.upload_folder(folder_path=d, repo_id=args.repo,
                                  commit_message="TürkMetinLab 2.5: doğrulanmış E5-small ONNX FP32")
    sonuc = {"TML_E5_ONNX_REPO": args.repo, "TML_E5_ONNX_REV": bilgi.oid,
             "TML_E5_ONNX_SHA256": kunye["dosyalar"]["model_fp32.onnx"]["sha256"],
             "gizli_depo": args.gizli.lower() == "true",
             "not": "Bu üç değeri Streamlit staging uygulamasının Secrets bölümüne yazın (gizli depoda HF_TOKEN da)."}
    json.dump(sonuc, open(args.cikti, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(sonuc, ensure_ascii=False))


if __name__ == "__main__":
    ana()
