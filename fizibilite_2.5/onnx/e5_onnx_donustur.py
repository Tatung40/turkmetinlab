"""E5-small → ONNX (FP32) ve dinamik INT8 nicemleme. Yalnızca dönüştürmede torch kullanılır.

    python fizibilite_2.5/onnx/e5_onnx_donustur.py --cikti ONNX_KLASORU [--model intfloat/multilingual-e5-small]
                                                    [--onbellek HF_ONBELLEK] [--kunye sonuc.json]

Çıktı klasörü: model_fp32.onnx, model_int8.onnx, tokenizer.json, onnx_kunye.json (boyut, SHA-256, dışa aktarıcı).
INT8 başarısız olursa hata künyeye yazılır, INT8 elenir; FP32 başarısızsa 1 ile çıkılır.
--onbellekten-devam: klasördeki dosyalar künyeyle (SHA-256, kaynak sürüm) doğrulanır; eksik/uyumsuzsa yeniden
oluşturulur. Yeni dosyalar geçici klasörde üretilir ve yalnızca tamamlanınca yerine taşınır.
Pooling ve normalizasyon modele gömülmez; kodlayıcıda (e5_onnx_kodlayici.py) sentence-transformers ile aynı
biçimde yapılır: dikkat maskesiyle ortalama havuzlama + L2 normalizasyonu.
"""
import argparse
import hashlib
import json
import os
import shutil
import time


KUNYE_SURUMU = 2


def onbellek_denetle(klasor, kaynak_surum):
    """(künye, None) geçerliyse; (None, neden) değilse."""
    try:
        kunye = json.load(open(os.path.join(klasor, "onnx_kunye.json"), encoding="utf-8"))
        if kunye.get("kunye_surumu") != KUNYE_SURUMU:
            return None, "künye sürümü farklı"
        if kunye.get("kaynak_surum") != kaynak_surum:
            return None, f"kaynak model sürümü farklı ({kunye.get('kaynak_surum')} ≠ {kaynak_surum})"
        if kunye["dosyalar"].get("model_fp32.onnx", {}).get("durum") != "hazır":
            return None, "künyede hazır FP32 yok"
        for ad, b in kunye["dosyalar"].items():
            if b.get("durum") != "hazır":
                continue  # INT8 önceden başarısız olduysa kayıtlı; aynı kodla yeniden denenmez
            f = os.path.join(klasor, ad)
            if not os.path.isfile(f):
                return None, f"{ad} eksik"
            if os.path.getsize(f) != b["bayt"] or sha(f) != b["sha256"]:
                return None, f"{ad} künyeyle uyumsuz (boyut/SHA-256)"
        return kunye, None
    except Exception as e:  # bozuk/eksik künye
        return None, f"künye okunamadı: {e!r}"[:300]


def dosya_bilgisi(f):
    return {"durum": "hazır", "bayt": os.path.getsize(f), "MB": round(os.path.getsize(f) / 2**20, 1), "sha256": sha(f)}


def yuklenebilir_mi(f):
    import onnxruntime as ort
    ort.InferenceSession(f, providers=["CPUExecutionProvider"])


def sha(yol):
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()


def ana():
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="intfloat/multilingual-e5-small")
    a.add_argument("--onbellek", default=os.path.expanduser("~/.cache/turkmetinlab_modeller"))
    a.add_argument("--cikti", required=True)
    a.add_argument("--kunye")
    a.add_argument("--azami-token", type=int, help="yoksa modelin sentence_bert_config.json değeri (E5: 512)")
    a.add_argument("--onbellekten-devam", action="store_true", help="geçerli önbellek varsa dönüştürme yapma")
    args = a.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    yol = args.model if os.path.isdir(args.model) else snapshot_download(
        args.model, cache_dir=args.onbellek, allow_patterns=["*.json", "*.txt", "model.safetensors", "*.model"])
    kaynak_surum = os.path.basename(os.path.normpath(yol))
    hedef = os.path.abspath(args.cikti)
    onbellek_durumu = "yeni dönüşüm"
    if args.onbellekten_devam and os.path.isdir(hedef):
        kunye, neden = onbellek_denetle(hedef, kaynak_surum)
        if kunye:
            kunye["onbellek"] = "doğrulandı ve kullanıldı"
            yaz(kunye, args.kunye)
            return
        onbellek_durumu = f"önbellek yeniden oluşturuldu: {neden}"
        print(onbellek_durumu)
    gecici = hedef + ".yapim"
    shutil.rmtree(gecici, ignore_errors=True)
    os.makedirs(gecici)
    tok = AutoTokenizer.from_pretrained(yol)
    model = AutoModel.from_pretrained(yol).eval()
    fp32, int8 = os.path.join(gecici, "model_fp32.onnx"), os.path.join(gecici, "model_int8.onnx")
    ornek = tok(["query: örnek bir cümle", "passage: daha uzun bir örnek metin parçası"], padding=True,
                return_tensors="pt")
    girdi = (ornek["input_ids"], ornek["attention_mask"])

    class Sarmal(torch.nn.Module):  # yalnızca last_hidden_state; token_type_ids sıfır kabul edilir (tek bölüt)
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    t = time.perf_counter()
    disa = None
    try:  # TorchScript tabanlı dışa aktarıcı (ek bağımlılık gerektirmez)
        torch.onnx.export(Sarmal(model), girdi, fp32, input_names=["input_ids", "attention_mask"],
                          output_names=["last_hidden_state"], opset_version=17, dynamo=False,
                          dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                                        "last_hidden_state": {0: "b", 1: "s"}})
        disa = "torch.onnx.export (TorchScript, opset 17)"
    except Exception as e:  # yalnızca bir kez alternatif dene: dynamo dışa aktarıcı
        print("TorchScript dışa aktarıcı başarısız:", repr(e)[:200])
        b, s = torch.export.Dim("b"), torch.export.Dim("s", max=512)
        torch.onnx.export(Sarmal(model), girdi, fp32, input_names=["input_ids", "attention_mask"],
                          output_names=["last_hidden_state"], dynamo=True, external_data=False,
                          dynamic_shapes={"input_ids": {0: b, 1: s}, "attention_mask": {0: b, 1: s}})
        disa = "torch.onnx.export (dynamo)"
    sure = time.perf_counter() - t
    yuklenebilir_mi(fp32)  # FP32 açılamıyorsa hata ile durulur (önbellek değişmez)
    dosyalar = {"model_fp32.onnx": dosya_bilgisi(fp32)}
    try:  # INT8 başarısızlığı FP32'yi durdurmaz; hata kaydedilir, INT8 elenir
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
        yuklenebilir_mi(int8)
        dosyalar["model_int8.onnx"] = dosya_bilgisi(int8)
    except Exception as e:
        print("INT8 başarısız, elendi:", repr(e)[:300])
        if os.path.exists(int8):
            os.remove(int8)
        dosyalar["model_int8.onnx"] = {"durum": "başarısız", "hata": repr(e)[:500]}
    tj = os.path.join(yol, "tokenizer.json")
    if not os.path.exists(tj):
        tok.save_pretrained(gecici)
    else:
        shutil.copy(tj, os.path.join(gecici, "tokenizer.json"))
    dosyalar["tokenizer.json"] = dosya_bilgisi(os.path.join(gecici, "tokenizer.json"))
    sbc = os.path.join(yol, "sentence_bert_config.json")
    azami = args.azami_token or (json.load(open(sbc)).get("max_seq_length") if os.path.exists(sbc) else None) or 512
    kunye = {"kunye_surumu": KUNYE_SURUMU, "kaynak_model": args.model, "azami_token": int(azami),
             "kaynak_surum": kaynak_surum, "onbellek": onbellek_durumu,
             "lisans": "MIT (intfloat/multilingual-e5-small)", "disa_aktarici": disa,
             "torch": torch.__version__, "donusturme_sn": round(sure, 1),
             "nicemleme": "onnxruntime.quantization.quantize_dynamic, ağırlıklar QInt8",
             "dosyalar": dosyalar}
    json.dump(kunye, open(os.path.join(gecici, "onnx_kunye.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    shutil.rmtree(hedef, ignore_errors=True)  # künye en son yazıldı; tamamlanmış klasör yerine taşınır
    os.replace(gecici, hedef)
    yaz(kunye, args.kunye)


def yaz(kunye, yol):
    if yol:
        json.dump(kunye, open(yol, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(kunye, ensure_ascii=False))


if __name__ == "__main__":
    ana()
