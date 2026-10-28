"""20 örnekte PyTorch (sentence-transformers) ↔ ONNX (FP32, INT8) E5 uyumu. Torch gerektirir.

Ölçütler: örnek başına kosinüs (ST vektörü · ONNX vektörü), en büyük mutlak fark, 20×20 başlık→özet skor
matrisinde ilk-1 aynılığı ve ilk-5 örtüşmesi. Önceden belirlenen geçme koşulları:
  FP32: en küçük kosinüs ≥ 0,999 ve ilk-1 sıralamalar %100 aynı.
  INT8: en küçük kosinüs ≥ 0,98 ve ilk-5 örtüşmesi ortalaması ≥ 0,80.
Geçen adaylar --adaylar dosyasına yazılır (ör. e5onnx32_ozgun,e5onnx8_ozgun). FP32 geçemezse 1 ile çıkar.
INT8 dönüştürülememiş ya da hata vermişse hata kaydedilir ve yalnızca INT8 elenir.
"""
import argparse
import json
import os
import sys

import numpy as np

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
sys.path.insert(0, os.path.dirname(BURA))


def ana():
    a = argparse.ArgumentParser()
    a.add_argument("--onnx-klasor", required=True)
    a.add_argument("--model", default="intfloat/multilingual-e5-small")
    a.add_argument("--onbellek", default=os.path.expanduser("~/.cache/turkmetinlab_modeller"))
    a.add_argument("--veri")
    a.add_argument("--cikti", required=True)
    a.add_argument("--adaylar", required=True)
    args = a.parse_args()
    from anlamsal_karsilastirma import veri_oku
    from e5_onnx_kodlayici import E5Onnx
    from sentence_transformers import SentenceTransformer

    basliklar, ozetler, _, _ = veri_oku(args.veri, 20)
    st = SentenceTransformer(args.model, device="cpu", cache_folder=args.onbellek)
    kunye = json.load(open(os.path.join(args.onnx_klasor, "onnx_kunye.json")))
    P = [("passage: " + x) for x in ozetler]
    Qm = [("query: " + x) for x in basliklar]
    ref_d = st.encode(P, normalize_embeddings=True, convert_to_numpy=True)
    ref_q = st.encode(Qm, normalize_embeddings=True, convert_to_numpy=True)
    S_ref = ref_q @ ref_d.T
    sonuc, adaylar = {"model_azami_token_st": int(st.max_seq_length), "onnx_azami_token": kunye.get("azami_token")}, []
    for tur, ad, esik_cos, esik_ust in (("fp32", "e5onnx32_ozgun", 0.999, None), ("int8", "e5onnx8_ozgun", 0.98, 0.80)):
        dk = kunye.get("dosyalar", {}).get(f"model_{tur}.onnx", {})
        if dk.get("durum") != "hazır":
            sonuc[tur] = {"gecti": False, "hata": dk.get("hata", "künyede yok")}
            continue
        try:
            m = E5Onnx(args.onnx_klasor, tur, kunye.get("azami_token", 512))
            d, q = m.encode(P), m.encode(Qm)
        except Exception as e:
            sonuc[tur] = {"gecti": False, "hata": repr(e)[:500]}
            continue
        cos = np.concatenate([(d * ref_d).sum(1), (q * ref_q).sum(1)])
        S = q @ d.T
        ilk1 = float(np.mean(np.argmax(S, 1) == np.argmax(S_ref, 1)))
        ust5 = float(np.mean([len(set(np.argsort(-S[i])[:5]) & set(np.argsort(-S_ref[i])[:5])) / 5 for i in range(20)]))
        gecti = cos.min() >= esik_cos and (ilk1 == 1.0 if esik_ust is None else ust5 >= esik_ust)
        sonuc[tur] = {"kosinus_en_kucuk": round(float(cos.min()), 6), "kosinus_ortalama": round(float(cos.mean()), 6),
                      "en_buyuk_mutlak_fark": round(float(np.abs(np.vstack([d - ref_d, q - ref_q])).max()), 6),
                      "ilk1_ayni_orani": ilk1, "ilk5_ortusme_ort": round(ust5, 4), "gecti": bool(gecti)}
        if gecti:
            adaylar.append(ad)
    os.makedirs(args.cikti, exist_ok=True)
    json.dump(sonuc, open(os.path.join(args.cikti, "onnx_uyum_20.json"), "w"), ensure_ascii=False, indent=1)
    open(args.adaylar, "w").write(",".join(adaylar))
    print(json.dumps(sonuc, ensure_ascii=False))
    if not sonuc.get("fp32", {}).get("gecti"):
        sys.exit("FP32 ONNX, PyTorch ile uyumlu değil; karşılaştırma yapılmayacak.")


if __name__ == "__main__":
    ana()
