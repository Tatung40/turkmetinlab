"""Testler için küçük, rastgele ağırlıklı E5 benzeri ONNX modeli (indirme yok; gerçek E5 değildir).

Çizge: input_ids → Gather(gömme tablosu) → Tanh → last_hidden_state. Kodlayıcının belirteçleme, kırpma, dolgu,
maskeli ortalama havuzlama, L2 normalizasyonu ve sıralama adımlarını sınamak için yeterlidir.
"""
import hashlib
import json
import os

import numpy as np


def _sha(yol):
    return hashlib.sha256(open(yol, "rb").read()).hexdigest()


def kucuk_model_olustur(klasor, azami_token=32, boyut=16, tohum=0):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from tokenizers.processors import TemplateProcessing

    os.makedirs(klasor, exist_ok=True)
    harfler = list("abcçdefgğhıijklmnoöprsştuüvyzqwx0123456789:.,;-'’()")
    sozluk = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, **{h: i + 4 for i, h in enumerate(harfler)}}
    tok = Tokenizer(models.WordLevel(sozluk, unk_token="<unk>"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")  # her karakter bir belirteç
    tok.post_processor = TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 2), ("</s>", 3)])
    tok.save(os.path.join(klasor, "tokenizer.json"))
    E = np.random.default_rng(tohum).normal(size=(len(sozluk), boyut)).astype(np.float32)
    dugum = [helper.make_node("Gather", ["E", "input_ids"], ["g"]),
             helper.make_node("Tanh", ["g"], ["last_hidden_state"])]
    cizge = helper.make_graph(
        dugum, "kucuk_e5",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"]),
         helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "s"])],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["b", "s", boyut])],
        [numpy_helper.from_array(E, "E")])
    model = helper.make_model(cizge, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, os.path.join(klasor, "model_fp32.onnx"))
    kunye = {"kunye_surumu": 2, "kaynak_model": "test", "azami_token": azami_token,
             "dosyalar": {ad: {"durum": "hazır", "sha256": _sha(os.path.join(klasor, ad))}
                          for ad in ("model_fp32.onnx", "tokenizer.json")}}
    json.dump(kunye, open(os.path.join(klasor, "onnx_kunye.json"), "w"))
    return E


def referans_vektor(E, metin, azami_token, tok_yolu):
    """Bağımsız hesap: belirteçle, kırp, gömme ortalaması, L2."""
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(tok_yolu)
    t.enable_truncation(azami_token)
    v = np.tanh(E[t.encode(metin).ids]).mean(0)
    return v / np.linalg.norm(v)
