"""Küçük, rastgele ağırlıklı bir BERT modeliyle ONNX dönüştürme ve PyTorch'suz kodlayıcı testi.

Model indirmez; gerçek E5 doğruluğunu ölçmez. Çalıştırma: python -m pytest fizibilite_2.5/onnx/test_onnx.py
(torch, transformers, sentence-transformers, onnx, onnxruntime gerekir).
"""
import os
import subprocess
import sys

import numpy as np
import pytest

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)


@pytest.fixture(scope="module")
def kucuk_model(tmp_path_factory):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import models as stm
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("kucuk")
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer, tok.pre_tokenizer = normalizers.Lowercase(), pre_tokenizers.Whitespace()
    metin = ["uzaktan eğitim öğrencileri", "yapay zekâ ve makine öğrenmesi", "query: passage: örnek metin"] * 20
    tok.train_from_iterator(metin, trainers.WordPieceTrainer(vocab_size=200, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
    tok.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                            sep_token="[SEP]").save_pretrained(d / "hf")
    BertModel(BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=130)).save_pretrained(d / "hf")
    SentenceTransformer(modules=[stm.Transformer(str(d / "hf"), max_seq_length=64),
                                 stm.Pooling(32, pooling_mode="mean")]).save(str(d / "st"))
    subprocess.run([sys.executable, os.path.join(BURA, "e5_onnx_donustur.py"), "--model", str(d / "hf"),
                    "--cikti", str(d / "onnx"), "--azami-token", "64"], check=True, capture_output=True)
    return d


def test_onnx_pytorch_ile_ayni(kucuk_model):
    from e5_onnx_kodlayici import E5Onnx
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(kucuk_model / "st"), device="cpu")
    metin = ["query: uzaktan eğitim", "passage: yapay zekâ ve makine öğrenmesi " * 20, "passage: örnek"]
    ref = st.encode(metin, normalize_embeddings=True)
    fp32 = E5Onnx(str(kucuk_model / "onnx"), "fp32", 64).encode(metin)
    int8 = E5Onnx(str(kucuk_model / "onnx"), "int8", 64).encode(metin)
    assert (fp32 * ref).sum(1).min() > 0.9999  # kesme (64 token), havuzlama ve L2 aynı
    assert (int8 * ref).sum(1).min() > 0.98
    assert np.allclose(np.linalg.norm(fp32, axis=1), 1, atol=1e-5)
    uz = E5Onnx(str(kucuk_model / "onnx"), "fp32", 64).token_uzunluklari(metin)  # dolgu sayılmamalı
    assert list(uz) == [len(st.tokenizer(m)["input_ids"]) for m in metin]


def test_siralama_ve_toplu_is_sonucu_degistirmez(kucuk_model):
    from e5_onnx_kodlayici import E5Onnx
    e = E5Onnx(str(kucuk_model / "onnx"), "fp32", 64)
    metin = [f"passage: {'yapay zekâ ' * k}örnek" for k in (1, 30, 3, 12, 0, 7, 25, 2, 9, 4, 18)]
    tek = np.vstack([e.encode([m]) for m in metin])  # dolgusuz tek tek
    assert np.abs(e.encode(metin) - tek).max() < 1e-5  # sıralı küçük toplu işlerde sıra korunur
    assert np.abs(e.encode(metin, batch_size=32) - tek).max() < 1e-5
    so = e.oturum.get_session_options()
    assert e.toplu_is == 8 and not so.enable_cpu_mem_arena and so.intra_op_num_threads >= 1


def test_kodlayici_torch_yuklemez(kucuk_model):
    kod = (f"import sys; sys.path.insert(0, {BURA!r}); from e5_onnx_kodlayici import E5Onnx; "
           f"E5Onnx({str(kucuk_model / 'onnx')!r}, 'int8', 64).encode(['passage: x']); print('torch' in sys.modules)")
    assert subprocess.run([sys.executable, "-c", kod], capture_output=True, text=True, check=True).stdout.strip() == "False"


# --- Dayanıklılık: INT8 hatası ve önbellek bütünlüğü -------------------------------------------------------
import json  # noqa: E402
import shutil  # noqa: E402


def _donustur(*arg):
    return subprocess.run([sys.executable, os.path.join(BURA, "e5_onnx_donustur.py"), *map(str, arg)],
                          capture_output=True, text=True)


def _kunye(klasor):
    return json.load(open(klasor / "onnx_kunye.json", encoding="utf-8"))


@pytest.fixture
def sahte_veri(tmp_path):
    satir = [f"{i}\tbaşlık {i} uzaktan eğitim\t" + " ".join(f"sözcük{i} yapay zekâ öğrenme" for _ in range(12))
             + f"\teğitim, konu{i % 4}" for i in range(25)]
    (tmp_path / "TRAll.txt").write_text("\n".join(satir), encoding="utf-8")
    return tmp_path / "TRAll.txt"


@pytest.mark.parametrize("bozma", ["gecerli", "int8_bozuk", "fp32_eksik", "kunye_bozuk", "kaynak_farkli"])
def test_onbellek_dogrulanir_ve_gerekirse_yeniden_olusur(kucuk_model, tmp_path, bozma):
    k = tmp_path / "onnx"
    shutil.copytree(kucuk_model / "onnx", k)
    if bozma == "int8_bozuk":
        with open(k / "model_int8.onnx", "ab") as f:
            f.write(b"x")
    elif bozma == "fp32_eksik":
        (k / "model_fp32.onnx").unlink()
    elif bozma == "kunye_bozuk":
        (k / "onnx_kunye.json").write_text("{yarım", encoding="utf-8")
    elif bozma == "kaynak_farkli":
        y = _kunye(k)
        y["kaynak_surum"] = "baska"
        (k / "onnx_kunye.json").write_text(json.dumps(y), encoding="utf-8")
    r = _donustur("--onbellekten-devam", "--model", kucuk_model / "hf", "--cikti", k, "--azami-token", "64",
                  "--kunye", tmp_path / "k.json")
    assert r.returncode == 0, r.stderr[-500:]
    durum = json.load(open(tmp_path / "k.json"))["onbellek"]
    assert durum.startswith("doğrulandı" if bozma == "gecerli" else "önbellek yeniden oluşturuldu"), durum
    y = _kunye(k)  # yeniden oluşturulan önbellek geçerli olmalı ve geçici klasör kalmamalı
    from e5_onnx_donustur import onbellek_denetle
    assert onbellek_denetle(str(k), "hf")[0] is not None
    assert not (tmp_path / "onnx.yapim").exists() and y["dosyalar"]["model_int8.onnx"]["durum"] == "hazır"


def test_int8_hatasi_yalnizca_int8i_eler(kucuk_model, tmp_path, monkeypatch, sahte_veri):
    import e5_onnx_donustur
    import onnxruntime.quantization as q

    def bozuk(*a, **k):
        raise RuntimeError("deneme: nicemleme hatası")
    monkeypatch.setattr(q, "quantize_dynamic", bozuk)
    k = tmp_path / "onnx"
    monkeypatch.setattr(sys, "argv", ["x", "--model", str(kucuk_model / "hf"), "--cikti", str(k), "--azami-token", "64"])
    e5_onnx_donustur.ana()
    y = _kunye(k)
    assert y["dosyalar"]["model_fp32.onnx"]["durum"] == "hazır" and not (k / "model_int8.onnx").exists()
    assert y["dosyalar"]["model_int8.onnx"]["durum"] == "başarısız" and "nicemleme" in y["dosyalar"]["model_int8.onnx"]["hata"]
    # Kayıtlı INT8 hatası önbelleği geçersiz kılmaz (aynı kodla yeniden denenmez)
    assert e5_onnx_donustur.onbellek_denetle(str(k), "hf")[0] is not None
    r = subprocess.run([sys.executable, os.path.join(BURA, "e5_onnx_uyum.py"), "--onnx-klasor", k, "--model",
                        kucuk_model / "st", "--veri", sahte_veri, "--cikti", tmp_path, "--adaylar", tmp_path / "a.txt"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    assert (tmp_path / "a.txt").read_text() == "e5onnx32_ozgun"
    u = json.load(open(tmp_path / "onnx_uyum_20.json"))
    assert u["fp32"]["gecti"] and not u["int8"]["gecti"] and "nicemleme" in u["int8"]["hata"]


def test_fp32_uyumsuzsa_durur(kucuk_model, tmp_path, sahte_veri):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import models as stm
    from transformers import BertConfig, BertModel

    hf = tmp_path / "hf"  # aynı tokenizer, farklı rastgele ağırlıklar → FP32 referansla uyuşmamalı
    shutil.copytree(kucuk_model / "hf", hf)
    BertModel(BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=130)).save_pretrained(hf)
    SentenceTransformer(modules=[stm.Transformer(str(hf), max_seq_length=64),
                                 stm.Pooling(32, pooling_mode="mean")]).save(str(tmp_path / "st"))
    r = subprocess.run([sys.executable, os.path.join(BURA, "e5_onnx_uyum.py"), "--onnx-klasor", kucuk_model / "onnx",
                        "--model", tmp_path / "st", "--veri", sahte_veri, "--cikti", tmp_path, "--adaylar",
                        tmp_path / "a.txt"], capture_output=True, text=True)
    assert r.returncode == 1 and "FP32" in r.stderr
    assert "e5onnx32_ozgun" not in (tmp_path / "a.txt").read_text()
