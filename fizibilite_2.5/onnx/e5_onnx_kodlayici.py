"""PyTorch'suz E5 kodlayıcı: tokenizers + onnxruntime + numpy.

sentence-transformers'daki E5 yapılandırmasıyla aynı adımlar: azami 512 token kesme, dikkat maskesiyle ortalama
havuzlama, L2 normalizasyonu. Önekler ("query: ", "passage: ") çağıran tarafından eklenir.

Bellek: dışa aktarılan çizgede dikkat matrisi açıkça oluşur (toplu iş × 12 baş × uzunluk² × 4 bayt, katman başına).
Bu yüzden toplu iş küçük tutulur (8), metinler uzunluğa göre sıralanır (dolgu azalır; sentence-transformers da
sıralar) ve ONNX Runtime bellek havuzu (arena) kapatılır: havuz tepe bloklarını sürece geri vermez.
"""
import os

import numpy as np


class E5Onnx:
    def __init__(self, klasor: str, tur: str = "fp32", azami_token: int = 512, is_parcacigi: int = 0,
                 toplu_is: int = 8):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(os.path.join(klasor, "tokenizer.json"))
        self.tok.enable_truncation(azami_token)
        pad = self.tok.token_to_id("<pad>")
        self.tok.enable_padding(pad_id=pad if pad is not None else 0, pad_token="<pad>" if pad is not None else "[PAD]")
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False  # tepe bellek sürece geri verilir
        so.enable_mem_pattern = False  # değişken uzunlukta önceden ayrılan bellek kalıbı işe yaramaz
        so.intra_op_num_threads = is_parcacigi or min(4, os.cpu_count() or 1)
        so.inter_op_num_threads = 1
        self.toplu_is = toplu_is
        self.oturum = ort.InferenceSession(os.path.join(klasor, f"model_{tur}.onnx"), so,
                                           providers=["CPUExecutionProvider"])
        self.girdiler = {g.name for g in self.oturum.get_inputs()}
        self.azami_token = azami_token
        self.dosya = os.path.join(klasor, f"model_{tur}.onnx")

    def token_uzunluklari(self, metinler):
        self.tok.no_truncation()
        u = [sum(e.attention_mask) for e in self.tok.encode_batch(list(metinler))]  # dolgu hariç
        self.tok.enable_truncation(self.azami_token)
        return np.array(u)

    def encode(self, metinler, batch_size: int | None = None):
        batch_size = batch_size or self.toplu_is
        metinler = list(metinler)
        sira = np.argsort([-len(x) for x in metinler], kind="stable")  # uzunluğa göre; sonra eski sıraya dönülür
        cikti = []
        for i in range(0, len(metinler), batch_size):
            enc = self.tok.encode_batch([metinler[j] for j in sira[i:i + batch_size]])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            besle = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.girdiler:
                besle["token_type_ids"] = np.zeros_like(ids)
            h = self.oturum.run(None, {k: v for k, v in besle.items() if k in self.girdiler})[0]
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            cikti.append(v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None))
        V = np.vstack(cikti).astype(np.float32)
        geri = np.empty_like(V)
        geri[sira] = V
        return geri
