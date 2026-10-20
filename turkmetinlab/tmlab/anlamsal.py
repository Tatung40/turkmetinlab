"""DENEYSEL (2.5) — E5-small ONNX FP32 ile anlamsal belge benzerliği ve arama.

Bilimsel dayanak (fizibilite_2.5; GitHub Actions ONNX #3, 500 DergiPark Türkçe özeti):
* ONNX FP32, PyTorch E5 ile aynı sıralamaları üretti. INT8 uyum eşiğini geçemedi; bu yüzden kullanılmaz.
* Özet → özet görevinde nDCG@5: TF-IDF 0,317; E5 0,304; TF-IDF + E5 hibrit 0,342 (+0,025). Kazanç önceden
  belirlenen 0,03 eşiğinin altındadır. İlgililik etiketi yazar anahtar sözcüklerinden türetilmiştir; uzman
  yargısı değildir. Sonuçlar keşif amaçlıdır.

Yöntem (doğrulanan ayarlar değiştirilmeden):
* Önekler: belge → belge benzerliğinde iki taraf "query: ", aramada sorgu "query: ", belgeler "passage: ".
* En fazla 512 belirteç. Daha uzun belgeler SESSİZCE kırpılmaz: belirteç sayıları önceden hesaplanır ve
  arayüz kullanıcıya kırpma ya da dışarıda bırakma seçeneği sunar.
* Dikkat maskeli ortalama havuzlama + L2 normalizasyonu; toplu iş 8, uzunluğa göre sıralama, ONNX Runtime bellek
  havuzu kapalı (fizibilite bellek düzeltmesi).
* Hibrit: her sorgu satırında TF-IDF ve E5 benzerliklerinin z-puanları toplanır (sorgunun kendisi hariç).

Gizlilik ve kaynak:
* Model dosyaları (kullanıcı verisi içermez) diskte önbelleğe alınır ve tüm oturumlarca paylaşılır.
* Vektörler ve sonuçlar yalnızca çağıranın oturum belleğinde tutulur; bu modül diske kullanıcı verisi yazmaz.
* Model aynı anda tek bir istek için çalışır (süreç genelinde kilit): eşzamanlı kullanıcılar sıraya girer,
  böylece tepe bellek kullanıcı sayısıyla katlanmaz.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

MODEL_ADI = "intfloat/multilingual-e5-small (ONNX FP32)"
LISANS = "MIT"
AZAMI_TOKEN = 512
TOPLU_IS = 8
BELGE_SINIRI = int(os.environ.get("TML_E5_BELGE_SINIRI") or 500)  # doğrulanan ölçek: 500 belge
SORGU_ONEKI, BELGE_ONEKI = "query: ", "passage: "
DOSYALAR = ("onnx_kunye.json", "tokenizer.json", "model_fp32.onnx")
GECERLI_SONUCLAR = {"TF-IDF": 0.317, "E5": 0.304, "Hibrit": 0.342}
YONTEMLER = ("TF-IDF", "E5", "Hibrit")

_INDIRME_KILIDI = threading.Lock()
_CALISMA_KILIDI = threading.BoundedSemaphore(1)


class ModelHatasi(RuntimeError):
    """Kullanıcıya gösterilebilir model hatası (metin içermez)."""


# ---------------------------------------------------------------------------
# Model kaynağı ve indirme
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelKaynagi:
    repo: str = ""
    surum: str = "main"
    sha256: str = ""
    yerel: str = ""  # önceden hazırlanmış klasör (test/CI); indirme yapılmaz
    belirtec: str = ""  # özel Hugging Face deposu için (isteğe bağlı)
    adres: str = "https://huggingface.co"  # testlerde yerel sunucu

    @classmethod
    def ortamdan(cls) -> "ModelKaynagi":
        e = os.environ.get
        return cls(e("TML_E5_ONNX_REPO", "").strip(), e("TML_E5_ONNX_REV", "main").strip() or "main",
                   e("TML_E5_ONNX_SHA256", "").strip().lower(), e("TML_E5_ONNX_YEREL", "").strip(),
                   e("HF_TOKEN", "").strip(), (e("TML_E5_ONNX_URL") or "https://huggingface.co").rstrip("/"))

    def eksik(self) -> Optional[str]:
        if self.yerel:
            return None if os.path.isdir(self.yerel) else "Yerel model klasörü bulunamadı (TML_E5_ONNX_YEREL)."
        if not self.repo or len(self.sha256) != 64:
            return ("Model kaynağı yapılandırılmamış. TML_E5_ONNX_REPO ve doğrulanmış model özeti "
                    "TML_E5_ONNX_SHA256 tanımlanmalıdır (kurulum: staging/README.md).")
        return None

    @property
    def kimlik(self) -> str:
        return self.yerel or f"{self.repo}@{self.surum}#{self.sha256[:12]}"


def onbellek_klasoru(kaynak: ModelKaynagi) -> str:
    if kaynak.yerel:
        return kaynak.yerel
    kok = os.environ.get("TML_MODEL_ONBELLEK") or os.path.join(os.path.expanduser("~"), ".cache", "turkmetinlab_modeller")
    return os.path.join(kok, "e5onnx", kaynak.sha256[:16])


def sha256_dosya(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()


def _kunye_denetle(klasor: str, kaynak: ModelKaynagi) -> dict:
    """Künye, model ve belirteçleyici özetlerini denetler; uyumsuzlukta ModelHatasi."""
    try:
        with open(os.path.join(klasor, "onnx_kunye.json"), encoding="utf-8") as f:
            kunye = json.load(f)
        dosyalar = kunye["dosyalar"]
        beklenen_model = kaynak.sha256 or dosyalar["model_fp32.onnx"]["sha256"]
        if dosyalar["model_fp32.onnx"]["sha256"] != beklenen_model:
            raise ModelHatasi("Model künyesi doğrulanmış özetle uyuşmuyor.")
        if sha256_dosya(os.path.join(klasor, "model_fp32.onnx")) != beklenen_model:
            raise ModelHatasi("Model dosyası doğrulanmış özetle uyuşmuyor.")
        if sha256_dosya(os.path.join(klasor, "tokenizer.json")) != dosyalar["tokenizer.json"]["sha256"]:
            raise ModelHatasi("Belirteçleyici dosyası künyeyle uyuşmuyor.")
        return kunye
    except ModelHatasi:
        raise
    except (OSError, KeyError, ValueError, TypeError) as e:
        raise ModelHatasi(f"Model dosyaları eksik ya da okunamıyor ({type(e).__name__}).") from None


def model_hazirla(kaynak: ModelKaynagi, ilerleme: Optional[Callable[[str, float], None]] = None,
                  zaman_asimi: int = 60) -> str:
    """Model dosyalarını (gerekirse) TEK denemeyle indirir, özetleri doğrular ve klasörü döndürür.

    Dosyalar geçici adla indirilir; yalnızca özet doğrulanınca yerine taşınır. Başarısız indirme yinelenmez.
    """
    neden = kaynak.eksik()
    if neden:
        raise ModelHatasi(neden)
    klasor = onbellek_klasoru(kaynak)
    if kaynak.yerel:
        _kunye_denetle(klasor, kaynak)
        return klasor
    with _INDIRME_KILIDI:  # aynı anda yalnızca bir indirme
        try:
            _kunye_denetle(klasor, kaynak)
            return klasor
        except ModelHatasi:
            pass
        os.makedirs(klasor, exist_ok=True)
        for ad in DOSYALAR:
            hedef = os.path.join(klasor, ad)
            gecici = hedef + ".indiriliyor"
            url = f"{kaynak.adres}/{kaynak.repo}/resolve/{kaynak.surum}/{ad}"
            istek = urllib.request.Request(url, headers={"User-Agent": "TurkMetinLab"})
            if kaynak.belirtec:
                istek.add_header("Authorization", f"Bearer {kaynak.belirtec}")
            try:
                with urllib.request.urlopen(istek, timeout=zaman_asimi) as yanit, open(gecici, "wb") as f:
                    toplam, alinan = int(yanit.headers.get("Content-Length") or 0), 0
                    for parca in iter(lambda: yanit.read(1 << 20), b""):
                        f.write(parca)
                        alinan += len(parca)
                        if ilerleme:
                            ilerleme(ad, alinan / toplam if toplam else 0.0)
                os.replace(gecici, hedef)
            except Exception as e:
                if os.path.exists(gecici):
                    os.remove(gecici)
                raise ModelHatasi(f"Model indirilemedi ({ad}: {type(e).__name__}). Ağ bağlantısını ve model "
                                  "deposu ayarlarını denetleyip daha sonra yeniden deneyin.") from None
        try:
            _kunye_denetle(klasor, kaynak)
        except ModelHatasi:
            for ad in DOSYALAR:  # doğrulanmamış dosya önbellekte bırakılmaz
                if os.path.exists(os.path.join(klasor, ad)):
                    os.remove(os.path.join(klasor, ad))
            raise
        return klasor


# ---------------------------------------------------------------------------
# Kodlayıcı
# ---------------------------------------------------------------------------

class E5Kodlayici:
    """PyTorch'suz E5 kodlayıcı (onnxruntime + tokenizers). fizibilite_2.5/onnx/e5_onnx_kodlayici.py ile aynı
    hesaplama; oturumlar arasında paylaşılabilir (çalıştırma süreç genelinde kilitlenir)."""

    def __init__(self, klasor: str, is_parcacigi: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        with open(os.path.join(klasor, "onnx_kunye.json"), encoding="utf-8") as f:
            self.kunye = json.load(f)
        self.azami_token = int(self.kunye.get("azami_token", AZAMI_TOKEN))
        self.tok = Tokenizer.from_file(os.path.join(klasor, "tokenizer.json"))
        pad = self.tok.token_to_id("<pad>")
        self._dolgu = (pad, "<pad>") if pad is not None else (0, "[PAD]")
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        so.intra_op_num_threads = is_parcacigi or min(2, os.cpu_count() or 1)
        so.inter_op_num_threads = 1
        self.oturum = ort.InferenceSession(os.path.join(klasor, "model_fp32.onnx"), so,
                                           providers=["CPUExecutionProvider"])
        self.girdiler = {g.name for g in self.oturum.get_inputs()}
        self._tok_kilidi = threading.Lock()  # tokenizers nesnesinin ayarları paylaşılan durumdur

    def belirtec_sayilari(self, metinler: Sequence[str], onek: str = BELGE_ONEKI) -> np.ndarray:
        """Özel belirteçler dahil, kırpmasız belirteç sayıları."""
        with self._tok_kilidi:
            self.tok.no_truncation()
            self.tok.no_padding()
            return np.array([len(e.ids) for e in self.tok.encode_batch([onek + m for m in metinler])], dtype=int)

    def _kodla_toplu(self, metinler: List[str]) -> np.ndarray:
        with self._tok_kilidi:
            self.tok.enable_truncation(self.azami_token)
            self.tok.enable_padding(pad_id=self._dolgu[0], pad_token=self._dolgu[1])
            enc = self.tok.encode_batch(metinler)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        besle = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        h = self.oturum.run(None, {k: v for k, v in besle.items() if k in self.girdiler})[0]
        m = mask[..., None].astype(np.float32)
        v = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)

    def kodla(self, metinler: Sequence[str], onek: str, ilerleme: Optional[Callable[[float], None]] = None,
              bekleme: Optional[Callable[[], None]] = None, toplu_is: int = TOPLU_IS) -> np.ndarray:
        """L2 normalleştirilmiş vektörler (sıra korunur). bekleme: model meşgulse bir kez çağrılır."""
        metinler = [onek + m for m in metinler]
        if not metinler:
            return np.zeros((0, 384), dtype=np.float32)
        sira = np.argsort([-len(x) for x in metinler], kind="stable")
        if not _CALISMA_KILIDI.acquire(blocking=False):
            if bekleme:
                bekleme()
            _CALISMA_KILIDI.acquire()
        try:
            parcalar = []
            for i in range(0, len(metinler), toplu_is):
                parcalar.append(self._kodla_toplu([metinler[j] for j in sira[i:i + toplu_is]]))
                if ilerleme:
                    ilerleme(min(1.0, (i + toplu_is) / len(metinler)))
        finally:
            _CALISMA_KILIDI.release()
        V = np.vstack(parcalar).astype(np.float32)
        geri = np.empty_like(V)
        geri[sira] = V
        return geri


# ---------------------------------------------------------------------------
# Skorlar
# ---------------------------------------------------------------------------

def znorm(S: np.ndarray) -> np.ndarray:
    """Satır başına z-puanı; sonsuz değerler (sorgunun kendisi) ortalamaya katılmaz (fizibilite ile aynı)."""
    S = np.where(np.isfinite(S), S, np.nan)
    m, s = np.nanmean(S, axis=1, keepdims=True), np.nanstd(S, axis=1, keepdims=True) + 1e-9
    return np.nan_to_num((S - m) / s, nan=-1e9)


def hibrit(S_tfidf: np.ndarray, S_e5: np.ndarray) -> np.ndarray:
    return znorm(S_tfidf) + znorm(S_e5)


def tfidf_sorgu_vektoru(matris, terimler: Sequence[str], sorgu_terimleri: Sequence[str], smooth_idf: bool,
                        sublinear_tf: bool) -> np.ndarray:
    """Derlemin TF-IDF uzayında sorgu vektörü (scikit-learn TfidfVectorizer.transform ile aynı formül).

    idf, matristeki belge sıklıklarından yeniden hesaplanır: smooth ise ln((1+N)/(1+df))+1, değilse ln(N/df)+1.
    """
    n = matris.shape[0]
    df = np.asarray((matris > 0).sum(axis=0)).ravel()
    idf = np.log((1 + n) / (1 + df)) + 1 if smooth_idf else np.log(n / df) + 1
    sozluk = {t: i for i, t in enumerate(terimler)}
    q = np.zeros(len(terimler))
    for t in sorgu_terimleri:
        if t in sozluk:
            q[sozluk[t]] += 1
    if sublinear_tf:
        q[q > 0] = 1 + np.log(q[q > 0])
    q *= idf
    norm = np.linalg.norm(q)
    return q / norm if norm else q


def en_yakinlar(skor: np.ndarray, k: int, haric: Optional[int] = None) -> List[int]:
    s = skor.astype(float).copy()
    if haric is not None:
        s[haric] = -np.inf
    k = min(k, int(np.isfinite(s).sum()))
    return [int(i) for i in np.argsort(-s, kind="stable")[:k]]


def ust_k_ortusme(A: np.ndarray, B: np.ndarray, k: int) -> float:
    """İki benzerlik matrisinde (köşegen hariç) her satırın ilk-k kümelerinin ortalama örtüşmesi (0–1)."""
    n = A.shape[0]
    if n < 2:
        return float("nan")
    k = min(k, n - 1)
    oran = []
    for i in range(n):
        a, b = set(en_yakinlar(A[i], k, i)), set(en_yakinlar(B[i], k, i))
        oran.append(len(a & b) / k)
    return float(np.mean(oran))


def sure_tahmini(belge_sayisi: int, olculen_sn: float = 64.21, olculen_belge: int = 500) -> float:
    """Kaba üst tahmin: GitHub Actions ölçümünden doğrusal ölçek (sunucu daha yavaş olabilir)."""
    return olculen_sn * belge_sayisi / olculen_belge
