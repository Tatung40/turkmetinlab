"""Oturuma özel bellek ve veri temizliği.

Neden st.cache_data kullanılmıyor?
---------------------------------
Streamlit belgelerine göre st.cache_data ile önbelleğe alınan değerler
"uygulamanın tüm kullanıcılarına açıktır" ve oturum kapandıktan sonra da
sunucu belleğinde kalır. Kullanıcının yüklediği metinler ve bunlardan üretilen
sonuçlar bu yüzden YALNIZCA ilgili oturumun st.session_state alanında tutulur:

* Başka bir oturum bu verilere erişemez.
* "Verilerimi temizle" düğmesi verileri anında siler.
* Oturum kapandığında (sekme kapatılınca) Streamlit oturum verisini ve
  yüklenen dosyaları en geç server.disconnectedSessionTTL süresi
  (varsayılan 120 saniye) dolduktan sonra siler.

st.cache_data yalnızca herkese açık örnek veri seti için kullanılır.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import traceback
from collections import OrderedDict
from typing import Any, Callable, MutableMapping

ONEK = "_tml_"
_BELLEK_ANAHTARI = ONEK + "bellek"


def ozet(*parcalar: Any) -> str:
    """Girdilerin SHA-256 özeti (önbellek anahtarı). Metnin kendisi saklanmaz."""
    h = hashlib.sha256()
    for p in parcalar:
        _guncelle(h, p)
        h.update(b"\x1e")
    return h.hexdigest()


def _guncelle(h, p):
    if isinstance(p, bytes):
        h.update(b"b"); h.update(len(p).to_bytes(8, "little")); h.update(p)
    elif isinstance(p, str):
        v = p.encode("utf-8", "surrogatepass")
        h.update(b"s"); h.update(len(v).to_bytes(8, "little")); h.update(v)
    elif isinstance(p, (tuple, list)):
        h.update(b"t"); h.update(len(p).to_bytes(8, "little"))
        for x in p:
            _guncelle(h, x)
    else:
        v = repr(p).encode("utf-8")
        h.update(b"r"); h.update(len(v).to_bytes(8, "little")); h.update(v)


class OturumBellegi:
    """st.session_state içinde, ad alanı başına sınırlı sayıda kayıt tutan bellek.

    depo: st.session_state (testlerde sıradan bir sözlük).
    """

    def __init__(self, depo: MutableMapping):
        self._depo = depo
        if _BELLEK_ANAHTARI not in depo:
            depo[_BELLEK_ANAHTARI] = {}

    @property
    def _alanlar(self) -> dict:
        return self._depo[_BELLEK_ANAHTARI]

    def al(self, ad: str, anahtar: str, hesapla: Callable[[], Any], en_fazla: int = 2) -> Any:
        alan = self._alanlar.setdefault(ad, OrderedDict())
        if anahtar in alan:
            alan.move_to_end(anahtar)
            return alan[anahtar]
        deger = hesapla()  # hata olursa hiçbir şey saklanmaz
        alan[anahtar] = deger
        while len(alan) > en_fazla:
            alan.popitem(last=False)
        return deger

    def kayit_sayisi(self) -> int:
        return sum(len(a) for a in self._alanlar.values())

    def temizle(self) -> None:
        self._alanlar.clear()


def oturum_verilerini_sil(depo: MutableMapping, korunacaklar=()) -> int:
    """Bu uygulamanın oturumdaki tüm verilerini siler; silinen anahtar sayısını döndürür."""
    silinecek = [k for k in list(depo.keys()) if str(k).startswith(ONEK) and k not in korunacaklar]
    for k in silinecek:
        del depo[k]
    return len(silinecek)


def yuklenen_dosyalari_sil() -> bool:
    """Bu oturumun sunucu belleğindeki yüklenmiş dosyalarını hemen siler.

    Streamlit'in iç API'sini kullanır; bulunamazsa False döner (dosyalar yine
    oturum kapandığında Streamlit tarafından silinir).
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is None:
            return False
        ctx.uploaded_file_mgr.remove_session_files(ctx.session_id)
        return True
    except Exception:
        return False


_LOGGER = logging.getLogger("turkmetinlab")


def guvenli_hata_kaydi(hata: BaseException) -> str:
    """Hata için kullanıcı verisi İÇERMEYEN bir kayıt üretir ve sunucu günlüğüne yazar.

    Hata mesajı (ör. pandas'ın ayrıştırma hatası) kullanıcının metninden parçalar
    içerebileceği için günlüğe yalnızca hata türü ve kod konumu yazılır.
    """
    cerceveler = traceback.extract_tb(hata.__traceback__)
    konum = " <- ".join(f"{c.filename.rsplit('/', 1)[-1]}:{c.lineno}:{c.name}" for c in reversed(cerceveler[-6:]))
    kayit = f"{type(hata).__name__} @ {konum}"
    print(f"[TürkMetinLab] Beklenmeyen hata: {kayit}", file=sys.stderr)
    return kayit
