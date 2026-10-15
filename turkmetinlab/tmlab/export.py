"""Sonuçların CSV ve Excel olarak dışa aktarılması.

Formül enjeksiyonu koruması
---------------------------
"=", "+", "-", "@" ile başlayan hücreler Excel'de formül olarak
çalıştırılabilir (CSV/formül enjeksiyonu). Sonuç tabloları kullanıcı metinlerini
(özgün metin, belge adları) içerdiği için:

* Excel çıktısında tüm metinler "metin" türünde yazılır; hiçbir hücre formül
  ya da bağlantıya dönüştürülmez (xlsxwriter: strings_to_formulas=False).
  Veriler değiştirilmez.
* Excel dosyası tamamen bellekte oluşturulur (in_memory=True); xlsxwriter'ın
  varsayılan davranışı olan geçici dosyaya yazma kapatılmıştır.
* CSV çıktısında, isteğe bağlı (varsayılan açık) olarak bu karakterlerle
  başlayan metin hücrelerinin başına tek tırnak (') eklenir (OWASP önerisi).
"""

from __future__ import annotations

import io
import re
from typing import Dict, List, Tuple

import pandas as pd

TEHLIKELI_BASLANGIC = ("=", "+", "-", "@", "\t", "\r", "＝", "＋", "－", "＠")
EXCEL_HUCRE_SINIRI = 32767
# Ücretsiz sunucunun bellek sınırı nedeniyle Excel'de sayfa başına en çok bu kadar satır
# yazılır (tablolar sıralı olduğundan en üstteki satırlar). Eksiksiz tablo CSV ile alınır.
EXCEL_SATIR_SINIRI = 50000
_KISALTMA_NOTU = " […Excel hücre sınırı nedeniyle kısaltıldı]"


def _metin_sutunlari(tablo: pd.DataFrame):
    return [c for c in tablo.columns if not pd.api.types.is_numeric_dtype(tablo[c])
            and not pd.api.types.is_bool_dtype(tablo[c])]


def formul_guvenli(tablo: pd.DataFrame) -> pd.DataFrame:
    """Formül gibi başlayan metin hücrelerinin başına ' ekler (yalnızca CSV için)."""
    sutunlar = _metin_sutunlari(tablo)
    if not sutunlar:
        return tablo
    kopya = tablo.copy()
    for c in sutunlar:
        kopya[c] = kopya[c].map(lambda v: "'" + v if isinstance(v, str) and v.startswith(TEHLIKELI_BASLANGIC) else v)
    return kopya


def csv_bayt(tablo: pd.DataFrame, ayirici: str = ",", formul_korumasi: bool = True) -> bytes:
    """UTF-8 (BOM'lu) CSV. BOM sayesinde Excel Türkçe karakterleri doğru gösterir.

    Ayırıcı ';' seçilirse ondalık ayırıcı virgül olur (Türkçe Excel ayarlarıyla uyumlu).
    """
    if formul_korumasi:
        tablo = formul_guvenli(tablo)
    ondalik = "," if ayirici == ";" else "."
    return tablo.to_csv(index=False, sep=ayirici, decimal=ondalik).encode("utf-8-sig")


def _sayfa_adi(ad: str, kullanilan: set) -> str:
    # Excel sayfa adı: en fazla 31 karakter, []:*?/\\ içeremez
    temiz = re.sub(r"[\[\]:*?/\\]", "_", ad)[:31] or "Sayfa"
    aday, i = temiz, 2
    while aday in kullanilan:
        ek = f" ({i})"
        aday = temiz[:31 - len(ek)] + ek
        i += 1
    kullanilan.add(aday)
    return aday


def _excel_sinirina_uydur(tablo: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Excel'in 32.767 karakterlik hücre sınırını aşan metinleri açıkça işaretleyerek kısaltır."""
    sayi = 0
    kopya = None
    for c in _metin_sutunlari(tablo):
        uzun = tablo[c].map(lambda v: isinstance(v, str) and len(v) > EXCEL_HUCRE_SINIRI)
        if uzun.any():
            if kopya is None:
                kopya = tablo.copy()
            sayi += int(uzun.sum())
            kopya.loc[uzun, c] = kopya.loc[uzun, c].map(
                lambda v: v[:EXCEL_HUCRE_SINIRI - len(_KISALTMA_NOTU)] + _KISALTMA_NOTU)
    return (kopya if kopya is not None else tablo), sayi


def excel_bayt_ve_uyarilar(tablolar: Dict[str, pd.DataFrame],
                           satir_siniri: int = EXCEL_SATIR_SINIRI) -> Tuple[bytes, List[str]]:
    """Birden çok tabloyu tek Excel çalışma kitabında ayrı sayfalar olarak yazar."""
    tampon = io.BytesIO()
    kullanilan: set = set()
    uyarilar = []
    with pd.ExcelWriter(tampon, engine="xlsxwriter",
                        engine_kwargs={"options": {"strings_to_formulas": False, "strings_to_urls": False,
                                                   "strings_to_numbers": False,
                                                   # Geçici dosya kullanma: veriler diske yazılmaz
                                                   "in_memory": True}}) as yazici:
        for ad, tablo in tablolar.items():
            sayfa = _sayfa_adi(ad, kullanilan)
            if satir_siniri and len(tablo) > satir_siniri:
                uyarilar.append(f"'{sayfa}' sayfasına {len(tablo):,} satırın ilk {satir_siniri:,} satırı yazıldı "
                                "(bellek sınırı). Tablonun tamamı için CSV indirin.".replace(",", "."))
                tablo = tablo.head(satir_siniri)
            tablo, kisaltilan = _excel_sinirina_uydur(tablo)
            if kisaltilan:
                uyarilar.append(f"'{sayfa}' sayfasında {kisaltilan} hücre Excel'in 32.767 karakter sınırını aştığı "
                                "için kısaltıldı (CSV çıktısında tam hali bulunur).")
            tablo.to_excel(yazici, sheet_name=sayfa, index=False)
            ws = yazici.sheets[sayfa]
            for i, sutun in enumerate(tablo.columns):
                uzunluk = max([len(str(sutun))] + [len(str(v)) for v in tablo[sutun].head(500)])
                ws.set_column(i, i, min(max(10, uzunluk + 2), 60))
            ws.freeze_panes(1, 0)
    return tampon.getvalue(), uyarilar


def excel_bayt(tablolar: Dict[str, pd.DataFrame]) -> bytes:
    return excel_bayt_ve_uyarilar(tablolar)[0]


def sozluk_tablosu(sozluk: dict, anahtar: str = "Ayar", deger: str = "Değer") -> pd.DataFrame:
    return pd.DataFrame([(k, str(v)) for k, v in sozluk.items()], columns=[anahtar, deger])
