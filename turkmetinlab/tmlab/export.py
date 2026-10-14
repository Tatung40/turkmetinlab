"""Sonuçların CSV ve Excel olarak dışa aktarılması."""

from __future__ import annotations

import io
import re
from typing import Dict

import pandas as pd


def csv_bayt(tablo: pd.DataFrame, ayirici: str = ",") -> bytes:
    """UTF-8 (BOM'lu) CSV. BOM sayesinde Excel Türkçe karakterleri doğru gösterir.

    Ayırıcı ';' seçilirse ondalık ayırıcı virgül olur (Türkçe Excel ayarlarıyla uyumlu).
    """
    ondalik = "," if ayirici == ";" else "."
    return tablo.to_csv(index=False, sep=ayirici, decimal=ondalik).encode("utf-8-sig")


def _sayfa_adi(ad: str, kullanilan: set) -> str:
    # Excel sayfa adı: en fazla 31 karakter, []:*?/\ içeremez
    temiz = re.sub(r"[\[\]:*?/\\]", "_", ad)[:31] or "Sayfa"
    aday, i = temiz, 2
    while aday in kullanilan:
        ek = f" ({i})"
        aday = temiz[:31 - len(ek)] + ek
        i += 1
    kullanilan.add(aday)
    return aday


def excel_bayt(tablolar: Dict[str, pd.DataFrame]) -> bytes:
    """Birden çok tabloyu tek Excel çalışma kitabında ayrı sayfalar olarak yazar."""
    tampon = io.BytesIO()
    kullanilan: set = set()
    with pd.ExcelWriter(tampon, engine="openpyxl") as yazici:
        for ad, tablo in tablolar.items():
            sayfa = _sayfa_adi(ad, kullanilan)
            tablo.to_excel(yazici, sheet_name=sayfa, index=False)
            ws = yazici.sheets[sayfa]
            # Sütun genişliklerini içeriğe göre ayarla
            for i, sutun in enumerate(tablo.columns, start=1):
                uzunluk = max([len(str(sutun))] + [len(str(v)) for v in tablo[sutun].head(500)])
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(max(10, uzunluk + 2), 60)
            ws.freeze_panes = "A2"
    return tampon.getvalue()


def sozluk_tablosu(sozluk: dict, anahtar: str = "Ayar", deger: str = "Değer") -> pd.DataFrame:
    return pd.DataFrame([(k, str(v)) for k, v in sozluk.items()], columns=[anahtar, deger])
