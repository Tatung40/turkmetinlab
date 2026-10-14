"""CSV, Excel ve TXT dosyalarını okuma.

Yüklenen dosyalar yalnızca bellekte (bayt olarak) işlenir; diske yazılmaz.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

# Denenecek kodlamalar (sıra önemlidir: UTF-8 katıdır, cp1254 neredeyse her
# baytı çözebildiği için sona yakın denenir).
KODLAMALAR = ("utf-8-sig", "utf-8", "cp1254", "iso-8859-9", "utf-16")

# Yanlış kodlamayla okunmuş UTF-8 metinde tipik bozuk dizgeler (mojibake)
_MOJIBAKE_RE = re.compile(r"Ã[\u0080-¿]|Ä[\u0080-¿]|Å[\u0080-¿]|Ä±|ÅŸ|Ã¼|Ã§|Ã¶|ÄŸ")

TXT_MODLARI = {
    "dosya": "Her dosya bir belge",
    "paragraf": "Her paragraf bir belge (boş satırla ayrılmış)",
    "satir": "Her satır bir belge",
}


class VeriOkumaHatasi(Exception):
    """Kullanıcıya gösterilecek, anlaşılır Türkçe hata."""


@dataclass
class OkumaSonucu:
    tablo: pd.DataFrame
    kodlama: Optional[str] = None
    ayirici: Optional[str] = None
    uyarilar: List[str] = field(default_factory=list)


def baytlari_coz(veri: bytes) -> Tuple[str, str]:
    """Baytları uygun kodlamayla metne çevirir. (metin, kodlama) döndürür."""
    if not veri:
        raise VeriOkumaHatasi("Dosya boş.")
    if veri[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return veri.decode("utf-16"), "utf-16"
    for kod in KODLAMALAR:
        if kod == "utf-16":
            continue
        try:
            return veri.decode(kod), kod
        except UnicodeDecodeError:
            continue
    raise VeriOkumaHatasi("Dosyanın karakter kodlaması tanınamadı. Lütfen dosyayı UTF-8 olarak kaydedin.")


def mojibake_var_mi(metin: str) -> bool:
    """Metinde 'Ã¼', 'ÅŸ' gibi bozuk Türkçe karakter izleri var mı?"""
    return bool(_MOJIBAKE_RE.search(metin[:200000]))


def _ayirici_bul(ornek: str) -> str:
    try:
        return csv.Sniffer().sniff(ornek, delimiters=",;\t|").delimiter
    except csv.Error:
        # Basit sayım yedeği
        ilk_satir = ornek.splitlines()[0] if ornek else ""
        adaylar = {a: ilk_satir.count(a) for a in [";", ",", "\t", "|"]}
        en_iyi = max(adaylar, key=adaylar.get)
        return en_iyi if adaylar[en_iyi] > 0 else ","


def csv_oku(veri: bytes) -> OkumaSonucu:
    metin, kodlama = baytlari_coz(veri)
    ayirici = _ayirici_bul(metin[:20000])
    # Ayrıştırma csv modülüyle açıkça yapılır: pandas, başlıktan fazla alan
    # içeren satırlarda sütunları sessizce kaydırabildiği için kullanılmaz.
    csv.field_size_limit(2**31 - 1)
    try:
        satirlar = list(csv.reader(io.StringIO(metin, newline=""), delimiter=ayirici))
    except csv.Error as e:
        raise VeriOkumaHatasi(f"CSV dosyası okunamadı: {e}") from e
    satirlar = [s for s in satirlar if any(h.strip() for h in s)]
    if not satirlar:
        raise VeriOkumaHatasi("Dosyada veri bulunamadı.")
    baslik, veri_satirlari = satirlar[0], satirlar[1:]
    genislik = len(baslik)
    gecerli, hatali = [], []
    for no, s in enumerate(veri_satirlari, start=2):
        if len(s) > genislik and any(h.strip() for h in s[genislik:]):
            hatali.append(no)
            continue
        s = (s + [""] * genislik)[:genislik]
        gecerli.append(s)
    tablo = pd.DataFrame(gecerli, columns=baslik, dtype=object)
    tablo = tablo.replace({"": None})
    uyarilar = []
    if hatali:
        ornek = ", ".join(map(str, hatali[:10])) + (" …" if len(hatali) > 10 else "")
        uyarilar.append(f"{len(hatali)} satır, başlıktan fazla sütun içerdiği için analize alınmadı "
                        f"(satır no: {ornek}). Metin içinde ayırıcı karakter varsa hücrenin tırnak içinde "
                        "olduğundan emin olun.")
    if mojibake_var_mi(metin):
        uyarilar.append("Dosyada bozuk Türkçe karakterler (ör. 'Ã¼', 'ÅŸ') görünüyor. Dosya daha önce "
                        "yanlış kodlamayla kaydedilmiş olabilir; sonuçları kontrol edin.")
    return OkumaSonucu(_temizle(tablo), kodlama, ayirici, uyarilar)


def excel_sayfalari(veri: bytes) -> List[str]:
    try:
        return list(pd.ExcelFile(io.BytesIO(veri), engine="openpyxl").sheet_names)
    except Exception as e:
        raise VeriOkumaHatasi("Excel dosyası açılamadı. Yalnızca .xlsx biçimi desteklenir "
                              "(eski .xls dosyalarını Excel'de 'Farklı Kaydet > .xlsx' ile dönüştürün).") from e


def excel_oku(veri: bytes, sayfa=0) -> OkumaSonucu:
    try:
        tablo = pd.read_excel(io.BytesIO(veri), sheet_name=sayfa, dtype=str, engine="openpyxl")
    except Exception as e:
        raise VeriOkumaHatasi("Excel dosyası okunamadı. Yalnızca .xlsx biçimi desteklenir.") from e
    return OkumaSonucu(_temizle(tablo))


def txt_belgelere_ayir(metin: str, mod: str = "dosya") -> List[str]:
    if mod not in TXT_MODLARI:
        raise ValueError(f"Geçersiz TXT modu: {mod}")
    metin = metin.replace("\r\n", "\n").replace("\r", "\n")
    if mod == "dosya":
        parcalar = [metin]
    elif mod == "paragraf":
        parcalar = re.split(r"\n\s*\n", metin)
    else:
        parcalar = metin.split("\n")
    return [p.strip() for p in parcalar if p and p.strip()]


def txt_oku(dosyalar: List[Tuple[str, bytes]], mod: str = "dosya") -> OkumaSonucu:
    """Bir veya daha fazla TXT dosyasını belge tablosuna çevirir.

    dosyalar: (dosya_adı, bayt_içerik) listesi
    """
    satirlar, uyarilar, kodlamalar = [], [], set()
    for ad, veri in dosyalar:
        try:
            metin, kod = baytlari_coz(veri)
        except VeriOkumaHatasi:
            uyarilar.append(f"'{ad}' boş ya da okunamadı; atlandı.")
            continue
        kodlamalar.add(kod)
        if mojibake_var_mi(metin):
            uyarilar.append(f"'{ad}' dosyasında bozuk Türkçe karakterler görünüyor.")
        parcalar = txt_belgelere_ayir(metin, mod)
        for i, p in enumerate(parcalar, start=1):
            etiket = ad if mod == "dosya" else f"{ad} — {'paragraf' if mod == 'paragraf' else 'satır'} {i}"
            satirlar.append({"belge": etiket, "metin": p})
    if not satirlar:
        raise VeriOkumaHatasi("Yüklenen TXT dosyalarında okunabilir metin bulunamadı.")
    return OkumaSonucu(pd.DataFrame(satirlar), ", ".join(sorted(kodlamalar)), None, uyarilar)


def _temizle(tablo: pd.DataFrame) -> pd.DataFrame:
    tablo = tablo.dropna(how="all").dropna(axis=1, how="all")
    tablo.columns = [str(c).strip() if str(c).strip() else f"Sütun {i + 1}" for i, c in enumerate(tablo.columns)]
    # Yinelenen sütun adlarını tekilleştir
    gorulen = {}
    yeni = []
    for c in tablo.columns:
        if c in gorulen:
            gorulen[c] += 1
            yeni.append(f"{c} ({gorulen[c]})")
        else:
            gorulen[c] = 0
            yeni.append(c)
    tablo.columns = yeni
    if tablo.empty:
        raise VeriOkumaHatasi("Dosyada veri bulunamadı.")
    return tablo.reset_index(drop=True)


def metin_sutunu_oner(tablo: pd.DataFrame) -> Optional[str]:
    """Ortalama uzunluğu en fazla olan metin sütununu önerir."""
    en_iyi, en_uzun = None, -1.0
    for c in tablo.columns:
        degerler = tablo[c].dropna().astype(str)
        if degerler.empty:
            continue
        # Çoğunluğu sayı olan sütunları atla
        sayisal_oran = degerler.str.fullmatch(r"\s*[-+]?\d+([.,]\d+)?\s*").mean()
        if sayisal_oran > 0.8:
            continue
        ort = degerler.str.len().mean()
        if ort > en_uzun:
            en_iyi, en_uzun = c, ort
    return en_iyi


def dosya_oku(ad: str, veri: bytes, sayfa=0, txt_modu: str = "dosya") -> OkumaSonucu:
    """Uzantıya göre uygun okuyucuyu çağırır."""
    uzanti = ad.lower().rsplit(".", 1)[-1] if "." in ad else ""
    if uzanti in ("csv", "tsv"):
        return csv_oku(veri)
    if uzanti in ("xlsx", "xlsm"):
        return excel_oku(veri, sayfa)
    if uzanti == "xls":
        raise VeriOkumaHatasi("Eski .xls biçimi desteklenmiyor. Excel'de 'Farklı Kaydet > Excel Çalışma "
                              "Kitabı (.xlsx)' seçeneğiyle dönüştürüp yeniden yükleyin.")
    if uzanti == "txt":
        return txt_oku([(ad, veri)], txt_modu)
    raise VeriOkumaHatasi(f"Desteklenmeyen dosya türü: .{uzanti}. CSV, XLSX veya TXT yükleyin.")
