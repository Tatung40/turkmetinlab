"""Veri boyutu denetimi ve ücretsiz plan uyarıları.

Sınırlar ölçüme dayanır (degerlendirme/performans_olcumu.py, 2 çekirdek):

    Derlem (gerçekçi dağarcık)       Süre    Tepe bellek (Excel öncesi)
    288 bin sözcük / 60 bin tür      ~4 sn   ~290 MB
    2,9 milyon sözcük / 175 bin tür  ~22 sn  ~700 MB (lemmatizasyonla ~850 MB)

Streamlit Community Cloud bir uygulamaya yaklaşık 690 MB ile 2,7 GB arası bellek
verir ve uygulamanın tüm kullanıcıları aynı süreci paylaşır: bir kullanıcının
belleği aşan analizi uygulamayı herkes için durdurabilir. Bu yüzden:

* UYARI_SOZCUK üzerindeki derlemler için kullanıcı uyarılır.
* EN_FAZLA_SOZCUK üzerindeki derlemler bulutta analiz edilmez; kullanıcıya
  veriyi bölmesi ya da uygulamayı kendi bilgisayarında çalıştırması önerilir.

Kendi bilgisayarında çalıştıranlar sınırları ortam değişkenleriyle
değiştirebilir: TML_UYARI_SOZCUK, TML_EN_FAZLA_SOZCUK.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable


def _ortam_sayisi(ad: str, varsayilan: int) -> int:
    try:
        return max(1, int(os.environ.get(ad, varsayilan)))
    except ValueError:
        return varsayilan


UYARI_SOZCUK = _ortam_sayisi("TML_UYARI_SOZCUK", 500_000)
EN_FAZLA_SOZCUK = _ortam_sayisi("TML_EN_FAZLA_SOZCUK", 1_500_000)
# Arayüzde gösterilen en fazla satır (tamamı indirilebilir)
GOSTERIM_SATIR_SINIRI = 5000
ONIZLEME_BELGE_SINIRI = 500


@dataclass
class BoyutDegerlendirmesi:
    duzey: str  # "normal" | "uyari" | "engel"
    sozcuk: int
    mesaj: str = ""


def sozcuk_tahmini(metinler: Iterable[str]) -> int:
    """Boşluklara göre hızlı sözcük sayımı (ön işleme öncesi)."""
    return sum(len(str(m).split()) for m in metinler)


def _bicim(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def boyut_degerlendir(sozcuk: int, lemmatizasyon: bool = False,
                      uyari: int = None, en_fazla: int = None) -> BoyutDegerlendirmesi:
    uyari = UYARI_SOZCUK if uyari is None else uyari
    en_fazla = EN_FAZLA_SOZCUK if en_fazla is None else en_fazla
    if sozcuk > en_fazla:
        return BoyutDegerlendirmesi("engel", sozcuk, (
            f"Bu veri yaklaşık {_bicim(sozcuk)} sözcük içeriyor. Ücretsiz sunucunun bellek sınırı nedeniyle "
            f"en fazla {_bicim(en_fazla)} sözcük analiz edilebilir; daha büyük bir analiz uygulamayı tüm "
            "kullanıcılar için durdurabilir. Seçenekler: (1) veriyi birkaç dosyaya bölün, (2) uygulamayı kendi "
            "bilgisayarınızda çalıştırın (README'de anlatılıyor; orada bu sınır yükseltilebilir)."))
    if sozcuk > uyari:
        ek = " Lemmatizasyon açık olduğu için işlem daha uzun sürer ve daha çok bellek kullanır." if lemmatizasyon else ""
        return BoyutDegerlendirmesi("uyari", sozcuk, (
            f"Büyük veri: yaklaşık {_bicim(sozcuk)} sözcük. Ücretsiz sunucuda analiz bir dakikayı bulabilir; "
            f"her ayar değişikliğinde hesaplama yinelenir.{ek} Arayüzdeki tablolar ilk "
            f"{_bicim(GOSTERIM_SATIR_SINIRI)} satırla sınırlıdır; tamamını İndir sekmesinden alabilirsiniz."))
    return BoyutDegerlendirmesi("normal", sozcuk)
