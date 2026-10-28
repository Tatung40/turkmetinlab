"""Bellek ölçümü ve süreç yaşam döngüsü testleri (model gerekmez). Gerçek E5 ölçümü değildir."""
import os
import sys
import time

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BURA))
from anlamsal_karsilastirma import _agac_bellek, _sinirli_calistir  # noqa: E402

TORUN = """import os, sys, time
if os.fork() == 0:  # torun: büyük bellek ayırır
    b = bytearray(int(sys.argv[1]) * 2**20); [b.__setitem__(i, 1) for i in range(0, len(b), 4096)]
    time.sleep(float(sys.argv[2])); os._exit(0)
time.sleep(float(sys.argv[3]))
"""


def test_torun_bellegi_sayilir_ve_sinirda_durur():
    bas = time.time()
    kod, _, _, sinir, olcum = _sinirli_calistir([sys.executable, "-c", TORUN, "400", "30", "30"], 60, 300)
    assert sinir and "bellek" in sinir and time.time() - bas < 20  # alt süreç küçük, torun büyük
    assert olcum["en_cok_surec"] == 2 and olcum["dis_tepe_rss_MB"] > 300


def test_artik_surec_sayilir_ve_sonlandirilir():
    kod, _, _, sinir, olcum = _sinirli_calistir([sys.executable, "-c", TORUN, "10", "60", "2"], 30, 5000)
    assert kod == 0 and sinir is None and olcum["artik_surec"] == 1  # alt süreç bitti, torun kaldı
    for _ in range(30):  # sonlandırılan torun en geç 3 sn içinde kaybolmalı
        if not _kalan():
            break
        time.sleep(0.1)
    assert not _kalan()


def _kalan():
    sonuc = []
    for p in filter(str.isdigit, os.listdir("/proc")):
        try:
            if TORUN[:20].encode() in open(f"/proc/{p}/cmdline", "rb").read():
                sonuc.append(p)
        except OSError:
            pass
    return sonuc


def test_pss_ve_rss_ayri_olculur():
    kod, _, _, sinir, olcum = _sinirli_calistir([sys.executable, "-c", "import time; time.sleep(1.5)"], 30, 5000)
    assert kod == 0 and 0 < olcum["dis_tepe_pss_MB"] <= olcum["dis_tepe_rss_MB"]
