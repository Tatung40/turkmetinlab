"""Testler için ortak yardımcılar: UD Turkish-BOUN örnek verisini okuma."""

import os

import pytest

UD_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "veri", "ud_boun_test_ornek.conllu")


def ud_cumleleri(yol: str = UD_YOLU):
    """CoNLL-U dosyasını okur.

    Her cümle için: {"id", "metin", "tur" (bio/ess/ins/news/pop), "sozcukler": [...]} döner.
    Çok parçalı sözcüklerde (ör. "böyledir" = "böyle" + "dir") yüzey biçim tek sözcük
    sayılır ve ilk parçanın lemması/etiketi kullanılır.
    """
    cumleler = []
    c = None
    mwt_son, mwt = 0, None
    with open(yol, encoding="utf-8") as f:
        for satir in f:
            satir = satir.rstrip("\n")
            if not satir:
                if c and c["sozcukler"]:
                    cumleler.append(c)
                c, mwt_son = None, 0
                continue
            if c is None:
                c = {"id": "", "metin": "", "tur": "", "sozcukler": []}
            if satir.startswith("# sent_id"):
                c["id"] = satir.split("=", 1)[1].strip()
                c["tur"] = c["id"].split("_")[0]
                continue
            if satir.startswith("# text"):
                c["metin"] = satir.split("=", 1)[1].strip()
                continue
            if satir.startswith("#"):
                continue
            a = satir.split("\t")
            if "-" in a[0]:
                mwt_son = int(a[0].split("-")[1])
                mwt = {"no": int(a[0].split("-")[0]), "bicim": a[1], "lemma": None, "upos": None,
                       "bas": 0, "iliski": "", "misc": a[9]}
                c["sozcukler"].append(mwt)
                continue
            if "." in a[0]:
                continue
            no = int(a[0])
            if mwt_son and no <= mwt_son:
                if mwt["lemma"] is None:
                    mwt.update(lemma=a[2], upos=a[3], bas=int(a[6]), iliski=a[7])
                continue
            c["sozcukler"].append({"no": no, "bicim": a[1], "lemma": a[2], "upos": a[3], "bas": int(a[6]),
                                   "iliski": a[7], "misc": a[9]})
    if c and c["sozcukler"]:
        cumleler.append(c)
    return cumleler


@pytest.fixture(scope="session")
def ud():
    return ud_cumleleri()
