"""Türkçe lemmatizasyon yöntemlerinin UD Türkçe altın standart verilerinde değerlendirilmesi.

Bu betik uygulamanın parçası değildir; TürkMetinLab 1.1'deki yöntem seçiminin nasıl yapıldığını
yeniden üretmek içindir (sonuçlar: docs/bilimsel_degerlendirme_1.1.md).

Gerekenler (ayrı bir sanal ortamda, Python 3.12):
    pip install simplemma snowballstemmer zeyrek zemberek-python "setuptools<81" psutil
Veri (CC BY-SA 4.0):
    https://github.com/UniversalDependencies/UD_Turkish-BOUN  -> tr_boun-ud-test.conllu
    https://github.com/UniversalDependencies/UD_Turkish-IMST  -> tr_imst-ud-test.conllu
Kullanım:
    python degerlendirme/lemma_degerlendirmesi.py tr_boun-ud-test.conllu 100000 kimlik,snowball,simplemma_kucuk,zemberek
"""
import gc, json, os, sys, time, unicodedata
import psutil

UD = sys.argv[1]
SINIR = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
proc = psutil.Process()


def kucuk(s):
    s = unicodedata.normalize("NFC", s).replace("İ", "i").replace("I", "ı").lower().replace("i̇", "i")
    return s.translate(str.maketrans("âîû", "aiu"))


def norm_lemma(s):
    s = kucuk(s)
    for ek in ("mek", "mak"):  # Zemberek fiil sözlük biçimi (gelmek) -> UD kök (gel)
        if len(s) > 4 and s.endswith(ek):
            return s[:-3]
    return s


def cumleler(yol):
    cumle, mwt_son, mwt = [], 0, None
    for satir in open(yol, encoding="utf-8"):
        satir = satir.rstrip("\n")
        if not satir:
            if cumle:
                yield cumle
            cumle, mwt_son = [], 0
            continue
        if satir.startswith("#"):
            continue
        a = satir.split("\t")
        if "-" in a[0]:
            bas, son = map(int, a[0].split("-"))
            mwt = {"form": a[1], "lemma": None, "upos": None}
            mwt_son = son
            cumle.append(mwt)
            continue
        if "." in a[0]:
            continue
        no = int(a[0])
        if mwt_son and no <= mwt_son:
            if mwt["lemma"] is None:  # ilk alt sözcüğün lemması ana lemmadır
                mwt["lemma"], mwt["upos"] = a[2], a[3]
            continue
        cumle.append({"form": a[1], "lemma": a[2], "upos": a[3]})
    if cumle:
        yield cumle


tum = list(cumleler(UD))[:SINIR]
ICERIK = {"NOUN", "VERB", "ADJ", "PROPN", "ADV"}


def degerlendir(ad, tahmin_fn, cumle_duzeyi=False):
    gc.collect()
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    hazirla = tahmin_fn()  # yükleme
    yukleme = time.perf_counter() - t0
    rss1 = proc.memory_info().rss
    dogru = toplam = dogru_ic = toplam_ic = 0
    ornek_hata = []
    t1 = time.perf_counter()
    for c in tum:
        sozcukler = [t for t in c]
        tahminler = hazirla(sozcukler) if cumle_duzeyi else [hazirla(t["form"]) for t in sozcukler]
        for t, p in zip(sozcukler, tahminler):
            if t["upos"] in ("PUNCT", "NUM", "SYM", None):
                continue
            g = norm_lemma(t["lemma"])
            ok = norm_lemma(p) == g
            dogru += ok; toplam += 1
            if t["upos"] in ICERIK:
                dogru_ic += ok; toplam_ic += 1
            if not ok and len(ornek_hata) < 12 and t["upos"] in ICERIK:
                ornek_hata.append(f"{t['form']} -> {p} (altın: {t['lemma']})")
    sure = time.perf_counter() - t1
    return {"yöntem": ad, "doğruluk_tüm": round(dogru / toplam, 4), "doğruluk_içerik": round(dogru_ic / toplam_ic, 4),
            "sözcük": toplam, "içerik_sözcük": toplam_ic, "yükleme_sn": round(yukleme, 2),
            "sözcük_başına_ms": round(sure / toplam * 1000, 3), "ek_bellek_MB": round((rss1 - rss0) / 2**20, 1),
            "örnek_hatalar": ornek_hata}


def simplemma_kucuk():
    import simplemma
    return lambda w: simplemma.lemmatize(kucuk(w), lang="tr")


def kimlik():
    return lambda w: w


def snowball():
    import snowballstemmer
    s = snowballstemmer.stemmer("turkish")
    return lambda w: s.stemWord(kucuk(w))


def simplemma_():
    import simplemma
    return lambda w: simplemma.lemmatize(w, lang="tr")


def zeyrek_():
    import zeyrek
    a = zeyrek.MorphAnalyzer()
    def f(w):
        r = [x for x in a._parse(w) if x is not None]
        return r[0].dict_item.lemma if r else w
    return f


def zemberek_():
    from zemberek import TurkishMorphology
    m = TurkishMorphology.create_with_defaults()
    def f(sozcukler):
        bicimler = [t["form"] for t in sozcukler]
        sonuc = m.disambiguate(" ".join(bicimler), [m.analyze(b) for b in bicimler]).best_analysis()
        return [b if a.is_unknown() else a.item.lemma for b, a in zip(bicimler, sonuc)]
    return f


yontemler = sys.argv[3].split(",") if len(sys.argv) > 3 else ["kimlik", "snowball", "simplemma", "zeyrek", "zemberek"]
sonuclar = []
for y in yontemler:
    fn = {"kimlik": kimlik, "snowball": snowball, "simplemma": simplemma_, "simplemma_kucuk": simplemma_kucuk, "zeyrek": zeyrek_, "zemberek": zemberek_}[y]
    r = degerlendir(y, fn, cumle_duzeyi=(y == "zemberek"))
    print(json.dumps({k: v for k, v in r.items() if k != "örnek_hatalar"}, ensure_ascii=False), flush=True)
    sonuclar.append(r)
print(json.dumps(sonuclar, ensure_ascii=False, indent=1))
