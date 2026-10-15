"""N-gram bölütleme kuralının gerekçesi (TürkMetinLab 1.1).

Kullanım: python degerlendirme/bolutleme_analizi.py tr_boun-ud-train.conllu tr_boun-ud-dev.conllu ...

Noktalama işaretinin iki yanındaki sözcükler sözdizimsel olarak aynı öbekte mi?

UD ağaç bankasında, aralarında tek bir noktalama işareti bulunan (veya hiç
bulunmayan) iki ardışık sözcük için doğrudan bağımlılık ilişkisi ve türü
incelenir. N-gramlar çok sözcüklü öbekleri (ör. "uzaktan eğitim") yakalamak
için kullanıldığından, bir sınırın iki yanındaki sözcükler nadiren öbek içi
ilişkiyle bağlıysa o sınır n-gram sınırı sayılmalıdır.
"""
import collections
import sys

OBEK_ICI = {"amod", "compound", "compound:lvc", "compound:redup", "nmod:poss", "nmod", "det", "nummod", "flat",
            "fixed", "case", "obj", "obl", "advmod", "nsubj", "aux", "cop", "acl", "advcl", "ccomp", "xcomp",
            "iobj", "mark", "clf", "discourse", "vocative", "dep", "orphan", "list", "appos", "dislocated",
            "csubj", "goeswith", "reparandum", "expl", "nmod:part", "advmod:emph", "aux:q", "obl:tmod",
            "nsubj:pass", "csubj:pass", "acl:relcl", "det:predet", "nummod:gov", "obl:agent"}
KOORD = {"conj", "cc", "parataxis", "root"}


def cumleler(yol):
    c = []
    for s in open(yol, encoding="utf-8"):
        s = s.rstrip("\n")
        if not s:
            if c:
                yield c
            c = []
            continue
        if s.startswith("#"):
            continue
        a = s.split("\t")
        if "-" in a[0] or "." in a[0]:
            continue
        c.append({"id": int(a[0]), "form": a[1], "upos": a[3], "head": int(a[6]), "rel": a[7]})
    if c:
        yield c


def iliski(a, b):
    if a["head"] == b["id"]:
        return a["rel"]
    if b["head"] == a["id"]:
        return b["rel"]
    return None


istat = collections.defaultdict(collections.Counter)
for yol in sys.argv[1:]:
    for c in cumleler(yol):
        i = 0
        while i < len(c) - 1:
            a = c[i]
            if a["upos"] == "PUNCT":
                i += 1
                continue
            j = i + 1
            arada = []
            while j < len(c) and c[j]["upos"] == "PUNCT":
                arada.append(c[j]["form"])
                j += 1
            if j >= len(c):
                break
            b = c[j]
            if not arada:
                tur = "noktalama yok"
            elif arada == [","]:
                tur = "virgül"
            elif arada == [";"]:
                tur = "noktalı virgül"
            elif arada == [":"]:
                tur = "iki nokta"
            elif arada in (["-"], ["–"], ["—"]):
                tur = "tire"
            else:
                tur = "diğer: " + " ".join(arada)
            r = iliski(a, b)
            if r is None:
                istat[tur]["doğrudan ilişki yok"] += 1
            elif r.split(":")[0] in {"conj", "cc", "parataxis"} or r in KOORD:
                istat[tur]["eşgüdüm/yan yana cümle (conj, parataxis)"] += 1
            else:
                istat[tur]["öbek içi ilişki"] += 1
                istat[tur + " · ilişki"][r] += 1
            i = j

for tur in ["noktalama yok", "virgül", "noktalı virgül", "iki nokta", "tire"]:
    s = istat[tur]
    n = sum(s.values())
    if not n:
        continue
    print(f"{tur:15s} n={n:6d}  " + "  ".join(f"{k}: {v / n:.1%}" for k, v in s.most_common()))
    if tur != "noktalama yok":
        print("      öbek içi ilişki türleri:", istat[tur + " · ilişki"].most_common(6))
