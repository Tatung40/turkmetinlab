# Karşılaştırma sonuçları (n = 100)

Veri: DergiPark Türkçe özetleri (humakilicunlu/DergiParkEn, TRAll.txt); uygun benzersiz belge 1395, örneklem 100 (tohum 2501); özet ortanca 135 sözcük. Görev B sorgu: 28 (ortalama 2.14 ilgili belge).

Değerler: ortalama [%95 bootstrap GA]. Fark sütunu: yöntem − tfidf_tml (eşleştirilmiş).

## Görev A — başlık → kendi özeti (tek ilgili belge)

| Yöntem | İsabet@1 | İsabet@5 | MRR | nDCG@5 | Fark nDCG@5 |
|---|---|---|---|---|---|
| tfidf_ham | 0.970 [0.930, 1.000] | 0.970 [0.930, 1.000] | 0.970 [0.931, 1.000] | 0.970 [0.930, 1.000] | +0.000 [+0.000, +0.000] |
| tfidf_tml | 0.970 [0.930, 1.000] | 0.970 [0.930, 1.000] | 0.971 [0.932, 1.000] | 0.970 [0.930, 1.000] | — |
| tfidf_tml_terim | 0.970 [0.930, 1.000] | 0.970 [0.930, 1.000] | 0.971 [0.932, 1.000] | 0.970 [0.930, 1.000] | +0.000 [+0.000, +0.000] |
| tfidf_tml_lemma | 0.970 [0.930, 1.000] | 0.970 [0.930, 1.000] | 0.973 [0.936, 1.000] | 0.970 [0.930, 1.000] | +0.000 [+0.000, +0.000] |

## Görev B — özet → özet, ortak yazar anahtar sözcüğü (derecelendirilmiş)

| Yöntem | İsabet@1 | İsabet@5 | MRR | nDCG@5 | Fark nDCG@5 |
|---|---|---|---|---|---|
| tfidf_ham | 0.393 [0.214, 0.571] | 0.607 [0.429, 0.786] | 0.468 [0.305, 0.628] | 0.405 [0.252, 0.559] | -0.052 [-0.099, -0.009] |
| tfidf_tml | 0.429 [0.250, 0.607] | 0.607 [0.429, 0.786] | 0.535 [0.375, 0.692] | 0.457 [0.304, 0.612] | — |
| tfidf_tml_terim | 0.429 [0.250, 0.607] | 0.607 [0.429, 0.786] | 0.541 [0.383, 0.695] | 0.459 [0.308, 0.615] | +0.002 [-0.002, +0.008] |
| tfidf_tml_lemma | 0.429 [0.250, 0.607] | 0.571 [0.393, 0.750] | 0.511 [0.346, 0.672] | 0.433 [0.283, 0.585] | -0.024 [-0.078, +0.019] |

## Kaynak ölçümleri (bu makinede, CPU)

| Yöntem | Yükleme sn | 100 belge sn | Tüm belgeler sn | Tepe RAM MB | Vektör boyutu | Azami token | Kesilen belge | Disk MB | Eşzamanlı |
|---|---|---|---|---|---|---|---|---|---|
| tfidf_ham | — | — | 0.02 | 143.0 | 5570 | — | — | — | — |
| tfidf_tml | — | — | 0.1 | 142.8 | 5294 | — | — | — | — |
| tfidf_tml_terim | — | — | 0.18 | 145.8 | 5300 | — | — | — | — |
| tfidf_tml_lemma | — | — | 1.16 | 263.7 | 2785 | — | — | — | — |
