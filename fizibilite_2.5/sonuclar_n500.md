# Karşılaştırma sonuçları (n = 500)

Veri: DergiPark Türkçe özetleri (humakilicunlu/DergiParkEn, TRAll.txt); uygun benzersiz belge 1395, örneklem 500 (tohum 2501); özet ortanca 139 sözcük. Görev B sorgu: 269 (ortalama 5.99 ilgili belge).

Değerler: ortalama [%95 bootstrap GA]. Fark sütunu: yöntem − tfidf_tml (eşleştirilmiş).

## Görev A — başlık → kendi özeti (tek ilgili belge)

| Yöntem | İsabet@1 | İsabet@5 | MRR | nDCG@5 | Fark nDCG@5 |
|---|---|---|---|---|---|
| tfidf_ham | 0.914 [0.890, 0.938] | 0.980 [0.966, 0.990] | 0.943 [0.925, 0.959] | 0.952 [0.935, 0.966] | +0.004 [-0.000, +0.009] |
| tfidf_tml | 0.904 [0.876, 0.928] | 0.980 [0.966, 0.990] | 0.937 [0.919, 0.953] | 0.947 [0.931, 0.962] | — |
| tfidf_tml_terim | 0.896 [0.870, 0.920] | 0.980 [0.966, 0.990] | 0.932 [0.914, 0.949] | 0.944 [0.927, 0.958] | -0.004 [-0.008, +0.000] |
| tfidf_tml_lemma | 0.930 [0.908, 0.950] | 0.984 [0.972, 0.994] | 0.954 [0.938, 0.968] | 0.962 [0.947, 0.974] | +0.014 [+0.006, +0.023] |
| minilm_ozgun | çalıştırılamadı: httpx.ProxyError: 403 Forbidden | | | | |

## Görev B — özet → özet, ortak yazar anahtar sözcüğü (derecelendirilmiş)

| Yöntem | İsabet@1 | İsabet@5 | MRR | nDCG@5 | Fark nDCG@5 |
|---|---|---|---|---|---|
| tfidf_ham | 0.349 [0.294, 0.405] | 0.606 [0.550, 0.665] | 0.471 [0.422, 0.520] | 0.322 [0.280, 0.364] | +0.004 [-0.007, +0.016] |
| tfidf_tml | 0.338 [0.282, 0.394] | 0.602 [0.543, 0.662] | 0.468 [0.421, 0.517] | 0.317 [0.277, 0.359] | — |
| tfidf_tml_terim | 0.342 [0.282, 0.398] | 0.599 [0.543, 0.658] | 0.469 [0.419, 0.518] | 0.316 [0.275, 0.357] | -0.002 [-0.012, +0.010] |
| tfidf_tml_lemma | 0.342 [0.282, 0.398] | 0.591 [0.532, 0.654] | 0.459 [0.406, 0.508] | 0.303 [0.261, 0.344] | -0.014 [-0.038, +0.008] |
| minilm_ozgun | çalıştırılamadı: httpx.ProxyError: 403 Forbidden | | | | |

## Kaynak ölçümleri (bu makinede, CPU)

| Yöntem | Yükleme sn | 100 belge sn | Tüm belgeler sn | Tepe RAM MB | Vektör boyutu | Azami token | Kesilen belge | Disk MB | Eşzamanlı |
|---|---|---|---|---|---|---|---|---|---|
| tfidf_ham | — | — | 0.08 | 158.8 | 16609 | — | — | — | — |
| tfidf_tml | — | — | 0.36 | 155.4 | 16081 | — | — | — | — |
| tfidf_tml_terim | — | — | 0.75 | 165.3 | 16123 | — | — | — | — |
| tfidf_tml_lemma | — | — | 1.76 | 270.2 | 6858 | — | — | — | — |
