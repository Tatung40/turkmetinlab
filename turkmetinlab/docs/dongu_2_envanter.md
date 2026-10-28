# Döngü 2 — kompakt karar (28 Ekim 2026)

Taban: gelistirme-3-0-nitel-mvp `c1a7ddb` (PR #11 squash ile birleştirildi; CI 4/4 yeşildi).

| Alan | Karar | Gerekçe |
| --- | --- | --- |
| `birliktelik.dugum_tablosu` derece/strength | NETWORKX'E TAŞINDI — bu PR | `nx.from_pandas_edgelist` + `Graph.degree(weight=)`. kenar_tablosu çiftleri bir kez ve öz-ilmeksiz verir (triu k=1); yinelenen çift/öz-ilmek artık açıkça reddedilir (sessiz çökme yok). Normalleştirme d/(n−1), n = gösterilen ağ. pandas concat/groupby özel toplama kaldırıldı. |
| `birliktelik._birim_matrisi, _pencere_sayimi, _pmi`, Jaccard | KORU | Birim/pencere/bölüt sınırı ve PMI paydaları yöntem tanımı; kütüphane karşılığı eşdeğer değil. |
| `grup._z_degerleri`, permütasyon eşikleri | DENEYSEL / KORU | Monroe vd. bilgilendirici önselli log-odds; scipy/statsmodels standart testleri aynı yöntem değil. `_grup_sayimlari` zaten seyrek gösterge çarpımı. |
| `benzerlik` | Döngü 1'de bitti | linear_kernel (PR #11). |

Eşdeğerlik: `tests/test_birliktelik_networkx.py` eski groupby kodunu dondurulmuş oracle olarak tutar;
3 bağlam × 3 ölçüt × 3 ağ boyutu (1–25 kenar, Türkçe İ/ı/ç/ğ/ş, None boşluk, tekil belge),
atol=1e−12, sıra ve sütunlar birebir. Ek: boş şema, yinelenen/öz-ilmek reddi. Mevcut
`test_derece_networkx_ile_ayni` de geçer. Yeni bağımlılık yok (networkx zaten requirements'ta).

## Döngü 3 öncelikleri
1. `anlamsal.tfidf_sorgu_vektoru` ↔ fitted `vectorizer.transform` eşdeğerlik oracle'ı.
2. NPMI/UMass/ortalama Jaccard ve log-odds z için bağımsız sabit veri doğrulaması.
3. BERTopic wheel + sözlük/model lisans denetimi; kullanıcı bilgisayarında tek Qwen3.5 kanıt testi.
