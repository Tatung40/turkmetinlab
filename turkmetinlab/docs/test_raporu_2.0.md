# TürkMetinLab 2.0 — Test Sonuç Raporu

Tarih: 15 Ekim 2026 · Dal: `gelistirme-2.0` · `main` dalı 1.1 sürümünde, değiştirilmeden bırakılmıştır.

## Özet

| Ortam | Sonuç |
|---|---|
| Python 3.12.13 (sıfırdan kurulmuş ortam, `requirements-dev.txt`) | **312 geçti**, 2 atlandı (isteğe bağlı büyük veri testleri), 0 başarısız |
| Python 3.13.16 (sıfırdan kurulmuş ortam) | **312 geçti**, 2 atlandı, 0 başarısız |
| İsteğe bağlı büyük veri testleri (`TML_BUYUK_TEST=1`, ~2,9 milyon sözcük) | geçti |

**1.1 testleri:** 232 testin (1'i isteğe bağlı) tamamı çalıştırıldı ve geçti. Git karşılaştırmasına göre
1.1 test dosyalarından silinen ya da değiştirilen satır sayısı **0**'dır; yalnızca yeni testler eklenmiştir.

## Dosyalara göre testler

| Dosya | Test | Kapsam | Sürüm |
|---|---|---|---|
| test_preprocess.py | 49 | Türkçe harf dönüşümü, kesme işareti, durak sözcükler, sayılar, web adresleri | 1.0 |
| test_loader.py | 30 | CSV/Excel/TXT okuma, karakter kodlamaları | 1.0 |
| test_analysis.py | 22 | Sıklık, n-gram, TF-IDF (bağımsız formülle) | 1.0 |
| test_export_visualize.py | 12 | CSV/Excel çıktısı, grafikler, kelime bulutu | 1.0 |
| test_app.py | 9 | Uygulamanın uçtan uca çalışması | 1.0 |
| test_gizlilik.py | 14 | Paylaşılan önbellek, disk, oturum yalıtımı, hata günlüğü | 1.1 |
| test_bolutleme.py | 18 | N-gram bölütleme ve dilbilimsel gerekçesi (UD) | 1.1 |
| test_lemma.py | 15 | Lemmatizasyon (UD altın standart) | 1.1 |
| test_dosya_guvenligi.py | 18 | Dosya türü, sıkıştırma bombası, formüller, tırnak | 1.1 |
| test_tfidf_referans.py | 9 | TF-IDF ↔ bağımsız uygulama, UD referans verisi | 1.1 |
| test_kalite.py | 28 | Veri kalitesi (26) + İngilizce içerik tespiti (2) | 1.1 + 2.0 |
| test_performans.py | 13 | Sınırlar, süre ve bellek (10) + 2.0 modülleri ve düğmeyle hesaplama (3) | 1.1 + 2.0 |
| **test_benzerlik.py** | **14** | Elle kosinüs; sklearn ve saf Python karşılaştırması; parçalı en yakın komşu = tam sıralama; scipy kümeleme; uygulama | 2.0 |
| **test_birliktelik.py** | **24** | Elle pencere/PMI/Jaccard; kaba kuvvet sayımla 6 bağlam; PPMI/Jaccard formülü; networkx derecesi; uygulama | 2.0 |
| **test_konu.py** | **21** | Elle NPMI/UMass; bağımsız tutarlılık; ekilmiş konuların geri bulunması (LDA, NMF); belirlenimcilik; uyarılar; uygulama | 2.0 |
| **test_grup.py** | **18** | Elle log-odds (2 önsel); FightingWords karşılaştırması; çok grup ↔ iki grup; permütasyon; uyarılar; uygulama | 2.0 |
| **Toplam** | **314** | | |

## Otomatik test dışındaki denemeler

| Deneme | Sonuç |
|---|---|
| Gerçek tarayıcıda (Chromium) dört yeni sekme, kümeleme, konu adlandırma, Excel indirme | Hata yok. Excel'de 23 sayfa var: yeni modüllerin tabloları, araştırmacının konu adı ve tüm yöntem ayarları. Hiçbir hücre formül değil. |
| 4 eşzamanlı kullanıcı (her biri farklı dosya ve gizli sözcük) | Oturum yalıtımı 4/4. Hata yok. Sunucu günlüğünde kullanıcı verisi yok. Sunucunun tepe belleği 566 MB. |
| Gerçek akademik veriyle validasyon (1.417 DergiPark özeti) | `degerlendirme/v2_dogrulama_sonuclari.json`. Özet: `docs/bilimsel_degerlendirme_2.0.md` |

## Geliştirme sırasında testlerin yakaladığı hatalar (düzeltildi)

1. **Benzerlik:** İstenen sayı belge sayısına ulaştığında, belgenin kendisi "en benzer belgeler"
   listesine giriyordu.
2. **Grup:** Kimlik sütunu varsayılan grup sütunu olarak öneriliyordu; artık yalnızca kategorik
   sütunlar öneriliyor. "Her grupta tek belge" durumu hata yerine uyarı oldu, çünkü iki uzun belgeyi
   karşılaştırmak meşru bir kullanım.
3. **Ağ:** Belge bağlamında kenar tablosu 1,2 milyon satıra çıkıp belleği 965 MB'a yükseltiyordu.
   Artık sıralama NumPy ile yapılıyor ve tablo sınırlı; tepe bellek 519 MB.
4. **En yakın komşu:** 20 bin belgede 64 sn sürüyordu; aday seçimiyle 28 sn'ye indi. Eşitliklerin
   yoğun olduğu durumlarda sonucun tam sıralamayla aynı kaldığı ayrıca test edildi.
5. **Konu:** LDA'nın büyük derlemlerde çok yavaş olduğu ölçüldü. Sınır ve tahmini süre gösterimi
   eklendi; NMF'nin yakınsayıp yakınsamadığı artık raporlanıyor.
6. **Gerçek veri:** Türkçe akademik kayıtlardaki İngilizce özetlerin bir "konu" oluşturduğu görüldü.
   İngilizce içerik tespiti ve İngilizce durak sözcük seçeneği eklendi.
