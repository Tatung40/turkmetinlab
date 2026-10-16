# TürkMetinLab 2.1 — Test Sonuç Raporu

Tarih: 16 Ekim 2026 · Dal: `gelistirme-2.1` · GitHub `main` dalına ve canlı uygulamaya dokunulmamıştır.

## Özet

| Ortam | Sonuç |
|---|---|
| Python 3.12.13: sıfırdan kurulmuş ortam (`uv`, `requirements-dev.txt`) | **458 geçti**, 3 atlandı (isteğe bağlı büyük veri testleri), 0 başarısız |
| Python 3.13.16: sıfırdan kurulmuş ortam | **458 geçti**, 3 atlandı, 0 başarısız |
| Son commit (yalnızca metin ve belge değişiklikleri), Python 3.13 | 458 geçti, 3 atlandı, 0 başarısız |
| İsteğe bağlı büyük veri testleri (`TML_BUYUK_TEST=1`, ~2,95 milyon sözcük, sözlük dahil) | 15/15 geçti |

**2.0 testleri:** 314 testin tamamı çalıştırıldı ve geçti. Git karşılaştırmasına göre (`v2.0-aday` →
`gelistirme-2.1`) test dosyalarından silinen ya da değiştirilen satır sayısı **0**'dır. Yalnızca yeni testler
eklendi; `test_performans.py`'nin sonuna 2 test eklendi.

## Yeni testler

| Dosya | Test | Kapsam |
|---|---|---|
| **test_terimler.py** | **135** | Aşağıda ayrıntılı |
| **test_terimler_arayuz.py** | **10** | Uygulama arayüzü (AppTest), aşağıda ayrıntılı |
| test_performans.py (ek) | 2 (1'i isteğe bağlı) | Sözlükle ön işleme süresi ve tepe bellek (orta ve büyük derlem) |

`test_terimler.py` şunları sınar:
- **Sözlük yapısı:** 637 terim, 6 alan, alan seçimi, terimi devre dışı bırakma.
- **Örnekler:** entegrasyon notlarındaki ve istemdeki örnekler; zorunlu 6 terimin 9 farklı ek biçimi
  (elle yazılmış doğru biçimler).
- **Çekim ve türetim ayrımı:** 9 sözcük için elle yazılmış kabul ve ret listeleri.
- **Yanlış pozitif kontrolleri:**
  - kesme işareti;
  - noktalama ve paragraf sınırları;
  - tireli bileşik ve satır sonu tiresi;
  - "eğitme / ölçme" tuzağı;
  - büyük harfli başlıklar ve küçük harfli "yem".
- **Konumlar:** NFKC, ayrık aksan ve yumuşak tire içeren metinlerde konumların özgün metne geri eşlenmesi.
- **Çakışmalar ve kısaltmalar:** iç içe ve kesişen terimler, öncelik sırası, kısaltmalar.
- **Kullanıcı sözlüğü:** elle ekleme, dosyadan yükleme, formüllü hücre, yeniden yükleme.
- **Ön işleme ve analizler:**
  - tek birim temsili lemma, durak sözcük ve uzunluk süzgeçlerinden etkilenmiyor;
  - "işaretle" kipi ve sözlüksüz çıktı 2.0 ile aynı;
  - n-gram, TF-IDF ve ağ analizinde terim tek öğe.
- **Gizlilik:** kullanıcı terimleri modül düzeyinde saklanmıyor.
- **Alt sınır:** bağımsız yapay test kümesinde F1 ≥ 0,95 ve en fazla 2 yanlış pozitif.

`test_terimler_arayuz.py` şunları sınar:
- sözlük varsayılan olarak kapalı ve 2.0 ile aynı sonuç;
- sözlük açıkken metriklerin doğrudan hesapla eşleşmesi;
- "işaretle" kipinin analizleri değiştirmemesi;
- alan seçimi;
- terim ekleme ve silme;
- oturum yalıtımı;
- geçersiz terim;
- terimi devre dışı bırakma;
- konu karşılaştırması;
- "Verilerimi temizle";
- kısaltmalar.

## Otomatik test dışındaki denemeler

| Deneme | Sonuç |
|---|---|
| Gerçek tarayıcı (Chromium) | Sözlüğü açma, sözlük dosyası yükleme (formül biçimli terim dahil), Terimler sekmesinin bütün alt sekmeleri, Excel ve TXT indirme çalıştı. Hata yok. Excel'de 22 sayfa var; 5'i terim sayfası. *Ayarlar* sayfasında sözlük sürümü ve SHA-256 özeti yazılı. Hiçbir hücre formül değil. |
| Bilimsel validasyon | Yapay küme, DergiPark altın standardı, UD BOUN, ön işleme etkisi, analiz etkisi: `docs/bilimsel_degerlendirme_2.1.md`, `degerlendirme/v21_terim_sonuclari.json` |

## Geliştirme sırasında bulunan ve düzeltilen hatalar

1. **Varyant ikilemesi:** Katlandıktan sonra standart biçimle aynı olan varyantlar (zekâ / zeka) iki kez
   derleniyor, eşleşmeyen biçimler ve çakışmalar ikileniyordu.
2. **Başlık kuralı:** Kısa satırlarda kısaltmalar yanlışlıkla başlık sayılıyordu ("KEKK-YEM yöntemiyle").
   Artık yalnızca 6 harf ve daha uzun büyük harfli sözcükler sayılıyor.
3. **DergiPark kimlikleri:** Kimlikler benzersiz değildi; değerlendirme betiği eşleşmeleri yanlış belgeye
   atıyordu. Kaçırılmış gibi görünen eşleşmelerin nedeni buydu. Satır sırası kimliğine geçildi ve yinelenen
   metinler çıkarıldı.
4. **Geliştirme yarısında bulunan dilbilgisi boşlukları:**
   - zincirleme tamlamada iyelik düşmesi ("makine öğrenme yöntemleri");
   - büyük harfli "IOT";
   - "log-normal dağılım".
5. **Test yarısında bulunanlar** (bunlar ayrıca raporlandı; bağımsız değil):
   - kesmeden önce çekim eki;
   - tek harfli önek ("k-en yakın komşu");
   - iyeliği düşmüş gövdeye ek.
6. **Ölçüm hataları:**
   - **Bellek:** Performans ölçümünde sözlüklü senaryo pandas'ı ayrıca yüklüyordu; bu, bellek farkını olduğundan
     büyük gösteriyordu (86 MB yerine gerçekte 14 MB).
   - **Alan seçimi:** Seçilmeyen alandaki uzun terimin içindeki geçerli alt terim, yanlış pozitif sayılıyordu.
   - **Bağlamsız ölçüm:** Ön işleme etkisi ölçümünde iyeliksiz biçimler bağlamsız işlenince tek birim
     olmuyordu.

   Üçü de düzeltildi.
7. **Belge:** README'de yükleme sınırı 50 MB yazıyordu; gerçek sınır 30 MB.
