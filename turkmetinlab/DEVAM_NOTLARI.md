# TürkMetinLab 2.5 staging — Devam notları (20 Ekim 2026)

Dal: `gelistirme-2.5-staging`. `main` ve canlı uygulama **değişmedi**. Ayrıntı: `staging/README.md`.

- **Yeni modüller:**
  - `tmlab/anlamsal.py`: ONNX FP32 kodlayıcı, indirme ve SHA-256 doğrulaması, TF-IDF sorgu vektörü, hibrit.
  - `tmlab/anlamsal_sekme.py`: arayüz.
- **Uygulama değişikliği:** `app.py` içinde yalnızca bayrakla açılan sekme (+8 satır). Bayrak kapalıyken sekmeler
  2.1 ile aynı (test edildi).
- **Testler (küçük sınama modeli):**
  - `tests/test_anlamsal.py` (13) ve `tests/test_anlamsal_arayuz.py` (9).
  - Tüm paket: 480+ test.
  - `tests/test_gizlilik.py`'ye iki dar istisna eklendi:
    - `_kodlayici` önbelleği yalnızca model klasörünü alır;
    - `model_hazirla` yalnızca model dosyası yazar.
- **Gerçek E5 ile doğrulanmadı.** `onnx-2.5.yml` *staging* kipi `staging/entegrasyon_testi.py`'yi çalıştırır:
  - uygulama kodlayıcısı ↔ fizibilite kodlayıcısı (20 belge);
  - 100/500 belge ölçek ve bellek ölçümü;
  - n=500 görev B'de ONNX #3 değerlerini yeniden üretme (±0,002);
  - AppTest.

  Geçerse model Hugging Face'e yüklenir (HF_TOKEN varsa).
- **Sıradaki adımlar:**
  1. İş akışını *staging* kipinde çalıştır.
  2. Streamlit'te ayrı staging uygulaması kur ve bellek/süreyi gözlemle.
  3. Üretim kararı ancak bundan sonra verilmeli.

# TürkMetinLab 2.1 — Devam notları

Son güncelleme: 16 Ekim 2026 · Dal: `gelistirme-2.1` · `main` dalına ve canlı Streamlit uygulamasına
**dokunulmadı**.

## 1. Durum

`DEVAM_NOTLARI.md`'nin 15 Ekim sürümündeki 10 görevin durumu:

| # | Görev | Durum |
|---|---|---|
| 1 | Terim tanıma modülü (`tmlab/terimler.py`) | ✅ |
| 2 | Testler (`tests/test_terimler.py`) ve 2.0 regresyonu | ✅ 135 test; sözlük kapalıyken çıktı 2.0 ile aynı |
| 3 | Ön işleme entegrasyonu (`terimci`, yer tutucu, `IslenmisBelge.terim_sonucu`) | ✅ |
| 4 | Arayüz | ✅ Kenar çubuğu, 📚 Terimler sekmesi, sözlük düzenleme ve yükleme, karşılaştırma, indirmeler, Ayarlar. Testler: `tests/test_terimler_arayuz.py` (10) + tarayıcı denemesi |
| 5 | Validasyon betiği ve altın standart | ✅ `degerlendirme/v21_terim_dogrulama.py`, `degerlendirme/veri/` |
| 6 | AkademikDerlem örneklemi | ❌ **Yapılamadı.** Ortamın ağ politikası Hugging Face'i engelliyor (403); hiç veri indirilmedi. Yöntem bölüm 4'te. |
| 7 | Performans ölçümü | ✅ `performans_olcumu.py --terim`; 2 test |
| 8 | Python 3.12 ve 3.13 testleri | ✅ Sonuçlar `docs/test_raporu_2.1.md`'de |
| 9 | Belgeler | ✅ `docs/bilimsel_degerlendirme_2.1.md`, `docs/sozluk_denetimi_2.1.md`, `docs/test_raporu_2.1.md`, `DEGISIKLIKLER.md`, README; `__version__ = "2.1.0"` |
| 10 | `turkmetinlab-2.1.zip` | ✅ Kullanıcıya teslim edildi. `main` dalına birleştirme yalnızca kullanıcı onayıyla yapılmalı. |

## 2. Önemli tasarım kararları ve nedenleri

- **Varsayılan olarak kapalı.** Sözlük açıldığında da kullanıcıya terimlerin aday terim olduğu açıklanıyor.
- **Asıl doğruluk sonucu dondurulmuş eşleştiriciye ait** (commit `ab6b3dd`). Test verisi görüldükten sonra
  yapılan 3 kural düzeltmesi raporda ayrıca ve "bağımsız değil" notuyla verildi.
- **Kısaltmalar isteğe bağlı ve büyük/küçük harfe duyarlı.** Gerçek veride "yem" (hayvan yemi) ile "YEM"
  (yapısal eşitlik modeli) aynı metinlerde geçti.
- **Alan seçimi önemli.** Eğitimdeki "derin öğrenme" terimi mühendislik metinlerinde hep başka anlamda
  geçti; alan seçimi kavram düzeyi kesinliği %90'dan %99,5'e çıkardı. Arayüz bu yüzden uyarıyor.
- **Sözlük analizleri kendiliğinden iyileştirmiyor** (bölüm 7, bilimsel değerlendirme). Arayüzde
  açık/kapalı karşılaştırması bu yüzden var.

## 3. Açık işler ve öneriler (sonraki sürüm için)

1. **AkademikDerlem ya da başka alanlardan gerçek metinle değerlendirme.** Gerçek veri kanıtı bugün
   mühendislikte güçlü; eğitim, sağlık, psikoloji ve ekonomide öğe sayısı az (19–31).
2. **Alan uzmanı incelemesi:** `docs/sozluk_denetimi_2.1.md`'deki eş anlamlılar, eksik varyantlar
   ("makine öğrenimi", "bilgisayarlı görme"…), çoğul standart biçimler ve anlamı alana göre değişen kayıtlar.
3. **İkinci etiketleyici:** Etiketleyiciler arası uyum ölçülmedi.
4. **Olası geliştirmeler:**
   - "Hem sözcükler hem birim" temsili: kısmi örtüşmeyi korumak için;
   - türetilmiş biçimleri ("tip 2 diyabetli") ayrı etiketle saymak;
   - genel ünlü düşmesi için sözlüğe dayalı denetim.

## 4. AkademikDerlem için önerilen yöntem (erişim olan ortamda)

- **Kaynak:** Hugging Face `turkish-nlp-suite/AkademikDerlem`. Bir veri seti kataloğuna göre CC BY-SA 4.0
  lisanslı. Lisans, kullanılmadan önce veri setinin kendi sayfasından doğrulanmalıdır.
- **Örneklem:** Veri setinin tamamı indirilmez.
  - Akış kipi kullanılır: `load_dataset(..., split="train", streaming=True)`.
  - Kayıtlar karıştırılır: `.shuffle(seed=2101, buffer_size=10_000)`.
  - Yalnızca ilk 600 kayıt alınır.
- **Saklama:** Ham metin depoya konmaz. Yalnızca kayıt kimlikleri, konumlar, etiketler ve özet ölçümler
  saklanır (`terim_altin_dergipark.tsv` gibi).
- **Etiketleme:** `degerlendirme/v21_terim_dogrulama.py` içindeki `adaylari_bul` (gevşek arama) +
  eşleştirici çıktısı (havuzlama) ile bölgeler üretilir. Bölgeler eşleştiricinin kararı görülmeden
  etiketlenir; sonra `gercek_degerlendir` ile aynı ölçütler hesaplanır.

## 5. Kaldığı yerden devam etmek için

```bash
git checkout gelistirme-2.1
python -m pytest -q                                   # bütün testler
TML_BUYUK_TEST=1 python -m pytest tests/test_performans.py   # isteğe bağlı büyük ölçümler
python degerlendirme/v21_terim_dogrulama.py hepsi     # validasyon (DergiPark'ı indirir)
```
