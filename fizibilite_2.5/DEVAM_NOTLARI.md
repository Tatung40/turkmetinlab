# 2.5 fizibilite — devam notları

**Durum:** TF-IDF referansı ölçüldü. Betik uçtan uca sınandı (sahte modelle). Çalışma zamanı belleği ölçüldü.
**Gömme modelleri çalıştırılamadı**, çünkü bu ortamdan Hugging Face'e erişim yok (403).

## Yapılanlar

- `anlamsal_karsilastirma.py`: 8 yöntem ve hibrit; görev A ve B; bootstrap güven aralıkları; RAM, süre, kesilme
  ve eşzamanlılık ölçümleri.
- Gerçek veriyle TF-IDF sonuçları: `sonuclar/sonuclar_n500.*`, `sonuclar/sonuclar_n100.*`.
- Çalışma zamanı ölçümleri: torch + sentence-transformers 765 MB; onnxruntime + tokenizers 53 MB.
- Uygulama kodu değişmedi. Bu yüzden 2.1 regresyon testleri yeniden çalıştırılmadı.

## GitHub Actions hazırlığı (19 Ekim 2026)

- Yeni dosyalar: `.github/workflows/fizibilite-2.5.yml` (yalnızca elle başlatılır), `ortam_denetimi.py`,
  `model_indir.py` ve `requirements-fizibilite.txt`. Bağımlılık dosyası artık pakette; daha önce `.gitignore`'daki
  `*.txt` kuralı yüzünden eksikti.
- Bağımlılıklar `uv pip compile` ile Python 3.12 için çözüldü. PyPI'deki torch 15 NVIDIA paketi getirdiği için CPU
  dizini zorunlu. torch 2.14.1'in CPU dizininde bulunduğu **buradan doğrulanamadı**; iş akışındaki ortam denetimi
  bunu ilk çalıştırmada kendisi denetler.
- Betiğe yöntem başına süre ve bellek sınırı ile `--kati` kipi eklendi. Sahte modelle sınandı.
- 2.1 ön işleme modüllerinin kopyası `vendor/` klasöründe; depoda hangi uygulama sürümü olursa olsun aynı ön
  işleme kullanılır.
- **Modellerle gerçek çalıştırma hâlâ yapılmadı.** İş akışının çalıştırılması ve artifact'in incelenmesi gerekiyor.

## ONNX aşaması (19 Ekim 2026) — ayrıntı: `onnx/RAPOR_ONNX.md`

- Kullanıcının GitHub n=500 sonucu (JSON görülmedi): TF-IDF 0,317; E5 0,304; hibrit 0,342 (+0,025 < 0,03 eşiği);
  E5 PyTorch ~2.631 MB, 63,84 sn.
- Yeni: `onnx/` (dönüştürme, torch'suz kodlayıcı, 20 örnek uyum, test), `.github/workflows/onnx-2.5.yml` (elle).
  `anlamsal_karsilastirma.py`'ye `e5onnx32_*`/`e5onnx8_*` yöntemleri, ONNX↔PyTorch uyum ve fark tablosu eklendi.
- Küçük rastgele modelle sınandı: `pytest fizibilite_2.5/onnx/test_onnx.py` (2 geçti) ve iş akışı adımlarının
  yerel benzetimi (çıkış 0). **Gerçek E5 ONNX ölçümü yok** (Hugging Face 403).
- Düzeltilen hata: ONNX token uzunluğu dolgu dahil sayılıyordu.
- Dayanıklılık (19 Ekim, inceleme sonrası): INT8 hatası yalnızca INT8'i eler (FP32 sürer, FP32 hatası işi
  durdurur); ONNX önbelleği SHA-256 ve kaynak sürümle doğrulanır, bozuksa güvenli yeniden oluşturulur.
  `test_onnx.py`: 9 test geçti (küçük rastgele model; gerçek E5 değil).

## Bellek düzeltmesi (GitHub #2 sonrası) — ayrıntı: `onnx/RAPOR_ONNX.md` §8

- #2: FP32 uyumlu, INT8 elendi (ilk-5 0,76). ONNX FP32 n=100'de 6.078 MB ile sınırda durdu.
- Neden: birleştirilmemiş dikkat (toplu iş × 12 × uzunluk²) × 3 eşzamanlı oturum + geri verilmeyen arena.
- Düzeltme: toplu iş 8, uzunluk sıralaması, arena kapalı, iş parçacığı ≤ 4. Ölçüm: süreç ağacı RSS + PSS,
  artık süreç denetimi, yükleme/çıkarım ayrımı.
- E5 boyutlu rastgele modelle: 3 eşzamanlı tepe >5.000 → 1.786 MB. Gerçek E5 için GitHub #3 gerekli.

## Sıradaki adımlar

1. `onnx-2.5.yml` dosyasını `main`'e ekle (yalnızca bu dosya; düğmenin görünmesi için), iş akışını
   `gelistirme-2.5-fizibilite` dalında çalıştır.
2. `onnx-2.5-sonuclar` artifact'indeki `sonuclar_n500_onnx.json` ile `onnx/RAPOR_ONNX.md` 4–5. bölümlerini doldur.
3. Hibrit +0,03 eşiğini geçmedikçe uygulama entegrasyonu yapma. Geçerse: ayrı dal, deneysel sekme, yalnızca
   ONNX, `st.cache_resource` yalnızca modeli tutar; Streamlit'te bellek denemesi ve 2.1 regresyon testleri.
4. Daha güçlü değerlendirme: insan etiketli küçük bir küme (ör. 50 sorgu × ilk 10, iki etiketleyici).
