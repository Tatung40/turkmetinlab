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

## Sıradaki adımlar

1. **Modelleri ölçmek:** Kullanıcının kendi bilgisayarında README'deki komutlar çalıştırılmalı: önce `--n 100`,
   sonra `--n 500`. Ortaya çıkan `sonuclar_n500.json` ile `RAPOR.md`'nin 1–3. ve 5. bölümleri güncellenmeli;
   karar, README'deki önceden belirlenmiş ölçütlere göre verilmeli.
2. **ONNX karşılaştırması (bir model ölçütleri geçerse):** Model ONNX'e dönüştürülüp (int8) ONNX Runtime ile
   yeniden ölçülmeli. Bunun için betiğe bir `onnx` yöntemi eklenmeli.
3. **Uygulama entegrasyonu (yalnızca bundan sonra):** Ayrı bir dalda (`gelistirme-2.5`) yapılmalı.
   `st.cache_resource` yalnızca modeli tutmalı; vektörler oturum belleğinde kalmalı. Entegrasyondan sonra
   2.1 regresyon testleri çalıştırılmalı.
4. **Daha güçlü değerlendirme:** Mümkünse insan tarafından ilgililik etiketi verilmiş bir değerlendirme kümesi
   hazırlanmalı. Örneğin 50 sorgu × ilk 10 sonuç, iki etiketleyiciyle.
