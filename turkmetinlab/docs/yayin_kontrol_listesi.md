# 3.0.0rc1 yayın kontrol listesi (Döngü 5)

Dal: gelistirme-3-0-nitel-mvp. main'e ve Streamlit canlı uygulamasına alınmadı.

| Kapı | Durum | Not |
| --- | --- | --- |
| CI (Python 3.12 / 3.13, tam test) | ✅ PR #11–#15 | Her PR'da yeşil |
| TF-IDF, benzerlik, ağ, grup, ön işleme regresyonu | ✅ | Dondurulmuş oracle testleri |
| NMF/LDA (scikit-learn) | ✅ | Ölçüt adları hesaplamayla örtüşüyor |
| NPMI/UMass/çeşitlilik/kararlılık | ⚠️ Keşif amaçlı | Kütüphane eşdeğeri yok; arayüzde ve README'de belirtiliyor |
| Nitel kanıt doğrulaması + araştırmacı onayı | ✅ | Hiyerarşi önerisi de onay kuyruğunda |
| Nitel rapor sınırlılıkları / sürüm izi | ✅ | Excel `Yontem_Sinirliliklar`, `Ozet` |
| Gerçek Qwen3.5 kabul testi | ⏸️ Ertelendi | Kullanıcı bilgisayarında: README "Nitel analiz" |
| ELECTRA sürüm + lisans | ⏸️ Doğrulanmadı | Deneysel, varsayılan kapalı (staging) |
| BERTopic-KMeans | ⚠️ Deneysel | Ağır paketler yalnız staging gereksinimlerinde |
| Yeni ücretli servis / GPU / zorunlu ağır bağımlılık | ✅ Yok | requirements.txt değişmedi |

## Canlıya almadan önce (araştırmacı/bakımcı kararı)
1. PR #7 zinciri ile nitel dalı staging'e, ardından main'e taşımak; PR #6 (BERTopic kalite) kararı.
2. Staging'de `streamlit run app.py` ve `streamlit run nitel_demo.py` ile elle duman testi.
3. Qwen3.5 kabul testi ya da nitel ekranın yalnız "elle JSON + onay" modunda yayımlanması.
4. ELECTRA'yı doğrulamak ya da staging bayrağı arkasında tutmak.

## Son entegrasyon denetimi (main ← gelistirme-3-0-nitel-mvp)

- **Dal ilişkisi:** main, nitel dalının atasıdır; birleştirme çakışmasızdır. staging (PR #7'nin hedefi) de nitel dalının atası; PR #7'nin içeriği bu dalda zaten var.
- **PR #6 dahil edilmedi:** BERTopic–NMF/LDA keşif karşılaştırması 8 commit'tir. `konu.py`'dekinden farklı tanımlı, ayrı bir NPMI belirteçleyicisi getiriyor; tek başına staging hedefli. Aynı adı taşıyan iki ayrı NPMI tanımını yayına sokmamak için dışarıda bırakıldı. Ayrı karar gerektirir.
- **Canlı bağımlılık duman testi:** Yalnız `requirements.txt` kurulu, onnxruntime/tokenizers/bertopic/torch/transformers içe aktarımı engelliyken `app.py` (15 sekme) ve `nitel_demo.py`, AppTest altında istisnasız açılıyor. Hiçbir `tmlab` modülü ağır paketleri modül düzeyinde içe aktarmıyor.
- **Deneysel bayrak:** Anlamsal, duygu (SWNetTR++, TES-Lex, ELECTRA) ve BERTopic arayüzleri yalnız `TML_DENEYSEL_ANLAMSAL=1` ile görünür; canlıda varsayılan olarak kapalı.
- **Nitel ekran:** Ayrı giriş noktası (`nitel_demo.py`), "Pilot" ve deneysel olarak etiketli. Yerel model yalnız 127.0.0.1 üzerinden çalışır; gerçek Qwen kabul testi ertelendi.
- **Gizli bilgi:** Diff'te açık anahtar yok; iş akışları yalnız `secrets.*` referansı içeriyor.
- **Bilinen kalıntı:** `fizibilite_2.5/` kökte araştırma artefaktlarını ve `vendor/` altında kod kopyalarını taşıyor (uygulama çalışma yolunda değil). `staging/vendor/` içinde BERTopic wheel'i var (MIT iddiası; yukarı akışla içerik eşliği denetlenmedi). Yayından sonra temizlenebilir.
