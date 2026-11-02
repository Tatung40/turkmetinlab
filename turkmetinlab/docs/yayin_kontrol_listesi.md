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

## Yayın sonrası kabul (main `c1e3434` + PR #19, 29 Ekim 2026)

| Kontrol | Sonuç | Kanıt |
| --- | --- | --- |
| main CI (Python 3.12 / 3.13) | ✅ | `c1e3434` için testler.yml başarılı |
| PR #19 CI | ✅ | 4/4 kontrol başarılı |
| Yerel Streamlit sunucusu (depo kökünden `turkmetinlab/app.py`) | ✅ | `/_stcore/health` = ok, `/` = 200 |
| Açılış (yalnız `requirements.txt`, ağır paketler engelli) | ✅ | 15 sekme, istisna yok |
| CSV yükleme → analiz | ✅ | 16 belge, 836 sözcük, 540 farklı sözcük |
| TXT yükleme → analiz | ✅ | 3 belge; konu modeli küçük derlemi açık uyarıyla reddediyor (beklenen) |
| Rapor indirme | ✅ | Her senaryoda 6 indirme düğmesi; Excel dosyaları openpyxl ile açılıyor |
| **Canlı uygulama (Streamlit Cloud)** | ⛔ Doğrulanamadı | Bu ortamdan `*.streamlit.app` ve share.streamlit.io erişimi engelli (bağlantı kurulamadı) |

Canlıda yapılması gerekenler (sahip):
1. PR #19'u birleştirmek.
2. share.streamlit.io → uygulama ⋮ → Settings: Branch `main`, Main file path `turkmetinlab/app.py`, Python 3.12.
3. Canlı adreste örnek veriyle bir analiz çalıştırmak, küçük bir CSV yüklemek ve Excel raporunu indirmek.

## İnsan ilgililik doğrulaması (yöntem; gerçek veri depoda değil)

- Betik: `degerlendirme/insan_ilgililik.py` (testler: `tests/test_insan_ilgililik.py`, yalnız sentetik veri).
- Puanlayıcı dosyaları satır sırasıyla değil, normalize metin özetiyle eşleştirilir; eksik puan doldurulmaz.
- Uyum: Cohen kappa (ağırlıksız, doğrusal), Krippendorff alfa (ordinal/nominal/aralık; Krippendorff 2011
  örneğiyle doğrulanmış bağımsız uygulama), sorgu düzeyinde küme bootstrap.
- Sıralama: yalnız verilen aday listeleri içinde; ölçütler sorgu başına hesaplanıp ortalanır. Hibrit sabit
  eşit ağırlıklıdır (veriyle ayarlanmaz). E5 modeli yapılandırılmamışsa E5/Hibrit `NOT_RUN` raporlanır.
- Gerçek puanlayıcı Excel'leri ve raporlar `.gitignore` ile dışlanır; çıktı depo dışına yazılmalıdır.

## E5 + Hibrit: model kaynağı ve etkinleştirme

- **Doğrulanmış kaynak (Actions günlüğü, 20 Ekim 2026):** `intfloat/multilingual-e5-small` (MIT), commit
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`. ONNX FP32 (opset 17), 20 örnekte PyTorch ile kosinüs 1,0 ve en büyük
  mutlak fark 0. O çalıştırmadaki `model_fp32.onnx` SHA-256 `a1b6428e37d1b6370b0f22d843f3153da145cb401dfcda77ab0b15722752b652`
  (470.233.634 bayt); `tokenizer.json` `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39`. INT8 bu denetimi geçemedi; kullanılmaz.
- **Yayın:** `.github/workflows/e5-onnx-yayin.yml` yalnız elle başlatılır. Modeli aynı commit'ten yeniden üretir,
  PyTorch'a ve uygulama kodlayıcısına karşı doğrular, ardından `e5-onnx-fp32-614241f` ön sürüm release'ine yükler.
  Ağırlıklar git geçmişine girmez. Gerçek SHA-256 değeri release notlarında yazar; yukarıdakinden farklıysa o geçerlidir.
- **Streamlit Cloud (deneysel, isteğe bağlı) Secrets:**
  `TML_E5_ONNX_URL = "https://github.com/Tatung40/turkmetinlab/releases/download/e5-onnx-fp32-614241f/{dosya}"`,
  `TML_E5_ONNX_SHA256 = "<release notlarındaki değer>"`, `TML_DENEYSEL_ANLAMSAL = "1"`.
  `requirements.txt` ONNX paketlerini içermez; canlıda E5 için `onnxruntime` ve `tokenizers` gerekir (staging gereksinimleri).
  Model yaklaşık 450 MB indirir ve önbelleğe alır; ücretsiz sunucunun belleği sınırlıdır. Canlı tarayıcı testi yapılmadan "canlıda çalışıyor" denmez.
- **İnsan değerlendirmesi protokolü:** Belge–belge benzerliğinde E5 iki tarafta da `query: ` öneki kullanır (uygulamanın
  simetrik kipi). Hibrit, sabit eşit ağırlıklı z-puanı toplamıdır; 75 çift üzerinde ağırlık ayarlanmaz.
## İnsan duygu doğrulaması (yöntem; gerçek veri depoda değil)

- Betik: `degerlendirme/insan_duygu.py`; testler `tests/test_insan_duygu.py` (yalnız sentetik veri, ağ yok).
- Etiketler değiştirilmez: 0 = NÖTR, 1 = OLUMLU, 2 = OLUMSUZ. Kayıtlar kimlik + normalize metinle eşleşir; eksikler doldurulmaz.
- Uyum: Krippendorff alfa (nominal), Fleiss kappa (tam kayıt; Fleiss 1971 örneğiyle sınanmış formül ya da statsmodels),
  çift Cohen kappa. Referans: oy birliği / güçlü çoğunluk (oran ≥ 5/6) / mutlak çoğunluk; çoğunluksuz yorumlar uzlaşmaya.
- Sözlük yöntemleri (SWNetTR++, TES-Lex) önceden sabitlenmiş sayım kuralıyla 3 sınıfa eşlenir; ayar yapılmaz.
  ELECTRA modeli yüklenemezse `NOT_RUN`. TürkMetinLab'de olumsuzlamayı ele alan bir sınıflandırıcı yoktur.
- `--haric` ile bir puanlayıcının etkisi duyarlılık analizi olarak ayrıca raporlanır; etiketleri değiştirilmez.
