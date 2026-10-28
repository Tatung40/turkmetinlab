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
