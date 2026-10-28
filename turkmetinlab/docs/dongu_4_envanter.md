# Döngü 4 — bilimsel risk kapatma ve nitel onay kapısı (28 Ekim 2026)

Taban: gelistirme-3-0-nitel-mvp `c080e3a` (PR #11–#14 birleşik). main, PR #6/#7 değişmedi.

| Alan | Karar |
| --- | --- |
| Nitel hiyerarşi YZ önerisi (`nitel_demo.py`) | DÜZELTİLDİ. Yerel model ya da yapıştırılan JSON önerisi eskiden onaylı kayıtların üzerine doğrudan yazılıyordu; ayrıca `except Exception` hataları yutuyordu. Artık öneri bir onay kuyruğuna gidiyor: `hierarchy_changes` ile mevcut → önerilen farkı gösteriliyor ve ancak "onayla" ile uygulanıyor. Reddedilen ya da geçersiz öneri kayıtları değiştirmiyor; hata görünür. |
| Nitel rapor sınırlılıkları | EKLENDİ. `nitel_rapor.METHOD_NOTES`: yalnız onaylı atamalar, birebir kanıt, katılımcı paydası, kodlayıcılar arası uyumun hesaplanmadığı, genelleme sınırı. Excel'de `Yontem_Sinirliliklar` sayfası. |
| NPMI / UMass | DENEYSEL/SINIRLI etiketi. Eşdeğer kütüphane yok (gensim tanımları farklı); arayüz yardım metni tanımı, kaynağı ve "keşif amaçlı" uyarısını gösteriyor. Değerler değişmedi. |
| ELECTRA | DENEYSEL. Hugging Face'e erişilemediği için commit SHA'sı ve lisans doğrulanmadı; arayüz bunu açıkça yazıyor. Sabit sürüm veya lisans iddiası yok. |
| BERTopic-KMeans | DENEYSEL, değişmedi (Döngü 1–3 kararı). HDBSCAN varsayılanıyla eşdeğer sunulmuyor; ağır paket üretim kurulumunda yok. |
| Nitel kalite uyarıları | KORUNDU. Yalnız tanısal uyarı; puan üretmiyor. |
| Qwen3.5 bağlayıcısı | DEĞİŞMEDİ. Düşünme kapalı (`reasoning_effort=none`), 128–4096 token, `finish_reason=length` reddi, 12.000 karakter sınırı, loopback/yönlendirme engeli. Gerçek model bu ortamda çalıştırılmadı. |

Gerçek Qwen kabul testi (kullanıcı bilgisayarında, `turkmetinlab/` dizininde):
`ollama pull qwen3.5:4b && python scripts/nitel_gercek_model_pilot.py --model qwen3.5:4b --limit 1 --output pilot_gercek.csv`
Bu tek görüş geçerli kanıtlı kod üretirse `--limit 20` ile tekrarlanır.

## Son yayın için kalanlar
1. Gerçek Qwen3.5 kabul testi; çıktı araştırmacı tarafından okunmalı.
2. ELECTRA sürüm/lisans doğrulaması erişimi olan bir makinede yapılmalı; yapılmazsa modül deneysel kalır.
3. Nitel dalın (PR #7 zinciri) staging/main yayın kapısı; PR #6 kararı.
