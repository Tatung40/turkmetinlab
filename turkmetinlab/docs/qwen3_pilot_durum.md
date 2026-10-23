# Qwen3 4B pilot durumu — 23 Ekim 2026

Kaynak dal: gelistirme-3-0-nitel-mvp; temel commit: adf0dad1a11ee8414a786a125bd8da664f72e5d1.
Git fetch/proxy engeli nedeniyle kaynaklar GitHub bağlantısından bu commit sabitlenerek alındı. Önceki scratch kopyaları yeni geliştirme sayılmadı ve PR'a eklenmedi.

## Sayaçlar

- processed: işleme alınan/denenen görüş sayısı; başarılı analiz sayısı değildir.
- succeeded: model yanıtı ayrıştırılan ve tüm önerileri doğrulanan görüş sayısı. Geçerli boş codes listesi dahildir; bu durum kod bulunduğunu ifade etmez.
- failed: model çağrısı, ayrıştırma veya öneri doğrulamasında hata olan görüş sayısı. Bir görüşte birden fazla hata olsa da bir kez sayılır.
- processed = succeeded + failed. Kısmen geçerli öneriler korunur ancak görüş failed sayılır.
- Hiyerarşi yeniden gruplama hatası errors içinde _hierarchy olarak kalır; görüş sayaçlarını değiştirmez.
- CSV/JSON çıktıları hata halinde de korunur; CLI failed veya errors varsa 1 döndürür.
- Frekans paydası eskisi gibi denenen tüm benzersiz katılımcılardır. Hatalı kayıtları dışlayarak yüzdeler şişirilmez; eksik analiz bulunan tablolar tamamlanmış pilot sonucu sayılmaz.

## Gerçek model erişimi

Mevcut ortam yapılandırmasında capabilities, secrets, runtime_variables ve outbound_identities boş. Kullanılabilir araçlar içinde yetkilendirilmiş Qwen çıkarım servisi yok. Önceki incelemede Ollama/torch/transformers/llama_cpp ve GPU bulunmamış; socket oluşturma PermissionError, betik URLError, dış erişim proxy bağlantı hatası vermişti. Bu başarısız girişimler tekrar edilmedi. Ücretsiz ve yetkilendirilmiş, gerçekten erişilebilir bir Qwen3 4B çıkarım yolu doğrulanamadı.

Hugging Face PRO sınırsız ücretsiz API/GPU olarak değerlendirilmedi. Ücretli API/GPU, bulut işi veya GitHub Actions çıkarım işi başlatılmadı; model/paket indirilmedi.

## Çıktı ve kalan iş

Bu değişiklik sırasında gerçek model çağrısı yapılmadı; yeni gerçek model sonucu yok. Önceki tek görüş başarısızdır. 20 görüş aşaması çalıştırılmadı. Kod/kategori/tema/birebir kanıt, frekans/yüzde veya kalite başarısı iddia edilmez. Boş öneriler üzerindeki boş kalite uyarıları doğrulama değildir.

Ücretsiz yerel model sunucusu ve ağırlıklar erişilebilir olduğunda önce --limit 1 ile model yanıtı, birebir kanıtlar ve rapor hataları incelenmeli; başarılıysa 20 görüşe geçilmeli. Geçerli boş yanıtın succeeded sayılması, anlamlı kod üretiminin doğrulandığı anlamına gelmez. Mevcut betik 20 kayıt çalıştırıldığında ilk kaydı tekrar işler; başarılı ilk sonucu yeniden kullanma işi bu sayaç düzeltmesinin dışında kalmıştır.

## Hedefli doğrulama

`python -m pytest -q --noconftest tests/test_nitel_gercek_pilot.py`

Mevcut sanal ortamla 9 test geçti. Test completion yanıtları simülasyondur; gerçek Qwen3 üretimi veya model kalitesi kanıtı değildir. Testler başarı, geçerli boş yanıt, URLError, bozuk JSON, çoklu reddetme, kısmi kanıt kabulü, hiyerarşi hatası ve CLI sayaç/çıkış kodlarını kapsar. Başka test veya indirme yapılmadı.

PR hedefi geliştirme dalıdır; ana dala birleştirme yapılmadı.
