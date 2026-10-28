# TürkMetinLab 3.0 — Nitel Veri Analizi MVP
**Durum:** Ayrı geliştirme dalında deneysel çekirdek. Arayüz ve LLM bağlantısı henüz yoktur.

## Maliyet önceliği
1. Mevcut 2.5 kodunu ve CPU/Streamlit'i koru; ağır model dosyası yükleme.
2. Modeli zorunlu kılma: önce kullanıcı kod kitabı + yerel kod/kanıt kayıtları ve manuel düzenleme.
3. LLM kullanımı isteğe bağlı olsun; tek katılımcı için kısa, tek JSON yanıtı; aynı girdiye tekrar istek yapma.
4. Gizli yorum/görüşme verilerini kullanıcı onayı ve uygun veri işleme koşulları olmadan dış API'ye gönderme. Harici sağlayıcıda ücret doğabilir.
5. Model yanıtlarını insan onayı olmadan bilimsel kod kabul etme.

## Çekirdek işlevler
- `prepare_prompt(metin, kod_kitabi)`: kısa komut ve JSON şeması.
- `parse_response(json)`: katı JSON ayrıştırma.
- `validate_codes(kimlik, metin, kodlar)`: harfiyen kanıt eşleşmesi; başarısız kodlar ayrılır.
- `analyze_model_response(...)`: önerilerin doğrulanmış kaydı (insan incelemesi gerekir).
- `frequencies(...)`: kod başına benzersiz katılımcı oranları.

## İzleyen işler
- Streamlit ekranında onayla/değiştir/reddet; özgün kanıtı renklendirme.
- Katılımcı sayısı için eksik/hiç kod verilmeyen kayıtları da sayabilen veri modeli.
- Araştırmacı onaylı kod-kategori-tema ilişkileri ve Excel dışa aktarımı.
- Yerel veya harici **OpenAI uyumlu** servis adaptörü: sağlayıcıdan bağımsız, açıkça maliyet onaylı; ayrı bağlantı ayarları.
- İlk olarak Turkish-Llama-8B-Instruct ve küçük Türkçe instruct modelini az sayıda anonimleştirilmiş görüşmede karşılaştır.
- Gerçek örneklerle doğruluk/kanıt isabeti, iki uzman arası uyum, tutarlılık, maliyet, gecikme.

## Bilimsel sınırlama
Alıntının metinde geçmesi, önerilen kodun anlamca doğru olduğu anlamına gelmez.
LLMCode ve Thematic Analyzer iş akışlarından yararlanılabilir; üçüncü taraf kodlarını lisans incelemesi olmadan kopyalama.
