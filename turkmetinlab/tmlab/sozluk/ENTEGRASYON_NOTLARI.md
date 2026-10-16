# TürkMetinLab 2.1 — Akademik terimler sözlüğü

## İçerik
637 benzersiz aday terim; 6 disiplin grubu. Bu sözlük uzman doğrulaması bekleyen **ilk çekirdek taslaktır**. Terimler bilimsel terminoloji standardı olarak sunulmamalıdır. Eğitim ve mühendislik gibi alanlarda aynı kavram birden fazla alana uygun olabilir; bu sürümde birincil alan atanmıştır.

## Veri şeması
- `terim_id`: sabit benzersiz kimlik
- `alan`: birincil akademik alan
- `standart_terim`: korunacak yüzey biçimi / sözlük etiketi
- `yazim_varyantlari`: yalnızca açıkça doğrulanmış yazım alternatifleri; çoklu varyant varsa `;` ayracı kullanılabilir
- `tur`: çok sözcüklü / tek sözcüklü terim
- `dogrulama_durumu`: uzman incelemesi durumu
- `isleme_notu`: çekimlerin kuralla tanınması gereği

## Entegrasyon ilkeleri
1. Seçenek varsayılan **kapalı** olsun. Alan seçimi ve kullanıcı sözlüğü desteklensin.
2. Orijinal metni değiştirmeyin; eşleşmeyi `terim_id`, özgün yüzey, standart etiket, güven puanı ve başlangıç/bitiş karakter konumlarıyla kaydedin.
3. Uzun ve tam terim eşleşmesine öncelik verin. Sözcük sınırlarını koruyun. İsim tamlamalarının son sözcüğüne ek gelmesini özel bir morfolojik kural ile değerlendirin.
4. Yalnızca son sözcükte ek var diye terimi otomatik olarak eşlemeyin; yanlış pozitifleri test edin. `ölçme ve değerlendirmede` ≈ `ölçme ve değerlendirme` aday eşleşmedir, `ölçme ve değerlendirmeci` değildir.
5. Kullanıcı terimleri > seçilen alan sözlüğü > genel sözlük önceliği izleyin. Eşleşme çakışmalarını görüntüleyin.
6. N-gram, TF-IDF, ağ analizi ve konu modellemesi için **korunmuş terimi tek kavramsal birim** olarak işleyen isteğe bağlı ayrı token temsili üretin; ham token çıktısı ile karşılaştırma sunun.
7. Yanlış pozitifler ve yanlış negatifler için uzmanlarca etiketlenmiş altın standart test kümesi oluşturun. Precision, recall, F1'i tüm alanlarda ve ekli biçimlerde ayrı raporlayın.
8. Özellikle `uzaktan eğitim`, `yapay zekâ`, `ölçme ve değerlendirme`, `yapısal eşitlik modellemesi`, `madde tepki kuramı`, `doğal dil işleme` ifadelerini çekimli ve çekimsiz örneklerle sınayın.
9. Kaynak listesinin özel veriye dayanmadan oluşturulduğunu ve manuel aday listesi olduğunu bildirin. Bu sürümde doğrulanmış tanım, eş anlam ve ontoloji hiyerarşisi yoktur.
10. GitHub `main` dalında değil ayrı `gelistirme-2.1` dalında geliştirin; bütün regresyon testleri geçmeden ana sürümle birleştirmeyin.

## Test örnekleri (beklenen)
| Metin | Beklenen terim | Not |
|---|---|---|
| Uzaktan eğitime katılan öğrenci | uzaktan eğitim | çekimli biçim, morfolojik doğrulama |
| Yapay zekânın etkileri | yapay zekâ | kesme işaretli çekim |
| Ölçme ve değerlendirmede yenilik | ölçme ve değerlendirme | son sözcükte ek |
| Yapısal eşitlik modellemesinin sonuçları | yapısal eşitlik modellemesi | iyelik/genitif |
| Uzaktan eğitimci bir araştırmacı | eşleşme yok | aşırı eşleştirme önlenmeli |
| Yapay zekâsız bir ortam | eşleşme yok veya ayrı türetim etiketi | olumsuzluk/türetim anlamını koru |

## Kontrol listesi
- [ ] Alan uzmanı terimleri ve varyantları gözden geçirdi.
- [ ] Çekimli terimlerin eşleştirilmesi ayrı ölçüldü.
- [ ] Yanlış pozitif örnekler testi geçti.
- [ ] Kullanıcı eklediği terimi silebiliyor.
- [ ] Oturum gizliliği ve dışa aktarım korunuyor.
- [ ] Mevcut 2.0 analizleri regresyon testlerini geçti.
