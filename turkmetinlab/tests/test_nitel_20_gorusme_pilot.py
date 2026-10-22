"""20 sentetik Türkçe öğretmen görüşüyle uçtan uca nitel analiz pilotu.

Bu bir MOCK model testidir, gerçek LLM bilimsel performansı değildir.
"""
import json
from tmlab.nitel_toplu import batch_analyze, regroup_once

SAMPLES = [
 ("T01","Yapay zekâ ders planı hazırlarken zaman kazanmamı sağlıyor.","Zaman tasarrufu","Verimlilik","YZ'nin faydaları"),
 ("T02","Öğrencilerin düzeyine göre farklı etkinlikler hazırlamak kolaylaşıyor.","Farklılaştırılmış öğretim","Uyarlama","Öğretimsel fırsatlar"),
 ("T03","Üretilen bazı bilgiler yanlış olduğu için mutlaka kontrol ediyorum.","Bilgi doğrulama","Güvenilirlik","YZ riskleri"),
 ("T04","İnternet erişimi zayıf olan öğrenciler bu araçlardan yararlanamıyor.","Erişim eşitsizliği","Erişim","Uygulama engelleri"),
 ("T05","Yapay zekâ ile hazırlanan çalışma kâğıtları iş yükümü azaltıyor.","İş yükü","Verimlilik","YZ'nin faydaları"),
 ("T06","Hazır cevaplara alışan öğrenciler kendi düşüncelerini geliştirmeyebilir.","Düşünme becerileri","Öğrenme riskleri","YZ riskleri"),
 ("T07","Kendi dersime uygun örnek sorular oluşturabiliyorum.","Soru hazırlama","Materyal","Öğretimsel fırsatlar"),
 ("T08","Kişisel öğrenci verilerini sisteme yüklemekten çekiniyorum.","Gizlilik","Mahremiyet","YZ riskleri"),
 ("T09","Kısa sınavların hazırlanması çok daha hızlı oluyor.","Ölçme hazırlığı","Verimlilik","YZ'nin faydaları"),
 ("T10","Öğretmenler için uygulamalı eğitim verilmesi gerektiğini düşünüyorum.","Mesleki eğitim","Mesleki gelişim","Uygulama engelleri"),
 ("T11","Bazen verilen örnekler kazanımlarla örtüşmüyor.","Kazanım uyumu","İçerik","YZ riskleri"),
 ("T12","Öğrenciler zorlandıkları konuları ek örneklerle tekrar edebiliyor.","Ek destek","Uyarlama","Öğretimsel fırsatlar"),
 ("T13","Sınıf içinde teknoloji kullanmak öğrencilerin ilgisini artırıyor.","Öğrenci ilgisi","Katılım","Öğretimsel fırsatlar"),
 ("T14","Hazırladığım etkinlikleri son hâline getirmek için yeniden düzenleme yapıyorum.","Düzenleme","Öğretmen denetimi","Uygulama engelleri"),
 ("T15","Ücretsiz araçların günlük kullanım sınırı ders hazırlığını aksatıyor.","Kota sınırı","Teknoloji","Uygulama engelleri"),
 ("T16","Veliler çocuklarının yapay zekâdan çok yardım almasına endişeyle yaklaşıyor.","Veli kaygısı","Tutumlar","YZ riskleri"),
 ("T17","Ders için görsel ve örnek metin bulmak kolaylaştı.","Kaynak bulma","Materyal","Öğretimsel fırsatlar"),
 ("T18","Öğrencinin kendi yazdığı metinle otomatik üretilen metni ayırt etmek zorlaşıyor.","Özgünlük","Etik","YZ riskleri"),
 ("T19","Daha önce saatler süren hazırlıkları artık daha kısa sürede tamamlıyorum.","Zaman tasarrufu","Verimlilik","YZ'nin faydaları"),
 ("T20","Okulumuzdaki bilgisayar sayısı sınırlı olduğu için bütün öğrenciler kullanamıyor.","Donanım","Erişim","Uygulama engelleri"),
]

def test_20_turkish_synthetic_interviews_end_to_end():
    by_text={text:(code,category,theme) for _,text,code,category,theme in SAMPLES}
    calls=[]
    def mock_completion(prompt):
        calls.append(prompt)
        source=prompt.split("KAYNAK:\n",1)[1]
        code,category,theme=by_text[source]
        return json.dumps({"codes":[{"quote":source,"code":code,"category":category,"theme":theme}]},ensure_ascii=False)

    result=batch_analyze([(pid,text) for pid,text,_,_,_ in SAMPLES],mock_completion)
    assert result["processed"]==20
    assert len(result["proposals"])==20
    assert not result["errors"]
    assert result["hierarchy"]["denominator"]==20
    assert len(calls)==20
    for item in result["proposals"]:
        text=next(text for pid,text,_,_,_ in SAMPLES if pid==item["participant_id"])
        assert text[item["start"]:item["end"]]==item["quote"]

    def group_once(prompt):
        assert "KAYNAK:" not in prompt
        labels=sorted({item["code"] for item in result["proposals"]})
        return json.dumps({"mapping":[{"code":label,"category":"İncelenecek kategori","theme":"İncelenecek tema"}
                                       for label in labels]}, ensure_ascii=False)

    grouped=regroup_once(result,group_once)
    assert len(grouped["proposals"])==20
    assert grouped["hierarchy"]["denominator"]==20
    assert grouped["proposals"][0]["theme"]=="İncelenecek tema"
    assert grouped["requires_researcher_approval"]
