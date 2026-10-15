"""Türkçe durak sözcük (stopword) listeleri.

Liste; bağlaçlar, edatlar, zamirler, soru ekleri, belirteçler ve sık
kullanılan yardımcı fiil biçimlerinden oluşur. Tüm sözcükler Türkçe
küçük harf kurallarına göre yazılmıştır (ı/i ayrımı korunur).

İki liste vardır:
* GENEL_DURAK_SOZCUKLER: Hemen her Türkçe metinde anlam taşımayan sözcükler.
* AKADEMIK_KALIP_SOZCUKLER: Akademik özetlerde çok sık geçen kalıp sözcükler.
  Bunlar bazı araştırmalarda anlamlı olabileceği için varsayılan olarak
  çıkarılmaz; kullanıcı isterse arayüzden etkinleştirir.
"""

GENEL_DURAK_SOZCUKLER = frozenset(
    """
    a acaba altı altmış ama ancak arada artık asla aslında az bana bazen bazı bazıları
    belki ben benden beni benim beş bile bin bir biraz birçok birçoğu birkaç birkez
    birşey birşeyi biri birileri birisi bize bizden bizi bizim bu buna bunda bundan
    bunlar bunları bunların bunu bunun burada böyle böylece bütün da daha dahi de
    defa değil diğer diğeri diğerleri diye doksan dokuz dolayı dolayısıyla dört edilen
    eden ederek edildi ediliyor edilmiş edilmesi eğer elli en etmek etmesi etti ettiği
    ettiğini fakat gene gibi göre hala halde hangi hangisi hani hatta hem henüz hep
    hepsi her herhangi herkes herkesin hiç hiçbir için iki ile ilgili ise işte itibaren
    kadar karşın katrilyon kendi kendine kendini kendisi kendisine kendisini kez
    ki kim kimden kime kimi kimse kırk milyar milyon mi mı mu mü mısın misin musun
    müsün mıdır midir mudur müdür nasıl ne neden nedenle nerde nerede nereye niçin
    niye o olan olarak oldu olduğu olduğunu olduklarını olmadı olmadığı olmak olması
    olmayan olmaz olsa olsun olup olur olursa oluyor on ona ondan onlar onlardan onları
    onların onu onun orada otuz öyle önce ötürü pek rağmen sadece sanki sekiz seksen
    sen senden seni senin siz sizden sizi sizin sonra şayet şey şeyden şeyi şeyler
    şöyle şu şuna şunda şundan şunları şunu tarafından trilyon tüm üç üzere var vardı
    ve veya ya yani yapacak yapılan yapılması yapıyor yapmak yaptı yaptığı yaptığını
    yaptıkları yedi yerine yetmiş yine yirmi yoksa yüz zaten zira
    ayrıca böylelikle çoğu çok çünkü dair değin denli gerek gerekir gereken göre
    hâlâ hepimiz hiçbiri ile birlikte iken ilk ister kimisi madem mademki meğer nitekim
    oysa oysaki öte ötürü sırf şimdi tabii üzerine üzerinde veyahut ya da yahut
    vb vs vd bkz bk örn krş
    """.split()
)

AKADEMIK_KALIP_SOZCUKLER = frozenset(
    """
    amaç amacı amacıyla amaçlanmaktadır amaçlamaktadır araştırma araştırmada araştırmanın
    bulgular bulgulara bulguları bulgulara göre çalışma çalışmada çalışmanın çalışmanın
    elde edilen ele alınmıştır incelenmiştir kapsamında makale makalede sonuç sonuçlar
    sonuçları sonucunda tespit tespit edilmiştir yöntem yöntemi yöntemiyle
    belirlenmiştir değerlendirilmiştir görülmüştür ortaya konmuştur saptanmıştır
    bulunmaktadır olmaktadır yapılmıştır kullanılmıştır gerçekleştirilmiştir
    """.split()
)


def stopword_set(genel: bool = True, akademik: bool = False, ozel=None) -> frozenset:
    """İstenen durak sözcük kümelerini birleştirir.

    ozel: kullanıcının eklediği sözcükler (zaten Türkçe küçük harfe çevrilmiş olmalı).
    """
    kume = set()
    if genel:
        kume |= GENEL_DURAK_SOZCUKLER
    if akademik:
        kume |= AKADEMIK_KALIP_SOZCUKLER
    if ozel:
        kume |= set(ozel)
    return frozenset(kume)
