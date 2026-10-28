"""Conservative extended Turkish stopword *candidate* set.

This list is hand-selected for experimental comparison. It is NOT the full
nezahatkorkmaz/turkish-stopwords dataset; that source's raw words require
separate encoding and provenance audit. Negators are intentionally excluded.
"""
from .stopwords import GENEL_DURAK_SOZCUKLER

KORUNACAK = frozenset(("değil", "yok", "hiç", "asla", "hayır", "olumsuz", "olumlu", "yoksun"))
ADAYLAR = frozenset("""
acaba ayrıca beraber birçoğu birkaçı bunca bunlar bunları bunların burada buraya buradan
böylece çoğunca dolayısıyla evvela genellikle gerçekte giderek herhangi herkes herkese
herkesin herşey herkesçe iken ileride kısacası kimi kimileri kimisi kolayca
muhtemelen neredeyse neticede nihayetinde öbürü öncelikle örneğin peşinden
sadece sonrasında süresince şayet takiben tümü tümünün tümüne tümünü üstelik
velev velhasıl yeniden yukarıda yukarıya ziyade çoğunlukla özetle
""".split())
EK_STOPWORD = frozenset(x for x in ADAYLAR if x not in GENEL_DURAK_SOZCUKLER and x not in KORUNACAK)
