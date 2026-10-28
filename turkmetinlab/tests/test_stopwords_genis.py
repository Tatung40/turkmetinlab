"""Extended list opt-in and negation preservation."""
from tmlab.stopwords_genis import EK_STOPWORD, KORUNACAK
from tmlab.stopwords import GENEL_DURAK_SOZCUKLER, stopword_set

def test_no_negators_in_optional_extension():
    assert EK_STOPWORD.isdisjoint(KORUNACAK)
    assert EK_STOPWORD.isdisjoint(GENEL_DURAK_SOZCUKLER)
    assert len(EK_STOPWORD)>5

def test_default_unchanged_and_opt_in_effect():
    base=stopword_set(True,False,())
    extra=stopword_set(True,False,EK_STOPWORD)
    assert len(extra)>len(base)
    assert extra.difference(base)==EK_STOPWORD
