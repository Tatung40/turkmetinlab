"""TES-Lex separate emotion columns can be combined without false positive empty/zero."""
import pandas as pd
from tmlab import teslex

def test_separate_emotion_columns():
    frame=pd.DataFrame({"Kelime":["Korku","Neşe"],"Polarite":["negatif","pozitif"],
                       "Fear":[1,0],"Joy":[0,1],"Trust":["Hayır","evet"]})
    l=teslex.compile_lexicon(frame,"Kelime","Polarite",emotion_columns=["Fear","Joy","Trust"])
    res=teslex.analyze("Korku ve neşe",l)
    assert res["Korku"]==1 and res["Neşe"]==1
    assert res["Güven"]==1
    assert res["Öfke"]==0
