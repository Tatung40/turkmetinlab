"""Offline tests for TES-Lex mapping and lexicon-based emotion signals."""
import pandas as pd
from tmlab import teslex

def test_mapping_and_plutchik_labels():
    df=pd.DataFrame({"word":["SEVİNÇ","korku"],"pol":["Positive","Negative"],
                     "em":["Joy;Trust","Fear"],"power":["5","4"]})
    lex=teslex.compile_lexicon(df,"word","pol","em","power")
    x=teslex.analyze("Sevinç ve korku",lex)
    assert x["Eşleşen ifade"]==2
    assert x["Neşe"]==1 and x["Güven"]==1 and x["Korku"]==1
    assert x["Kapsama (%)"] > 0
    assert x["eşleşmeler"][0]["Yoğunluk"]=="5"

def test_no_invented_emotions_for_unmatched():
    lex=teslex.compile_lexicon(pd.DataFrame({"w":["mutlu"],"p":["positive"]}),"w","p")
    s,d=teslex.analyze_corpus(["bilinmeyen","mutlu"],["1","2"],lex)
    assert s[0]["Eşleşen ifade"]==0 and not d[0]["Belge"]=="1"
    assert len(d)==1
