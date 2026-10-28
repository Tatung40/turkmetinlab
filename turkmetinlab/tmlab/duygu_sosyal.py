"""Opt-in, auditable tweet cleanup and TES-Lex lexical contribution diagnostics."""
from __future__ import annotations
import re
import unicodedata
from collections import Counter
import pandas as pd

MENTION=re.compile(r"(?<![\w])@[\w_]+",re.UNICODE)
URL=re.compile(r"https?://[^\s]+|www\.[^\s]+",re.IGNORECASE)
RT=re.compile(r"(?m)^\s*RT\s+(?=@)",re.IGNORECASE)
TAG=re.compile(r"(?<!\w)#([^\W_][\w]*)",re.UNICODE)
# Selective lexical filtering, not source dictionary modification.
FUNCTION_WORDS=frozenset("ve veya ile için en çok ancak ama gibi kadar diye de da ki mi mı mu mü bir bu şu o".split())

def clean_tweet(text, remove_mentions=True, remove_urls=True, strip_hashtag_sign=True,
                remove_emojis=False):
    """Preserve hashtag words and emoji by default; do not collapse negation or sentiment."""
    s=unicodedata.normalize("NFC",str(text))
    if remove_urls: s=URL.sub(" ",s)
    if remove_mentions:
        s=RT.sub(" ",s)
        s=MENTION.sub(" ",s)
    if strip_hashtag_sign: s=TAG.sub(lambda m:m.group(1),s)
    if remove_emojis:
        # Remove Unicode pictographs conservatively. Normally keep them for possible sentiment.
        s="".join(c for c in s if not (0x1F000<=ord(c)<=0x1FAFF or 0x2600<=ord(c)<=0x27BF))
    return re.sub(r"\s+"," ",s).strip()

def filtered_lexicon(lexicon, excluded=FUNCTION_WORDS):
    return {k:v for k,v in lexicon.items() if not (len(k)==1 and k[0] in excluded)}

def contribution_table(details):
    c=Counter()
    for item in details:
        word=unicodedata.normalize("NFC",str(item["Sözcük/ifade"])).casefold()
        for emotion in str(item.get("Duygular","")).split(","):
            emotion=emotion.strip()
            if emotion: c[(emotion,word)]+=1
    totals=Counter()
    for (emotion,_),n in c.items():totals[emotion]+=n
    rows=[{"Duygu":e,"Sözcük":w,"Frekans":n,
           "Duygu içindeki pay (%)":round(100*n/totals[e],2)}
           for (e,w),n in c.items()]
    return pd.DataFrame(rows,columns=["Duygu","Sözcük","Frekans","Duygu içindeki pay (%)"]).sort_values(
        ["Duygu","Frekans","Sözcük"],ascending=[True,False,True]).reset_index(drop=True)

def excluded_contributions(raw_details,filtered_details):
    f=Counter((x["Belge"],x["Başlangıç"],x["Bitiş"]) for x in filtered_details)
    removed=[]
    for x in raw_details:
        k=(x["Belge"],x["Başlangıç"],x["Bitiş"])
        if f[k]:f[k]-=1
        else:removed.append(x)
    return removed
