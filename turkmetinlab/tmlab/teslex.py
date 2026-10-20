"""TES-Lex academic-use emotion evidence; source is not redistributed.

Kumova Metin & Aka Uymaz (2025), IT Professional 27(5), 52–58.
Source repository README limits dataset to academic/research use. The repository
also has MIT LICENSE; seek rights-holder clarification before other use.
"""
from __future__ import annotations
from io import BytesIO
import re
import unicodedata
import urllib.request
import pandas as pd

SOURCE = "https://raw.githubusercontent.com/HandeAkaUymaz/TES-Lex/1cce07edc410dac4aede739296c94d80ad2aa58c/Turkish%20Emotion%20and%20Sentiment%20Lexicon%20%28TES-Lex%29.xlsx"
EMOTIONS = ("Öfke","Neşe","Üzüntü","Korku","İğrenme","Şaşkınlık","Beklenti","Güven")
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
TR = str.maketrans({"I":"ı", "İ":"i"})
MAX_BYTES = 2_500_000

def normalize(s):
    return unicodedata.normalize("NFKC", str(s)).translate(TR).lower().strip()

def read_xlsx(blob: bytes):
    if len(blob) > MAX_BYTES:
        raise ValueError("TES-Lex veri boyutu beklenenden büyük")
    return pd.read_excel(BytesIO(blob), dtype=str).fillna("")

def load_source():
    req = urllib.request.Request(SOURCE,headers={"User-Agent":"TurkMetinLab/2.5-academic-evaluation"})
    with urllib.request.urlopen(req,timeout=25) as response:
        blob=response.read(MAX_BYTES+1)
    return read_xlsx(blob)

def _split_emotions(s):
    low=normalize(s)
    aliases={"öfke":("öfke","anger"),"neşe":("neşe","joy"),
             "üzüntü":("üzüntü","sadness"),"korku":("korku","fear"),
             "iğrenme":("iğrenme","disgust"),"şaşkınlık":("şaşkınlık","surprise"),
             "beklenti":("beklenti","anticipation"),"güven":("güven","trust")}
    return tuple(name for name,keys in aliases.items() if any(
        re.search(r"(?<!\w)"+re.escape(key)+r"(?!\w)",low) for key in keys))

def compile_lexicon(frame, word_col, polarity_col, emotion_col=None, intensity_col=None):
    """Mapping is explicit: never assume actual unpublished XLSX column names."""
    if word_col not in frame or polarity_col not in frame:
        raise ValueError("Kelime ve polarite sütunları seçilmeli")
    out={}
    for _, row in frame.iterrows():
        word=normalize(row[word_col])
        key=tuple(TOKEN.findall(word))
        if not key or len(key)>6:
            continue
        polar=str(row[polarity_col]).strip()
        emotion=str(row[emotion_col]) if emotion_col else ""
        strength=str(row[intensity_col]) if intensity_col else ""
        out[key]={"polarite":polar,"duygular":_split_emotions(emotion),
                  "yogunluk":strength}
    if not out: raise ValueError("Eşleştirilebilir sözlük kaydı yok")
    return out

def analyze(text, lexicon):
    raw=str(text)
    matches=list(TOKEN.finditer(raw))
    tokens=[normalize(m.group()) for m in matches]
    maxn=max((len(x) for x in lexicon), default=1)
    rows=[]
    i=0
    while i<len(tokens):
        hit=None
        for n in range(min(maxn,len(tokens)-i),0,-1):
            hit=lexicon.get(tuple(tokens[i:i+n]))
            if hit is not None: break
        if hit is None:
            i+=1
            continue
        start,end=matches[i].start(),matches[i+n-1].end()
        rows.append({"Sözcük/ifade":raw[start:end],"Polarite":hit["polarite"],
                     "Duygular":", ".join(hit["duygular"]),
                     "Yoğunluk":hit["yogunluk"],"Başlangıç":start,"Bitiş":end})
        i+=n
    denom=len(tokens)
    covered=sum(len(TOKEN.findall(r["Sözcük/ifade"])) for r in rows)
    emotions={name:sum(name.lower() in r["Duygular"].lower().split(", ") for r in rows) for name in EMOTIONS}
    return {"Sözcük sayısı":denom,"Eşleşen ifade":len(rows),
            "Kapsama (%)":round(100*covered/denom,2) if denom else 0.0,
            **emotions,"eşleşmeler":rows}

def analyze_corpus(texts,names,lexicon):
    summary,detail=[],[]
    for name,raw in zip(names,texts):
        r=analyze(raw,lexicon)
        summary.append({"Belge":str(name),**{k:v for k,v in r.items() if k!="eşleşmeler"}})
        detail.extend({"Belge":str(name),**record} for record in r["eşleşmeler"])
    return summary,detail
