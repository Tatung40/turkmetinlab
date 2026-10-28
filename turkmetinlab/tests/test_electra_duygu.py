"""Offline ELECTRA ONNX contract tests: no Hugging Face or network calls."""
import numpy as np
from tmlab import electra_duygu as e

class Token:
    def __init__(self): self.ids=[2,200,201,3];self.type_ids=[0,0,0,0];self.overflowing=[]
class Tok:
    truncation=None
    def enable_truncation(self,max_length): self.truncation={"max_length":max_length}
    def encode(self,txt): return Token()
class Input:
    def __init__(self,n): self.name=n
class Sess:
    def get_inputs(self): return [Input("input_ids"),Input("attention_mask")]
    def run(self,outs,feeds):
        assert feeds["input_ids"].shape==(1,4)
        return [np.asarray([[0.,0.,3.]])]
def test_local_inference_and_scores():
    ans=e.infer("çok güzel",(Tok(),Sess(),{0:"Negative",1:"Neutral",2:"Positive"}))
    assert ans["Etiket"]=="Positive"
    assert ans["En yüksek model skoru"]>0.9
    assert ans["Kesildi mi"] is False
def test_corpus_limit():
    try:
        e.classify_corpus(["x"]*101,[str(i) for i in range(101)],None)
    except ValueError: pass
    else: assert False


# --- Library truncation/softmax vs frozen pre-refactor code (offline, tiny local vocab) ---
import pytest

def _tokenizer():
    tk=pytest.importorskip("tokenizers")
    vocab={"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"çok":4,"güzel":5,"ışık":6,"İstanbul":7,"değil":8}
    t=tk.Tokenizer(tk.models.WordPiece(vocab,unk_token="[UNK]"))
    t.pre_tokenizer=tk.pre_tokenizers.Whitespace()
    t.post_processor=tk.processors.TemplateProcessing(single="[CLS] $A [SEP]",special_tokens=[("[CLS]",2),("[SEP]",3)])
    return t

class EchoSess:
    def __init__(self,logits): self.logits=logits; self.seen=None
    def get_inputs(self): return [Input("input_ids"),Input("attention_mask")]
    def run(self,outs,feeds): self.seen=feeds["input_ids"][0].tolist(); return [np.asarray([self.logits],dtype=np.float32)]

LABELS={0:"Negative",1:"Neutral",2:"Positive"}

def old_scores(logits):
    logits=np.asarray(logits,dtype=np.float32); s=np.exp(logits-np.max(logits)); return s/s.sum()

@pytest.mark.parametrize("logits",[[0.,0.,3.],[-2.5,1.25,0.5],[7.,7.,7.],[100.,-100.,0.]])
def test_softmax_matches_old_manual_formula(logits):
    ans=e.infer("çok güzel",(_tokenizer(),EchoSess(logits),LABELS))
    old=old_scores(logits)
    for i,lab in LABELS.items():
        assert ans["Skor: "+lab]==pytest.approx(round(float(old[i]),4),abs=1e-4)
    assert ans["Etiket"]==LABELS[int(np.argmax(old))] and ans["Model"]==e.MODEL_ID

def test_short_text_unchanged_and_truncation_keeps_sep():
    sess=EchoSess([0.,1.,0.])
    ans=e.infer("İstanbul çok güzel",(_tokenizer(),sess,LABELS),max_length=8)
    assert sess.seen==[2,7,4,5,3] and ans["Kesildi mi"] is False and ans["İşlenen token"]==5
    ans=e.infer("çok güzel ışık değil çok güzel",(_tokenizer(),sess,LABELS),max_length=4)
    assert sess.seen==[2,4,5,3]  # old ids[:4] slice gave [2,4,5,6]: no closing [SEP]
    assert ans["Kesildi mi"] is True and ans["İşlenen token"]==4

def test_unexpected_label_count_is_rejected():
    with pytest.raises(ValueError,match="etiket"):
        e.infer("çok güzel",(_tokenizer(),EchoSess([0.,1.,2.,3.]),LABELS))
