"""Offline ELECTRA ONNX contract tests: no Hugging Face or network calls."""
import numpy as np
from tmlab import electra_duygu as e

class Token:
    def __init__(self): self.ids=[2,200,201,3];self.type_ids=[0,0,0,0]
class Tok:
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
