"""Public synthetic smoke strings only; real Hugging Face ONNX model."""
from tmlab import electra_duygu as model
state = model.load_model()
print("Yuklendi: labels",state[2])
print("ONNX girisleri:",[(x.name,x.type,x.shape) for x in state[1].get_inputs()])
for txt in ("Bu harika bir haber, çok mutluyum.", "Çok kötü bir deneyimdi.", "Toplantı dün yapıldı."):
    result = model.infer(txt,state)
    assert 0 <= result["En yüksek model skoru"] <= 1
    assert result["Etiket"] in state[2].values()
    print("Sinif:",result["Etiket"],"skor:",result["En yüksek model skoru"])
print("GERCEK_ELECTRA_CPU_SMOKE_BASARILI")
