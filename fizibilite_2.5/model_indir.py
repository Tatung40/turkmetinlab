"""İki modeli TEK deneme ile indirir ve sürümlerini (commit) kaydeder. Model ağırlıkları önbellekte kalır;
sonuç klasörüne yalnızca künye yazılır."""
import json
import os
import sys

from huggingface_hub import snapshot_download

onbellek, cikti = sys.argv[1], sys.argv[2]
MODELLER = sys.argv[3:] or ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "intfloat/multilingual-e5-small"]
DESEN = ["*.json", "*.txt", "model.safetensors", "sentencepiece.bpe.model", "1_Pooling/*", "README.md"]
kunye = {}
for m in MODELLER:
    yol = snapshot_download(m, cache_dir=onbellek, allow_patterns=DESEN)
    boyut = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(yol) for f in fs)
    kunye[m] = {"surum_commit": os.path.basename(yol), "boyut_MB": round(boyut / 2**20, 1),
                "dosyalar": sorted(os.listdir(yol))}
os.makedirs(cikti, exist_ok=True)
with open(os.path.join(cikti, "modeller.json"), "w") as f:
    json.dump(kunye, f, ensure_ascii=False, indent=1)
print(json.dumps(kunye, ensure_ascii=False))
