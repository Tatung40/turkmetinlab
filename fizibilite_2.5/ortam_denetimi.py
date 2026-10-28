"""CI ortam denetimi: Python 3.12, CPU'ya özel torch, CUDA/NVIDIA paketi yok. Sorun varsa 1 ile çıkar."""
import importlib.metadata as md
import json
import os
import sys

import torch

r = {"python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda,
     "cuda_kullanilabilir": torch.cuda.is_available(),
     "nvidia_paketleri": sorted(d.metadata["Name"] for d in md.distributions() if d.metadata["Name"].lower().startswith("nvidia")),
     "surumler": {p: md.version(p) for p in ("sentence-transformers", "transformers", "tokenizers", "huggingface-hub",
                                              "numpy", "scikit-learn", "onnxruntime")}}
os.makedirs(sys.argv[1] if len(sys.argv) > 1 else ".", exist_ok=True)
with open(os.path.join(sys.argv[1] if len(sys.argv) > 1 else ".", "ortam.json"), "w") as f:
    json.dump(r, f, ensure_ascii=False, indent=1)
print(json.dumps(r, ensure_ascii=False))
hatalar = []
if not r["python"].startswith("3.12"):
    hatalar.append("Python 3.12 değil")
if r["torch_cuda"] is not None or "+cpu" not in r["torch"]:
    hatalar.append("torch CPU sürümü değil")
if r["nvidia_paketleri"]:
    hatalar.append("NVIDIA/CUDA paketleri kurulmuş")
if hatalar:
    sys.exit("ORTAM HATASI: " + "; ".join(hatalar))
