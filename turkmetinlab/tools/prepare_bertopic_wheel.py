"""Build a lightweight, audited BERTopic wheel from an upstream MIT release.

NO Python source files are modified; only dependency METADATA is rewritten,
because TürkMetinLab supplies ONNX embeddings and KMeans without Torch/UMAP.
The original license and every other upstream file remain in the wheel.
"""
import base64
import csv
import hashlib
import io
import os
import sys
import zipfile

PIN = "0.17.4"
UPSTREAM_SHA256 = "0dbe7ee22c18fb76efe7f4fbf3a14b51c1e67d9fccc569e3f81e75b8f32646d1"

def prepare(source, output):
    with open(source, "rb") as f:
        blob = f.read()
    if hashlib.sha256(blob).hexdigest() != UPSTREAM_SHA256:
        raise SystemExit("Upstream BERTopic wheel checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        files = {i.filename: z.read(i.filename) for i in z.infolist()}
    metadata_path = next(k for k in files if k.endswith(".dist-info/METADATA"))
    record_path = next(k for k in files if k.endswith(".dist-info/RECORD"))
    raw = files[metadata_path].decode("utf-8")
    lines = raw.splitlines(keepends=True)
    new = "".join(line for line in lines if not line.startswith("Requires-Dist:"))
    assert len(new) < len(raw), "No dependencies found to remove"
    files[metadata_path] = new.encode("utf-8")
    notice_path = metadata_path.rsplit("/", 1)[0] + "/TURKMETINLAB_NOTICE.txt"
    files[notice_path] = (
        "BERTopic 0.17.4 (MIT) from https://github.com/MaartenGr/BERTopic\n"
        "Upstream wheel SHA256: " + UPSTREAM_SHA256 + "\n"
        "Upstream Python source code is unchanged. ONLY package dependency declarations\n"
        "were removed to allow a documented lightweight installation with external\n"
        "ONNX embeddings, no UMAP, and KMeans. Missing alternative backends cannot work.\n"
        "See TürkMetinLab's BERTopic integration guide for limitations.\n"
    ).encode("utf-8")
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for name, content in sorted(files.items()):
        if name == record_path:
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
        writer.writerow((name, "sha256=" + digest, str(len(content))))
    writer.writerow((record_path, "", ""))
    files[record_path] = out.getvalue().encode("utf-8")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    print("Built lightweight wheel:", output, os.path.getsize(output), "bytes")
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: prepare_bertopic_wheel.py upstream.whl output.whl")
    prepare(sys.argv[1], sys.argv[2])
