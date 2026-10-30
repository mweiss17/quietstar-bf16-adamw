"""Prefetch model + eval datasets into HF_HOME and build the 1,000-row open-web-math subset.

Streaming the first 1,000 rows walks the same shard list in the same order as
split="train[:1000]", so the rows are identical without the 27 GB download.
"""
import os, sys, json, time
QSTAR = os.environ.get("QSTAR_ROOT", "/scratch/mweiss10/qstar")
os.environ.setdefault("HF_HOME", f"{QSTAR}/hf_home")
from huggingface_hub import snapshot_download
from datasets import load_dataset, Dataset, load_from_disk

t0 = time.time()
p = snapshot_download("mistralai/Mistral-7B-v0.1",
                      allow_patterns=["*.safetensors", "*.json", "tokenizer.model"],
                      max_workers=8)
print("model snapshot:", p, "files:", sorted(os.listdir(p)), f"({time.time()-t0:.0f}s)", flush=True)

gsm = load_dataset("gsm8k", "main", split="test")
print("gsm8k test rows:", len(gsm), flush=True)
csqa = load_dataset("tau/commonsense_qa", "default", split="validation")
print("csqa validation rows:", len(csqa), flush=True)

out = f"{QSTAR}/data/owm_first1000"
if not os.path.exists(os.path.join(out, "dataset_info.json")):
    ds = load_dataset("open-web-math/open-web-math", "default", split="train", streaming=True)
    rows = []
    for r in ds:
        rows.append(r)
        if len(rows) == 1000:
            break
    d = Dataset.from_list(rows)
    d.save_to_disk(out)
    print("saved owm subset", flush=True)
d2 = load_from_disk(out)
print(d2)
print("row0 url:", d2[0]["url"])
print("row0 text[:120]:", repr(d2[0]["text"][:120]))
print("row999 url:", d2[999]["url"])
print("PREP_DATA_DONE", f"({time.time()-t0:.0f}s)")
