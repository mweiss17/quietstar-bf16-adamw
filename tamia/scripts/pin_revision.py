"""Pin mistralai/Mistral-7B-v0.1 to the last pre-July-2024 revision (tokenizer.json compatible with tokenizers 0.15,
i.e. what the Quiet-STaR authors used). Weights are identical across revisions (only tokenizer/README changed)."""
import os, json, urllib.request
QSTAR = os.environ.get("QSTAR_ROOT", "/scratch/mweiss10/qstar")
os.environ.setdefault("HF_HOME", f"{QSTAR}/hf_home")
from huggingface_hub import snapshot_download
commits = json.load(urllib.request.urlopen("https://huggingface.co/api/models/mistralai/Mistral-7B-v0.1/commits/main", timeout=30))
old = [c for c in commits if c["id"].startswith("26bca36bde83")][0]
REV = old["id"]
print("pinning to", REV, old["date"], old["title"])
p = snapshot_download("mistralai/Mistral-7B-v0.1", revision=REV, allow_patterns=["*.safetensors", "*.json", "tokenizer.model"], max_workers=8)
print("snapshot:", p, sorted(os.listdir(p)))
refs = os.path.join(os.environ["HF_HOME"], "hub", "models--mistralai--Mistral-7B-v0.1", "refs", "main")
open(refs, "w").write(REV)
print("refs/main ->", open(refs).read())
os.makedirs(f"{QSTAR}/data", exist_ok=True)
with open(f"{QSTAR}/data/MISTRAL_REVISION", "w") as f:
    f.write(REV + "\n")
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", revision=REV)
print("fast tokenizer loaded:", type(tok).__name__, "len", len(tok))
print("ids->tokens", tok.convert_ids_to_tokens([28740, 28750, 28734, 13, 330, 365, 334, 384, 413]))
print("encode 'Q: 2+2\\nA: 4\\n' ->", tok.encode("Q: 2+2\nA: 4\n"))
tok.add_special_tokens({"additional_special_tokens": ["<|startthought|>", "<|endthought|>"]})
print("after add:", len(tok), tok.convert_tokens_to_ids("<|startthought|>"), tok.convert_tokens_to_ids("<|endthought|>"))
tok2 = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")  # offline, via refs/main
print("default-revision offline load ok:", len(tok2))
print("PIN_DONE")
