


import os
print([f for f in os.listdir(".") if "torch" in f.lower()])   # should be []
import torch
print(torch.__version__, torch.cuda.is_available(), hasattr(torch, "device"))
print(torch.cuda.get_device_name(0))





import os, glob
from pathlib import Path
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["CS781_WORK"] = "/kaggle/working/cs781_phase3"

# locate the Phase-1 package you attached as a Kaggle Model
tars = glob.glob("/kaggle/input/**/*.tar.gz", recursive=True)
pts  = glob.glob("/kaggle/input/**/model.pt", recursive=True)
os.environ["CS781_FOUNDATION"] = tars[0] if tars else str(Path(pts[0]).parent)
print("FOUNDATION =", os.environ["CS781_FOUNDATION"])


get_ipython().run_line_magic("pip", " install -q \"tokenizers>=0.20,<1\" \"huggingface_hub>=0.26,<2\" pandas")


import os, re, json, math, time, copy, random, zipfile, tarfile, gc, shutil, getpass
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, fields
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ============================== EDIT THESE ==============================
WORK = Path(os.environ.get("CS781_WORK", str(Path.home() / "cs781_phase3")))   # persistent EBS in SageMaker Studio
FOUNDATION = os.environ.get("CS781_FOUNDATION", str(WORK / "lexy-pytorch-default-v1.tar.gz"))  # .tar.gz / .zip / dir / s3://
DATA_DIR = WORK / "data"              # Phase_3_RR is downloaded here from HF if not already present
HF_REPO = "Exploration-Lab/CS-781-Capstone"
TIME_BUDGET_H = 6.0                   # the queue will not START a new experiment after this many hours
SEED = 42
# =======================================================================
OUT = WORK / "runs"
for p in (WORK, DATA_DIR, OUT): p.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    print(torch.cuda.get_device_name(), f"| {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
print("device:", DEVICE, "| amp:", AMP_DTYPE if DEVICE == "cuda" else "off", "| torch", torch.__version__)

SESSION_START = time.time()
LOG = WORK / "phase3_log.txt"

def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def amp():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if DEVICE == "cuda" else torch.autocast("cpu", enabled=False)

def no_amp():
    return torch.autocast("cuda" if DEVICE == "cuda" else "cpu", enabled=False)

seed_all(SEED)





class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        z = x.float()
        return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight


class Rotary(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        assert head_dim % 2 == 0
        inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv', inv, persistent=False)

    def forward(self, x, positions):
        a = torch.outer(positions.float(), self.inv.float())
        c = a.cos().to(x.dtype)[None, None]
        s = a.sin().to(x.dtype)[None, None]
        u, v = x[..., ::2], x[..., 1::2]
        return torch.stack((u*c-v*s, u*s+v*c), -1).flatten(-2)


class Attention(nn.Module):
    """Exact causal token attention is retained for names, numbers and citations."""
    def __init__(self, dim, heads, kv_heads):
        super().__init__()
        assert dim % heads == 0 and heads % kv_heads == 0
        self.h, self.k, self.dh = heads, kv_heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * kv_heads * self.dh, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.rope = Rotary(self.dh)

    def forward(self, x):
        b, t, d = x.shape
        q = self.q(x).view(b, t, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(x).view(b, t, 2, self.k, self.dh).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        pos = torch.arange(t, device=x.device)
        q, k = self.rope(q, pos), self.rope(k, pos)
        if self.h != self.k:
            k = k.repeat_interleave(self.h // self.k, dim=1)
            v = v.repeat_interleave(self.h // self.k, dim=1)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.)
        return self.out(z.transpose(1, 2).contiguous().view(b, t, d))


class ChunkMemory(nn.Module):
    """Learn a salience-weighted vector per chunk; read only COMPLETED past chunks."""
    def __init__(self, dim, mem_dim=128, heads=4, chunk=32, learned_pool=True):
        super().__init__()
        assert mem_dim % heads == 0
        self.dim, self.md, self.h = dim, mem_dim, heads
        self.dh, self.chunk = mem_dim // heads, chunk
        self.score = nn.Linear(dim, 1, bias=False) if learned_pool else None
        self.q = nn.Linear(dim, mem_dim, bias=False)
        self.kv = nn.Linear(dim, 2*mem_dim, bias=False)
        self.out = nn.Linear(mem_dim, dim, bias=False)
        self.rope = Rotary(self.dh)

    def pool(self, x):
        b, t, d = x.shape
        c = self.chunk
        n = (t+c-1)//c
        padded = F.pad(x, (0, 0, 0, n*c-t)).view(b, n, c, d)
        valid = torch.arange(n*c, device=x.device).view(1, n, c) < t
        if self.score is None:
            w = valid.float().expand(b, -1, -1)
            w = w / w.sum(-1, keepdim=True).clamp_min(1)
        else:
            logits = self.score(padded).squeeze(-1).float()
            logits = logits.masked_fill(~valid, -float('inf'))
            w = logits.softmax(-1)
        summaries = (padded * w.to(padded.dtype).unsqueeze(-1)).sum(-2)
        ends = (torch.arange(n, device=x.device)+1)*c - 1
        return summaries, ends

    def forward(self, x):
        b, t, _ = x.shape
        summary, ends = self.pool(x)
        n = summary.shape[1]
        q = self.q(x).view(b, t, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(summary).view(b, n, 2, self.h, self.dh).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        pos = torch.arange(t, device=x.device)
        q, k = self.rope(q, pos), self.rope(k, ends)
        k = torch.cat((k.new_zeros(b, self.h, 1, self.dh), k), dim=2)
        v = torch.cat((v.new_zeros(b, self.h, 1, self.dh), v), dim=2)
        allowed = ends[None, :] < pos[:, None]
        allowed = torch.cat((torch.ones(t, 1, dtype=torch.bool, device=x.device), allowed), -1)
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[None, None], dropout_p=0.)
        return self.out(z.transpose(1, 2).contiguous().view(b, t, self.md))


class PhraseMixer(nn.Module):
    """Token-gated, left-padded depthwise convolutions at multiple spacings."""
    def __init__(self, dim, inner=128, dilations=(1, 4, 16)):
        super().__init__()
        self.dilations = tuple(dilations)
        self.up = nn.Linear(dim, inner, bias=False)
        self.conv = nn.ModuleList([nn.Conv1d(inner, inner, 3, dilation=r,
                                            groups=inner, bias=False) for r in self.dilations])
        self.scale_gate = nn.Linear(inner, len(self.dilations), bias=False) if len(self.dilations)>1 else None
        self.out = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        z = F.silu(self.up(x))
        gate = self.scale_gate(z).float().softmax(-1).to(z.dtype) if self.scale_gate is not None else None
        y = torch.zeros_like(z)
        for i, (conv, r) in enumerate(zip(self.conv, self.dilations)):
            branch = conv(F.pad(z.transpose(1, 2), (2*r, 0))).transpose(1, 2)
            y = y + (branch * gate[..., i:i+1] if gate is not None else branch)
        return self.out(F.silu(y))


class WeaveBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, a = cfg['dim'], cfg['ablation']
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, cfg['heads'], cfg['kv_heads'])
        self.memory = None if a in ('no_memory', 'plain') else ChunkMemory(
            d, cfg['mem_dim'], cfg['mem_heads'], cfg['chunk'], learned_pool=a != 'mean_pool')
        self.phrase = None if a in ('no_conv', 'plain') else PhraseMixer(
            d, cfg['conv_dim'], (1,) if a == 'single_scale' else cfg['dilations'])
        self.active = [0] + ([1] if self.memory is not None else []) + ([2] if self.phrase is not None else [])
        self.router = nn.Linear(d, 3, bias=False) if a not in ('fixed_mix', 'plain') else None
        self.register_buffer('route_prior', torch.tensor([2., 0., 0.]), persistent=True)
        self.chunk = cfg['chunk']
        self.up = nn.Linear(d, 2*cfg['ff_dim'], bias=False)
        self.down = nn.Linear(cfg['ff_dim'], d, bias=False)
        self.record_routes = False
        self.last_routes = None

    def route_weights(self, x):
        b, t, _ = x.shape
        logits = self.route_prior[None, None].expand(b, t, 3)
        if self.router is not None: logits = logits + self.router(x).float()
        enabled = torch.zeros(t, 3, dtype=torch.bool, device=x.device)
        enabled[:, self.active] = True
        enabled[:, 1] &= torch.arange(t, device=x.device) >= self.chunk
        return logits.masked_fill(~enabled[None], -float('inf')).softmax(-1).to(x.dtype)

    def forward(self, x):
        z = self.n1(x)
        weights = self.route_weights(z)
        mixed = weights[..., 0:1] * self.attn(z)
        if self.memory is not None: mixed = mixed + weights[..., 1:2] * self.memory(z)
        if self.phrase is not None: mixed = mixed + weights[..., 2:3] * self.phrase(z)
        if self.record_routes and not self.training:
            self.last_routes = weights.detach().float().mean((0, 1)).cpu()
        x = x + mixed
        a, b = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(a) * b)


class LegalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.kind = dict(cfg), 'lexweave'
        self.grad_ckpt = cfg.get('grad_checkpoint', True)
        self.emb = nn.Embedding(cfg['vocab'], cfg['dim'])
        self.blocks = nn.ModuleList([WeaveBlock(cfg) for _ in range(cfg['layers'])])
        self.norm = RMSNorm(cfg['dim'])
        self.apply(self.init_weights)
        residual_std = 0.02 / math.sqrt(2*cfg['layers'])
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=residual_std)
            nn.init.normal_(block.down.weight, std=residual_std)
            if block.memory is not None:
                nn.init.normal_(block.memory.out.weight, std=residual_std)
                if block.memory.score is not None: nn.init.zeros_(block.memory.score.weight)
            if block.phrase is not None:
                nn.init.normal_(block.phrase.out.weight, std=residual_std)
                if block.phrase.scale_gate is not None: nn.init.zeros_(block.phrase.scale_gate.weight)
            if block.router is not None: nn.init.zeros_(block.router.weight)

    @staticmethod
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, 'bias', None) is not None: nn.init.zeros_(m.bias)

    def hidden(self, ids, src=None, memory=None):
        if ids.shape[1] > self.cfg['context']: raise ValueError('Input exceeds configured context.')
        x = self.emb(ids)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training and self.grad_ckpt else block(x)
        return self.norm(x)

    def forward(self, ids, src=None, targets=None, memory=None):
        x = self.hidden(ids)
        if targets is None: return F.linear(x[:, -1], self.emb.weight).float()
        def ce(h, y):
            logits = F.linear(h, self.emb.weight).float()
            return F.cross_entropy(logits.reshape(-1, self.cfg['vocab']), y.reshape(-1), reduction='sum')
        total = x.new_zeros((), dtype=torch.float32)
        for j in range(0, x.shape[1], 128):
            h, y = x[:, j:j+128], targets[:, j:j+128]
            total = total + (checkpoint(ce, h, y, use_reentrant=False) if self.training and self.grad_ckpt else ce(h, y))
        return total / targets.ne(-100).sum().clamp_min(1)


def prepare_foundation(src=FOUNDATION, dest=WORK / "foundation"):
    """Extract model.pt + tokenizer.json from the Phase-1 package (tar.gz / zip / directory / s3:// URI)."""
    need = ("model.pt", "tokenizer.json")
    if all((dest / n).exists() for n in need): return dest
    dest.mkdir(parents=True, exist_ok=True); src = str(src)
    if src.startswith("s3://"):
        import boto3
        bucket, key = src[5:].split("/", 1); local = WORK / Path(key).name
        if not local.exists(): boto3.client("s3").download_file(bucket, key, str(local))
        src = str(local)
    p = Path(src)
    assert p.exists(), f"Foundation package not found: {p}. Upload lexy-pytorch-default-v1.tar.gz to {WORK}."
    def _copy(fi, n):
        with open(dest / (n + ".tmp"), "wb") as fo: shutil.copyfileobj(fi, fo)
        os.replace(dest / (n + ".tmp"), dest / n)
    if p.is_dir():
        for n in need:
            with open(next(p.rglob(n)), "rb") as fi: _copy(fi, n)
    elif zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            for n in need:
                with z.open(next(m for m in z.namelist() if m.split("/")[-1] == n)) as fi: _copy(fi, n)
    else:
        with tarfile.open(p) as t:
            members = {m.name.split("/")[-1]: m for m in t.getmembers() if m.isfile()}
            for n in need:
                with t.extractfile(members[n]) as fi: _copy(fi, n)
    return dest

FOUND = prepare_foundation()
_ck = torch.load(FOUND / "model.pt", map_location="cpu", weights_only=False)
LM_CFG = dict(_ck["model_config"])
MASTER_LM = LegalLM(LM_CFG)
MASTER_LM.load_state_dict(_ck["model"], strict=True)          # our Phase-1 weights, nothing else
MASTER_LM.eval()
del _ck; gc.collect()
CONTEXT = int(LM_CFG["context"]); HID = int(LM_CFG["dim"])
print({k: LM_CFG[k] for k in ("arch", "vocab", "context", "dim", "heads", "kv_heads", "layers", "ff_dim", "ablation") if k in LM_CFG})
print(f"Phase-1 LEXWeave parameters: {sum(p.numel() for p in MASTER_LM.parameters()):,}")

from tokenizers import Tokenizer
TOK = Tokenizer.from_file(str(FOUND / "tokenizer.json"))
PAD, BOS, EOS, UNK = [TOK.token_to_id(s) for s in ("<pad>", "<bos>", "<eos>", "<unk>")]
assert TOK.get_vocab_size() == LM_CFG["vocab"], "tokenizer / model vocab mismatch"
print("special ids:", dict(pad=PAD, bos=BOS, eos=EOS, unk=UNK))





def ensure_data():
    hits = list(DATA_DIR.rglob("label_vocab.json"))
    if not hits:
        from huggingface_hub import snapshot_download
        token = os.environ.get("HF_TOKEN") or getpass.getpass("Hugging Face read token: ")
        snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=["Phase_3_RR/*"],
                          local_dir=str(DATA_DIR), token=token)
        hits = list(DATA_DIR.rglob("label_vocab.json"))
    assert hits, f"label_vocab.json not found under {DATA_DIR}"
    return hits[0].parent.parent              # .../Phase_3_RR

RR_ROOT = ensure_data()
def find_file(name):
    hits = sorted(RR_ROOT.rglob(name)); assert hits, f"{name} not found under {RR_ROOT}"; return hits[0]

_lv = json.load(open(find_file("label_vocab.json"), encoding="utf-8"))
if isinstance(_lv, dict):
    if all(isinstance(v, int) for v in _lv.values()): LABELS = [k for k, _ in sorted(_lv.items(), key=lambda kv: kv[1])]
    elif "labels" in _lv: LABELS = list(_lv["labels"])
    else: LABELS = [v for _, v in sorted(_lv.items(), key=lambda kv: int(kv[0]))]
else:
    LABELS = list(_lv)
LABELS = [str(l) for l in LABELS]
L2ID = {l: i for i, l in enumerate(LABELS)}; C = len(LABELS)
DOMAINS = ["CL", "IT"]; DOM2ID = {d: i for i, d in enumerate(DOMAINS)}
print(C, "labels:", LABELS)

def _read_json(path):
    try: return json.load(open(path, encoding="utf-8"))
    except json.JSONDecodeError:               # JSON-lines fallback
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def load_docs(fname, domain, with_labels):
    raw = _read_json(find_file(fname))
    if isinstance(raw, dict):
        if isinstance(raw.get("data"), list): items = raw["data"]
        elif isinstance(raw.get("documents"), list): items = raw["documents"]
        else: items = [dict(v, id=v.get("id", k)) if isinstance(v, dict) else {"id": k, "text": v} for k, v in raw.items()]
    else:
        items = raw
    docs = []
    for it in items:
        did = it.get("id", it.get("doc_id", it.get("document_id")))
        sents = it.get("text", it.get("sentences"))
        assert did is not None and isinstance(sents, list), f"unexpected record layout in {fname}: {list(it)[:6]}"
        d = dict(id=str(did), domain=domain, sents=[str(s) for s in sents])
        if with_labels:
            labs = it.get("labels", it.get("label", it.get("roles")))
            labs = [LABELS[l] if isinstance(l, int) else str(l) for l in labs]
            assert len(labs) == len(sents), f"{fname}:{did} has {len(sents)} sentences but {len(labs)} labels"
            bad = set(labs) - set(LABELS)
            assert not bad, f"{fname}:{did} unknown labels {bad}"
            d["labels"] = labs
        docs.append(d)
    return docs

TRAIN_DOCS = load_docs("CL_train.json", "CL", True) + load_docs("IT_train.json", "IT", True)
DEV_DOCS = load_docs("CL_dev.json", "CL", True) + load_docs("IT_dev.json", "IT", True)
TEST_DOCS = load_docs("CL_test.json", "CL", False) + load_docs("IT_test.json", "IT", False)
_ids = [d["id"] for d in TEST_DOCS]
assert len(_ids) == len(set(_ids)), "test document ids collide between CL and IT, so they cannot share one prediction.json"
for name, ds in (("train", TRAIN_DOCS), ("dev", DEV_DOCS), ("test", TEST_DOCS)):
    for dom in DOMAINS:
        sub = [d for d in ds if d["domain"] == dom]
        print(f"{name:<5} {dom}: {len(sub):4d} docs | {sum(len(d['sents']) for d in sub):6d} sentences")





import pandas as pd
try: display
except NameError: display = print

def label_table(docs):
    t = {dom: Counter(l for d in docs if d["domain"] == dom for l in d["labels"]) for dom in DOMAINS}
    df = pd.DataFrame({dom: [t[dom][l] for l in LABELS] for dom in DOMAINS}, index=LABELS)
    for dom in DOMAINS: df[dom + "_%"] = (100 * df[dom] / max(1, df[dom].sum())).round(2)
    return df
print("Train label distribution:"); display(label_table(TRAIN_DOCS))
print("Dev label distribution:");   display(label_table(DEV_DOCS))

def sent_ids(s):
    """Sentences are encoded with a leading space, the way mid-document words were seen in Phase-1 pretraining."""
    return TOK.encode(" " + " ".join(s.split()), add_special_tokens=False).ids or [UNK]

for d in TRAIN_DOCS + DEV_DOCS + TEST_DOCS: d["tok"] = [sent_ids(s) for s in d["sents"]]

_sl = np.array([len(t) for d in TRAIN_DOCS for t in d["tok"]]); _dl = np.array([sum(map(len, d["tok"])) for d in TRAIN_DOCS])
print(f"tokens / sentence: p50 {np.median(_sl):.0f} | p95 {np.percentile(_sl, 95):.0f} | max {_sl.max()} | >128: {100*(_sl > 128).mean():.2f}%")
print(f"tokens / document: p50 {np.median(_dl):.0f} | max {_dl.max()} | windows of {CONTEXT} needed per doc ≈ {np.median(_dl)/CONTEXT:.1f}")
print(f"sentences / doc  : p50 {np.median([len(d['sents']) for d in TRAIN_DOCS]):.0f}")
# How strongly do labels persist across neighbouring sentences? (motivates document-level context)
_same = [a == b for d in TRAIN_DOCS for a, b in zip(d["labels"], d["labels"][1:])]
print(f"P(label == previous sentence's label) = {np.mean(_same):.3f}")
_pos = defaultdict(list)
for d in TRAIN_DOCS:
    for i, l in enumerate(d["labels"]): _pos[l].append(i / max(1, len(d["labels"]) - 1))
print("median relative position in document:", {l: round(float(np.median(v)), 2) for l, v in _pos.items()})





@torch.no_grad()
def lm_loss_on(docs, max_windows=24):
    m = MASTER_LM.to(DEVICE).eval(); tot, n = 0.0, 0
    for d in docs:
        ids = [t for s in d["tok"] for t in s]
        for a in range(0, max(1, len(ids) - 1), CONTEXT):
            x = torch.tensor([ids[a:a + CONTEXT + 1]], device=DEVICE)
            if x.size(1) < 2: continue
            with amp(): loss = m(x[:, :-1], targets=x[:, 1:])
            tot += float(loss) * (x.size(1) - 1); n += x.size(1) - 1; max_windows -= 1
            if max_windows <= 0: break
        if max_windows <= 0: break
    MASTER_LM.cpu()
    return tot / max(1, n)
_l = lm_loss_on(DEV_DOCS)
print(f"Phase-1 LM loss on RR dev text: {_l:.3f} (ppl {math.exp(_l):.1f}) vs uniform ln|V| = {math.log(LM_CFG['vocab']):.3f}")
if DEVICE == "cuda": torch.cuda.empty_cache()





def cap_ids(ids, cap):
    return ids if len(ids) <= cap else ids[:cap - 32] + ids[-32:]

def sentence_windows(lens, budget, overlap):
    n, i, out = len(lens), 0, []
    while i < n:
        j, tot = i, 0
        while j < n and (j == i or tot + lens[j] <= budget):
            tot += lens[j]; j += 1
        out.append((i, j))
        if j >= n: break
        i = max(i + 1, j - int((j - i) * overlap))
    return out

def make_examples(docs, cfg, with_labels, reserve=0):
    budget = min(cfg.window, CONTEXT - reserve)
    cap = min(cfg.max_sent_tokens, budget)
    exs = []
    for di, d in enumerate(docs):
        toks = [cap_ids(t, cap) for t in d["tok"]]
        spans = [(k, k + 1) for k in range(len(toks))] if cfg.context == "none" else \
                sentence_windows([len(t) for t in toks], budget, cfg.overlap)
        for i, j in spans:
            ids, seg, last = [], [], []
            for k in range(i, j):
                ids += toks[k]; seg += [k - i] * len(toks[k]); last.append(len(ids) - 1)
            ex = dict(doc=di, lo=i, hi=j, ids=ids, seg=seg, last=last, dom=DOM2ID[d["domain"]])
            if with_labels: ex["y"] = [L2ID[l] for l in d["labels"][i:j]]
            exs.append(ex)
    return exs

def collate(batch):
    B = len(batch); T = max(len(x["ids"]) for x in batch); S = max(len(x["last"]) for x in batch)
    ids = torch.full((B, T), PAD, dtype=torch.long); seg = torch.full((B, T), -1, dtype=torch.long)
    last = torch.zeros((B, S), dtype=torch.long); smask = torch.zeros((B, S), dtype=torch.bool)
    y = torch.full((B, S), -100, dtype=torch.long)
    for b, x in enumerate(batch):
        n, s = len(x["ids"]), len(x["last"])
        ids[b, :n] = torch.tensor(x["ids"]); seg[b, :n] = torch.tensor(x["seg"])
        last[b, :s] = torch.tensor(x["last"]); smask[b, :s] = True
        if "y" in x: y[b, :s] = torch.tensor(x["y"])
    return dict(ids=ids, seg=seg, last=last, smask=smask, y=y, dom=torch.tensor([x["dom"] for x in batch]))

def to_dev(b): return {k: v.to(DEVICE, non_blocking=True) for k, v in b.items()}

# --- self-test: every sentence covered, sentence pointers correct --------------------------------------
@dataclass
class _W: window: int = 1024; overlap: float = 0.25; max_sent_tokens: int = 128; context: str = "doc"
for _ctx in ("doc", "none"):
    _ex = make_examples(DEV_DOCS, _W(context=_ctx), True)
    _cov = [np.zeros(len(d["sents"]), int) for d in DEV_DOCS]
    for x in _ex:
        _cov[x["doc"]][x["lo"]:x["hi"]] += 1
        assert len(x["ids"]) <= CONTEXT and x["seg"][x["last"][-1]] == len(x["last"]) - 1
    assert all((c >= 1).all() for c in _cov)
    print(f"context={_ctx}: {len(_ex)} dev windows | mean sentences/window {np.mean([x['hi']-x['lo'] for x in _ex]):.1f} "
          f"| mean coverage {np.mean(np.concatenate(_cov)):.2f}")





class Backbone(nn.Module):
    """Runs LEXWeave and returns final-norm token states (the LM head is unused).
    Hosts the soft prompt (prompt tuning) and the prefix encoder (prefix tuning)."""
    def __init__(self, lm):
        super().__init__()
        self.lm, self.prompt, self.prefix, self.grad_ckpt = lm, None, None, False

    @property
    def n_virtual(self): return 0 if self.prompt is None else self.prompt.shape[0]

    def forward(self, ids):
        x = self.lm.emb(ids)
        if self.prompt is not None:
            x = torch.cat([self.prompt.to(x.dtype).unsqueeze(0).expand(x.size(0), -1, -1), x], 1)
        if self.prefix is not None:
            for blk, kv in zip(self.lm.blocks, self.prefix()): blk.attn.cur = kv
        for blk in self.lm.blocks:
            x = checkpoint(blk, x, use_reentrant=False) if (self.training and self.grad_ckpt) else blk(x)
        return self.lm.norm(x)[:, self.n_virtual:]

def pool_sentences(h, seg, last):
    """h [B,T,H] -> [B,S,2H] = [mean over the sentence's tokens ‖ state at its last token]."""
    B, T, H = h.shape; S = last.size(1); h = h.float()
    m = (seg >= 0); idx = seg.clamp(min=0)
    sums = h.new_zeros(B, S, H).scatter_add_(1, idx[..., None].expand(-1, -1, H), h * m[..., None])
    cnt = h.new_zeros(B, S).scatter_add_(1, idx, m.float())
    mean = sums / cnt.clamp(min=1)[..., None]
    lastv = h.gather(1, last[..., None].expand(-1, -1, H))
    return torch.cat([mean, lastv], -1)

class RRHead(nn.Module):
    def __init__(self, H, C, d=256, rnn=192, use_rnn=True, dropout=0.2, n_dom=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * H, d), nn.GELU())
        self.dom = nn.Embedding(n_dom, d); nn.init.zeros_(self.dom.weight)
        self.rnn = nn.LSTM(d, rnn, batch_first=True, bidirectional=True) if use_rnn else None
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * rnn if use_rnn else d, C))

    def forward(self, rep, smask, dom):
        z = self.proj(rep) + self.dom(dom)[:, None]
        if self.rnn is not None:
            lens = smask.sum(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(z, lens, batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            z, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=smask.size(1))
        return self.cls(z)

class RRModel(nn.Module):
    def __init__(self, bb, head):
        super().__init__(); self.bb, self.head = bb, head
    def forward(self, b):
        h = self.bb(b["ids"])
        rep = pool_sentences(h, b["seg"], b["last"])
        with no_amp():                                   # small head in fp32 (stable LSTM, exact softmax)
            logits = self.head(rep, b["smask"], b["dom"])
        return logits, rep





LOWRANK_TARGETS = ("attn.q", "attn.kv", "attn.out", "up", "down")

def set_module(root, dotted, new):
    parts = dotted.split("."); parent = root
    for p in parts[:-1]: parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    if parts[-1].isdigit(): parent[int(parts[-1])] = new
    else: setattr(parent, parts[-1], new)

def block_linears(lm, targets):
    return [(f"blocks.{bi}.{n}", m) for bi, blk in enumerate(lm.blocks)
            for n, m in blk.named_modules() if isinstance(m, nn.Linear) and n in targets]

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout, rs=False):
        super().__init__(); self.base = base
        self.A = nn.Parameter(torch.empty(r, base.in_features)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / math.sqrt(r) if rs else alpha / r
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.scale

class LoRMALinear(nn.Module):
    """LoRMA₊ (multiplicative): W' = (I + s·BA) W. Computed as h = W x ; y = h + s·B(A h). Linear layers here have no bias."""
    def __init__(self, base, r, alpha, dropout):
        super().__init__(); self.base = base; o = base.out_features
        self.A = nn.Parameter(torch.empty(r, o)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(o, r))
        self.scale = alpha / r; self.drop = nn.Dropout(dropout)
    def forward(self, x):
        h = self.base(x)
        return h + (self.drop(h) @ self.A.t().to(h.dtype)) @ self.B.t().to(h.dtype) * self.scale

NF4_TABLE = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                          -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                          0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                          0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])

def _autocast_on():
    try: return torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu")
    except TypeError: return torch.is_autocast_enabled()

class _NF4MatMul(torch.autograd.Function):
    """y = x Wᵀ with W de-quantised from NF4 in forward and again in backward (never stored in high precision)."""
    @staticmethod
    def forward(ctx, x, mod):
        cdt = AMP_DTYPE if (_autocast_on() and x.is_cuda) else x.dtype
        ctx.mod, ctx.cdt, ctx.xdt = mod, cdt, x.dtype
        return x.to(cdt) @ mod.dequant(cdt).t()
    @staticmethod
    def backward(ctx, g):
        return (g.to(ctx.cdt) @ ctx.mod.dequant(ctx.cdt)).to(ctx.xdt), None

class NF4Linear(nn.Module):
    def __init__(self, base, block=64):
        super().__init__()
        W = base.weight.detach().float(); self.out_features, self.in_features = W.shape
        self.numel, self.block = W.numel(), block
        flat = F.pad(W.flatten(), (0, (-W.numel()) % block)).view(-1, block)
        absmax = flat.abs().amax(1).clamp_min(1e-12)
        mids = (NF4_TABLE[1:] + NF4_TABLE[:-1]) / 2
        codes = torch.bucketize((flat / absmax[:, None]).contiguous(), mids).to(torch.uint8).flatten()
        self.register_buffer("packed", (codes[0::2] << 4) | codes[1::2])       # two 4-bit codes per byte
        self.register_buffer("absmax", absmax)
        self.register_buffer("table", NF4_TABLE.clone())
    def dequant(self, dtype):
        codes = torch.stack([self.packed >> 4, self.packed & 15], -1).flatten().long()
        w = (self.table[codes].view(-1, self.block) * self.absmax[:, None]).flatten()[:self.numel]
        return w.view(self.out_features, self.in_features).to(dtype)
    def forward(self, x): return _NF4MatMul.apply(x, self)

class IA3Out(nn.Module):
    def __init__(self, base):
        super().__init__(); self.base = base; self.l = nn.Parameter(torch.ones(base.out_features))
    def forward(self, x): return self.base(x) * self.l.to(x.dtype)

class IA3In(nn.Module):
    def __init__(self, base):
        super().__init__(); self.base = base; self.l = nn.Parameter(torch.ones(base.in_features))
    def forward(self, x): return self.base(x * self.l.to(x.dtype))

class PrefixEncoder(nn.Module):
    def __init__(self, L, kv, dh, P, d=256):
        super().__init__(); self.L, self.kv, self.dh, self.P = L, kv, dh, P
        self.emb = nn.Parameter(torch.randn(P, d))
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, L * 2 * kv * dh))
        nn.init.normal_(self.mlp[-1].weight, std=0.02); nn.init.zeros_(self.mlp[-1].bias)
    def forward(self):                                   # -> [L, 2, kv, P, dh]
        return self.mlp(self.emb).view(self.P, self.L, 2, self.kv, self.dh).permute(1, 2, 3, 0, 4)

class PrefixAttention(nn.Module):
    """LEXWeave token attention plus P learned key/value slots that every position may attend to."""
    def __init__(self, attn):
        super().__init__(); self.a = attn; self.cur = None
    def forward(self, x):
        a = self.a; b, t, d = x.shape
        q = a.q(x).view(b, t, a.h, a.dh).transpose(1, 2)
        k, v = a.kv(x).view(b, t, 2, a.k, a.dh).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        pos = torch.arange(t, device=x.device)
        q, k = a.rope(q, pos), a.rope(k, pos)
        pk, pv = self.cur[0], self.cur[1]; P = pk.size(1)
        k = torch.cat([pk.to(k.dtype).unsqueeze(0).expand(b, -1, -1, -1), k], 2)
        v = torch.cat([pv.to(v.dtype).unsqueeze(0).expand(b, -1, -1, -1), v], 2)
        if a.h != a.k:
            k = k.repeat_interleave(a.h // a.k, dim=1); v = v.repeat_interleave(a.h // a.k, dim=1)
        mask = torch.cat([torch.ones(t, P, dtype=torch.bool, device=x.device),
                          torch.ones(t, t, dtype=torch.bool, device=x.device).tril()], 1)
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None])
        return a.out(z.transpose(1, 2).contiguous().view(b, t, d))

def apply_method(bb, cfg):
    lm, m = bb.lm, cfg.method
    if m in ("full", "kd"):
        for p in lm.parameters(): p.requires_grad = True
        return
    for p in lm.parameters(): p.requires_grad = False
    if m in ("lora", "rslora", "lorma", "qlora"):
        targets = block_linears(lm, tuple(cfg.targets))
        for name, lin in targets:
            if m == "lorma": new = LoRMALinear(lin, cfg.r, cfg.alpha, cfg.peft_dropout)
            else: new = LoRALinear(NF4Linear(lin) if m == "qlora" else lin, cfg.r, cfg.alpha, cfg.peft_dropout, rs=(m == "rslora"))
            set_module(lm, name, new)
    elif m == "ia3":
        for blk in lm.blocks:
            blk.attn.kv = IA3Out(blk.attn.kv); blk.down = IA3In(blk.down)
            if blk.memory is not None: blk.memory.kv = IA3Out(blk.memory.kv)
    elif m == "prompt":
        g = torch.Generator().manual_seed(cfg.seed)
        rows = torch.randint(4, lm.emb.num_embeddings, (cfg.n_virtual,), generator=g)
        bb.prompt = nn.Parameter(lm.emb.weight.detach()[rows].clone())
    elif m == "prefix":
        a0 = lm.blocks[0].attn
        bb.prefix = PrefixEncoder(len(lm.blocks), a0.k, a0.dh, cfg.n_virtual)
        for blk in lm.blocks: blk.attn = PrefixAttention(blk.attn)
    else:
        raise ValueError(m)





@dataclass
class ExpConfig:
    name: str
    method: str                       # full | lora | rslora | lorma | qlora | ia3 | prompt | prefix | kd
    lr: float = 1e-3                  # backbone / adapter parameters
    head_lr: float = 1e-3
    epochs: int = 12
    patience: int = 4
    batch_size: int = 4               # windows per micro-batch
    grad_accum: int = 2
    warmup: float = 0.06
    wd: float = 0.01
    dropout: float = 0.2
    label_smoothing: float = 0.05
    class_weight: str = "sqrt_inv"    # none | sqrt_inv   (macro-F1 rewards rare roles)
    context: str = "doc"              # doc (windows + BiLSTM) | none (sentence-only ablation)
    window: int = 1024
    overlap: float = 0.25
    max_sent_tokens: int = 128
    head_dim: int = 256
    rnn_dim: int = 192
    r: int = 16
    alpha: float = 32.0
    peft_dropout: float = 0.05
    targets: tuple = LOWRANK_TARGETS
    n_virtual: int = 32               # prompt / prefix length
    init: str = "phase1"              # phase1 | random  (random = transfer ablation)
    grad_ckpt: bool = False
    kd_teacher: str = ""
    kd_keep: str = "alternate"        # alternate | first_half | "0,3,6,9"
    kd_alpha: float = 0.5             # weight of the gold loss
    kd_T: float = 2.0
    kd_hidden: float = 0.1            # MSE between teacher and student sentence vectors
    seed: int = SEED
    note: str = ""

def student_keep(n, mode):
    if mode == "alternate": return list(range(0, n, 2))
    if mode == "first_half": return list(range(n // 2))
    return [int(x) for x in mode.split(",")]

def build_model(cfg):
    seed_all(cfg.seed)
    lm = copy.deepcopy(MASTER_LM) if cfg.init == "phase1" else LegalLM(LM_CFG)
    keep = None
    if cfg.method == "kd":
        keep = student_keep(len(lm.blocks), cfg.kd_keep)
        lm.blocks = nn.ModuleList([lm.blocks[i] for i in keep])
    bb = Backbone(lm); bb.grad_ckpt = cfg.grad_ckpt
    apply_method(bb, cfg)
    head = RRHead(HID, C, cfg.head_dim, cfg.rnn_dim, use_rnn=(cfg.context == "doc"), dropout=cfg.dropout)
    return RRModel(bb, head), keep

def n_params(m, trainable=False):
    n = sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable)
    if not trainable: n += sum(q.numel for q in m.modules() if isinstance(q, NF4Linear))   # 4-bit base weights are buffers
    return n
def n_bytes(m): return sum(t.numel() * t.element_size() for t in list(m.parameters()) + list(m.buffers()))

def class_weights(docs, mode):
    if mode == "none": return None
    cnt = np.bincount([L2ID[l] for d in docs for l in d["labels"]], minlength=C).astype(float)
    w = np.where(cnt > 0, np.sqrt(cnt.sum() / np.maximum(cnt, 1)), 1.0)
    w = w / w[cnt > 0].mean()
    return torch.tensor(w, dtype=torch.float, device=DEVICE)

def macro_scores(gold, pred):
    """gold/pred: flat lists of label ids. macro_F1 averages over labels present in gold ∪ pred (sklearn default).
    macro_F1_all13 averages over every official label. Check against the official script on dev."""
    gold, pred = np.asarray(gold), np.asarray(pred); per = {}
    for c in range(C):
        tp = int(((pred == c) & (gold == c)).sum()); fp = int(((pred == c) & (gold != c)).sum()); fn = int(((pred != c) & (gold == c)).sum())
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
        per[LABELS[c]] = dict(P=P, R=R, F1=2 * P * R / (P + R) if P + R else 0.0, support=int((gold == c).sum()))
    present = [LABELS[c] for c in sorted(set(gold.tolist()) | set(pred.tolist()))]
    mean = lambda k, ls: float(np.mean([per[l][k] for l in ls])) if ls else 0.0
    return dict(macro_F1=mean("F1", present), macro_P=mean("P", present), macro_R=mean("R", present),
                macro_F1_all13=mean("F1", LABELS), accuracy=float((gold == pred).mean()) if len(gold) else 0.0,
                per_label=per)

@torch.no_grad()
def predict(model, docs, cfg, reserve=0, bs=8):
    model.eval()
    exs = make_examples(docs, cfg, with_labels=False, reserve=reserve)
    order = sorted(range(len(exs)), key=lambda i: len(exs[i]["ids"]))
    acc = [np.zeros((len(d["sents"]), C), np.float32) for d in docs]; cnt = [np.zeros(len(d["sents"]), np.float32) for d in docs]
    bs = bs * 8 if cfg.context == "none" else bs
    for k in range(0, len(order), bs):
        chunk = [exs[i] for i in order[k:k + bs]]
        with amp(): logits, _ = model(to_dev(collate(chunk)))
        lp = F.log_softmax(logits.float(), -1).cpu().numpy()
        for x, l in zip(chunk, lp):
            n = x["hi"] - x["lo"]; acc[x["doc"]][x["lo"]:x["hi"]] += l[:n]; cnt[x["doc"]][x["lo"]:x["hi"]] += 1
    assert all((c > 0).all() for c in cnt), "a sentence got no prediction"
    return [(a / c[:, None]).argmax(-1).tolist() for a, c in zip(acc, cnt)]

def evaluate(model, docs, cfg, reserve=0):
    preds = predict(model, docs, cfg, reserve)
    out = {"all": macro_scores([L2ID[l] for d in docs for l in d["labels"]], [p for ps in preds for p in ps])}
    for dom in DOMAINS:
        idx = [i for i, d in enumerate(docs) if d["domain"] == dom]
        if idx: out[dom] = macro_scores([L2ID[l] for i in idx for l in docs[i]["labels"]], [p for i in idx for p in preds[i]])
    return out, preds

def train_model(model, cfg, trn_docs, dev_docs, teacher=None, reserve=0):
    seed_all(cfg.seed)
    exs = make_examples(trn_docs, cfg, with_labels=True, reserve=reserve)
    head_ids = {id(p) for p in model.head.parameters()}
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [dict(params=[p for p in body if p.ndim >= 2], lr=cfg.lr, weight_decay=cfg.wd),
              dict(params=[p for p in body if p.ndim < 2], lr=cfg.lr, weight_decay=0.0),
              dict(params=list(model.head.parameters()), lr=cfg.head_lr, weight_decay=0.01)]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.98))
    for g in opt.param_groups: g["base_lr"] = g["lr"]
    bs = cfg.batch_size
    steps_per_ep = math.ceil(math.ceil(len(exs) / bs) / cfg.grad_accum)
    total = steps_per_ep * cfg.epochs; warm = max(1, int(cfg.warmup * total))
    lr_at = lambda s: (s + 1) / warm if s < warm else max(0.02, (total - s) / max(1, total - warm))
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda" and AMP_DTYPE == torch.float16))
    cw = class_weights(trn_docs, cfg.class_weight)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"  train windows {len(exs)} | optimizer steps {total} | trainable {n_params(model, True)/1e6:.3f}M "
        f"of {n_params(model)/1e6:.1f}M ({100*n_params(model, True)/n_params(model):.2f}%)")
    best, best_state, best_ep, bad, hist, step = -1.0, None, 0, 0, [], 0
    if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train(); order = list(range(len(exs))); random.shuffle(order); tot = nb = 0
        nbatches = math.ceil(len(order) / bs)
        for bi, k in enumerate(range(0, len(order), bs)):
            b = to_dev(collate([exs[i] for i in order[k:k + bs]]))
            with amp(): logits, rep = model(b)
            msk = b["y"] != -100; y = b["y"][msk]
            loss = F.cross_entropy(logits[msk], y, weight=cw, label_smoothing=cfg.label_smoothing)
            if teacher is not None:
                with torch.no_grad(), amp(): t_logits, t_rep = teacher(b)
                T = cfg.kd_T
                kl = F.kl_div(F.log_softmax(logits[msk] / T, -1), F.softmax(t_logits[msk].float() / T, -1),
                              reduction="batchmean") * T * T
                loss = cfg.kd_alpha * loss + (1 - cfg.kd_alpha) * kl
                if cfg.kd_hidden > 0: loss = loss + cfg.kd_hidden * F.mse_loss(rep[msk], t_rep[msk].float())
            if not torch.isfinite(loss): raise FloatingPointError("non-finite loss")
            scaler.scale(loss / cfg.grad_accum).backward(); tot += loss.item(); nb += 1
            if (bi + 1) % cfg.grad_accum == 0 or bi + 1 == nbatches:
                for g in opt.param_groups: g["lr"] = g["base_lr"] * lr_at(step)
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
        sc, _ = evaluate(model, dev_docs, cfg, reserve)
        row = dict(epoch=ep + 1, loss=tot / max(1, nb), dev_macro_F1=sc["all"]["macro_F1"],
                   **{f"dev_{d}_F1": sc[d]["macro_F1"] for d in DOMAINS if d in sc}, minutes=(time.time() - t0) / 60)
        hist.append(row)
        log(f"  ep {ep+1:2d} | loss {row['loss']:.4f} | dev macro-F1 {row['dev_macro_F1']:.4f} "
            + " ".join(f"{d} {row.get(f'dev_{d}_F1', 0):.4f}" for d in DOMAINS) + f" | {row['minutes']:.1f} min")
        if row["dev_macro_F1"] > best:
            best, best_ep, bad = row["dev_macro_F1"], ep + 1, 0
            best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        else:
            bad += 1
            if bad >= cfg.patience: log("  early stop"); break
    model.load_state_dict(best_state, strict=False)
    stats = dict(train_minutes=(time.time() - t0) / 60, best_epoch=best_ep,
                 peak_mem_GB=torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0)
    return hist, stats, best_state





RESULTS = OUT / "results.jsonl"

def cfg_from_dict(d):
    names = {f.name for f in fields(ExpConfig)}
    d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
    return ExpConfig(**d)

def load_experiment(name, out_dir=OUT):
    ck = torch.load(out_dir / name / "model.pt", map_location="cpu", weights_only=False)
    cfg = cfg_from_dict(ck["cfg"]); model, _ = build_model(cfg)
    missing = {n for n, p in model.named_parameters() if p.requires_grad} - set(ck["state"])
    assert not missing, f"checkpoint lacks trainable tensors: {sorted(missing)[:5]}"
    _, unexpected = model.load_state_dict(ck["state"], strict=False); assert not unexpected, unexpected[:5]
    return model.to(DEVICE).eval(), cfg

def write_submission(test_preds, docs, dest):
    out = {}
    for d, p in zip(docs, test_preds):
        assert len(p) == len(d["sents"]), (d["id"], len(p), len(d["sents"]))
        out[d["id"]] = [LABELS[i] for i in p]
    dest.mkdir(parents=True, exist_ok=True)
    pj = dest / "prediction.json"; json.dump(out, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    zp = dest / "submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z: z.write(pj, arcname="prediction.json")
    check_submission(zp, docs)
    return zp

def check_submission(zp, docs):
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
        pred = json.loads(z.read("prediction.json"))
    assert set(pred) == {d["id"] for d in docs}, "missing or extra document ids"
    for d in docs:
        assert len(pred[d["id"]]) == len(d["sents"]), d["id"]
        assert all(l in L2ID for l in pred[d["id"]]), d["id"]
    return True

def run_experiment(cfg, trn=None, dev=None, test=None, out_dir=OUT, submit=True, record=True):
    trn, dev, test = trn or TRAIN_DOCS, dev or DEV_DOCS, test or TEST_DOCS
    d = out_dir / cfg.name
    if (d / "done.json").exists():
        log(f"[skip] {cfg.name} already finished"); return json.load(open(d / "done.json"))
    log(f"\n=== {cfg.name} | method={cfg.method} context={cfg.context} init={cfg.init} ===")
    teacher = None
    if cfg.method == "kd":
        teacher, tcfg = load_experiment(cfg.kd_teacher, out_dir)
        for p in teacher.parameters(): p.requires_grad = False
        cfg.window, cfg.overlap, cfg.max_sent_tokens, cfg.context = tcfg.window, tcfg.overlap, tcfg.max_sent_tokens, tcfg.context
    model, keep = build_model(cfg); model = model.to(DEVICE)
    reserve = max(model.bb.n_virtual, teacher.bb.n_virtual if teacher is not None else 0)
    if teacher is not None:
        log(f"  KD: teacher {cfg.kd_teacher} {n_params(teacher)/1e6:.1f}M -> student {n_params(model)/1e6:.1f}M (blocks {keep})")
    hist, stats, state = train_model(model, cfg, trn, dev, teacher, reserve)

    t0 = time.time(); dev_sc, dev_pred = evaluate(model, dev, cfg, reserve); dev_s = time.time() - t0
    gold = [L2ID[l] for x in dev for l in x["labels"]]; flat = [p for ps in dev_pred for p in ps]
    conf = np.zeros((C, C), int)
    for g, p in zip(gold, flat): conf[g, p] += 1
    d.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cfg=asdict(cfg), state=state, keep_layers=keep), d / "model.pt")
    json.dump(dict(dev_ids=[x["id"] for x in dev], pred=[[LABELS[i] for i in ps] for ps in dev_pred]),
              open(d / "dev_predictions.json", "w"))
    if submit:
        t0 = time.time(); test_pred = predict(model, test, cfg, reserve); stats["test_infer_s"] = time.time() - t0
        zp = write_submission(test_pred, test, d); log(f"  submission -> {zp}")
    row = dict(name=cfg.name, method=cfg.method, context=cfg.context, init=cfg.init,
               total_params_M=n_params(model) / 1e6, trainable_params_M=n_params(model, True) / 1e6,
               trainable_pct=100 * n_params(model, True) / n_params(model), backbone_MB=n_bytes(model.bb) / 2**20,
               dev_macro_F1=dev_sc["all"]["macro_F1"], dev_macro_P=dev_sc["all"]["macro_P"], dev_macro_R=dev_sc["all"]["macro_R"],
               dev_macro_F1_all13=dev_sc["all"]["macro_F1_all13"], dev_acc=dev_sc["all"]["accuracy"],
               **{f"dev_{dm}_macro_F1": dev_sc[dm]["macro_F1"] for dm in DOMAINS if dm in dev_sc},
               dev_infer_s_per_doc=dev_s / max(1, len(dev)), **stats,
               per_label_F1={l: round(v["F1"], 4) for l, v in dev_sc["all"]["per_label"].items()},
               confusion=conf.tolist(), history=hist, cfg=asdict(cfg))
    if record:
        with open(RESULTS, "a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")
    json.dump(row, open(d / "done.json", "w"))
    log(f"  DEV macro-F1 {row['dev_macro_F1']:.4f} (P {row['dev_macro_P']:.4f} R {row['dev_macro_R']:.4f}) | "
        + " ".join(f"{dm} {row.get(f'dev_{dm}_macro_F1', 0):.4f}" for dm in DOMAINS)
        + f" | trainable {row['trainable_params_M']:.3f}M ({row['trainable_pct']:.2f}%) | peak {stats['peak_mem_GB']:.1f} GB"
        + f" | {stats['train_minutes']:.1f} min")
    del model, teacher; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return row





def smoke_test(methods=("full", "lorma", "lora", "rslora", "ia3", "prompt", "prefix", "qlora", "kd"), keep=False):
    sd = OUT / "_smoke"; shutil.rmtree(sd, ignore_errors=True)
    trn = TRAIN_DOCS[:2] + [d for d in TRAIN_DOCS if d["domain"] == "IT"][:2]
    dev = DEV_DOCS[:1] + [d for d in DEV_DOCS if d["domain"] == "IT"][:1]
    test = TEST_DOCS[:2]
    t0 = time.time()
    for m in methods:
        cfg = ExpConfig(f"smoke_{m}", m, epochs=1, window=256, n_virtual=8, kd_teacher="smoke_full")
        row = run_experiment(cfg, trn, dev, test, out_dir=sd, submit=True, record=False)
        if m != "kd":
            model, _ = load_experiment(cfg.name, sd)           # reload round-trip must reproduce the dev score
            sc, _ = evaluate(model, dev, cfg, model.bb.n_virtual)
            assert abs(sc["all"]["macro_F1"] - row["dev_macro_F1"]) < 1e-6, (m, sc["all"]["macro_F1"], row["dev_macro_F1"])
            del model
    cfg = ExpConfig("smoke_nocontext", "lorma", epochs=1, context="none", batch_size=32, grad_accum=1)
    run_experiment(cfg, trn, dev, test, out_dir=sd, submit=True, record=False)
    if not keep: shutil.rmtree(sd, ignore_errors=True)
    log(f"SMOKE TEST PASSED in {(time.time()-t0)/60:.1f} min")

SMOKE = False
if SMOKE: smoke_test()





EXPERIMENTS = [
    ExpConfig("E0_full",    "full",   lr=5e-5, epochs=10),
    ExpConfig("E1_lorma",   "lorma",  lr=1e-3, r=16, alpha=32),
    ExpConfig("E2_lora",    "lora",   lr=1e-3, r=16, alpha=32),
    ExpConfig("E3_rslora",  "rslora", lr=1e-3, r=64, alpha=16),
    ExpConfig("E4_ia3",     "ia3",    lr=5e-3),
    ExpConfig("E5_prompt",  "prompt", lr=1e-2, n_virtual=32, wd=0.0),
    ExpConfig("E6_kd",      "kd",     lr=8e-5, epochs=12, kd_teacher="E0_full", kd_keep="alternate",
              kd_alpha=0.5, kd_T=2.0, kd_hidden=0.1),
    # ---- optional extra submissions / analysis ablations ----
    ExpConfig("E7_prefix",  "prefix", lr=5e-3, n_virtual=16, wd=0.0),
    ExpConfig("E8_qlora",   "qlora",  lr=1e-3, r=16, alpha=32),
    ExpConfig("E9_lorma_sentonly", "lorma", lr=1e-3, r=16, alpha=32, context="none", batch_size=32, grad_accum=1,
              note="ablation: no document context"),
]
# Not for submission, but answers "are Phase-1 representations useful?" (same recipe as E0, random backbone):
TRANSFER_ABLATION = ExpConfig("A1_full_randominit", "full", lr=3e-4, epochs=10, init="random")


RUN_QUEUE = [c.name for c in EXPERIMENTS]          # trim this list if you are short on time
RUN_TRANSFER_ABLATION = False

for cfg in EXPERIMENTS:
    if cfg.name not in RUN_QUEUE: continue
    hours = (time.time() - SESSION_START) / 3600
    if hours > TIME_BUDGET_H:
        log(f"time budget reached ({hours:.2f} h); not starting {cfg.name}"); break
    oom = False
    try:
        run_experiment(copy.deepcopy(cfg))
    except torch.cuda.OutOfMemoryError:
        oom = True                                   # retry outside the except block so the failed graph is freed
    if oom:
        log(f"OOM in {cfg.name}: retrying with gradient checkpointing and half micro-batch")
        gc.collect(); torch.cuda.empty_cache()
        c2 = copy.deepcopy(cfg); c2.grad_ckpt = True; c2.batch_size = max(1, c2.batch_size // 2); c2.grad_accum *= 2
        run_experiment(c2)
if RUN_TRANSFER_ABLATION: run_experiment(copy.deepcopy(TRANSFER_ABLATION), submit=False)





rows = [json.loads(l) for l in open(RESULTS, encoding="utf-8")] if RESULTS.exists() else []
df = pd.DataFrame(rows).drop_duplicates("name", keep="last").set_index("name")
cols = ["method", "context", "dev_macro_F1", "dev_macro_P", "dev_macro_R", "dev_macro_F1_all13", "dev_acc",
        "dev_CL_macro_F1", "dev_IT_macro_F1", "total_params_M", "trainable_params_M", "trainable_pct", "backbone_MB",
        "peak_mem_GB", "train_minutes", "best_epoch", "dev_infer_s_per_doc"]
print("Main comparison (sorted by dev macro-F1)")
display(df[[c for c in cols if c in df]].sort_values("dev_macro_F1", ascending=False).round(4))

print("Per-label dev F1: which rhetorical roles stay hard?")
display(pd.DataFrame({n: r["per_label_F1"] for n, r in df.iterrows()}).round(3))


# Confusions of the best model: which closely related roles get mixed up?
best_name = df["dev_macro_F1"].idxmax()
conf = np.array(df.loc[best_name, "confusion"])
pairs = sorted(((conf[i, j], LABELS[i], LABELS[j]) for i in range(C) for j in range(C) if i != j and conf[i, j] > 0), reverse=True)
print(f"best = {best_name}. Top confusions (gold -> predicted):")
for n, g, p in pairs[:12]: print(f"  {g:>24} -> {p:<24} {n}")
display(pd.DataFrame(conf, index=LABELS, columns=LABELS))


# KD: how much accuracy does the student keep for its size and speed?
if {"E0_full", "E6_kd"} <= set(df.index):
    t, s = df.loc["E0_full"], df.loc["E6_kd"]
    print(f"KD student keeps {100*s.dev_macro_F1/t.dev_macro_F1:.1f}% of teacher macro-F1 with "
          f"{100*s.total_params_M/t.total_params_M:.0f}% of parameters, "
          f"{100*s.dev_infer_s_per_doc/t.dev_infer_s_per_doc:.0f}% of inference time")
# Context: E1 (windows + BiLSTM) vs E9 (sentence-only), same LoRMA adaptation
if {"E1_lorma", "E9_lorma_sentonly"} <= set(df.index):
    print(f"document context: {df.loc['E1_lorma','dev_macro_F1']:.4f} vs sentence-only {df.loc['E9_lorma_sentonly','dev_macro_F1']:.4f}")
if {"E0_full", "A1_full_randominit"} <= set(df.index):
    print(f"Phase-1 init {df.loc['E0_full','dev_macro_F1']:.4f} vs random init {df.loc['A1_full_randominit','dev_macro_F1']:.4f}")


# Learning curves (dev macro-F1 per epoch) for the report
display(pd.DataFrame({n: pd.Series([h["dev_macro_F1"] for h in r["history"]], index=range(1, len(r["history"]) + 1))
                      for n, r in df.iterrows()}).round(4))
print("Submissions:")
for n in df.index:
    zp = OUT / n / "submission.zip"
    if zp.exists(): print(" ", zp)


from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
from IPython.display import FileLink, display
import os
import shutil

phase = "phase3"
work = Path("/kaggle/working")
archive = work / f"{phase}_kaggle_output.zip"

# Exclude the archive itself, including a copy from an earlier run.
files = [p for p in work.rglob("*") if p.is_file() and p != archive]
size = sum(p.stat().st_size for p in files)
free = shutil.disk_usage(work).free

print(f"{len(files):,} files · {size / 2**30:.2f} GiB to archive")
if free < size + 64 * 2**20:
    raise RuntimeError(
        f"Only {free / 2**30:.2f} GiB free. "
        "Use Kaggle's Save Version to preserve the outputs without making a ZIP."
    )

# ZIP_STORED avoids spending time recompressing checkpoints and submission ZIPs.
with ZipFile(archive, "w", compression=ZIP_STORED, allowZip64=True) as z:
    for p in files:
        z.write(p, p.relative_to(work))

os.chdir(work)
print(f"Ready: {archive.name} ({archive.stat().st_size / 2**30:.2f} GiB)")
display(FileLink(archive.name))



