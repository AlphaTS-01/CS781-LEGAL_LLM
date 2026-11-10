





get_ipython().run_line_magic("pip", " install -q \"tokenizers>=0.20,<1\" \"huggingface_hub>=0.26,<2\" pandas")


import os, re, json, math, time, copy, random, zipfile, tarfile, gc, shutil, getpass, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, fields
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ============================== EDIT THESE ==============================
WORK = Path("/kaggle/working/cs781_phase4")
FOUNDATION = "/kaggle/input/models/tejass73/lexyp1/pytorch/default/1"
DATA_DIR = WORK / "data"
HF_REPO = "Exploration-Lab/CS-781-Capstone"
TIME_BUDGET_H = 8.0                   # the queue will not START a new experiment after this many hours
TRAIN_DOCS_PER_EPOCH = 8000           # cases sampled per epoch (0 = all). Main speed knob.
EPOCHS = 2
SEED = 42
# =======================================================================
OUT = WORK / "runs"; CACHE = WORK / "cache"
for p in (WORK, DATA_DIR, OUT, CACHE): p.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    print(torch.cuda.get_device_name(), f"| {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
print("device:", DEVICE, "| amp:", AMP_DTYPE if DEVICE == "cuda" else "off", "| torch", torch.__version__)

SESSION_START = time.time()
LOG = WORK / "phase4_log.txt"

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





PHASE1_MODELS = {
    "LEXWeave": FOUNDATION,
    # "LEXWeave_plain":    str(WORK / "lexy-plain.tar.gz"),       # e.g. Phase-1 ablation='plain' (attention only)
    # "LEXWeave_nomemory": str(WORK / "lexy-nomemory.tar.gz"),
    # "LEXWeave_noconv":   str(WORK / "lexy-noconv.tar.gz"),
}

def prepare_foundation(src, dest):
    """Extract model.pt + tokenizer.json from a Phase-1 package (tar.gz / zip / directory / s3:// URI)."""
    need = ("model.pt", "tokenizer.json")
    if all((dest / n).exists() for n in need): return dest
    dest.mkdir(parents=True, exist_ok=True); src = str(src)
    if src.startswith("s3://"):
        import boto3
        bucket, key = src[5:].split("/", 1); local = WORK / Path(key).name
        if not local.exists(): boto3.client("s3").download_file(bucket, key, str(local))
        src = str(local)
    p = Path(src)
    assert p.exists(), f"Foundation package not found: {p}. Upload it to {WORK}."
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

from tokenizers import Tokenizer
P1 = {}                                      # key -> cfg, CPU state dict, tokenizer, special ids
for key, src in PHASE1_MODELS.items():
    fd = prepare_foundation(src, WORK / "foundation" / key)
    ck = torch.load(fd / "model.pt", map_location="cpu", weights_only=False)
    cfg_ = dict(ck["model_config"]); lm_ = LegalLM(cfg_)
    lm_.load_state_dict(ck["model"], strict=True)            # our Phase-1 weights, nothing else
    tok_ = Tokenizer.from_file(str(fd / "tokenizer.json"))
    assert tok_.get_vocab_size() == cfg_["vocab"], f"{key}: tokenizer / model vocab mismatch"
    P1[key] = dict(cfg=cfg_, state={k: v.clone() for k, v in lm_.state_dict().items()}, tok=tok_,
                   tok_key=hashlib.sha1((fd / "tokenizer.json").read_bytes()).hexdigest()[:12],
                   pad=tok_.token_to_id("<pad>"), unk=tok_.token_to_id("<unk>"),
                   hid=int(cfg_["dim"]), context=int(cfg_["context"]),
                   params=sum(p.numel() for p in lm_.parameters()))
    print(f"{key}: {P1[key]['params']:,} params |",
          {k: cfg_[k] for k in ("arch", "vocab", "context", "dim", "heads", "kv_heads", "layers", "ff_dim", "ablation") if k in cfg_})
    del ck, lm_; gc.collect()
MODEL_KEYS = list(P1)
if len(MODEL_KEYS) < 4:
    print(f"⚠ only {len(MODEL_KEYS)} Phase-1 model(s) registered. Doc §10.1 asks for experiments with all four "
          f"Phase-1 models/architectures: add the others to PHASE1_MODELS.")

def load_lm(key, init="phase1"):
    lm = LegalLM(P1[key]["cfg"])
    if init == "phase1": lm.load_state_dict(P1[key]["state"], strict=True)
    return lm





def ensure_data():
    hits = [p for p in DATA_DIR.rglob("train.json") if "Phase_4_CJPE" in str(p)]
    if not hits:
        from huggingface_hub import snapshot_download
        token = os.environ.get("HF_TOKEN") or getpass.getpass("Hugging Face read token: ")
        snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=["Phase_4_CJPE/*"],
                          local_dir=str(DATA_DIR), token=token)
        hits = [p for p in DATA_DIR.rglob("train.json") if "Phase_4_CJPE" in str(p)]
    assert hits, f"Phase_4_CJPE/Train/train.json not found under {DATA_DIR}"
    return hits[0].parent.parent              # .../Phase_4_CJPE

CJ_ROOT = ensure_data()
def find_file(name):
    hits = sorted(CJ_ROOT.rglob(name)); assert hits, f"{name} not found under {CJ_ROOT}"; return hits[0]

def _read_json(path):
    try: return json.load(open(path, encoding="utf-8"))
    except json.JSONDecodeError:               # JSON-lines fallback
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

TEXT_KEYS = ("text", "document", "case_text", "judgment", "content", "facts")
LABEL_KEYS = ("label", "labels", "y", "decision", "outcome", "pred")

def to_label(v):
    if isinstance(v, list) and len(v) == 1: v = v[0]
    if isinstance(v, bool): return int(v)
    if isinstance(v, (int, float)) and int(v) in (0, 1): return int(v)
    if isinstance(v, str) and v.strip() in ("0", "1"): return int(v.strip())
    raise ValueError(f"unexpected label value {v!r}")

def load_cases(fname, with_labels):
    raw = _read_json(find_file(fname))
    if isinstance(raw, dict):
        if isinstance(raw.get("data"), list): items = raw["data"]
        else: items = [dict(v, id=v.get("id", k)) if isinstance(v, dict) else {"id": k, "text": v} for k, v in raw.items()]
    else:
        items = raw
    cases = []
    for it in items:
        rid = next((it[k] for k in ("id", "case_id", "doc_id", "name", "uid") if k in it), None)
        text = next((it[k] for k in TEXT_KEYS if k in it), None)
        assert rid is not None and text is not None, f"unexpected record layout in {fname}: {list(it)[:8]}"
        if isinstance(text, list): text = " ".join(map(str, text))
        c = dict(id=str(rid), rid=rid, text=" ".join(str(text).split()))
        if with_labels:
            lk = next((k for k in LABEL_KEYS if k in it), None)
            assert lk is not None, f"{fname}:{rid} has no label field ({list(it)[:8]})"
            c["y"] = to_label(it[lk])
        cases.append(c)
    return cases

TRAIN = load_cases("train.json", True)
DEV = load_cases("dev.json", True)
TEST = load_cases("test.json", False)
assert len({c["id"] for c in TEST}) == len(TEST), "duplicate ids in test.json"

# Leakage / duplicate check on exact (whitespace-normalised) text
_h = lambda c: hashlib.md5(c["text"].encode("utf-8")).hexdigest()
_dev_h, _test_h = {_h(c) for c in DEV}, {_h(c) for c in TEST}
_tr_h = [_h(c) for c in TRAIN]
_lab_by_h = defaultdict(set)
for c, h in zip(TRAIN, _tr_h): _lab_by_h[h].add(c["y"])
print(f"train texts also in dev: {sum(h in _dev_h for h in _tr_h)} | also in test: {sum(h in _test_h for h in _tr_h)} | "
      f"duplicate texts inside train: {len(_tr_h) - len(set(_tr_h))} (with conflicting labels: {sum(len(v) > 1 for v in _lab_by_h.values())})")
TRAIN = [c for c, h in zip(TRAIN, _tr_h) if h not in _dev_h]     # keep dev an honest held-out set
for name, ds in (("train", TRAIN), ("dev", DEV), ("test", TEST)):
    print(f"{name:<5}: {len(ds):6d} cases" + (f" | label counts {dict(sorted(Counter(c['y'] for c in ds).items()))}" if "y" in ds[0] else ""))





import pandas as pd
try: display
except NameError: display = print

def encode_split(key, split, cases):
    """Token arrays for `cases` with the tokenizer of Phase-1 model `key` (disk-cached per tokenizer)."""
    tk = P1[key]["tok_key"]; f = CACHE / f"tok_{tk}_{split}.npz"
    ids_sig = hashlib.md5("\n".join(c["id"] for c in cases).encode()).hexdigest()
    if f.exists():
        z = np.load(f, allow_pickle=False)
        if str(z["sig"]) == ids_sig:
            flat, offs = z["flat"], z["offs"]
            return [flat[offs[i]:offs[i + 1]] for i in range(len(cases))]
    tok, unk = P1[key]["tok"], P1[key]["unk"]
    dt = np.uint16 if P1[key]["cfg"]["vocab"] <= 65535 else np.int32
    arrs = []
    for a in range(0, len(cases), 256):                 # small batches: Encoding objects are memory-heavy
        enc = tok.encode_batch([" " + c["text"] for c in cases[a:a + 256]], add_special_tokens=False)
        arrs += [np.asarray(e.ids or [unk], dtype=dt) for e in enc]
    offs = np.concatenate([[0], np.cumsum([len(x) for x in arrs])]).astype(np.int64)
    np.savez(f, flat=np.concatenate(arrs), offs=offs, sig=np.array(ids_sig))
    return arrs

TOKS = {}                                               # tok_key -> {split: [np arrays]}
for key in MODEL_KEYS:
    tk = P1[key]["tok_key"]
    if tk in TOKS: continue
    t0 = time.time()
    TOKS[tk] = {s: encode_split(key, s, cs) for s, cs in (("train", TRAIN), ("dev", DEV), ("test", TEST))}
    print(f"tokenised with {key}'s tokenizer ({tk}) in {time.time() - t0:.0f}s")
def toks_for(key): return TOKS[P1[key]["tok_key"]]

_T0 = toks_for(MODEL_KEYS[0])
_len = {s: np.array([len(x) for x in _T0[s]]) for s in _T0}
rows = []
for s in ("train", "dev", "test"):
    l = _len[s]
    rows.append(dict(split=s, cases=len(l), p10=int(np.percentile(l, 10)), p50=int(np.median(l)), p90=int(np.percentile(l, 90)),
                     max=int(l.max()), **{f">{k}": f"{100*(l > k).mean():.1f}%" for k in (512, 1024, 2048, 4096, 8192)}))
print("Tokens per case:"); display(pd.DataFrame(rows).set_index("split"))
_ytr = np.array([c["y"] for c in TRAIN]); _ydv = np.array([c["y"] for c in DEV])
print(f"train P(y=1) = {_ytr.mean():.3f} | dev P(y=1) = {_ydv.mean():.3f}")
print("median tokens by label (train):", {int(y): int(np.median(_len['train'][_ytr == y])) for y in (0, 1)})
print("P(y=1) by train length quartile:",
      [round(float(_ytr[(_len['train'] >= a) & (_len['train'] <= b)].mean()), 3)
       for a, b in zip(np.percentile(_len['train'], [0, 25, 50, 75]), np.percentile(_len['train'], [25, 50, 75, 100]))])
print("a 4×512 tail window covers on average "
      f"{100*np.mean(np.minimum(1, (512 + 3*448) / _len['dev'])):.1f}% of a dev case's tokens")





@torch.no_grad()
def lm_loss_on(key, n_docs=48):
    m = load_lm(key).to(DEVICE).eval(); ctx = P1[key]["context"]; tot = n = 0
    for arr in toks_for(key)["dev"][:n_docs]:
        x = torch.tensor(arr[-(ctx + 1):].astype(np.int64), device=DEVICE)[None]
        if x.size(1) < 2: continue
        with amp(): loss = m(x[:, :-1], targets=x[:, 1:])
        tot += float(loss) * (x.size(1) - 1); n += x.size(1) - 1
    del m; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return tot / max(1, n)
for key in MODEL_KEYS:
    _l = lm_loss_on(key)
    print(f"{key}: LM loss on CJPE dev text {_l:.3f} (ppl {math.exp(_l):.1f}) vs uniform ln|V| = {math.log(P1[key]['cfg']['vocab']):.3f}")





def chunk_spans(n, L, K, overlap, mode):
    if n <= L: return [(0, n)]
    step = max(1, L - overlap)
    if mode == "head":       starts = [min(i * step, n - L) for i in range(K)]
    elif mode == "tail":     starts = [max(n - L - i * step, 0) for i in range(K)]
    elif mode == "middle":
        c = max(0, (n - (L + (K - 1) * step)) // 2); starts = [min(c + i * step, n - L) for i in range(K)]
    elif mode == "headtail": starts = [0] + [max(n - L - i * step, 0) for i in range(K - 1)]
    elif mode == "uniform":  starts = [round(i * (n - L) / (K - 1)) for i in range(K)] if K > 1 else [n - L]
    else: raise ValueError(mode)
    return [(s, s + L) for s in sorted(set(starts))]

def rel_bucket(a, b, n): return min(9, int(10 * (a + b) / 2 / max(1, n)))

def make_examples(cases, toks, cfg, with_labels, select=None, K=None):
    select, K = select or cfg.select, K or cfg.n_chunks
    exs = []
    for i, (c, arr) in enumerate(zip(cases, toks)):
        sp = chunk_spans(len(arr), cfg.chunk_len, K, cfg.overlap, select)
        ex = dict(doc=i, chunks=[arr[a:b] for a, b in sp], rel=[rel_bucket(a, b, len(arr)) for a, b in sp])
        if with_labels: ex["y"] = c["y"]
        exs.append(ex)
    return exs

def collate(batch, pad):
    chunks = [(bi, k, ch, r) for bi, ex in enumerate(batch) for k, (ch, r) in enumerate(zip(ex["chunks"], ex["rel"]))]
    T = max(len(ch) for _, _, ch, _ in chunks)
    ids = torch.full((len(chunks), T), pad, dtype=torch.long)
    for j, (_, _, ch, _) in enumerate(chunks): ids[j, :len(ch)] = torch.from_numpy(ch.astype(np.int64))
    return dict(ids=ids, lens=torch.tensor([len(ch) for _, _, ch, _ in chunks]),
                doc_of=torch.tensor([bi for bi, _, _, _ in chunks]), kidx=torch.tensor([k for _, k, _, _ in chunks]),
                rel=torch.tensor([r for _, _, _, r in chunks]), y=torch.tensor([ex.get("y", -1) for ex in batch]))

def to_dev(b): return {k: v.to(DEVICE, non_blocking=True) for k, v in b.items()}

# --- self-test of the chunker ---------------------------------------------------------------------------
for _n in (1, 300, 512, 513, 1500, 9000):
    for _m in ("head", "tail", "middle", "headtail", "uniform"):
        _sp = chunk_spans(_n, 512, 4, 64, _m)
        assert all(0 <= a < b <= _n and b - a == min(512, _n) for a, b in _sp) and 1 <= len(_sp) <= 4, (_n, _m, _sp)
    if _n > 2048: assert chunk_spans(_n, 512, 4, 64, "tail")[-1][1] == _n and chunk_spans(_n, 512, 4, 64, "head")[0][0] == 0
print("chunker OK | example spans for a 9000-token case:", {m: chunk_spans(9000, 512, 4, 64, m) for m in ("tail", "uniform")})





class Backbone(nn.Module):
    """Runs LEXWeave and returns final-norm token states (the LM head is unused). Right padding is safe because
    every LEXWeave mixer (attention, chunk memory, convolutions) is causal."""
    def __init__(self, lm):
        super().__init__(); self.lm, self.grad_ckpt = lm, False
    def forward(self, ids):
        x = self.lm.emb(ids)
        for blk in self.lm.blocks:
            x = checkpoint(blk, x, use_reentrant=False) if (self.training and self.grad_ckpt) else blk(x)
        return self.lm.norm(x)

class TokenPool(nn.Module):
    """[N,T,H] -> [N,H*]: attn (learned weights over tokens) | mean | last (state at the last real token) | attn+last."""
    def __init__(self, H, mode):
        super().__init__(); self.mode = mode
        self.score = nn.Linear(H, 1) if "attn" in mode else None
        self.out_dim = 2 * H if mode == "attn+last" else H
    def forward(self, h, lens):
        N, T, H = h.shape; m = torch.arange(T, device=h.device)[None] < lens[:, None]
        last = h[torch.arange(N, device=h.device), lens - 1]
        if self.mode == "last": return last
        if self.mode == "mean": return (h * m[..., None]).sum(1) / lens[:, None].float()
        w = self.score(h).squeeze(-1).masked_fill(~m, -float("inf")).softmax(-1)
        v = (w[..., None] * h).sum(1)
        return torch.cat([v, last], -1) if self.mode == "attn+last" else v

class DocHead(nn.Module):
    def __init__(self, H, d=256, pool="attn", doc_pool="attn", relpos=True, dropout=0.1):
        super().__init__()
        self.tpool = TokenPool(H, pool)
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.tpool.out_dim, d), nn.GELU())
        self.rel = nn.Embedding(10, d) if relpos else None
        if relpos: nn.init.zeros_(self.rel.weight)
        self.doc_pool = doc_pool
        self.enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True, norm_first=True) \
            if doc_pool == "transformer" else None
        self.att = nn.Linear(d, 1) if doc_pool in ("attn", "transformer") else None
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 2))
    def forward(self, h, b):
        c = self.proj(self.tpool(h, b["lens"]))
        if self.rel is not None: c = c + self.rel(b["rel"])
        B, K = b["y"].size(0), int(b["kidx"].max()) + 1
        z = c.new_zeros(B, K, c.size(-1)).index_put((b["doc_of"], b["kidx"]), c)
        m = torch.zeros(B, K, dtype=torch.bool, device=c.device).index_put((b["doc_of"], b["kidx"]),
                                                                           torch.ones_like(b["kidx"], dtype=torch.bool))
        if self.enc is not None: z = self.enc(z, src_key_padding_mask=~m)
        if self.doc_pool == "max":
            v = z.masked_fill(~m[..., None], -float("inf")).max(1).values; w = m.float() / m.sum(1, keepdim=True)
        else:
            w = self.att(z).squeeze(-1).masked_fill(~m, -float("inf")).softmax(-1) if self.att is not None \
                else m.float() / m.sum(1, keepdim=True)
            v = (w[..., None] * z).sum(1)
        return self.cls(v), v, w

class CJPEModel(nn.Module):
    def __init__(self, bb, head):
        super().__init__(); self.bb, self.head = bb, head
        self.trainable_names, self.ada, self.schedule = set(), None, None
    def forward(self, b):
        h = self.bb(b["ids"])
        with no_amp():                                       # small head in fp32 (exact softmax)
            return self.head(h.float(), b)
    def on_step(self, step, total):
        if self.schedule is not None: self.schedule(step, total)





LOWRANK_TARGETS = ("attn.q", "attn.kv", "attn.out", "up", "down")

def set_module(root, dotted, new):
    parts = dotted.split("."); parent = root
    for p in parts[:-1]: parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    if parts[-1].isdigit(): parent[int(parts[-1])] = new
    else: setattr(parent, parts[-1], new)

def block_linears(lm, targets=None):
    return [(f"blocks.{bi}.{n}", m) for bi, blk in enumerate(lm.blocks)
            for n, m in blk.named_modules() if isinstance(m, nn.Linear) and (targets is None or n in targets)]

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__(); self.base = base
        self.A = nn.Parameter(torch.empty(r, base.in_features)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r; self.drop = nn.Dropout(dropout)
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

class AdaLoRALinear(nn.Module):
    """AdaLoRA: ΔW = s · P diag(λ ⊙ mask) Q. The mask (a buffer) is set by AdaLoRAController."""
    def __init__(self, base, r, alpha, dropout):
        super().__init__(); self.base = base
        self.P = nn.Parameter(torch.randn(base.out_features, r) * 0.02)
        self.E = nn.Parameter(torch.zeros(r))
        self.Q = nn.Parameter(torch.randn(r, base.in_features) * 0.02)
        self.register_buffer("ada_mask", torch.ones(r))
        self.scale = alpha / r; self.drop = nn.Dropout(dropout)
    def forward(self, x):
        lam = (self.E * self.ada_mask).to(x.dtype)
        return self.base(x) + ((self.drop(x) @ self.Q.t().to(x.dtype)) * lam) @ self.P.t().to(x.dtype) * self.scale
    def orth(self):
        P, Q = self.P.float(), self.Q.float(); I = torch.eye(P.size(1), device=P.device)
        return (P.t() @ P - I).square().sum() + (Q @ Q.t() - I).square().sum()

class AdaLoRAController:
    """Importance scoring + global cubic budget schedule (Zhang et al. 2023, Alg. 1)."""
    def __init__(self, mods, r0, r_target, t_init, t_final, dt, beta1, beta2):
        self.mods, self.r0 = mods, r0
        self.b0, self.bT = r0 * len(mods), r_target * len(mods)
        self.t_init, self.t_final, self.dt, self.b1, self.b2 = t_init, t_final, dt, beta1, beta2
        self.Ibar, self.Ubar = {}, {}
    def orth_loss(self):
        return sum(m.orth() for m in self.mods) / len(self.mods)
    def budget(self, t, T):
        ti, tf = int(self.t_init * T), int(self.t_final * T)
        if t < ti: return self.b0
        if t >= T - tf: return self.bT
        return int(self.bT + (self.b0 - self.bT) * (1 - (t - ti) / max(1, T - tf - ti)) ** 3)
    @torch.no_grad()
    def _score(self, name, p):
        s = (p * p.grad).abs().float()
        if not torch.isfinite(s).all(): return None
        if name not in self.Ibar: self.Ibar[name], self.Ubar[name] = s.clone(), torch.zeros_like(s)
        self.Ubar[name].mul_(self.b2).add_((1 - self.b2) * (s - self.Ibar[name]).abs())
        self.Ibar[name].mul_(self.b1).add_((1 - self.b1) * s)
        return self.Ibar[name] * self.Ubar[name]
    @torch.no_grad()
    def update(self, t, T):
        scores = []
        for i, m in enumerate(self.mods):
            if m.E.grad is None or m.P.grad is None or m.Q.grad is None: return
            sE, sP, sQ = self._score((i, "E"), m.E), self._score((i, "P"), m.P), self._score((i, "Q"), m.Q)
            if sE is None or sP is None or sQ is None: return
            scores.append(sE + sP.mean(0) + sQ.mean(1))                    # one score per triplet (P_:,k, λ_k, Q_k,:)
        b = self.budget(t, T)
        if t < int(self.t_init * T) or not (t % self.dt == 0 or t >= T - int(self.t_final * T)): return
        allsc = torch.cat(scores)
        if b >= allsc.numel(): keep = torch.ones_like(allsc, dtype=torch.bool)
        else: keep = allsc >= allsc.topk(b).values.min()
        for m, k in zip(self.mods, keep.split(self.r0)): m.ada_mask.copy_(k.float())
    def ranks(self): return [int(m.ada_mask.sum()) for m in self.mods]

class BiasOut(nn.Module):
    """BitFit: adds a zero-initialised trainable bias to a frozen module's output."""
    def __init__(self, base, dim):
        super().__init__(); self.base = base; self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        y = self.base(x); return y + self.bias.to(y.dtype)

class AdapterOut(nn.Module):
    """Houlsby adapter on a sub-layer output: y + U·GELU(D·y). U = 0, so it starts as the identity."""
    def __init__(self, base, dim, bottleneck):
        super().__init__(); self.base = base
        self.down = nn.Linear(dim, bottleneck); self.up = nn.Linear(bottleneck, dim)
        nn.init.normal_(self.down.weight, std=0.02); nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self, x):
        y = self.base(x)
        return y + self.up(F.gelu(self.down(y)))

def apply_method(model, cfg):
    lm, m = model.bb.lm, cfg.method
    for p in lm.parameters(): p.requires_grad = m in ("full", "kd")
    if m in ("full", "kd"): pass
    elif m in ("lora", "lorma", "adalora"):
        cls = dict(lora=LoRALinear, lorma=LoRMALinear, adalora=AdaLoRALinear)[m]
        r = cfg.ada_r_init if m == "adalora" else cfg.r
        new = []
        for name, lin in block_linears(lm, tuple(cfg.targets)):
            mod = cls(lin, r, cfg.alpha, cfg.peft_dropout); set_module(lm, name, mod); new.append(mod)
        if m == "adalora":
            model.ada = AdaLoRAController(new, cfg.ada_r_init, cfg.ada_r_target, cfg.ada_tinit, cfg.ada_tfinal,
                                          cfg.ada_dt, cfg.ada_beta1, cfg.ada_beta2)
    elif m == "adapter":
        for blk in lm.blocks:
            d = blk.attn.out.out_features
            blk.attn.out = AdapterOut(blk.attn.out, d, cfg.adapter_dim); blk.down = AdapterOut(blk.down, d, cfg.adapter_dim)
    elif m == "bitfit":
        for name, lin in block_linears(lm): set_module(lm, name, BiasOut(lin, lin.out_features))
        lm.norm = BiasOut(lm.norm, lm.norm.weight.numel())
    elif m == "partial":
        L = len(lm.blocks)
        for i in range(L - cfg.partial_k, L):
            for p in lm.blocks[i].parameters(): p.requires_grad = True
        for p in lm.norm.parameters(): p.requires_grad = True
    elif m == "unfreeze":
        L = len(lm.blocks)
        for p in list(lm.blocks.parameters()) + list(lm.norm.parameters()): p.requires_grad = True
        def schedule(step, total, lm=lm, L=L):
            every = max(1, int(cfg.unfreeze_every * total))
            k = min(L, cfg.unfreeze_group * (1 + step // every))
            for i, blk in enumerate(lm.blocks):
                for p in blk.parameters(): p.requires_grad = i >= L - k
        model.schedule = schedule
    else:
        raise ValueError(m)





@dataclass
class ExpConfig:
    name: str
    method: str                       # full | lorma | lora | adalora | adapter | bitfit | partial | unfreeze | kd
    model: str = ""                   # Phase-1 model key ("" = first registered)
    lr: float = 1e-3                  # backbone / adapter parameters
    head_lr: float = 1e-3
    epochs: int = EPOCHS
    train_docs_per_epoch: int = TRAIN_DOCS_PER_EPOCH
    patience: int = 2                 # evaluations without improvement before stopping
    eval_every: int = 0               # optimizer steps between extra dev evaluations (0 = end of epoch only)
    batch_size: int = 2               # cases per micro-batch
    grad_accum: int = 8
    warmup: float = 0.06
    wd: float = 0.01
    dropout: float = 0.1
    label_smoothing: float = 0.0
    class_weight: str = "balanced"    # none | balanced   (macro-F1 rewards the minority outcome)
    chunk_len: int = 512
    n_chunks: int = 4
    overlap: int = 64
    select: str = "tail"              # tail | head | middle | headtail | uniform
    pool: str = "attn"                # attn | mean | last | attn+last   (tokens -> chunk)
    doc_pool: str = "attn"            # attn | mean | max | transformer  (chunks -> case)
    relpos: bool = True               # section (relative position) embedding
    head_dim: int = 256
    r: int = 16
    alpha: float = 32.0
    peft_dropout: float = 0.05
    targets: tuple = LOWRANK_TARGETS
    ada_r_init: int = 12
    ada_r_target: int = 4
    ada_tinit: float = 0.1
    ada_tfinal: float = 0.2
    ada_dt: int = 10
    ada_orth: float = 0.1
    ada_beta1: float = 0.85
    ada_beta2: float = 0.85
    adapter_dim: int = 64
    partial_k: int = 4
    llrd: float = 1.0                 # layer-wise LR decay factor per block from the top (1 = off)
    unfreeze_every: float = 0.08
    unfreeze_group: int = 2
    init: str = "phase1"              # phase1 | random  (random = transfer ablation)
    grad_ckpt: bool = False
    kd_teacher: str = ""
    kd_keep: str = "alternate"        # alternate | first_half | all | "0,3,6,9"
    kd_alpha: float = 0.5             # weight of the gold loss
    kd_T: float = 2.0
    kd_hidden: float = 0.1            # MSE between teacher and student document vectors
    tune_threshold: bool = True
    seed: int = SEED
    note: str = ""

def student_keep(n, mode):
    if mode == "all": return list(range(n))
    if mode == "alternate": return list(range(0, n, 2))
    if mode == "first_half": return list(range(n // 2))
    return [int(x) for x in mode.split(",")]

def build_model(cfg):
    seed_all(cfg.seed)
    cfg.model = cfg.model or MODEL_KEYS[0]
    assert cfg.chunk_len <= P1[cfg.model]["context"], "chunk_len exceeds the Phase-1 context"
    lm = load_lm(cfg.model, cfg.init); keep = None
    if cfg.method == "kd":
        keep = student_keep(len(lm.blocks), cfg.kd_keep)
        lm.blocks = nn.ModuleList([lm.blocks[i] for i in keep])
    bb = Backbone(lm); bb.grad_ckpt = cfg.grad_ckpt
    model = CJPEModel(bb, DocHead(P1[cfg.model]["hid"], cfg.head_dim, cfg.pool, cfg.doc_pool, cfg.relpos, cfg.dropout))
    apply_method(model, cfg)
    model.trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    return model, keep

def n_params(m, trainable=False):
    return sum(p.numel() for n, p in m.named_parameters() if not trainable or n in m.trainable_names)
def n_bytes(m): return sum(t.numel() * t.element_size() for t in list(m.parameters()) + list(m.buffers()))
def snapshot(model):
    s = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n in model.trainable_names}
    s.update({n: b.detach().cpu().clone() for n, b in model.named_buffers() if n.endswith("ada_mask")})
    return s

def binary_scores(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred); per = {}
    for c in (0, 1):
        tp = int(((pred == c) & (gold == c)).sum()); fp = int(((pred == c) & (gold != c)).sum()); fn = int(((pred != c) & (gold == c)).sum())
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
        per[c] = dict(P=P, R=R, F1=2 * P * R / (P + R) if P + R else 0.0, support=int((gold == c).sum()))
    return dict(macro_F1=(per[0]["F1"] + per[1]["F1"]) / 2, macro_P=(per[0]["P"] + per[1]["P"]) / 2,
                macro_R=(per[0]["R"] + per[1]["R"]) / 2, accuracy=float((gold == pred).mean()) if len(gold) else 0.0,
                F1_0=per[0]["F1"], F1_1=per[1]["F1"], pred_pos_rate=float(pred.mean()) if len(pred) else 0.0)

def tune_threshold(gold, prob):
    """τ maximising dev macro-F1 for pred = [P(y=1) ≥ τ]; ties broken towards 0.5."""
    cands = np.unique(np.concatenate([np.quantile(prob, np.linspace(0.01, 0.99, 197)), [0.5]]))
    best = max(cands, key=lambda t: (round(binary_scores(gold, prob >= t)["macro_F1"], 6), -abs(t - 0.5)))
    return float(best)

@torch.no_grad()
def predict_proba(model, cases, toks, cfg, select=None, K=None, bs=8, return_weights=False):
    model.eval(); pad = P1[cfg.model]["pad"]
    exs = make_examples(cases, toks, cfg, with_labels=False, select=select, K=K)
    order = sorted(range(len(exs)), key=lambda i: sum(len(c) for c in exs[i]["chunks"]))
    prob = np.zeros(len(exs), np.float32); W = [None] * len(exs)
    for k in range(0, len(order), bs):
        chunk = [exs[i] for i in order[k:k + bs]]
        with amp(): logits, _, w = model(to_dev(collate(chunk, pad)))
        p = logits.float().softmax(-1)[:, 1].cpu().numpy(); w = w.float().cpu().numpy()
        for j, i in enumerate(order[k:k + bs]):
            prob[i] = p[j]
            if return_weights: W[i] = list(zip(exs[i]["rel"], w[j][:len(exs[i]["rel"])].tolist()))
    return (prob, W) if return_weights else prob

def evaluate(model, cases, toks, cfg, tau=None, **kw):
    prob = predict_proba(model, cases, toks, cfg, **kw); gold = np.array([c["y"] for c in cases])
    t = tune_threshold(gold, prob) if tau is None else tau
    return dict(at05=binary_scores(gold, prob >= 0.5), tuned=binary_scores(gold, prob >= t), tau=t), prob

def class_weights(cases, mode):
    if mode == "none": return None
    cnt = np.bincount([c["y"] for c in cases], minlength=2).astype(float)
    return torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float, device=DEVICE)

def train_model(model, cfg, trn, trn_tok, dev, dev_tok, teacher=None):
    seed_all(cfg.seed)
    exs = make_examples(trn, trn_tok, cfg, with_labels=True); pad = P1[cfg.model]["pad"]
    named = dict(model.named_parameters()); L = len(model.bb.lm.blocks)
    def lr_scale(n):
        mm = re.match(r"bb\.lm\.blocks\.(\d+)\.", n)
        return cfg.llrd ** (L - 1 - int(mm.group(1))) if (mm and cfg.llrd < 1) else 1.0
    buckets = defaultdict(list)
    for n in sorted(model.trainable_names):
        p = named[n]
        key = ("head", cfg.head_lr, 0.01) if n.startswith("head.") else ("body", cfg.lr * lr_scale(n), cfg.wd if p.ndim >= 2 else 0.0)
        buckets[key].append(p)
    opt = torch.optim.AdamW([dict(params=ps, lr=lr, weight_decay=wd) for (_, lr, wd), ps in buckets.items()], betas=(0.9, 0.98))
    for g in opt.param_groups: g["base_lr"] = g["lr"]
    bs = cfg.batch_size; n_ep = min(cfg.train_docs_per_epoch or len(exs), len(exs))
    nbatches = math.ceil(n_ep / bs); steps_per_ep = math.ceil(nbatches / cfg.grad_accum)
    total = steps_per_ep * cfg.epochs; warm = max(1, int(cfg.warmup * total))
    lr_at = lambda s: (s + 1) / warm if s < warm else max(0.02, (total - s) / max(1, total - warm))
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda" and AMP_DTYPE == torch.float16))
    cw = class_weights(trn, cfg.class_weight)
    tparams = [named[n] for n in sorted(model.trainable_names)]
    log(f"  train cases {len(exs)} (sample {n_ep}/epoch) | optimizer steps {total} | trainable {n_params(model, True)/1e6:.3f}M "
        f"of {n_params(model)/1e6:.1f}M ({100*n_params(model, True)/n_params(model):.2f}%)")
    best, best_state, best_tag, bad, hist, step = -1.0, None, "", 0, [], 0
    if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); model.on_step(0, total); stop = False

    def do_eval(tag, run_loss):
        nonlocal best, best_state, best_tag, bad, stop
        sc, _ = evaluate(model, dev, dev_tok, cfg)
        f1 = sc["tuned"]["macro_F1"] if cfg.tune_threshold else sc["at05"]["macro_F1"]
        row = dict(tag=tag, step=step, loss=run_loss, dev_macro_F1=f1, dev_macro_F1_at05=sc["at05"]["macro_F1"],
                   tau=sc["tau"], minutes=(time.time() - t0) / 60)
        hist.append(row)
        log(f"  {tag:>9} | step {step:5d} | loss {run_loss:.4f} | dev macro-F1 {f1:.4f} (τ={sc['tau']:.3f}; "
            f"@0.5 {sc['at05']['macro_F1']:.4f}) | {row['minutes']:.1f} min")
        if f1 > best: best, best_tag, bad, best_state = f1, tag, 0, snapshot(model)
        else:
            bad += 1
            if bad >= cfg.patience: log("  early stop"); stop = True
        model.train()

    for ep in range(cfg.epochs):
        model.train(); order = random.Random(cfg.seed * 1000 + ep).sample(range(len(exs)), n_ep); tot = nb = 0
        for bi, k in enumerate(range(0, n_ep, bs)):
            b = to_dev(collate([exs[i] for i in order[k:k + bs]], pad))
            with amp(): logits, v, _ = model(b)
            loss = F.cross_entropy(logits, b["y"], weight=cw, label_smoothing=cfg.label_smoothing)
            if teacher is not None:
                with torch.no_grad(), amp(): t_logits, t_v, _ = teacher(b)
                T = cfg.kd_T
                kl = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(t_logits.float() / T, -1), reduction="batchmean") * T * T
                loss = cfg.kd_alpha * loss + (1 - cfg.kd_alpha) * kl
                if cfg.kd_hidden > 0: loss = loss + cfg.kd_hidden * F.mse_loss(v, t_v.float())
            if model.ada is not None: loss = loss + cfg.ada_orth * model.ada.orth_loss()
            if not torch.isfinite(loss): raise FloatingPointError("non-finite loss")
            scaler.scale(loss / cfg.grad_accum).backward(); tot += loss.item(); nb += 1
            if (bi + 1) % cfg.grad_accum == 0 or bi + 1 == nbatches:
                for g in opt.param_groups: g["lr"] = g["base_lr"] * lr_at(step)
                scaler.unscale_(opt)
                if model.ada is not None: model.ada.update(step, total)
                torch.nn.utils.clip_grad_norm_([p for p in tparams if p.grad is not None], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
                model.on_step(step, total)
                if step == 20 or step % 200 == 0:
                    el = time.time() - t0
                    log(f"    step {step}/{total} | loss {tot/max(1, nb):.4f} | {step*cfg.grad_accum*bs/el:.1f} cases/s "
                        f"| ETA training {el/step*(total-step)/60:.1f} min")
                if cfg.eval_every and step % cfg.eval_every == 0 and step < steps_per_ep * (ep + 1):
                    do_eval(f"step{step}", tot / max(1, nb))
                    if stop: break
        if stop: break
        do_eval(f"epoch{ep+1}", tot / max(1, nb))
        if stop: break
    model.on_step(total, total)
    model.load_state_dict(best_state, strict=False)
    stats = dict(train_minutes=(time.time() - t0) / 60, best_eval=best_tag,
                 peak_mem_GB=torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0)
    if model.ada is not None: stats["adalora_ranks"] = model.ada.ranks()
    return hist, stats, best_state





RESULTS = OUT / "results.jsonl"

def cfg_from_dict(d):
    names = {f.name for f in fields(ExpConfig)}
    d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
    return ExpConfig(**d)

def load_experiment(name, out_dir=OUT):
    ck = torch.load(out_dir / name / "model.pt", map_location="cpu", weights_only=False)
    cfg = cfg_from_dict(ck["cfg"]); model, _ = build_model(cfg)
    missing = model.trainable_names - set(ck["state"])
    assert not missing, f"checkpoint lacks trainable tensors: {sorted(missing)[:5]}"
    _, unexpected = model.load_state_dict(ck["state"], strict=False); assert not unexpected, unexpected[:5]
    return model.to(DEVICE).eval(), cfg, ck.get("tau", 0.5)

def write_submission(prob, cases, tau, dest):
    out = [{"id": c["rid"], "pred": int(p >= tau)} for c, p in zip(cases, prob)]
    dest.mkdir(parents=True, exist_ok=True)
    pj = dest / "prediction.json"; json.dump(out, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    zp = dest / "submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z: z.write(pj, arcname="prediction.json")
    check_submission(zp, cases)
    return zp

def check_submission(zp, cases):
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
        pred = json.loads(z.read("prediction.json"))
    assert isinstance(pred, list) and len(pred) == len(cases), "one prediction per test case expected"
    ids = [str(p["id"]) for p in pred]
    assert len(set(ids)) == len(ids) and set(ids) == {c["id"] for c in cases}, "missing, extra or duplicate ids"
    assert all(type(p["pred"]) is int and p["pred"] in (0, 1) for p in pred), "pred must be the integer 0 or 1"
    return True

def run_experiment(cfg, trn=None, dev=None, test=None, out_dir=OUT, submit=True, record=True):
    trn, dev, test = trn or TRAIN, dev or DEV, test or TEST
    cfg.model = cfg.model or MODEL_KEYS[0]
    d = out_dir / cfg.name
    if (d / "done.json").exists():
        log(f"[skip] {cfg.name} already finished"); return json.load(open(d / "done.json"))
    log(f"\n=== {cfg.name} | model={cfg.model} method={cfg.method} select={cfg.select} K={cfg.n_chunks}x{cfg.chunk_len} init={cfg.init} ===")
    teacher = None
    if cfg.method == "kd":
        teacher, tcfg, _ = load_experiment(cfg.kd_teacher, out_dir)
        for p in teacher.parameters(): p.requires_grad = False
        assert P1[tcfg.model]["tok_key"] == P1[cfg.model]["tok_key"], "KD teacher and student must share a tokenizer"
        for f in ("chunk_len", "n_chunks", "overlap", "select", "head_dim"): setattr(cfg, f, getattr(tcfg, f))
    T = toks_for(cfg.model)
    tok_of = lambda cases, split: T[split] if cases is {"train": TRAIN, "dev": DEV, "test": TEST}[split] else \
        [T[split][IDX[split][c["id"]]] for c in cases]
    trn_tok, dev_tok, test_tok = tok_of(trn, "train"), tok_of(dev, "dev"), tok_of(test, "test")
    model, keep = build_model(cfg); model = model.to(DEVICE)
    if teacher is not None:
        log(f"  KD: teacher {cfg.kd_teacher} {n_params(teacher)/1e6:.1f}M -> student {n_params(model)/1e6:.1f}M (blocks {keep})")
    hist, stats, state = train_model(model, cfg, trn, trn_tok, dev, dev_tok, teacher)

    t0 = time.time(); dev_sc, dev_prob = evaluate(model, dev, dev_tok, cfg); dev_s = time.time() - t0
    tau = dev_sc["tau"] if cfg.tune_threshold else 0.5
    final = dev_sc["tuned"] if cfg.tune_threshold else dev_sc["at05"]
    d.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cfg=asdict(cfg), state=state, keep_layers=keep, tau=tau), d / "model.pt")
    json.dump(dict(ids=[c["id"] for c in dev], prob=dev_prob.tolist(), gold=[c["y"] for c in dev], tau=tau),
              open(d / "dev_predictions.json", "w"))
    if submit:
        t0 = time.time(); test_prob = predict_proba(model, test, test_tok, cfg); stats["test_infer_s"] = time.time() - t0
        json.dump(dict(ids=[c["id"] for c in test], prob=test_prob.tolist()), open(d / "test_probs.json", "w"))
        zp = write_submission(test_prob, test, tau, d); log(f"  submission -> {zp}")
    row = dict(name=cfg.name, model=cfg.model, method=cfg.method, select=cfg.select, n_chunks=cfg.n_chunks,
               chunk_len=cfg.chunk_len, init=cfg.init,
               total_params_M=n_params(model) / 1e6, trainable_params_M=n_params(model, True) / 1e6,
               trainable_pct=100 * n_params(model, True) / n_params(model), backbone_MB=n_bytes(model.bb) / 2**20,
               dev_macro_F1=final["macro_F1"], dev_macro_P=final["macro_P"], dev_macro_R=final["macro_R"],
               dev_acc=final["accuracy"], dev_F1_0=final["F1_0"], dev_F1_1=final["F1_1"],
               dev_pred_pos_rate=final["pred_pos_rate"], dev_macro_F1_at05=dev_sc["at05"]["macro_F1"], tau=tau,
               dev_infer_ms_per_case=1000 * dev_s / max(1, len(dev)), **stats, history=hist, cfg=asdict(cfg))
    if record:
        with open(RESULTS, "a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")
    json.dump(row, open(d / "done.json", "w"))
    log(f"  DEV macro-F1 {row['dev_macro_F1']:.4f} (P {row['dev_macro_P']:.4f} R {row['dev_macro_R']:.4f}; τ={tau:.3f}; "
        f"@0.5 {row['dev_macro_F1_at05']:.4f}) | trainable {row['trainable_params_M']:.3f}M ({row['trainable_pct']:.2f}%) "
        f"| peak {stats['peak_mem_GB']:.1f} GB | {stats['train_minutes']:.1f} min")
    del model, teacher; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return row

IDX = {s: {c["id"]: i for i, c in enumerate(cs)} for s, cs in (("train", TRAIN), ("dev", DEV), ("test", TEST))}





def smoke_test(methods=("full", "lorma", "adalora", "adapter", "bitfit", "partial", "unfreeze", "lora", "kd"), keep=False):
    sd = OUT / "_smoke"; shutil.rmtree(sd, ignore_errors=True)
    trn, dev, test = TRAIN[:16], DEV[:8], TEST[:4]
    t0 = time.time()
    for i, m in enumerate(methods):
        key = MODEL_KEYS[i % len(MODEL_KEYS)] if m != "kd" else MODEL_KEYS[0]
        cfg = ExpConfig(f"smoke_{m}", m, model=key, epochs=1, train_docs_per_epoch=0, chunk_len=128, n_chunks=2,
                        overlap=16, batch_size=2, grad_accum=2, ada_dt=1, kd_teacher="smoke_full")
        row = run_experiment(cfg, trn, dev, test, out_dir=sd, submit=True, record=False)
        if m != "kd":
            model, c2, tau = load_experiment(cfg.name, sd)       # reload round-trip must reproduce the dev score
            sc, _ = evaluate(model, dev, [toks_for(c2.model)["dev"][IDX["dev"][c["id"]]] for c in dev], c2, tau=tau)
            assert abs(sc["tuned"]["macro_F1"] - row["dev_macro_F1"]) < 1e-6, (m, sc["tuned"]["macro_F1"], row["dev_macro_F1"])
            del model
    cfg = ExpConfig("smoke_flat", "lorma", epochs=1, train_docs_per_epoch=0, chunk_len=256, n_chunks=1,
                    doc_pool="transformer", pool="attn+last", select="head", batch_size=2, grad_accum=1)
    run_experiment(cfg, trn, dev, test, out_dir=sd, submit=True, record=False)
    if not keep: shutil.rmtree(sd, ignore_errors=True)
    log(f"SMOKE TEST PASSED in {(time.time()-t0)/60:.1f} min")

SMOKE = False
if SMOKE: smoke_test()





M = lambda i: MODEL_KEYS[i % len(MODEL_KEYS)]
EXPERIMENTS = [
    ExpConfig("E0_full",     "full",     model=M(0), lr=3e-5),
    ExpConfig("E1_lorma",    "lorma",    model=M(0), lr=1e-3, r=16, alpha=32),
    ExpConfig("E2_adalora",  "adalora",  model=M(1), lr=1e-3, ada_r_init=12, ada_r_target=4, alpha=32),
    ExpConfig("E3_adapter",  "adapter",  model=M(2), lr=1e-3, adapter_dim=64),
    ExpConfig("E4_bitfit",   "bitfit",   model=M(3), lr=2e-3),
    ExpConfig("E5_partial",  "partial",  model=M(1), lr=1e-4, partial_k=4, llrd=0.8),
    ExpConfig("E6_unfreeze", "unfreeze", model=M(2), lr=5e-5, llrd=0.85, unfreeze_group=2, unfreeze_every=0.08),
    ExpConfig("E7_kd",       "kd",       model=M(0), lr=8e-5, kd_teacher="E0_full", kd_keep="alternate",
              kd_alpha=0.5, kd_T=2.0, kd_hidden=0.1),
    # ---- optional extra submissions / analysis ablations ----
    ExpConfig("E8_lorma_flat", "lorma",  model=M(0), lr=1e-3, n_chunks=1, chunk_len=1024, batch_size=4, grad_accum=4,
              note="ablation: one flat 1024-token tail chunk (no hierarchy)"),
    ExpConfig("E9_lorma_head", "lorma",  model=M(0), lr=1e-3, select="head", note="ablation: first chunks instead of last"),
    ExpConfig("E10_lora",    "lora",     model=M(0), lr=1e-3, r=16, alpha=32, note="additive vs multiplicative low-rank"),
]
# LoRMA on every other registered Phase-1 model (direct architecture comparison under the compulsory technique)
EXPERIMENTS += [ExpConfig(f"E11_lorma_{k}", "lorma", model=k, lr=1e-3) for k in MODEL_KEYS[1:]]
# Not for submission, but answers "does Phase-1 pretraining transfer?" (same recipe as E1, random backbone):
TRANSFER_ABLATION = ExpConfig("A1_lorma_randominit", "lorma", model=M(0), lr=1e-3, init="random")


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
cols = ["model", "method", "select", "n_chunks", "chunk_len", "dev_macro_F1", "dev_macro_P", "dev_macro_R", "dev_macro_F1_at05",
        "tau", "dev_F1_0", "dev_F1_1", "dev_pred_pos_rate", "dev_acc", "total_params_M", "trainable_params_M", "trainable_pct",
        "backbone_MB", "peak_mem_GB", "train_minutes", "dev_infer_ms_per_case"]
print("Main comparison (sorted by dev macro-F1 at the tuned threshold)")
display(df[[c for c in cols if c in df]].sort_values("dev_macro_F1", ascending=False).round(4))
print("Note: τ is tuned on dev, so dev scores are slightly optimistic. Codabench is the unbiased check.")
best_name = df.drop([n for n in df.index if n.startswith("A1")], errors="ignore")["dev_macro_F1"].idxmax()
print("best:", best_name)





bm, bcfg, btau = load_experiment(best_name)
_dt = toks_for(bcfg.model)["dev"]
probe = []
for sel, K in [("tail", 1), ("tail", bcfg.n_chunks), ("head", bcfg.n_chunks), ("middle", bcfg.n_chunks),
               ("headtail", bcfg.n_chunks), ("uniform", bcfg.n_chunks), ("uniform", 12)]:
    sc, _ = evaluate(bm, DEV, _dt, bcfg, tau=btau, select=sel, K=K, bs=4)
    sct, _ = evaluate(bm, DEV, _dt, bcfg, select=sel, K=K, bs=4)
    probe.append(dict(select=sel, K=K, macro_F1_at_model_tau=sc["tuned"]["macro_F1"], macro_F1_retuned=sct["tuned"]["macro_F1"]))
display(pd.DataFrame(probe).round(4))
_, W = predict_proba(bm, DEV, _dt, bcfg, select="uniform", K=10, bs=4, return_weights=True)
_acc = defaultdict(list)
for wl in W:
    for r, w in wl: _acc[r].append(w * len(wl))          # ×K: 1.0 = uniform attention
print("mean chunk-attention by relative position (1.0 = uniform; >1 = the model relies on this part):")
display(pd.DataFrame({"rel_position": [f"{10*r}-{10*r+10}%" for r in sorted(_acc)],
                      "attention_x_K": [round(float(np.mean(_acc[r])), 3) for r in sorted(_acc)]}).set_index("rel_position").T)





dp = json.load(open(OUT / best_name / "dev_predictions.json"))
prob, gold, tau = np.array(dp["prob"]), np.array(dp["gold"]), dp["tau"]
pred = (prob >= tau).astype(int)
print(f"dev label balance: P(y=1) = {gold.mean():.3f} | predicted P(y=1) = {pred.mean():.3f}")
print("macro-F1 vs decision threshold:")
display(pd.DataFrame([dict(tau=t, **{k: round(v, 4) for k, v in binary_scores(gold, prob >= t).items()})
                      for t in sorted({0.3, 0.4, 0.5, 0.6, 0.7, round(tau, 3)})]).set_index("tau"))
conf = pd.DataFrame([[int(((gold == g) & (pred == p)).sum()) for p in (0, 1)] for g in (0, 1)],
                    index=["gold 0", "gold 1"], columns=["pred 0", "pred 1"])
display(conf)
# effect of class weighting on this model: compare against the unweighted decision at τ = 0.5
print("at τ=0.5:", {k: round(v, 4) for k, v in binary_scores(gold, prob >= 0.5).items()})

dl = np.array([len(x) for x in _dt]); ids = dp["ids"]; by_id = {c["id"]: c for c in DEV}
qs = np.percentile(dl, [0, 25, 50, 75, 100])
print("macro-F1 by case length (tokens):")
display(pd.DataFrame([dict(bucket=f"{int(a)}–{int(b)}", n=int(((dl >= a) & (dl <= b)).sum()),
                           **{k: round(v, 4) for k, v in binary_scores(gold[(dl >= a) & (dl <= b)], pred[(dl >= a) & (dl <= b)]).items()
                              if k in ("macro_F1", "accuracy", "pred_pos_rate")})
                      for a, b in zip(qs[:-1], qs[1:])]).set_index("bucket"))

fp, fn = np.where((pred == 1) & (gold == 0))[0], np.where((pred == 0) & (gold == 1))[0]
tp_tn = np.where(pred == gold)[0]
print(pd.DataFrame([dict(group=g, n=len(ix), median_tokens=int(np.median(dl[ix])) if len(ix) else 0,
                         mean_confidence=round(float(np.mean(np.abs(prob[ix] - tau))), 3) if len(ix) else 0)
                    for g, ix in (("false positives", fp), ("false negatives", fn), ("correct", tp_tn))]).to_string(index=False))
for g, ix in (("FALSE POSITIVE", fp), ("FALSE NEGATIVE", fn)):
    for i in ix[np.argsort(-np.abs(prob[ix] - tau))][:2]:           # most confident mistakes
        print(f"\n[{g}] id={ids[i]} P(y=1)={prob[i]:.3f} tokens={dl[i]}\n  ...{by_id[ids[i]]['text'][-500:]}")
del bm; gc.collect()
if DEVICE == "cuda": torch.cuda.empty_cache()





if {"E0_full", "E7_kd"} <= set(df.index):
    t, s = df.loc["E0_full"], df.loc["E7_kd"]
    print(f"KD student keeps {100*s.dev_macro_F1/t.dev_macro_F1:.1f}% of teacher macro-F1 with "
          f"{100*s.total_params_M/t.total_params_M:.0f}% of parameters, {100*s.backbone_MB/t.backbone_MB:.0f}% of memory, "
          f"{100*s.dev_infer_ms_per_case/t.dev_infer_ms_per_case:.0f}% of inference time")
if "E0_full" in df.index:
    ref = df.loc["E0_full", "dev_macro_F1"]
    print("macro-F1 retained relative to full FT:")
    display(pd.DataFrame({"trainable_pct": df["trainable_pct"], "retained_%": 100 * df["dev_macro_F1"] / ref}).round(2)
            .sort_values("retained_%", ascending=False))
if {"E1_lorma", "E8_lorma_flat"} <= set(df.index):
    print(f"hierarchical {df.loc['E1_lorma','dev_macro_F1']:.4f} vs flat tail-1024 {df.loc['E8_lorma_flat','dev_macro_F1']:.4f}")
if {"E1_lorma", "E9_lorma_head"} <= set(df.index):
    print(f"tail chunks {df.loc['E1_lorma','dev_macro_F1']:.4f} vs head chunks {df.loc['E9_lorma_head','dev_macro_F1']:.4f}")
if {"E1_lorma", "E10_lora"} <= set(df.index):
    print(f"LoRMA (multiplicative) {df.loc['E1_lorma','dev_macro_F1']:.4f} vs LoRA (additive) {df.loc['E10_lora','dev_macro_F1']:.4f}")
if {"E1_lorma", "A1_lorma_randominit"} <= set(df.index):
    print(f"Phase-1 init {df.loc['E1_lorma','dev_macro_F1']:.4f} vs random init {df.loc['A1_lorma_randominit','dev_macro_F1']:.4f}")
if "E2_adalora" in df.index and isinstance(df.loc["E2_adalora"].get("adalora_ranks"), list):
    rk = np.array(df.loc["E2_adalora", "adalora_ranks"]).reshape(-1, len(LOWRANK_TARGETS))
    print("AdaLoRA final rank per block (rows) and target matrix (columns):")
    display(pd.DataFrame(rk, columns=LOWRANK_TARGETS, index=[f"block {i}" for i in range(len(rk))]))


# Learning curves (dev macro-F1 per evaluation) for the report
display(pd.DataFrame({n: pd.Series([h["dev_macro_F1"] for h in r["history"]], index=[h["tag"] for h in r["history"]])
                      for n, r in df.iterrows()}).round(4))
print("Submissions:")
for n in df.index:
    zp = OUT / n / "submission.zip"
    if zp.exists(): print(" ", zp)





MAKE_ENSEMBLE = False
if MAKE_ENSEMBLE:
    top = [n for n in df.sort_values("dev_macro_F1", ascending=False).index if (OUT / n / "test_probs.json").exists()][:3]
    dps = [json.load(open(OUT / n / "dev_predictions.json")) for n in top]
    tps = [json.load(open(OUT / n / "test_probs.json")) for n in top]
    assert all(d["ids"] == dps[0]["ids"] for d in dps) and all(t["ids"] == tps[0]["ids"] for t in tps)
    dprob, tprob = np.mean([d["prob"] for d in dps], 0), np.mean([t["prob"] for t in tps], 0)
    tau_e = tune_threshold(np.array(dps[0]["gold"]), dprob)
    print("ensemble of", top, "| dev macro-F1", round(binary_scores(np.array(dps[0]["gold"]), dprob >= tau_e)["macro_F1"], 4))
    print("->", write_submission(tprob, TEST, tau_e, OUT / "ENS_top3"))





from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
from IPython.display import FileLink, display
import os
import shutil

phase = "phase4"
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



