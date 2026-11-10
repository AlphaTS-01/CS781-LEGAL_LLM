


import importlib.util, subprocess, sys
requirements = {'numpy':'numpy>=1.26,<3', 'tokenizers':'tokenizers>=0.20,<1',
                'huggingface_hub':'huggingface_hub>=0.26,<2', 'rouge_score':'rouge-score==0.1.2'}
missing = [v for k,v in requirements.items() if importlib.util.find_spec(k) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
assert importlib.util.find_spec('torch'), 'Choose a PyTorch GPU kernel first.'






from pathlib import Path
import os, json, math, time, random, hashlib, tarfile, shutil, gc, zipfile, csv
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tokenizers import Tokenizer

SMOKE = False
CODE_HASH = '069ebff42d73e40896b4e5b6827a8f0d3503d7433404d26f33242ddf81ebe69e'  # populated by notebook builder
WORK = Path('/kaggle/working/phase6_summ' if Path('/kaggle').exists() else 'phase6_summ')
FOUNDATION_ARCHIVE = None  # explicit Path to your own Phase 1 archive if needed
FOUNDATION_DIR = None      # alternatively, directory with model.pt and tokenizer.json
LOCAL_DATA = None          # explicit Path to Phase_6_Summ, or discover/download
RESUME_FROM = None
EXPERIMENTS = ['lora', 'lorma', 'houlsby', 'pfeiffer', 'prompt', 'kd']
CFG = dict(seed=42, epochs=2, max_train=5000, val_fraction=0.1, val_docs=32,
           source_tokens=512, target_tokens=192, prompt_tokens=16,
           context_strategy='spread', rank=8, alpha=8, adapter_dim=64,
           lr=3e-4, lr_kd=2e-5, accumulation=8, minutes=60,
           checkpoint_seconds=600, temperature=2.0, kd_weight=0.5,
           student_fraction=0.5, min_new_tokens=16, no_repeat_ngram=3)
if SMOKE:
    CFG.update(epochs=1, max_train=8, val_docs=2, source_tokens=48, target_tokens=16,
               prompt_tokens=4, accumulation=2, minutes=2, min_new_tokens=2)
if RESUME_FROM is not None:
    if WORK.exists(): raise ValueError('Choose a new WORK directory when restoring saved output.')
    shutil.copytree(RESUME_FROM, WORK)
WORK.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_DTYPE = torch.bfloat16 if DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
def amp():
    return torch.autocast('cuda', dtype=AMP_DTYPE) if DEVICE.type == 'cuda' else nullcontext()
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def digest(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()
def save_json(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8'); tmp.replace(path)
def save_pt(obj,path):
    tmp=Path(str(path)+'.tmp'); torch.save(obj,tmp); tmp.replace(path)
seed_all(CFG['seed'])
print('Device:', DEVICE, '| Smoke:',SMOKE)
if DEVICE.type == 'cpu' and not SMOKE: print('WARNING: real training and generation should use a GPU.')






"""LEXWeave: a project-specific experimental causal LM, not a pretrained model.

Parallel token attention, strictly completed-chunk memory, and routed dilated
convolution, combined by a token-dependent gate. See notebook for hypotheses,
prior work, limitations, and controlled ablations. No claim of literature novelty.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
    """Learn a salience-weighted vector per chunk; read only COMPLETED past chunks.

    Memory is recomputed inside the current context, not carried between batches.
    A fixed zero slot makes the first chunk well-defined without all-masked rows.
    """
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
        # Null slot: exactly zero, no learned information from any future token.
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
        # Chunk memory carries no past summary during the first chunk.
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
        # Mild residual initialization keeps the extra paths from overwhelming input.
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



def discover_foundation():
    if FOUNDATION_DIR is not None: return Path(FOUNDATION_DIR)
    archive = FOUNDATION_ARCHIVE
    if archive is None:
        roots=[Path('/kaggle/input')] if Path('/kaggle/input').exists() else [Path.cwd()]
        hits=sorted({p.resolve() for r in roots for p in r.rglob('lexy-pytorch-default-v1.tar.gz')})
        if len(hits)==1: archive=hits[0]
        elif len(hits)>1: raise ValueError(f'Set FOUNDATION_ARCHIVE explicitly: {hits}')
        else:
            dirs=sorted({p.parent for r in roots for p in r.rglob('model.pt')
                         if (p.parent/'tokenizer.json').is_file() and p.parent.name!='foundation'})
            if len(dirs)==1: return dirs[0]
            raise FileNotFoundError('Attach Phase 1 archive or set FOUNDATION_DIR with model.pt/tokenizer.json.')
    dest=WORK/'foundation'; dest.mkdir(exist_ok=True)
    with tarfile.open(archive,'r:gz') as tf:
        for name in ('model.pt','tokenizer.json'):
            hits=[m for m in tf.getmembers() if m.isfile() and Path(m.name).name==name]
            if len(hits)!=1: raise ValueError(f'Expected exactly one {name}')
            with tf.extractfile(hits[0]) as src, open(dest/name,'wb') as dst: shutil.copyfileobj(src,dst)
    return dest
FOUNDATION=discover_foundation()
# Only load your own trusted checkpoint: this archive includes Python metadata.
BASE=torch.load(FOUNDATION/'model.pt',map_location='cpu',weights_only=False)
BASE_CFG=dict(BASE['model_config'])
tokenizer=Tokenizer.from_file(str(FOUNDATION/'tokenizer.json'))
tokenizer.no_padding(); tokenizer.no_truncation()
PAD,BOS,EOS=[tokenizer.token_to_id(t) for t in ('<pad>','<bos>','<eos>')]
assert None not in (PAD,BOS,EOS) and tokenizer.get_vocab_size()==BASE_CFG['vocab']
assert CFG['source_tokens']+CFG['target_tokens']+CFG['prompt_tokens'] <= BASE_CFG['context']
BASE_SHA=digest(FOUNDATION/'model.pt'); TOK_SHA=digest(FOUNDATION/'tokenizer.json')
probe=LegalLM(BASE_CFG); probe.load_state_dict(BASE['model'],strict=True)
print('Strict Phase 1 load passed:',sum(p.numel() for p in probe.parameters()),'parameters')
del probe; gc.collect()






def resolve_data():
    if LOCAL_DATA is not None: return Path(LOCAL_DATA)
    hits=[]
    local=Path('/kaggle/working/capstone_data/Phase_6_Summ')
    if (local/'Train/train.json').is_file(): hits.append(local)
    for root in (Path('/kaggle/input'),Path.cwd()):
        if root.exists():
            hits.extend(p.parent.parent for p in root.glob('**/Phase_6_Summ/Train/train.json'))
    hits=sorted({p.resolve() for p in hits})
    if len(hits)==1: return hits[0]
    if len(hits)>1: raise ValueError(f'Set LOCAL_DATA explicitly: {hits}')
    from huggingface_hub import HfApi, snapshot_download
    from getpass import getpass
    token=os.environ.get('HF_TOKEN') or getpass('Hugging Face read token (dataset access required): ')
    repo='Exploration-Lab/CS-781-Capstone'
    revision=HfApi().dataset_info(repo,token=token).sha
    folder=snapshot_download(repo_id=repo,repo_type='dataset',revision=revision,token=token,
                             allow_patterns=['Phase_6_Summ/**'])
    save_json({'repository':repo,'revision':revision},WORK/'download.json')
    return Path(folder)/'Phase_6_Summ'
def read_records(path, labelled):
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a nonempty JSON list: {path}")

    ids, accepted, excluded = set(), [], []

    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("id"), str) or not r["id"]:
            raise ValueError("Invalid string ID")
        if r["id"] in ids:
            raise ValueError(f"Duplicate ID: {r['id']}")
        ids.add(r["id"])

        if not isinstance(r.get("document"), str) or not r["document"].strip():
            raise ValueError(f"Missing document: {r['id']}")

        if labelled:
            summary = r.get("summary")
            if summary is None or (
                isinstance(summary, str) and not summary.strip()
            ):
                excluded.append(r["id"])
                continue
            if not isinstance(summary, str):
                raise ValueError(
                    f"ID {r['id']}: summary has type "
                    f"{type(summary).__name__}, expected string. "
                    f"Fields: {list(r)}"
                )
        elif "summary" in r:
            raise ValueError("Test contains summaries; verify the test input.")

        accepted.append(r)

    if labelled:
        if not accepted:
            raise ValueError(
                f"No usable summaries found. Check schema: {list(rows[0])}"
            )
        save_json(
            {
                "total_rows": len(rows),
                "accepted_rows": len(accepted),
                "excluded_missing_summary_ids": excluded,
            },
            WORK / "data_quality.json",
        )
        print(f"Retained {len(accepted)}/{len(rows)} training records.")
        print(f"Skipped {len(excluded)} missing/blank summaries.")
        if excluded:
            print("Skipped IDs (first 20):", excluded[:20])

    return accepted


DATA = resolve_data()
all_train = read_records(DATA / "Train/train.json", True)
test_rows = read_records(DATA / "Test/test.json", False)

assert not (
    {r["id"] for r in all_train} & {r["id"] for r in test_rows}
), "Train/test IDs overlap"


def document_key(r):
    return hashlib.sha256(
        " ".join(r["document"].split()).encode()
    ).hexdigest()


groups = sorted({document_key(r) for r in all_train})
if len(groups) < 2:
    raise ValueError("Need at least two distinct training documents.")

random.Random(CFG["seed"]).shuffle(groups)
nval = min(
    len(groups) - 1,
    max(1, round(len(groups) * CFG["val_fraction"])),
)
held = set(groups[:nval])

train_rows = [r for r in all_train if document_key(r) not in held]
val_rows = [r for r in all_train if document_key(r) in held]

random.Random(CFG["seed"]).shuffle(train_rows)
random.Random(CFG["seed"]).shuffle(val_rows)

train_rows = train_rows[:CFG["max_train"]]
val_rows = val_rows[:CFG["val_docs"]]

if SMOKE:
    test_rows = test_rows[:2]

save_json(
    {
        "train": [r["id"] for r in train_rows],
        "validation": [r["id"] for r in val_rows],
    },
    WORK / "split.json",
)

PROVENANCE = {
    p: digest(DATA / p)
    for p in ("Train/train.json", "Test/test.json")
}
save_json(PROVENANCE, WORK / "data_hashes.json")

print(
    "Used train / held-out validation / test:",
    len(train_rows), len(val_rows), len(test_rows),
)






PREFIX=tokenizer.encode('Legal document:\n',add_special_tokens=False).ids
SUFFIX=tokenizer.encode('\nWrite a concise legal summary:\n',add_special_tokens=False).ids
def select_source(ids,budget,strategy):
    if len(ids)<=budget: return ids
    if strategy=='head': return ids[:budget]
    if strategy!='spread': raise ValueError('context_strategy must be head or spread')
    a=budget//3; b=budget//3; c=budget-a-b
    mid=(len(ids)-b)//2
    return ids[:a]+ids[mid:mid+b]+ids[-c:]
def encode_row(row,labelled=True):
    ids=tokenizer.encode(row['document'],add_special_tokens=False).ids
    budget=CFG['source_tokens']-1-len(PREFIX)-len(SUFFIX)
    if budget<3: raise ValueError('Increase source_tokens to fit the instruction.')
    src=[BOS]+PREFIX+select_source(ids,budget,CFG['context_strategy'])+SUFFIX
    out={'id':row['id'],'source':src,'coverage':min(1,budget/max(1,len(ids)))}
    if labelled:
        target=tokenizer.encode(row['summary'],add_special_tokens=False).ids
        out.update(target=target[:CFG['target_tokens']-1]+[EOS],truncated=len(target)>=CFG['target_tokens'])
    return out
train_enc=[encode_row(r) for r in train_rows]
val_enc=[encode_row(r) for r in val_rows]
test_enc=[encode_row(r,False) for r in test_rows]
save_json({s:{'documents':len(rows),'mean_source_coverage':float(np.mean([r['coverage'] for r in rows])),
              'truncated_targets':sum(r.get('truncated',False) for r in rows)}
           for s,rows in [('train',train_enc),('validation',val_enc),('test',test_enc)]},WORK/'coverage.json')
def tensors(row):
    seq=row['source']+row['target']
    x=torch.tensor([seq[:-1]],device=DEVICE)
    y=torch.tensor([seq[1:]],device=DEVICE)
    y[:,:len(row['source'])-1]=-100
    return x,y






class LowRank(nn.Module):
    def __init__(self,base,multiplicative):
        super().__init__(); self.base=base; self.multiplicative=multiplicative
        self.scale=CFG['alpha']/CFG['rank']
        self.A=nn.Parameter(torch.empty(CFG['rank'],base.out_features if multiplicative else base.in_features))
        self.B=nn.Parameter(torch.zeros(base.out_features,CFG['rank']))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
    def forward(self,x):
        z=F.linear(x,self.base.weight)
        z=z+self.scale*F.linear(F.linear(z if self.multiplicative else x,self.A),self.B)
        return z if self.base.bias is None else z+self.base.bias
def inject(module,multiplicative):
    for name,child in list(module.named_children()):
        if isinstance(child,nn.Linear): setattr(module,name,LowRank(child,multiplicative))
        else: inject(child,multiplicative)
class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__(); self.norm=nn.LayerNorm(BASE_CFG['dim'])
        self.down=nn.Linear(BASE_CFG['dim'],CFG['adapter_dim']); self.up=nn.Linear(CFG['adapter_dim'],BASE_CFG['dim'])
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self,x): return x+self.up(F.gelu(self.down(self.norm(x))))
class AdaptedBlock(nn.Module):
    def __init__(self,base,two):
        super().__init__(); self.base=base; self.first=Bottleneck() if two else nn.Identity(); self.second=Bottleneck()
    def forward(self,x):
        b=self.base; z=b.n1(x); w=b.route_weights(z); mix=w[...,0:1]*b.attn(z)
        if b.memory is not None: mix=mix+w[...,1:2]*b.memory(z)
        if b.phrase is not None: mix=mix+w[...,2:3]*b.phrase(z)
        x=self.first(x+mix); a,c=b.up(b.n2(x)).chunk(2,-1)
        return self.second(x+b.down(F.silu(a)*c))
class Summarizer(nn.Module):
    def __init__(self,method):
        super().__init__(); self.method=method; self.lm=LegalLM(BASE_CFG)
        self.lm.load_state_dict(BASE['model'],strict=True)
        if method=='kd':
            n=max(1,int(len(self.lm.blocks)*CFG['student_fraction']))
            if n>=len(self.lm.blocks): raise ValueError('KD needs fewer student layers than teacher.')
            keep=np.linspace(0,len(self.lm.blocks)-1,n,dtype=int)
            self.lm.blocks=nn.ModuleList([self.lm.blocks[int(i)] for i in keep])
        else:
            self.lm.requires_grad_(False)
            if method in ('lora','lorma'): inject(self.lm.blocks,method=='lorma')
            elif method in ('houlsby','pfeiffer'):
                self.lm.blocks=nn.ModuleList([AdaptedBlock(b,method=='houlsby') for b in self.lm.blocks])
            elif method=='prompt':
                self.prompt=nn.Parameter(torch.empty(CFG['prompt_tokens'],BASE_CFG['dim']))
                nn.init.normal_(self.prompt,std=0.02)
            else: raise ValueError(method)
    def hidden(self,ids):
        x=self.lm.emb(ids); p=0
        if self.method=='prompt':
            p=len(self.prompt); x=torch.cat([self.prompt.unsqueeze(0).expand(len(ids),-1,-1),x],dim=1)
        if x.shape[1]>BASE_CFG['context']: raise ValueError('Context overflow')
        for b in self.lm.blocks:
            x=checkpoint(b,x,use_reentrant=False) if self.training else b(x)
        return self.lm.norm(x)[:,p:]
    def loss(self,x,y,teacher=None):
        h=self.hidden(x)[y!=-100]; gold=y[y!=-100]
        th=None
        if teacher is not None:
            with torch.no_grad(): th=teacher.hidden(x)[y!=-100]
        total=h.new_zeros((),dtype=torch.float32)
        for i in range(0,len(gold),32):
            def chunk_loss(hc,gc,tc):
                logits=F.linear(hc,self.lm.emb.weight).float()
                ce=F.cross_entropy(logits,gc,reduction='sum')
                if tc is None: return ce
                with torch.no_grad(): tl=F.linear(tc,teacher.lm.emb.weight).float()
                t=CFG['temperature']; kd=F.kl_div(F.log_softmax(logits/t,dim=-1),F.softmax(tl/t,dim=-1),reduction='sum')*t*t
                return (1-CFG['kd_weight'])*ce+CFG['kd_weight']*kd
            total=total+checkpoint(chunk_loss,h[i:i+32],gold[i:i+32],None if th is None else th[i:i+32],use_reentrant=False)
        return total/len(gold)






@torch.inference_mode()
def generate(model,row):
    model.eval(); generated=[]; ids=list(row['source'])
    for step in range(CFG['target_tokens']):
        with amp(): logits=F.linear(model.hidden(torch.tensor([ids],device=DEVICE))[:,-1],model.lm.emb.weight)[0].float()
        logits[PAD]=logits[BOS]=-float('inf')
        if step<CFG['min_new_tokens']: logits[EOS]=-float('inf')
        n=CFG['no_repeat_ngram']
        if n>0 and len(generated)>=n-1:
            prefix=tuple(generated[-(n-1):]) if n>1 else ()
            for j in range(len(generated)-n+1):
                if tuple(generated[j:j+n-1])==prefix: logits[generated[j+n-1]]=-float('inf')
        if not torch.isfinite(logits).any(): raise RuntimeError('No finite decoding logits')
        nxt=int(logits.argmax())
        if nxt==EOS: break
        generated.append(nxt); ids.append(nxt)
    text=tokenizer.decode(generated,skip_special_tokens=True).strip()
    if not text: raise RuntimeError(f'Empty generated summary for {row["id"]}; inspect checkpoint/decoding.')
    return text
def evaluate(model,folder):
    from rouge_score import rouge_scorer
    scorer=rouge_scorer.RougeScorer(['rouge1','rougeL'],use_stemmer=True)
    results=[]; values=[]; started=time.monotonic()
    for raw,enc in zip(val_rows,val_enc):
        pred=generate(model,enc); scores=scorer.score(raw['summary'],pred)
        values.append([scores[k].fmeasure for k in ('rouge1','rougeL')])
        results.append({'id':raw['id'],'reference':raw['summary'],'summary':pred})
    save_json(results,folder/'validation_predictions.json')
    means=np.mean(values,axis=0)
    return dict(rouge1=float(means[0]),rougeL=float(means[1]),validation_seconds=time.monotonic()-started)






def identity(method,teacher_method):
    obj=dict(version=1,code=CODE_HASH,method=method,teacher=teacher_method,cfg=CFG,smoke=SMOKE,
             base=BASE_SHA,tokenizer=TOK_SHA,data=PROVENANCE,
             train=[r['id'] for r in train_enc],val=[r['id'] for r in val_enc])
    if teacher_method: obj['teacher_hash']=digest(WORK/'runs'/teacher_method/'last.pt')
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def load_model(method):
    model=Summarizer(method)
    state=torch.load(WORK/'runs'/method/'last.pt',map_location='cpu',weights_only=False)
    if not state['done']: raise ValueError('Teacher/inference checkpoint is incomplete.')
    if state['identity']!=identity(method,state['teacher']): raise ValueError('Checkpoint identity mismatch')
    model.load_state_dict(state['model'],strict=True)
    return model.to(DEVICE).eval()
def train_one(method,teacher_method=None):
    seed_all(CFG['seed']); folder=WORK/'runs'/method; folder.mkdir(parents=True,exist_ok=True)
    ident=identity(method,teacher_method); model=Summarizer(method).to(DEVICE)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=CFG['lr_kd'] if method=='kd' else CFG['lr'],weight_decay=0.01)
    scaler=torch.amp.GradScaler('cuda',enabled=DEVICE.type=='cuda' and AMP_DTYPE==torch.float16)
    epoch=cursor=steps=0; elapsed=0.; done=False
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location=DEVICE,weights_only=False)
        if state['identity']!=ident: raise ValueError('Changed run configuration/inputs/teacher. Use a new WORK directory.')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        epoch,cursor,steps,elapsed,done=[state[k] for k in ('epoch','cursor','steps','elapsed','done')]
        torch.set_rng_state(state['rng'].cpu())
        if DEVICE.type=='cuda' and state['cuda_rng'] is not None: torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        del state
    teacher=load_model(teacher_method) if method=='kd' and not done else None
    if teacher is not None:
        assert sum(p.numel() for p in model.parameters())<sum(p.numel() for p in teacher.parameters())
    start=time.monotonic(); last_save=start; prior_elapsed=elapsed
    def persist():
        save_pt(dict(identity=ident,teacher=teacher_method,model=model.state_dict(),optimizer=opt.state_dict(),
                     scaler=scaler.state_dict(),epoch=epoch,cursor=cursor,steps=steps,elapsed=elapsed,done=done,
                     rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else None),folder/'last.pt')
    while not done and epoch<CFG['epochs']:
        order=list(range(len(train_enc))); random.Random(CFG['seed']+epoch).shuffle(order)
        while cursor<len(order):
            group=order[cursor:cursor+CFG['accumulation']]; opt.zero_grad(set_to_none=True); model.train()
            losses=[]
            for idx in group:
                x,y=tensors(train_enc[idx])
                with amp(): loss=model.loss(x,y,teacher)
                if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss: {method}')
                losses.append(float(loss.detach())); scaler.scale(loss/len(group)).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params,1.0)
            scaler.step(opt); scaler.update(); cursor+=len(group); steps+=1
            elapsed=prior_elapsed+time.monotonic()-start
            if steps%20==0 or steps==1: print(method,'epoch',epoch+1,'documents',cursor,'loss',round(np.mean(losses),4),flush=True)
            if elapsed>=CFG['minutes']*60: done=True
            if done or time.monotonic()-last_save>=CFG['checkpoint_seconds']: persist(); last_save=time.monotonic()
            if done: break
        if not done: epoch+=1; cursor=0; done=epoch>=CFG['epochs']; persist()
    if teacher is not None: del teacher
    gc.collect()
    if DEVICE.type=='cuda': torch.cuda.empty_cache()
    metrics=evaluate(model,folder)
    metrics.update(method=method,teacher=teacher_method,train_seconds=elapsed,steps=steps,
                   total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in params))
    save_json(metrics,folder/'metrics.json'); del model,opt,params; gc.collect()
    if DEVICE.type=='cuda': torch.cuda.empty_cache()
    return metrics



assert len(EXPERIMENTS)==len(set(EXPERIMENTS)) and set(EXPERIMENTS)<=set(['lora','lorma','houlsby','pfeiffer','prompt','kd'])
results=[]
for method in [m for m in EXPERIMENTS if m!='kd']:
    results.append(train_one(method))
if 'kd' in EXPERIMENTS:
    candidates=[r for r in results if r['method']!='kd']
    if not candidates:
        raise ValueError('Include a teacher method in EXPERIMENTS, e.g. ["lorma","kd"]. Completed matching runs are reused.')
    teacher_method=max(candidates,key=lambda r:r['rougeL'])['method']
    results.append(train_one('kd',teacher_method))
save_json(results,WORK/'comparison.json')
with open(WORK/'comparison.csv','w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
for row in sorted(results,key=lambda r:r['rougeL'],reverse=True): print(row)






def validate_predictions(predictions,ids):
    if not isinstance(predictions,list) or len(predictions)!=len(ids): raise ValueError('Wrong prediction count')
    if [r.get('id') for r in predictions]!=ids: raise ValueError('Missing, reordered, or duplicate test IDs')
    for r in predictions:
        if set(r)!= {'id','summary'} or not isinstance(r['summary'],str) or not r['summary'].strip(): raise ValueError('Invalid summary record')
def export(method):
    folder=WORK/'runs'/method
    tag=hashlib.sha256((digest(folder/'last.pt')+PROVENANCE['Test/test.json']+str(SMOKE)).encode()).hexdigest()
    progress_path=folder/'test_progress.json'; predictions=[]
    if progress_path.exists():
        progress=json.loads(progress_path.read_text(encoding='utf-8'))
        if progress['identity']!=tag: raise ValueError('Stale generation cache; use a new output directory.')
        predictions=progress['predictions']
        validate_predictions(predictions,[r['id'] for r in test_enc[:len(predictions)]])
    model=load_model(method); started=time.monotonic()
    for row in test_enc[len(predictions):]:
        predictions.append({'id':row['id'],'summary':generate(model,row)})
        if len(predictions)%10==0:
            save_json({'identity':tag,'predictions':predictions},progress_path)
            print(method,'test',len(predictions),'/',len(test_enc),flush=True)
    validate_predictions(predictions,[r['id'] for r in test_enc])
    save_json({'identity':tag,'predictions':predictions},progress_path)
    save_json(predictions,folder/'prediction.json')
    output=folder/('SMOKE_NOT_FOR_SUBMISSION.zip' if SMOKE else 'submission.zip')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z: z.write(folder/'prediction.json','prediction.json')
    with zipfile.ZipFile(output) as z:
        assert z.namelist()==['prediction.json']
        validate_predictions(json.loads(z.read('prediction.json')),[r['id'] for r in test_enc])
    save_json({'generation_seconds_this_invocation':time.monotonic()-started,
               'documents':len(predictions),'checkpoint_hash':digest(folder/'last.pt')},folder/'inference.json')
    print('Saved:',output,'| generation seconds this invocation:',round(time.monotonic()-started))
    del model; gc.collect()
    if DEVICE.type=='cuda': torch.cuda.empty_cache()
    return str(output)
exports=[export(r['method']) for r in results]
save_json({'smoke':SMOKE,'methods':[r['method'] for r in results],'exports':exports,
           'six_methods_complete':len(results)==6,'local_primary_metric':'not computed; official BERTScore requires evaluator',
           'architecture':'LEXWeave only','context_strategy':CFG['context_strategy']},WORK/'completion.json')
print('Download the ZIPs above and submit manually. Preserve the full WORK directory for resume.')







