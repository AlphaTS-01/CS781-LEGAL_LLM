


import importlib.util, subprocess, sys
requirements = {'numpy': 'numpy>=1.26,<3', 'tokenizers': 'tokenizers>=0.20,<1',
                'huggingface_hub': 'huggingface_hub>=0.26,<2'}
missing = [v for k, v in requirements.items() if importlib.util.find_spec(k) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
assert importlib.util.find_spec('torch'), 'Select a PyTorch GPU runtime.'






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
CODE_HASH = 'a068fe264039cf690c911d7aa153121b051429b378fb984933197c07b4228930'
WORK = Path('/kaggle/working/phase7_pcr' if Path('/kaggle').exists() else 'phase7_pcr')
FOUNDATION_ARCHIVE = None
FOUNDATION_DIR = '/kaggle/input/models/tejass73/lexyp1/pytorch/default/1'
LOCAL_DATA = None  # Path to Phase_7_PCR, containing Train/, Dev/, Test/
RESUME_FROM = None
EXPERIMENTS = ['full', 'partial', 'lorma', 'prompt', 'ia3', 'kd']
CFG = dict(seed=42, epochs=2, max_train=2000, minutes=45, accumulation=2,
           chunk_tokens=256, max_chunks=2, chunk_strategy='spread', pooling='mean',
           prompt_tokens=16, rank=8, alpha=8, partial_layers=2,
           negatives=7, lr=3e-4, lr_full=1e-5, temperature=0.07,
           kd_temperature=2.0, kd_weight=0.5, student_fraction=0.5,
           checkpoint_seconds=300, index_save_every=100)
if SMOKE:
    CFG.update(epochs=1, max_train=4, minutes=2, chunk_tokens=48, max_chunks=1,
               prompt_tokens=4, negatives=2, accumulation=1)
assert CFG['epochs'] > 0 and CFG['minutes'] > 0 and CFG['accumulation'] > 0
assert CFG['max_chunks'] > 0 and CFG['chunk_tokens'] > 2 and CFG['negatives'] > 0
assert CFG['temperature'] > 0 and CFG['kd_temperature'] > 0 and 0 <= CFG['kd_weight'] <= 1
assert CFG['pooling'] in ('mean', 'last') and CFG['chunk_strategy'] in ('head', 'spread')
assert CFG['max_train'] is None or CFG['max_train'] > 0
assert EXPERIMENTS and len(EXPERIMENTS) == len(set(EXPERIMENTS))
assert set(EXPERIMENTS) <= {'full', 'partial', 'lorma', 'prompt', 'ia3', 'kd'}
if RESUME_FROM is not None:
    if WORK.exists(): raise ValueError('Choose a new WORK when restoring saved output.')
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
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()
def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8'); tmp.replace(path)
def save_pt(obj, path):
    tmp = Path(str(path) + '.tmp'); torch.save(obj, tmp); tmp.replace(path)
def sync():
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
def cleanup():
    gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
print('Device:', DEVICE, '| Smoke:', SMOKE)
if DEVICE.type == 'cpu' and not SMOKE: print('Use a GPU for the real experiments.')






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
assert CFG['chunk_tokens']+CFG['prompt_tokens'] <= BASE_CFG['context']
BASE_SHA=digest(FOUNDATION/'model.pt'); TOK_SHA=digest(FOUNDATION/'tokenizer.json')
probe=LegalLM(BASE_CFG); probe.load_state_dict(BASE['model'],strict=True)
print('Strict Phase 1 load passed:',sum(p.numel() for p in probe.parameters()),'parameters')
del probe; gc.collect()






FILES = {s: (f'{s}/{s.lower()}.json', f'{s}/{s.lower()}_candidates.json') for s in ('Train', 'Dev', 'Test')}
def resolve_data():
    if LOCAL_DATA is not None: return Path(LOCAL_DATA)
    roots = [Path('/kaggle/input'), Path('/kaggle/working/capstone_data'), Path.cwd()]
    hits = sorted({p.parent.parent.resolve() for r in roots if r.exists()
                   for p in r.glob('**/Phase_7_PCR/Train/train.json')})
    if len(hits) == 1: return hits[0]
    if len(hits) > 1: raise ValueError(f'Set LOCAL_DATA explicitly: {hits}')
    from huggingface_hub import HfApi, snapshot_download
    from getpass import getpass
    token = os.environ.get('HF_TOKEN')
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:
            pass
    if not token: token = getpass('HF read token with dataset access: ')
    repo = 'Exploration-Lab/CS-781-Capstone'
    revision = HfApi().dataset_info(repo, token=token).sha
    folder = snapshot_download(repo_id=repo, repo_type='dataset', revision=revision,
                               token=token, allow_patterns=['Phase_7_PCR/**'])
    save_json({'repository': repo, 'revision': revision}, WORK / 'download.json')
    return Path(folder) / 'Phase_7_PCR'
def valid_id(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{where}: expected nonempty string ID (preserve leading zeros), got {value!r}')
    return value
def document_text(value, where):
    if isinstance(value, list) and all(isinstance(x, str) for x in value): value = '\n'.join(value)
    if not isinstance(value, str) or not value.strip(): raise ValueError(f'{where}: expected text or sentence-string list')
    return value
def load_records(path, labelled=False):
    raw = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if isinstance(raw, dict):
        entries = []
        for key, value in raw.items():
            if isinstance(value, dict):
                if 'id' in value and value['id'] != key: raise ValueError(f'{path}: key/id conflict for {key}')
                entries.append(dict(value, id=key))
            else: entries.append({'id': key, 'text': value})
    elif isinstance(raw, list): entries = raw
    else: raise ValueError(f'{path}: expected ID-keyed object or record list')
    result = {}
    for i, row in enumerate(entries):
        where = f'{path}, row {i}'
        if not isinstance(row, dict): raise ValueError(f'{where}: expected object')
        key = valid_id(row.get('id'), where)
        if key in result: raise ValueError(f'{where}: duplicate ID {key}')
        obj = {'id': key, 'text': document_text(row.get('text'), where)}
        if labelled:
            labels = row.get('relevant_candidates')
            missing = labels is None or (
                isinstance(labels, str) and not labels.strip()
            )

            if missing:
                print(f'Skipping {where}, ID {key}: missing relevance labels')
                continue

            if not isinstance(labels, list):
                raise ValueError(
                    f'{where}, ID {key}: expected a list of candidate IDs, '
                    f'got {type(labels).__name__}: {repr(labels)[:120]}'
                )

            if not labels and Path(path).name == 'train.json':
                print(f'Skipping {where}, ID {key}: no positive training candidate')
                continue

            labels = [valid_id(x, where) for x in labels]
            if len(labels) != len(set(labels)):
                raise ValueError(f'{where}: duplicate relevance labels')

            obj['relevant_candidates'] = labels

        result[key] = obj
    if not result: raise ValueError(f'{path}: empty records')
    return result
DATA = resolve_data()
SPLITS = {}; PROVENANCE = {}
for split, (qfile, cfile) in FILES.items():
    queries = load_records(DATA / qfile, split != 'Test'); candidates = load_records(DATA / cfile)
    if len(candidates) < 7: raise ValueError(f'{split}: need at least seven candidates')
    for q in queries.values():
        missing = set(q.get('relevant_candidates', [])) - candidates.keys()
        if missing: raise ValueError(f'{split}/{q["id"]}: labels outside split pool: {missing}')
        if split == 'Train' and len(q['relevant_candidates']) == len(candidates):
            raise ValueError(f'{q["id"]}: no eligible negative candidates')
    SPLITS[split] = dict(queries=queries, candidates=candidates)
    for file in (qfile, cfile): PROVENANCE[file] = digest(DATA / file)
save_json(PROVENANCE, WORK / 'inputs.json')
train_ids = sorted(SPLITS['Train']['queries']); random.Random(CFG['seed']).shuffle(train_ids)
if CFG['max_train'] is not None: train_ids = train_ids[:CFG['max_train']]
dev_ids = list(SPLITS['Dev']['queries']); test_ids = list(SPLITS['Test']['queries'])
if SMOKE: dev_ids = dev_ids[:2]; test_ids = test_ids[:2]
save_json({'train': train_ids, 'dev': dev_ids, 'test': test_ids, 'smoke': SMOKE}, WORK / 'query_ids.json')
print({s: {k: len(v) for k, v in data.items()} for s, data in SPLITS.items()})






def encode_text(text):
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids: raise ValueError('Document tokenized to zero tokens')
    width = CFG['chunk_tokens'] - 2
    n = math.ceil(len(ids) / width)
    selected = list(range(min(n, CFG['max_chunks']))) if CFG['chunk_strategy'] == 'head' else sorted(set(np.linspace(0, n-1, min(n, CFG['max_chunks']), dtype=int).tolist()))
    chunks = [[BOS] + ids[j*width:(j+1)*width] + [EOS] for j in selected]
    return {'chunks': chunks, 'coverage': sum(len(c)-2 for c in chunks) / len(ids)}
ENC = {}; coverage = {}
for split, data in SPLITS.items():
    ENC[split] = {}
    for role, rows in data.items():
        ENC[split][role] = {key: encode_text(row['text']) for key, row in rows.items()}
        values = [r['coverage'] for r in ENC[split][role].values()]
        coverage[f'{split}/{role}'] = dict(documents=len(values), mean_coverage=float(np.mean(values)), truncated=sum(v < 1 for v in values))
save_json(coverage, WORK / 'coverage.json'); print(coverage)






class LoRMA(nn.Module):
    def __init__(self, base):
        super().__init__(); self.base = base; self.scale = CFG['alpha'] / CFG['rank']
        self.A = nn.Parameter(torch.empty(CFG['rank'], base.out_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, CFG['rank']))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    def forward(self, x):
        z = F.linear(x, self.base.weight)
        z = z + self.scale * F.linear(F.linear(z, self.A), self.B)
        return z if self.base.bias is None else z + self.base.bias
def inject_lorma(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear): setattr(module, name, LoRMA(child))
        else: inject_lorma(child)
class ScaleLinear(nn.Module):
    def __init__(self, base, on_input=False):
        super().__init__(); self.base = base; self.on_input = on_input
        self.scale = nn.Parameter(torch.ones(base.in_features if on_input else base.out_features))
    def forward(self, x):
        return self.base(x * self.scale) if self.on_input else self.base(x) * self.scale
class Retriever(nn.Module):
    def __init__(self, method):
        super().__init__(); self.method = method; self.lm = LegalLM(BASE_CFG)
        self.lm.load_state_dict(BASE['model'], strict=True)
        if method == 'kd':
            n = max(1, int(len(self.lm.blocks) * CFG['student_fraction']))
            if n >= len(self.lm.blocks): raise ValueError('Student must have fewer layers')
            keep = np.linspace(0, len(self.lm.blocks)-1, n, dtype=int)
            self.lm.blocks = nn.ModuleList([self.lm.blocks[int(i)] for i in keep])
        elif method != 'full':
            self.lm.requires_grad_(False)
            if method == 'partial':
                if not 0 < CFG['partial_layers'] < len(self.lm.blocks): raise ValueError('partial_layers must be between 1 and layers-1')
                for b in self.lm.blocks[-CFG['partial_layers']:]: b.requires_grad_(True)
                self.lm.norm.requires_grad_(True)
            elif method == 'lorma': inject_lorma(self.lm.blocks)
            elif method == 'prompt':
                self.prompt = nn.Parameter(torch.empty(CFG['prompt_tokens'], BASE_CFG['dim']))
                nn.init.normal_(self.prompt, std=0.02)
            elif method == 'ia3':
                for b in self.lm.blocks:
                    b.attn.kv = ScaleLinear(b.attn.kv)
                    if b.memory is not None: b.memory.kv = ScaleLinear(b.memory.kv)
                    b.down = ScaleLinear(b.down, on_input=True)
            else: raise ValueError(method)
    def chunk_vector(self, ids):
        x = self.lm.emb(ids); offset = 0
        if self.method == 'prompt':
            offset = len(self.prompt); x = torch.cat([self.prompt[None], x], dim=1)
        if x.shape[1] > BASE_CFG['context']: raise ValueError('Context overflow')
        for b in self.lm.blocks: x = b(x)
        x = self.lm.norm(x)[:, offset:]
        # Mean excludes BOS/EOS and any soft prompt; last includes the final EOS.
        return x[:, 1:-1].float().mean(1) if CFG['pooling'] == 'mean' else x[:, -1].float()
    def forward(self, encoded):
        vectors = []
        for chunk in encoded['chunks']:
            ids = torch.tensor([chunk], device=DEVICE)
            v = checkpoint(self.chunk_vector, ids, use_reentrant=False) if self.training else self.chunk_vector(ids)
            vectors.append(v)
        return F.normalize(torch.stack(vectors).mean(0), dim=-1)[0]






def metrics_at7(predictions, queries):
    tp = sum(len(set(predictions[k][:7]) & set(queries[k]['relevant_candidates'])) for k in predictions)
    predicted = sum(len(predictions[k][:7]) for k in predictions)
    gold = sum(len(set(queries[k]['relevant_candidates'])) for k in predictions)
    p = tp / predicted if predicted else 0.; r = tp / gold if gold else 0.
    return dict(micro_f1_at7=2*tp/(predicted+gold) if predicted+gold else 0.,
                micro_precision_at7=p, micro_recall_at7=r, true_positives=tp, gold_relevant=gold, queries=len(predictions))
def validate_predictions(predictions, query_ids, candidate_ids):
    if not isinstance(predictions, dict) or set(predictions) != set(query_ids): raise ValueError('Missing/extra query IDs')
    pool = set(candidate_ids)
    for key, values in predictions.items():
        if not isinstance(values, list) or len(values) != 7 or not all(isinstance(x, str) for x in values): raise ValueError(f'{key}: need seven string IDs')
        if len(set(values)) != 7 or not set(values) <= pool: raise ValueError(f'{key}: duplicate/out-of-pool candidates')
@torch.inference_mode()
def rank_split(model, split, query_ids, folder, cache_tag):
    model.eval(); ids = sorted(ENC[split]['candidates']); path = folder / f'{split.lower()}_index.pt'
    vectors = []; done = 0
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['identity'] != cache_tag or state['ids'] != ids: raise ValueError('Stale candidate index')
        vectors = list(state['vectors'].unbind(0)); done = len(vectors)
        if done > len(ids): raise ValueError('Invalid candidate index length')
        del state
    for key in ids[done:]:
        with amp(): vector = model(ENC[split]['candidates'][key])
        if not torch.isfinite(vector).all(): raise RuntimeError('Non-finite candidate embedding')
        vectors.append(vector.float().cpu())
        if len(vectors) % CFG['index_save_every'] == 0:
            save_pt(dict(identity=cache_tag, ids=ids, vectors=torch.stack(vectors)), path)
            print(split, 'indexed', len(vectors), '/', len(ids), flush=True)
    matrix = torch.stack(vectors)
    save_pt(dict(identity=cache_tag, ids=ids, vectors=matrix), path)
    progress_path = folder / f'{split.lower()}_rankings.json'; predictions = {}
    if progress_path.exists():
        state = json.loads(progress_path.read_text(encoding='utf-8'))
        if state['identity'] != cache_tag: raise ValueError('Stale query rankings')
        predictions = state['predictions']
        if not set(predictions) <= set(query_ids): raise ValueError('Stale query IDs')
        validate_predictions(predictions, list(predictions), ids)
    for key in query_ids:
        if key in predictions: continue
        with amp(): q = model(ENC[split]['queries'][key])
        scores = (matrix @ q.float().cpu()).numpy()
        if not np.isfinite(scores).all(): raise RuntimeError('Non-finite retrieval scores')
        order = np.argsort(-scores, kind='stable')[:7]
        predictions[key] = [ids[int(i)] for i in order]
        if len(predictions) % 10 == 0:
            save_json(dict(identity=cache_tag, predictions=predictions), progress_path)
    validate_predictions(predictions, query_ids, ids)
    save_json(dict(identity=cache_tag, predictions=predictions), progress_path)
    return predictions






def identity(method, teacher_method):
    obj = dict(code=CODE_HASH, method=method, teacher=teacher_method, cfg=CFG, smoke=SMOKE,
               base=BASE_SHA, tokenizer=TOK_SHA, data=PROVENANCE, train=train_ids, dev=dev_ids, test=test_ids)
    if teacher_method: obj['teacher_hash'] = digest(WORK / 'runs' / teacher_method / 'last.pt')
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
def load_model(method):
    state = torch.load(WORK / 'runs' / method / 'last.pt', map_location='cpu', weights_only=False)
    if not state['done'] or state['identity'] != identity(method, state['teacher']): raise ValueError('Incomplete/stale checkpoint')
    model = Retriever(method); model.load_state_dict(state['model'], strict=True)
    return model.to(DEVICE).eval()
def sample_candidates(key, rng):
    labels = SPLITS['Train']['queries'][key]['relevant_candidates']; positives = set(labels)
    negatives = [c for c in SPLITS['Train']['candidates'] if c not in positives]
    return [rng.choice(labels)] + rng.sample(negatives, min(CFG['negatives'], len(negatives)))
def ranking_logits(model, key, candidates):
    q = model(ENC['Train']['queries'][key])
    c = torch.stack([model(ENC['Train']['candidates'][i]) for i in candidates])
    return (c @ q) / CFG['temperature']
def retrieval_loss(model, teacher, key, candidates):
    logits = ranking_logits(model, key, candidates).float()
    ce = F.cross_entropy(logits[None], torch.zeros(1, dtype=torch.long, device=DEVICE))
    if teacher is None: return ce
    with torch.no_grad(): target = ranking_logits(teacher, key, candidates).float()
    t = CFG['kd_temperature']
    kd = F.kl_div(F.log_softmax(logits/t, -1), F.softmax(target/t, -1), reduction='sum') * t*t
    return (1-CFG['kd_weight'])*ce + CFG['kd_weight']*kd
def train_one(method, teacher_method=None):
    seed_all(CFG['seed']); folder = WORK / 'runs' / method; folder.mkdir(parents=True, exist_ok=True)
    ident = identity(method, teacher_method); model = Retriever(method).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG['lr_full'] if method in ('full', 'partial', 'kd') else CFG['lr'], weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda' and AMP_DTYPE == torch.float16)
    epoch = cursor = steps = 0; elapsed = 0.; done = False
    if (folder / 'last.pt').exists():
        state = torch.load(folder / 'last.pt', map_location=DEVICE, weights_only=False)
        if state['identity'] != ident: raise ValueError('Changed configuration/inputs/teacher; use a new WORK.')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        epoch, cursor, steps, elapsed, done = [state[k] for k in ('epoch', 'cursor', 'steps', 'elapsed', 'done')]
        torch.set_rng_state(state['rng'].cpu())
        if DEVICE.type == 'cuda' and state['cuda_rng'] is not None: torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        del state
    teacher = load_model(teacher_method) if method == 'kd' and not done else None
    if teacher is not None:
        teacher.requires_grad_(False)
        assert sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in teacher.parameters())
    if DEVICE.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    sync(); start = time.monotonic(); last_save = start; prior_elapsed = elapsed
    def persist():
        save_pt(dict(identity=ident, teacher=teacher_method, model=model.state_dict(), optimizer=opt.state_dict(),
                     scaler=scaler.state_dict(), epoch=epoch, cursor=cursor, steps=steps, elapsed=elapsed, done=done,
                     rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if DEVICE.type == 'cuda' else None), folder / 'last.pt')
    while not done and epoch < CFG['epochs']:
        order = list(train_ids); random.Random(CFG['seed']+epoch).shuffle(order)
        while cursor < len(order):
            group = order[cursor:cursor+CFG['accumulation']]; opt.zero_grad(set_to_none=True); model.train(); losses = []
            for offset, key in enumerate(group):
                rng = random.Random(CFG['seed'] + epoch*len(order) + cursor + offset)
                candidates = sample_candidates(key, rng)
                with amp(): loss = retrieval_loss(model, teacher, key, candidates)
                if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss: {method}')
                losses.append(float(loss.detach())); scaler.scale(loss/len(group)).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            scaler.step(opt); scaler.update(); cursor += len(group); steps += 1
            sync(); elapsed = prior_elapsed + time.monotonic() - start
            if steps == 1 or steps % 20 == 0: print(method, 'epoch', epoch+1, 'queries', cursor, 'loss', round(float(np.mean(losses)), 4), flush=True)
            if elapsed >= CFG['minutes']*60: done = True
            if done or time.monotonic()-last_save >= CFG['checkpoint_seconds']: persist(); last_save = time.monotonic()
            if done: break
        if not done: epoch += 1; cursor = 0; done = epoch >= CFG['epochs']; persist()
    peak = torch.cuda.max_memory_allocated() if DEVICE.type == 'cuda' else None
    del teacher, opt; cleanup()
    sync(); started = time.monotonic()
    predictions = rank_split(model, 'Dev', dev_ids, folder, digest(folder / 'last.pt') + ident + 'Dev')
    sync(); metrics = metrics_at7(predictions, SPLITS['Dev']['queries'])
    metrics.update(method=method, teacher=teacher_method, train_seconds=elapsed, steps=steps,
                   dev_seconds_this_invocation=time.monotonic()-started, peak_cuda_bytes_this_invocation=peak,
                   total_parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in params))
    save_json(metrics, folder / 'metrics.json')
    save_json([dict(id=k, relevant=SPLITS['Dev']['queries'][k]['relevant_candidates'], retrieved=v,
                    missed=sorted(set(SPLITS['Dev']['queries'][k]['relevant_candidates'])-set(v))) for k, v in predictions.items()], folder / 'dev_errors.json')
    del model, params; cleanup(); return metrics



results = []
for method in [m for m in EXPERIMENTS if m != 'kd']: results.append(train_one(method))
if 'kd' in EXPERIMENTS:
    if not results: raise ValueError('Include a teacher method, e.g. EXPERIMENTS=["lorma", "kd"].')
    teacher_method = max(results, key=lambda r: r['micro_f1_at7'])['method']
    results.append(train_one('kd', teacher_method))
save_json(results, WORK / 'comparison.json')
with open(WORK / 'comparison.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
for result in sorted(results, key=lambda r: r['micro_f1_at7'], reverse=True): print(result)






def export(method):
    folder = WORK / 'runs' / method; model = load_model(method)
    state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
    tag = digest(folder / 'last.pt') + state['identity'] + 'Test'; del state
    sync(); started = time.monotonic()
    predictions = rank_split(model, 'Test', test_ids, folder, tag)
    validate_predictions(predictions, test_ids, SPLITS['Test']['candidates'])
    save_json(predictions, folder / 'prediction.json')
    output = folder / ('SMOKE_NOT_FOR_SUBMISSION.zip' if SMOKE else 'submission.zip')
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z: z.write(folder / 'prediction.json', 'prediction.json')
    with zipfile.ZipFile(output) as z:
        assert z.namelist() == ['prediction.json']
        validate_predictions(json.loads(z.read('prediction.json')), test_ids, SPLITS['Test']['candidates'])
    sync(); save_json({'seconds_this_invocation': time.monotonic()-started, 'queries': len(test_ids),
                      'candidates': len(SPLITS['Test']['candidates']), 'cache_may_be_reused': True}, folder / 'inference.json')
    del model; cleanup(); print('Saved:', output); return str(output)
exports = [export(r['method']) for r in results]
save_json({'smoke': SMOKE, 'exports': exports, 'six_methods_complete': len(results) == 6,
           'metric': 'Dev Micro-F1@7', 'architecture': 'LEXWeave',
           'official_test_scores': 'Requires manual submission to official evaluator'}, WORK / 'completion.json')
print('Preserve the full WORK directory. Submit each real submission.zip manually.')






from pathlib import Path

work = Path(globals().get("WORK", "/kaggle/working/phase7_pcr"))
print("Checking", work)

for method in ("full", "partial", "lorma", "prompt", "ia3", "kd"):
    folder = work / "runs" / method
    for name in ("submission.zip", "SMOKE_NOT_FOR_SUBMISSION.zip",
                 "prediction.json", "test_rankings.json", "last.pt"):
        path = folder / name
        if path.is_file():
            print(f"{method:8} {path}")
            break
    else:
        print(f"{method:8} No recovery file found")


from pathlib import Path

names = {
    "submission.zip", "prediction.json", "test_rankings.json",
    "last.pt", "SMOKE_NOT_FOR_SUBMISSION.zip",
}

for root in (Path("/kaggle/working"), Path("/kaggle/input")):
    if root.is_dir():
        hits = sorted(p for p in root.rglob("*") if p.name in names and p.is_file())
        print(f"\n{root}: {len(hits)} matching files")
        for p in hits:
            print(p)

print("\nOriginal export paths:", globals().get("exports", "unavailable"))


# Capture any restored scores BEFORE overwriting results.
previous_results = globals().get("results", [])
eligible = [
    row for row in previous_results
    if isinstance(row, dict)
    and row.get("method") in {"full", "partial", "lorma", "prompt", "ia3"}
    and isinstance(row.get("micro_f1_at7"), (int, float))
    and math.isfinite(row["micro_f1_at7"])
]

if SMOKE:
    raise ValueError("Set SMOKE = False and rerun setup before training.")

if eligible:
    winner = max(eligible, key=lambda row: row["micro_f1_at7"])
    method = winner["method"]
    print(f"Selected {method} from saved Dev scores: {winner['micro_f1_at7']:.4f}")
else:
    method = "lorma"  # Set this to the winner if you can see your old printed scores.
    print(f"No previous Dev scores available. Using {method}.")

results = [train_one(method)]
save_json(results, WORK / "comparison.json")
print(f"Finished {method}. Dev Micro-F1@7 = {results[0]['micro_f1_at7']:.4f}")


# Paste this entire file into ONE Kaggle cell after the prime model's Section 8.
# Requires the Phase 7 notebook variables, prime checkpoint, Dev/Test indexes and rankings.
def make_five_variants(prime_method, training_queries=64, change_fraction=0.05):
    methods = ['full', 'partial', 'lorma', 'prompt', 'ia3', 'kd']
    if prime_method not in methods[:-1]:
        raise ValueError('Choose the completed non-KD prime method as the teacher.')
    if not 0 <= change_fraction <= 1 or training_queries < 1:
        raise ValueError('Check training_queries and change_fraction.')
    prime_dir = WORK / 'runs' / prime_method
    prime_zip = prime_dir / ('SMOKE_NOT_FOR_SUBMISSION.zip' if SMOKE else 'submission.zip')
    if not prime_zip.is_file():
        raise FileNotFoundError(f'Finish the prime export in Section 8 first: {prime_zip}')
    teacher = load_model(prime_method).requires_grad_(False)
    prime_hash = digest(prime_dir / 'last.pt')
    cache_prefix = prime_hash + identity(prime_method, None)
    base_rankings = {}
    shortlists = {}
    for split, keys in [('Dev', dev_ids), ('Test', test_ids)]:
        split_dir = prime_dir / split.lower()
        index = torch.load(str(split_dir) + '_index.pt', map_location='cpu', weights_only=True)
        saved = json.loads(Path(str(split_dir) + '_rankings.json').read_text())
        if index['identity'] != cache_prefix + split or saved['identity'] != index['identity']:
            raise ValueError(f'{split}: rerun prime evaluation/export to refresh its saved rankings.')
        ids, matrix = index['ids'], index['vectors']
        rankings = saved['predictions']
        validate_predictions(rankings, keys, SPLITS[split]['candidates'])
        if ids != sorted(SPLITS[split]['candidates']) or len(matrix) != len(ids):
            raise ValueError(f'{split}: incomplete prime index')
        # Keep the queries with the smallest gap between seventh and eighth place.
        margins = []
        if len(ids) > 7:
            with torch.inference_mode():
                for key in keys:
                    with amp():
                        vector = teacher(ENC[split]['queries'][key])
                    scores = (matrix @ vector.float().cpu()).numpy()
                    order = np.argsort(-scores, kind='stable')[:10]
                    top = [ids[int(i)] for i in order]
                    if top[:7] != rankings[key]:
                        raise ValueError(f'{split}/{key}: prime scores disagree with saved rankings')
                    margins.append((float(scores[order[6]] - scores[order[7]]), key, top[6:]))
        limit = math.floor(len(keys) * change_fraction)
        shortlists[split] = {key: choices for _, key, choices in sorted(margins)[:limit]}
        base_rankings[split] = rankings
    del matrix, index
    with zipfile.ZipFile(prime_zip) as archive:
        if json.loads(archive.read('prediction.json')) != base_rankings['Test']:
            raise ValueError('Prime submission does not match saved test rankings.')

    train_sample = random.Random(CFG['seed']).sample(train_ids, min(training_queries, len(train_ids)))
    # Key the output folder by the prime and training settings for safe reuse.
    signature = hashlib.sha256(json.dumps([prime_hash, train_sample, change_fraction, 'v1']).encode()).hexdigest()[:16]
    root = WORK / 'rerankers' / f'{prime_method}-{signature}'
    root.mkdir(parents=True, exist_ok=True)
    save_json(train_sample, root / 'training_queries.json')
    zips = {prime_method: prime_zip}
    baseline = metrics_at7(base_rankings['Dev'], SPLITS['Dev']['queries'])['micro_f1_at7']
    print('Prime Dev Micro-F1@7:', baseline, flush=True)
    for method in methods:
        if method == prime_method:
            continue
        folder = root / method
        folder.mkdir(exist_ok=True)
        seed_all(CFG['seed'])
        model = Retriever(method).to(DEVICE)
        weights = folder / 'trained_parameters.pt'
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if weights.exists():
            state = torch.load(weights, map_location='cpu', weights_only=True)
            if set(state) != set(params):
                raise ValueError(f'{method}: checkpoint parameters differ')
            with torch.no_grad():
                for name, value in state.items():
                    params[name].copy_(value)
            del state
        else:
            lr = CFG['lr_full'] if method in ('full', 'partial', 'kd') else CFG['lr']
            opt = torch.optim.AdamW(params.values(), lr=lr)
            rng = random.Random(CFG['seed'])
            model.train()
            for i, key in enumerate(train_sample, 1):
                candidates = sample_candidates(key, rng)[:4]  # positive + up to three negatives
                opt.zero_grad(set_to_none=True)
                loss = retrieval_loss(model, teacher if method == 'kd' else None, key, candidates)
                if not torch.isfinite(loss):
                    raise ValueError(f'{method}: nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params.values(), 1., error_if_nonfinite=True)
                opt.step()
                if i % 16 == 0:
                    print(method, 'trained', i, '/', len(train_sample), flush=True)
            save_pt({n: p.detach().cpu() for n, p in params.items()}, weights)
            del opt, loss

        model.eval()
        report = {'prime': prime_method, 'reranker': method, 'training_queries': len(train_sample),
                  'trainable_parameters': sum(p.numel() for p in params.values()), 'prime_dev_f1': baseline}
        for split in ('Dev', 'Test'):
            predictions = {k: list(v) for k, v in base_rankings[split].items()}
            changes = {}
            with torch.inference_mode(), amp():
                for key, choices in shortlists[split].items():
                    query = model(ENC[split]['queries'][key])
                    documents = torch.stack([model(ENC[split]['candidates'][c]) for c in choices])
                    scores = documents @ query
                    if not torch.isfinite(scores).all():
                        raise ValueError(f'{method}/{key}: nonfinite ranking score')
                    selected = choices[int(scores.argmax())]
                    if selected != choices[0]:
                        predictions[key][-1] = selected
                        changes[key] = {'old': choices[0], 'new': selected}
            validate_predictions(predictions, dev_ids if split == 'Dev' else test_ids, SPLITS[split]['candidates'])
            save_json(changes, folder / f'{split.lower()}_changes.json')
            report[f'{split.lower()}_changed_queries'] = len(changes)
            if split == 'Dev':
                report.update(metrics_at7(predictions, SPLITS['Dev']['queries']))
            else:
                save_json(predictions, folder / 'prediction.json')
        output = folder / prime_zip.name
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(folder / 'prediction.json', 'prediction.json')
        save_json(report, folder / 'metrics.json')
        print(method, 'Dev F1:', report['micro_f1_at7'], 'test queries changed:', report['test_changed_queries'], flush=True)
        zips[f'{prime_method}_plus_{method}'] = output
        del model, params
        cleanup()
    bundle = root / ('SMOKE_six_zips.zip' if SMOKE else 'six_submission_zips.zip')
    with zipfile.ZipFile(bundle, 'w') as archive:
        for name, path in zips.items():
            archive.write(path, f'{name}.zip')
    print('Download this bundle and extract its six individual ZIPs:', bundle)
    return bundle

# After the single-prime training snippet, results contains exactly the prime method.
if len(results) != 1:
    raise ValueError("Specify your prime method explicitly in the call below and remove this guard.")
variant_bundle = make_five_variants(results[0]['method'])






# Paste into a Kaggle cell after Section 8. Uses existing rankings and original case texts.
import json
import zipfile
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def check_phase7(method, uploaded_zip=None):
    if SMOKE:
        raise ValueError("This is a smoke run. Use the full run for score diagnosis.")
    folder = WORK / 'runs' / method
    def read_json(path):
        return json.loads(path.read_text(encoding='utf-8'))
    def read_zip(path):
        with zipfile.ZipFile(path) as z:
            if z.namelist() != ['prediction.json']:
                raise ValueError(f'{path}: expected one prediction.json at ZIP root')
            return json.loads(z.read('prediction.json'))

    prime_dev = read_json(folder / 'dev_rankings.json')['predictions']
    prime_test = read_zip(folder / 'submission.zip')
    for split, pred in [('Dev', prime_dev), ('Test', prime_test)]:
        validate_predictions(pred, list(SPLITS[split]['queries']), SPLITS[split]['candidates'])
    if prime_test != read_json(folder / 'test_rankings.json')['predictions']:
        raise ValueError('Prime ZIP differs from the saved Test rankings.')
    if uploaded_zip is not None:
        uploaded = read_zip(Path(uploaded_zip))
        validate_predictions(uploaded, list(prime_test), SPLITS['Test']['candidates'])
        changed = sum(set(uploaded[q]) != set(prime_test[q]) for q in prime_test)
        print('Submitted file: valid. Queries with different top-seven sets from prime:', changed)
    print('Valid prime Test ZIP. Queries:', len(prime_test), 'method:', method)
    queries = SPLITS['Dev']['queries']
    def f1(pred):
        return metrics_at7(pred, queries)['micro_f1_at7']
    prime_f1 = f1(prime_dev)
    for split, rows in [('Dev', prime_dev), ('Test', prime_test)]:
        pool = SPLITS[split]['candidates']
        print(f'{split} self-ID audit ({len(rows)} queries):',
              'present in pool', sum(q in pool for q in rows),
              '| ranked first', sum(v[0] == q for q, v in rows.items()),
              '| anywhere in top 7', sum(q in v for q, v in rows.items()))
    gold_self = sum(q in row['relevant_candidates'] for q, row in queries.items())
    print('Dev queries whose own ID is labelled relevant:', gold_self)
    stored = read_json(folder / 'metrics.json')
    if not np.isclose(prime_f1, stored['micro_f1_at7']):
        raise ValueError('Saved Dev score differs from the score recomputed from Dev rankings.')
    counts = Counter(c for row in prime_dev.values() for c in row)
    gold = sum(len(q['relevant_candidates']) for q in queries.values())
    cap = int(len(queries) * .05)
    print('Prime Dev F1:', round(prime_f1, 4), '| training steps:', stored['steps'])
    print('Most frequently returned Dev case:', counts.most_common(1),
          '| Dev queries:', len(queries))
    print('Maximum possible Dev F1 gain under the 5% one-swap rule:',
          round(2 * cap / (7 * len(queries) + gold), 5))
    if (WORK / 'coverage.json').exists():
        for role, info in read_json(WORK / 'coverage.json').items():
            if role.startswith('Dev/'):
                print(role, 'mean fraction of tokens seen:', round(info['mean_coverage'], 3))

    # Fit on candidate texts within each retrieval pool. Use no relevance labels in indexing.
    def lexical(split):
        data = SPLITS[split]
        ids = sorted(data['candidates'])
        vectorizer = TfidfVectorizer(sublinear_tf=True, dtype=np.float32)
        matrix = vectorizer.fit_transform(data['candidates'][c]['text'] for c in ids).T.tocsr()
        rankings = {}
        for key, q in data['queries'].items():
            scores = (vectorizer.transform([q['text']]) @ matrix).toarray()[0]
            order = np.argsort(-scores, kind='stable')[:50]
            rankings[key] = [ids[int(i)] for i in order]
        return rankings

    # Scores cover the prime top seven and the full-text top fifty.
    def combine(dense, sparse, weight, exclude_self=False):
        hybrid = {}
        for q, candidates in dense.items():
            scores = Counter({c: 1 / (10 + i) for i, c in enumerate(candidates, 1)})
            for i, c in enumerate(sparse[q], 1):
                scores[c] += weight / (10 + i)
            if exclude_self:
                scores.pop(q, None)
            # With zero lexical weight preserve dense order and backfill from lexical rank.
            if weight == 0:
                ordered = dict.fromkeys([*candidates, *sparse[q]])
                hybrid[q] = [c for c in ordered if not exclude_self or c != q][:7]
            else:
                hybrid[q] = sorted(scores, key=lambda c: (-scores[c], c))[:7]
        return hybrid

    lex_dev = lexical('Dev')
    print('Full-text TF-IDF alone, Dev F1:', round(f1({q: v[:7] for q, v in lex_dev.items()}), 4))
    best_weight, best_exclusion, best_f1 = 0., False, prime_f1
    enough_candidates = all(len(SPLITS[s]['candidates']) >= 8 or
                            not (SPLITS[s]['queries'].keys() & SPLITS[s]['candidates'].keys())
                            for s in ('Dev', 'Test'))
    for exclude in (False, True):
        if exclude and (gold_self or not enough_candidates):
            print('Skipping self exclusion: Dev contains relevant self IDs or the pool is too small.')
            continue
        for weight in (0., 0.5, 1., 2.):
            trial = combine(prime_dev, lex_dev, weight, exclude)
            validate_predictions(trial, list(queries), SPLITS['Dev']['candidates'])
            hybrid_f1 = f1(trial)
            print(f'lexical weight {weight}, exclude self {exclude}: Dev F1 {hybrid_f1:.4f}')
            if hybrid_f1 > best_f1:
                best_weight, best_exclusion, best_f1 = weight, exclude, hybrid_f1
    if best_f1 > prime_f1:
        predictions = combine(prime_test, lexical('Test'), best_weight, best_exclusion)
        validate_predictions(predictions, list(SPLITS['Test']['queries']), SPLITS['Test']['candidates'])
        out = folder / f'fulltext_w{best_weight}_exclude_self_{best_exclusion}.zip'
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('prediction.json', json.dumps(predictions))
        print('Best tested hybrid on Dev. Test ZIP:', out)
    else:
        print('None of the tested hybrids improved on the prime Dev score.')

# Set METHOD to the name of the run you actually submitted if several runs exist.
METHOD = 'lorma'
if METHOD is None:
    raise ValueError("Set METHOD above to your submitted prime method, e.g. 'lorma'.")
check_phase7(METHOD)  # Optional second argument: path to the exact ZIP you submitted.



import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

if SMOKE:
    raise ValueError("Use the full run with SMOKE=False.")

runs = Path(WORK) / "runs"
out = Path(WORK) / "final_submissions"
out.mkdir(parents=True, exist_ok=True)

prime = "lorma"
confirmed_zip = runs / prime / "fulltext_w2.0_exclude_self_True.zip"
methods = [
    m for m in ("lorma", "full", "partial", "prompt", "ia3", "kd")
    if all((runs / m / f"{s}_rankings.json").is_file() for s in ("dev", "test"))
]
if prime not in methods:
    raise FileNotFoundError("The saved LoRMA Dev and Test rankings are required.")

def lexical_rankings(split):
    data = SPLITS[split]
    ids = sorted(data["candidates"])
    vectorizer = TfidfVectorizer(sublinear_tf=True, dtype=np.float32)
    index = vectorizer.fit_transform(
        data["candidates"][cid]["text"] for cid in ids
    ).T.tocsr()

    rankings = {}
    for qid, row in data["queries"].items():
        query = vectorizer.transform([row["text"]])
        scores = (query @ index).toarray()[0]
        top = np.argsort(-scores, kind="stable")[:50]
        rankings[qid] = [ids[i] for i in top]
    return rankings

def load_dense(method, split):
    path = runs / method / f"{split.lower()}_rankings.json"
    predictions = json.loads(path.read_text())["predictions"]
    data = SPLITS[split]
    validate_predictions(predictions, list(data["queries"]), data["candidates"])
    return predictions

def fuse(dense, lexical, weight):
    predictions = {}
    for qid, dense_list in dense.items():
        # Keep the original rank positions so weight 2 reproduces the prior hybrid.
        scores = {
            cid: 1.0 / (10 + rank)
            for rank, cid in enumerate(dense_list, 1)
        }
        for rank, cid in enumerate(lexical[qid], 1):
            scores[cid] = scores.get(cid, 0.0) + weight / (10 + rank)
        scores.pop(qid, None)
        predictions[qid] = sorted(scores, key=lambda cid: (-scores[cid], cid))[:7]
    return predictions

def check(predictions, split):
    data = SPLITS[split]
    validate_predictions(predictions, list(data["queries"]), data["candidates"])
    if any(qid in ranking for qid, ranking in predictions.items()):
        raise ValueError(f"Self-match found in {split} predictions.")

def write_zip(path, predictions):
    check(predictions, "Test")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("prediction.json", json.dumps(predictions))
    return path

print("Building full-text rankings once for each split...")
lex_dev = lexical_rankings("Dev")
lex_test = lexical_rankings("Test")
dev_queries = SPLITS["Dev"]["queries"]

lex_only = {
    qid: [cid for cid in ranking if cid != qid][:7]
    for qid, ranking in lex_dev.items()
}
check(lex_only, "Dev")
lex_f1 = metrics_at7(lex_only, dev_queries)["micro_f1_at7"]
print(f"TF-IDF alone, self excluded: Dev F1 = {lex_f1:.4f}")

bundle_files = {}

if confirmed_zip.is_file():
    with zipfile.ZipFile(confirmed_zip) as archive:
        confirmed = json.loads(archive.read("prediction.json"))
    check(confirmed, "Test")
    bundle_files["lorma_confirmed_0.358.zip"] = confirmed_zip

for method in methods:
    dense_dev = load_dense(method, "Dev")
    dense_test = load_dense(method, "Test")

    best_weight, best_f1 = None, -1.0
    for weight in (2.0, 4.0, 8.0):
        trial = fuse(dense_dev, lex_dev, weight)
        check(trial, "Dev")
        score = metrics_at7(trial, dev_queries)["micro_f1_at7"]
        print(f"{method:8s}  lexical weight {weight:.0f}  Dev F1 {score:.4f}")
        if score > best_f1:  # A tie keeps the earlier, lower weight.
            best_weight, best_f1 = weight, score

    predictions = fuse(dense_test, lex_test, best_weight)
    check(predictions, "Test")

    if method == prime and confirmed_zip.is_file():
        # Micro-F1@7 depends on membership, so a reorder alone changes no score.
        changed = sum(
            set(predictions[qid]) != set(confirmed[qid])
            for qid in predictions
        )
        print(f"LoRMA: {changed} Test queries differ from the confirmed ZIP.")
        if changed == 0:
            continue

    name = f"{method}_w{best_weight:.0f}_dev_{best_f1:.4f}.zip"
    bundle_files[name] = write_zip(out / name, predictions)

bundle = out / "download_and_extract.zip"
with zipfile.ZipFile(bundle, "w") as archive:
    for name, path in bundle_files.items():
        archive.write(path, arcname=name)

print("\nAvailable submission ZIPs:")
for name in bundle_files:
    print(" ", name)
print("\nDownload and extract:", bundle)


from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

if SMOKE:
    raise ValueError("Use your real Phase 7 run with SMOKE=False.")

PRIME = "lorma"
METHODS = ["full", "partial", "prompt", "ia3", "kd"]
prime_dir = Path(WORK) / "runs" / PRIME
confirmed_path = prime_dir / "fulltext_w2.0_exclude_self_True.zip"
output_dir = Path(WORK) / "final_six_hybrid"
output_dir.mkdir(parents=True, exist_ok=True)

# Set this only if the cell finds multiple sets of mini-models.
MINI_DIR = None

def read_predictions(path):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("prediction.json"))

def check(pred, split):
    data = SPLITS[split]
    validate_predictions(pred, list(data["queries"]), data["candidates"])
    if any(qid in row for qid, row in pred.items()):
        raise ValueError(f"Self-ID in {split} predictions")

if not confirmed_path.is_file():
    raise FileNotFoundError(
        f"Confirmed LoRMA ZIP missing: {confirmed_path}\n"
        "If you downloaded it, attach that ZIP and set confirmed_path to its location."
    )

confirmed_test = read_predictions(confirmed_path)
check(confirmed_test, "Test")

# Locate either the short-run weights or a completed ordinary run for each method.
roots = sorted((Path(WORK) / "rerankers").glob(f"{PRIME}-*"))
roots = [p for p in roots if any(p.glob("*/trained_parameters.pt"))]
if MINI_DIR is None and len(roots) > 1:
    raise ValueError(f"Set MINI_DIR to the run you want: {roots}")
mini_root = Path(MINI_DIR) if MINI_DIR else (roots[0] if roots else None)

sources = {}
for method in METHODS:
    mini_file = mini_root / method / "trained_parameters.pt" if mini_root else None
    if mini_file is not None and mini_file.is_file():
        sources[method] = ("mini", mini_file)
    elif (Path(WORK) / "runs" / method / "last.pt").is_file():
        sources[method] = ("regular", None)

missing = [m for m in METHODS if m not in sources]
if missing:
    raise FileNotFoundError(
        f"Missing trained weights for {missing}. Looked under {WORK}/rerankers "
        f"and {WORK}/runs. Restore those files or run the five-reranker training "
        "cell first. The LoRMA ZIP alone contains case IDs, not these models."
    )

def get_model(method):
    kind, path = sources[method]
    if kind == "regular":
        return load_model(method).eval()

    model = Retriever(method)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if set(saved) != set(params):
        raise ValueError(f"{method}: saved parameters do not match the notebook config")
    with torch.no_grad():
        for name, value in saved.items():
            params[name].copy_(value)
    return model.to(DEVICE).eval()

def prepare_split(split):
    data = SPLITS[split]
    ids = sorted(data["candidates"])
    vectorizer = TfidfVectorizer(sublinear_tf=True, dtype=np.float32)
    matrix = vectorizer.fit_transform(
        data["candidates"][cid]["text"] for cid in ids
    ).T.tocsr()

    dense = json.loads(
        (prime_dir / f"{split.lower()}_rankings.json").read_text()
    )["predictions"]
    validate_predictions(dense, list(data["queries"]), data["candidates"])

    baseline, shortlists = {}, {}
    for qid, row in data["queries"].items():
        query = vectorizer.transform([row["text"]])
        scores = (query @ matrix).toarray()[0]
        order = np.argsort(-scores, kind="stable")[:50]
        lexical = [ids[i] for i in order]

        # Reproduce the exact weight-2 fusion used for the 0.358 submission.
        fused = {cid: 1 / (10 + i) for i, cid in enumerate(dense[qid], 1)}
        for i, cid in enumerate(lexical, 1):
            fused[cid] = fused.get(cid, 0) + 2 / (10 + i)
        fused.pop(qid, None)
        ranked = sorted(fused, key=lambda cid: (-fused[cid], cid))
        baseline[qid] = ranked[:7]

        # Keep the top five. Reconsider ranks 6–15 for the final two slots.
        shortlists[qid] = ranked[5:15]

    check(baseline, split)
    return baseline, shortlists

print("Preparing full-text shortlists...")
dev_base, dev_choices = prepare_split("Dev")
test_base, test_choices = prepare_split("Test")

if test_base != confirmed_test:
    raise ValueError(
        "The reconstructed Test ranking differs from your confirmed ZIP. "
        "Check that the LoRMA rankings and ZIP come from the same run."
    )

@torch.inference_mode()
def model_orders(model, split, choices):
    # Each distinct candidate is encoded only once per model and split.
    encoded = ENC[split]
    candidate_ids = sorted({cid for row in choices.values() for cid in row})
    vectors = {}
    with amp():
        for cid in candidate_ids:
            vectors[cid] = model(encoded["candidates"][cid]).float().cpu()

        orders = {}
        for qid, pool in choices.items():
            query = model(encoded["queries"][qid]).float().cpu()
            scores = torch.stack([vectors[c] for c in pool]) @ query
            if not torch.isfinite(scores).all():
                raise ValueError(f"Nonfinite score for {split}/{qid}")
            # Stable ties retain the original hybrid order.
            idx = np.argsort(-scores.numpy(), kind="stable")
            orders[qid] = [pool[i] for i in idx]
    return orders

def rerank(base, choices, orders, model_weight):
    pred = {}
    for qid, pool in choices.items():
        model_rank = {cid: i for i, cid in enumerate(orders[qid], 1)}
        # Combine the confirmed hybrid's order and this method's learned order.
        score = {
            cid: 1 / (5 + i) + model_weight / (5 + model_rank[cid])
            for i, cid in enumerate(pool, 1)
        }
        final_two = sorted(pool, key=lambda cid: -score[cid])[:2]
        pred[qid] = base[qid][:5] + final_two
    return pred

def f1(pred):
    return metrics_at7(pred, SPLITS["Dev"]["queries"])["micro_f1_at7"]

baseline_f1 = f1(dev_base)
files = {"lorma_confirmed.zip": confirmed_path}
report = []
seen = {}

def membership_hash(pred):
    # Reordering the same seven IDs has no effect on Micro-F1@7.
    canonical = {q: sorted(row) for q, row in pred.items()}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()

seen[membership_hash(confirmed_test)] = PRIME
print(f"Confirmed LoRMA Dev F1: {baseline_f1:.4f}")

for method in METHODS:
    print(f"\nScoring {method} ({sources[method][0]} checkpoint)...", flush=True)
    model = get_model(method)
    dev_order = model_orders(model, "Dev", dev_choices)

    # Select among actual reranker variants. Report whether it beat the prime.
    trials = []
    for weight in (0.5, 1.0, 2.0):
        pred = rerank(dev_base, dev_choices, dev_order, weight)
        check(pred, "Dev")
        trials.append((f1(pred), weight))
    best_f1, best_weight = max(trials, key=lambda pair: pair[0])

    test_order = model_orders(model, "Test", test_choices)
    pred = rerank(test_base, test_choices, test_order, best_weight)
    check(pred, "Test")

    changed = sum(set(pred[q]) != set(confirmed_test[q]) for q in pred)
    fingerprint = membership_hash(pred)
    same_as = seen.get(fingerprint)
    seen.setdefault(fingerprint, method)

    path = output_dir / f"{method}_hybrid.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction.json", json.dumps(pred))
    files[path.name] = path

    report.append({
        "method": method,
        "model_weight": best_weight,
        "dev_f1": best_f1,
        "dev_gain_vs_prime": best_f1 - baseline_f1,
        "changed_test_queries": changed,
        "same_top7_sets_as": same_as,
    })
    print(f"  Dev F1 {best_f1:.4f} ({best_f1-baseline_f1:+.4f} vs prime)")
    print(f"  Changed top-seven sets: {changed}/{len(pred)}")
    if same_as:
        print(f"  Identical scoring sets to {same_as}; expect the same score.")

    del model
    cleanup()

(output_dir / "comparison.json").write_text(json.dumps(report, indent=2))
bundle = output_dir / "six_individual_submissions.zip"
with zipfile.ZipFile(bundle, "w") as z:
    for name, path in files.items():
        z.write(path, arcname=name)

print("\nDownload and extract:", bundle)
print("Unique top-seven prediction sets across the six files:", len(seen))































































