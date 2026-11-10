





# !pip -q install -U huggingface_hub   # transformers / torch are preinstalled on Colab & Kaggle
import os, re, json, math, time, copy, random, zipfile, gc, glob
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = (torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16)
print("device:", DEVICE, "| amp:", AMP_DTYPE if DEVICE == "cuda" else "off")

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---- paths ---------------------------------------------------------------------------------
# If DATA_DIR doesn't contain the files, they are downloaded from the official HF dataset.
DATA_DIR = "/content/Phase_2_LNER"
OUT_DIR = "/content/drive/MyDrive/cs781_phase2"      # put on Drive so checkpoints survive disconnects
HF_REPO = "Exploration-Lab/CS-781-Capstone"
DEV_FRACTION = 0.10                                  # held-out docs from fold1+fold2 for model selection
SEED = 42
os.makedirs(OUT_DIR, exist_ok=True)
seed_all(SEED)


from pathlib import Path
# Phase-2 data is downloaded from HF (needs Internet ON in the notebook settings + HF_TOKEN secret)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ.setdefault("HF_TOKEN", UserSecretsClient().get_secret("HF_TOKEN"))
except Exception:
    print("No HF_TOKEN secret found")

DATA_DIR = "/kaggle/working/Phase_2_LNER"
OUT_DIR  = "/kaggle/working/cs781_phase2"
os.makedirs(OUT_DIR, exist_ok=True)

# find the three uploaded files wherever Kaggle mounted them
P1 = next(Path("/kaggle/input").rglob("lexweave_model.py")).parent
print("Phase-1 files in:", P1, [p.name for p in P1.iterdir()])





def ensure_data():
    global DATA_DIR
    if glob.glob(os.path.join(DATA_DIR, "**", "fold1-text.json"), recursive=True):
        return
    from huggingface_hub import snapshot_download
    root = snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=["*LNER*"],
                             token=os.environ.get("HF_TOKEN"))
    hits = glob.glob(os.path.join(root, "**", "fold1-text.json"), recursive=True)
    assert hits, f"fold1-text.json not found under {root}"
    DATA_DIR = os.path.dirname(os.path.dirname(hits[0]))     # .../Phase_2_LNER

def find_file(name):
    hits = glob.glob(os.path.join(DATA_DIR, "**", name), recursive=True)
    assert hits, f"{name} not found under {DATA_DIR}"
    return hits[0]

def _id_map(raw, value_keys):
    """Accept {id: value} or [{id:..., <value_key>:...}, ...]."""
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}
    out = {}
    for item in raw:
        did = item.get("id", item.get("doc_id"))
        val = next(item[k] for k in value_keys if k in item)
        out[str(did)] = val
    return out

def parse_entities(ents):
    out = []
    for e in ents:
        if isinstance(e, dict):
            s = e.get("start", e.get("start_char", e.get("begin")))
            en = e.get("end", e.get("end_char"))
            lab = e.get("label", e.get("type", e.get("entity")))
            txt = e.get("text", e.get("entity_text"))
        else:
            s, en, lab = e[0], e[1], e[2]
            txt = e[3] if len(e) > 3 else None
        out.append((int(s), int(en), str(lab), txt))
    return sorted(out)

ensure_data()
print("DATA_DIR:", DATA_DIR)

LABELS = [l.strip() for l in open(find_file("labels.txt"), encoding="utf-8") if l.strip() and l.strip() != "O"]
print(len(LABELS), "labels:", LABELS)

TRAIN_TEXT, TRAIN_GOLD = {}, {}
for fold in ("fold1", "fold2"):
    texts = _id_map(json.load(open(find_file(f"{fold}-text.json"), encoding="utf-8")), ["text"])
    gold = _id_map(json.load(open(find_file(f"{fold}-gold.json"), encoding="utf-8")), ["entities", "spans", "labels", "annotations"])
    for did, t in texts.items():
        TRAIN_TEXT[did] = t
        TRAIN_GOLD[did] = parse_entities(gold.get(did, []))
TEST_TEXT = _id_map(json.load(open(find_file("fold3-text.json"), encoding="utf-8")), ["text"])
print(f"train docs: {len(TRAIN_TEXT)} | test docs: {len(TEST_TEXT)} | train entities: {sum(map(len, TRAIN_GOLD.values()))}")


# --- Offset sanity check: is `end` exclusive? do the offsets match the supplied entity text? -----
def check_offsets(texts, gold):
    ok = incl = bad = notext = 0
    for did, ents in gold.items():
        t = texts[did]
        for s, e, lab, txt in ents:
            if txt is None: notext += 1
            elif t[s:e] == txt: ok += 1
            elif t[s:e + 1] == txt: incl += 1
            else: bad += 1
    print(f"exclusive-end matches: {ok} | inclusive-end matches: {incl} | mismatches: {bad} | no text: {notext}")
    return incl > ok

if check_offsets(TRAIN_TEXT, TRAIN_GOLD):
    print("!! offsets look end-INCLUSIVE -> converting to exclusive internally (and back on output)")
    END_INCLUSIVE = True
    TRAIN_GOLD = {d: [(s, e + 1, l, t) for s, e, l, t in ents] for d, ents in TRAIN_GOLD.items()}
else:
    END_INCLUSIVE = False

# drop labels not in the official list, invalid or overlapping spans (reported)
def clean_gold(texts, gold):
    dropped = Counter(); out = {}
    for did, ents in gold.items():
        keep, last_end = [], -1
        for s, e, l, t in ents:
            if l not in LABELS: dropped["unknown_label"] += 1; continue
            if not (0 <= s < e <= len(texts[did])): dropped["out_of_range"] += 1; continue
            if s < last_end: dropped["overlap"] += 1; continue
            keep.append((s, e, l)); last_end = e
        out[did] = keep
    print("dropped gold entities:", dict(dropped) or "none")
    return out
TRAIN_GOLD = clean_gold(TRAIN_TEXT, TRAIN_GOLD)


# --- Data analysis (report material) --------------------------------------------------------------
lab_cnt = Counter(l for ents in TRAIN_GOLD.values() for _, _, l in ents)
lens = defaultdict(list)
for did, ents in TRAIN_GOLD.items():
    for s, e, l in ents: lens[l].append(e - s)
doc_chars = [len(t) for t in TRAIN_TEXT.values()]
print(f"doc length (chars): mean {np.mean(doc_chars):.0f} | median {np.median(doc_chars):.0f} | max {max(doc_chars)}")
print(f"{'label':<11}{'count':>7}{'mean_len':>10}{'max_len':>9}")
for l in LABELS:
    print(f"{l:<11}{lab_cnt[l]:>7}{(np.mean(lens[l]) if lens[l] else 0):>10.1f}{(max(lens[l]) if lens[l] else 0):>9}")

# relative position of entities in the document (header entities vs body entities)
pos = defaultdict(list)
for did, ents in TRAIN_GOLD.items():
    for s, e, l in ents: pos[l].append(s / max(1, len(TRAIN_TEXT[did])))
print("median relative position:", {l: round(float(np.median(v)), 3) for l, v in pos.items()})

# --- train / dev split (document level) ------------------------------------------------------------
_ids = sorted(TRAIN_TEXT)
random.Random(SEED).shuffle(_ids)
n_dev = max(3, int(round(DEV_FRACTION * len(_ids))))
DEV_IDS, TRN_IDS = sorted(_ids[:n_dev]), sorted(_ids[n_dev:])
print(f"split -> train {len(TRN_IDS)} docs, dev {len(DEV_IDS)} docs")





def strict_scores(gold, pred, labels=LABELS):
    """gold/pred: {doc_id: [(s,e,label), ...]}. Strict = exact (s,e,label) match.
    Macro averages over all official labels. Check against the official script on your dev split."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for did in gold:
        g = set(map(tuple, gold[did])); p = set(map(tuple, pred.get(did, [])))
        for x in p & g: tp[x[2]] += 1
        for x in p - g: fp[x[2]] += 1
        for x in g - p: fn[x[2]] += 1
    per = {}
    for l in labels:
        P = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        R = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0.0
        per[l] = dict(P=P, R=R, F1=2 * P * R / (P + R) if P + R else 0.0, support=tp[l] + fn[l])
    T, FP_, FN_ = sum(tp.values()), sum(fp.values()), sum(fn.values())
    mi_p = T / (T + FP_) if T + FP_ else 0.0; mi_r = T / (T + FN_) if T + FN_ else 0.0
    return dict(macro_P=np.mean([v["P"] for v in per.values()]),
                macro_R=np.mean([v["R"] for v in per.values()]),
                macro_F1=np.mean([v["F1"] for v in per.values()]),
                micro_F1=2 * mi_p * mi_r / (mi_p + mi_r) if mi_p + mi_r else 0.0,
                per_label=per)

def error_analysis(gold, pred, texts=None, n_examples=0):
    """Splits errors into: label error (same span), boundary error (overlap, same label),
    boundary+label, spurious (no overlap with gold), missed (no overlapping prediction)."""
    cats = Counter(); confusion = Counter(); boundary_by_label = Counter(); examples = defaultdict(list)
    for did in gold:
        G = list(map(tuple, gold[did])); P = list(map(tuple, pred.get(did, [])))
        gset = set(G)
        for p in P:
            if p in gset: cats["correct"] += 1; continue
            ov = [g for g in G if g[0] < p[1] and p[0] < g[1]]
            if not ov: cats["spurious"] += 1; k = "spurious"
            elif any(g[0] == p[0] and g[1] == p[1] for g in ov):
                g = next(g for g in ov if g[0] == p[0] and g[1] == p[1])
                cats["label_error"] += 1; confusion[(g[2], p[2])] += 1; k = "label_error"
            elif any(g[2] == p[2] for g in ov):
                cats["boundary_error"] += 1; boundary_by_label[p[2]] += 1; k = "boundary_error"
            else: cats["boundary+label"] += 1; k = "boundary+label"
            if texts is not None and len(examples[k]) < n_examples:
                g = ov[0] if ov else None
                examples[k].append((p[2], texts[did][p[0]:p[1]], g and (g[2], texts[did][g[0]:g[1]])))
        for g in G:
            if not any(p[0] < g[1] and g[0] < p[1] for p in P): cats["missed"] += 1
    return dict(categories=dict(cats), label_confusion=confusion.most_common(15),
                boundary_errors_by_label=dict(boundary_by_label), examples=dict(examples))





import sys; sys.path.append(str(P1))
from lexweave_model import LegalLM

CKPT = str(P1 / "model.pt")
TOKENIZER_PATH = str(P1 / "tokenizer.json")

def build_lexweave():
    cfg = dict(torch.load(CKPT, map_location="cpu", weights_only=False)["model_config"])
    return LegalLM(cfg)

PHASE1_MODELS = {
  "M1_lexweave": dict(kind="custom", build=build_lexweave, ckpt=CKPT,
      hidden_size=768, hidden_module="norm", is_causal=True, max_len=1024,
      blocks_attr="blocks", call=lambda m, ids, mask: m.hidden(ids)),
}


class Backbone(nn.Module):
    def __init__(self, model, hidden_size, forward_fn, is_causal, max_len, blocks_attr=None):
        super().__init__()
        self.model, self.hidden_size, self.forward_fn = model, hidden_size, forward_fn
        self.is_causal, self.max_len, self.blocks_attr = is_causal, max_len, blocks_attr

    def forward(self, input_ids, attention_mask):
        return self.forward_fn(self.model, input_ids, attention_mask)

def hook_forward_fn(module_name, call=None):
    """Run the full model and return the output of `module_name`.
    The hook is registered per call, so deep copies of the model stay safe."""
    call = call or (lambda m, ids, mask: m(ids))
    def fn(model, ids, mask):
        store = {}
        mod = model.get_submodule(module_name)
        h = mod.register_forward_hook(lambda m, i, o: store.__setitem__("h", o[0] if isinstance(o, (tuple, list)) else o))
        try: call(model, ids, mask)
        finally: h.remove()
        return store["h"]
    return fn

def find_blocks(model):
    """Dotted name of the ModuleList holding the transformer layers (largest ModuleList by #params)."""
    best, best_n = None, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) > 1:
            n = sum(p.numel() for p in m.parameters())
            if n > best_n: best, best_n = name, n
    return best

ENCODER_ONLY_TYPES = {"bert", "roberta", "deberta", "deberta-v2", "electra", "distilbert", "albert", "modernbert"}

def load_backbone(name):
    spec = PHASE1_MODELS[name]
    if spec["kind"] == "hf":
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(spec["path"])
        model = AutoModel.from_pretrained(spec["path"])          # YOUR Phase-1 weights
        if getattr(cfg, "is_encoder_decoder", False):
            model = model.get_encoder(); causal = False
            fn = lambda m, ids, mask: m(input_ids=ids, attention_mask=mask).last_hidden_state
        else:
            causal = cfg.model_type not in ENCODER_ONLY_TYPES
            fn = (lambda m, ids, mask: m(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state) if causal \
                 else (lambda m, ids, mask: m(input_ids=ids, attention_mask=mask).last_hidden_state)
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or cfg.d_model
        max_len = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None) or 512
        bb = Backbone(model, hidden, fn, causal, max_len, spec.get("blocks_attr"))
    else:
        model = spec["build"]()
        sd = torch.load(spec["ckpt"], map_location="cpu")
        if isinstance(sd, dict):
            for k in ("model", "model_state_dict", "state_dict"):
                if k in sd and isinstance(sd[k], dict): sd = sd[k]; break
        # strip torch.compile ("_orig_mod.") and DDP ("module.") prefixes
        sd = {re.sub(r"^module\.", "", k.replace("_orig_mod.", "")): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[{name}] missing={len(missing)} unexpected={len(unexpected)} (first: {missing[:3]} {unexpected[:3]})")
        bb = Backbone(model, spec["hidden_size"], hook_forward_fn(spec["hidden_module"], spec.get("call")),
                      spec["is_causal"], spec["max_len"], spec.get("blocks_attr"))
    if bb.blocks_attr is None: bb.blocks_attr = find_blocks(bb.model)
    return bb

def n_params(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)

# ---- tokenizer with character offsets -----------------------------------------------------------
_TOKENIZERS = {}
def load_tokenizer(path):
    if path in _TOKENIZERS: return _TOKENIZERS[path]
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    tok = (PreTrainedTokenizerFast(tokenizer_file=path, pad_token="<pad>", bos_token="<bos>",
                               eos_token="<eos>", unk_token="<unk>")
       if path.endswith(".json") else AutoTokenizer.from_pretrained(path))
    if not getattr(tok, "is_fast", False):
        print("WARNING: slow tokenizer, using decode-based offsets (verify the alignment ceiling below!)")
    _TOKENIZERS[path] = tok
    return tok

def tokenize_with_offsets(tok, text):
    """Returns token ids and (start,end) char offsets into the ORIGINAL text. The text is never modified."""
    if getattr(tok, "is_fast", False):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        return enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    # fallback for slow / custom tokenizers: locate each decoded token sequentially
    ids = tok.encode(text, add_special_tokens=False); offs, ptr = [], 0
    for i in ids:
        piece = tok.decode([i]).strip()
        j = text.find(piece, ptr, ptr + 64 + len(piece)) if piece and "�" not in piece else -1
        if j < 0: offs.append((ptr, ptr))
        else: offs.append((j, j + len(piece))); ptr = j + len(piece)
    return ids, offs

def tokenizer_path_for(model_name):
    return PHASE1_MODELS[model_name].get("tokenizer", TOKENIZER_PATH)


# ---- Smoke test: load every Phase-1 model once (run this first!) --------------------------------
for name in PHASE1_MODELS:
    bb = load_backbone(name).to(DEVICE).eval()
    tok = load_tokenizer(tokenizer_path_for(name))
    ids, _ = tokenize_with_offsets(tok, "IN THE SUPREME COURT OF INDIA CIVIL APPELLATE JURISDICTION")
    x = torch.tensor([ids], device=DEVICE)
    with torch.no_grad(): h = bb(x, torch.ones_like(x))
    print(f"{name:<14} params={n_params(bb)/1e6:6.1f}M hidden={bb.hidden_size} causal={bb.is_causal} "
          f"max_len={bb.max_len} blocks='{bb.blocks_attr}' out={tuple(h.shape)}")
    del bb; gc.collect(); torch.cuda.empty_cache()





TAGS = ["O"] + [f"{p}-{l}" for l in LABELS for p in ("B", "I")]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
L2ID = {l: i for i, l in enumerate(LABELS)}

@dataclass
class DocEnc:
    doc_id: str
    text: str
    ids: list
    starts: np.ndarray      # trimmed char start per token
    ends: np.ndarray        # trimmed char end per token
    keep: np.ndarray        # indices of non-empty ("content") tokens
    tags: np.ndarray = None # BIO tag per content token
    spans: list = None      # gold (i, j, label_id) in content-token coordinates (inclusive)

def encode_doc(tok, did, text, ents=None):
    ids, offs = tokenize_with_offsets(tok, text)
    st = np.zeros(len(ids), dtype=np.int64); en = np.zeros(len(ids), dtype=np.int64)
    for k, (s, e) in enumerate(offs):
        while s < e and text[s].isspace(): s += 1
        while e > s and text[e - 1].isspace(): e -= 1
        st[k], en[k] = s, e
    keep = np.nonzero(en > st)[0]
    d = DocEnc(did, text, ids, st, en, keep)
    if ents is not None:
        owner = np.full(len(text), -1, dtype=np.int64)
        for k, (s, e, _) in enumerate(ents): owner[s:e] = k
        tags = np.zeros(len(keep), dtype=np.int64); first, last = {}, {}
        prev = -1
        for ci, t in enumerate(keep):
            o = owner[st[t]:en[t]]; o = o[o >= 0]
            if len(o) == 0: prev = -1; continue
            k = int(o[0]); lab = ents[k][2]
            tags[ci] = TAG2ID[("I-" if k == prev else "B-") + lab]
            first.setdefault(k, ci); last[k] = ci; prev = k
        d.tags = tags
        d.spans = [(first[k], last[k], L2ID[ents[k][2]]) for k in first]
    return d

def tags_to_token_spans(tags):
    spans, cur = [], None
    for i, t in enumerate(tags):
        name = TAGS[t]
        if name == "O":
            if cur: spans.append(tuple(cur)); cur = None
        elif name.startswith("B-") or cur is None or cur[2] != L2ID[name[2:]]:   # stray I- repaired to B-
            if cur: spans.append(tuple(cur))
            cur = [i, i, L2ID[name[2:]]]
        else:
            cur[1] = i
    if cur: spans.append(tuple(cur))
    return spans

TRIM = {"lead": "", "trail": ""}     # learned below

def token_spans_to_chars(d, spans, trim=None):
    trim = trim or TRIM; out = []
    for i, j, c in spans:
        s, e = int(d.starts[d.keep[i]]), int(d.ends[d.keep[j]])
        while s < e and (d.text[s].isspace() or d.text[s] in trim["lead"]): s += 1
        while e > s and (d.text[e - 1].isspace() or d.text[e - 1] in trim["trail"]): e -= 1
        if e > s: out.append((s, e, LABELS[c]))
    return out

def alignment_ceiling(encs, gold, trim=None):
    pred = {d.doc_id: token_spans_to_chars(d, tags_to_token_spans(d.tags), trim) for d in encs}
    return strict_scores({d.doc_id: gold[d.doc_id] for d in encs}, pred)

def fit_trim_rules(encs, gold):
    """Greedy, data-driven post-processing: add a char to the trim set only if it raises train exact matches."""
    trim = {"lead": "", "trail": ""}
    base = alignment_ceiling(encs, gold, trim)["micro_F1"]
    for side, cands in (("trail", ",;:).]'\"’”-/"), ("lead", "([\"'‘“,-")):
        for ch in cands:
            t2 = dict(trim); t2[side] += ch
            sc = alignment_ceiling(encs, gold, t2)["micro_F1"]
            if sc > base + 1e-6: trim, base = t2, sc
    return trim

ENC_CACHE = {}
def get_encodings(tok_path):
    """Tokenize train/test docs once per tokenizer (cached)."""
    if tok_path in ENC_CACHE: return ENC_CACHE[tok_path]
    tok = load_tokenizer(tok_path); t0 = time.time()
    train = {did: encode_doc(tok, did, TRAIN_TEXT[did], TRAIN_GOLD[did]) for did in TRAIN_TEXT}
    test = {did: encode_doc(tok, did, TEST_TEXT[did]) for did in TEST_TEXT}
    trn = [train[d] for d in TRN_IDS]
    trim = fit_trim_rules(trn, TRAIN_GOLD)
    before = alignment_ceiling(trn, TRAIN_GOLD, {"lead": "", "trail": ""})["micro_F1"]
    after = alignment_ceiling(trn, TRAIN_GOLD, trim)
    ntok = sum(len(d.ids) for d in train.values())
    print(f"[{os.path.basename(tok_path)}] tokenized in {time.time()-t0:.1f}s | train tokens {ntok:,} "
          f"(~{ntok/len(train):.0f}/doc) | trim rules {trim}")
    print(f"  alignment ceiling (train): micro-F1 {before:.4f} -> {after['micro_F1']:.4f} after trimming | macro-F1 {after['macro_F1']:.4f}")
    ENC_CACHE[tok_path] = (train, test, trim)
    return ENC_CACHE[tok_path]


# Run once per distinct tokenizer. If the ceiling is far below 1.0, inspect tokenization before training anything.
_train_encs, _test_encs, TRIM = get_encodings(TOKENIZER_PATH)
_tok_lens = [j - i + 1 for d in _train_encs.values() for i, j, _ in d.spans]
print("entity length in content tokens: p50 %d | p95 %d | p99 %d | max %d" %
      tuple(np.percentile(_tok_lens, [50, 95, 99, 100]).astype(int)))





def make_examples(encs, cfg, bb, tok, with_labels=True):
    bos = tok.bos_token_id if (cfg.add_bos and tok.bos_token_id is not None) else None
    prefix = [bos] if bos is not None else []
    budget = bb.max_len - len(prefix)
    win = min(cfg.window, budget // 2 if cfg.echo else budget)
    stride = min(cfg.stride, win)
    exs = []
    for d in encs:
        n, a = len(d.ids), 0
        while True:
            b = min(a + win, n)
            lo, hi = np.searchsorted(d.keep, a), np.searchsorted(d.keep, b)
            if hi > lo:
                ids = d.ids[a:b]
                inp = prefix + ids + ids if cfg.echo else prefix + ids
                base = len(prefix) + (len(ids) if cfg.echo else 0)
                ex = dict(doc=d.doc_id, input_ids=inp, content_pos=(d.keep[lo:hi] - a + base).tolist(), lo=int(lo), hi=int(hi))
                if with_labels:
                    ex["tags"] = d.tags[lo:hi].tolist()
                    ex["spans"] = [(i - lo, j - lo, c) for i, j, c in d.spans if i >= lo and j < hi]
                exs.append(ex)
            if b >= n: break
            a += stride
    return exs

def collate(batch, pad_id, max_span=None, with_labels=True):
    B = len(batch); T = max(len(x["input_ids"]) for x in batch); L = max(len(x["content_pos"]) for x in batch)
    ids = torch.full((B, T), pad_id, dtype=torch.long); att = torch.zeros((B, T), dtype=torch.long)
    pos = torch.zeros((B, L), dtype=torch.long); cmask = torch.zeros((B, L), dtype=torch.bool)
    tags = torch.full((B, L), -100, dtype=torch.long)
    for b, x in enumerate(batch):
        n, m = len(x["input_ids"]), len(x["content_pos"])
        ids[b, :n] = torch.tensor(x["input_ids"]); att[b, :n] = 1
        pos[b, :m] = torch.tensor(x["content_pos"]); cmask[b, :m] = True
        if with_labels: tags[b, :m] = torch.tensor(x["tags"])
    out = dict(input_ids=ids, attention_mask=att, content_pos=pos, content_mask=cmask)
    if with_labels:
        out["tags"] = tags
        if max_span:   # span targets: -100 invalid, 0 = no entity, c+1 = label c
            sp = torch.full((B, L, L), -100, dtype=torch.long)
            valid = span_valid_mask(cmask, max_span)
            sp[valid] = 0
            for b, x in enumerate(batch):
                for i, j, c in x["spans"]:
                    if j - i < max_span: sp[b, i, j] = c + 1
            out["span_tags"] = sp
    return out

def span_valid_mask(cmask, max_span):
    L = cmask.size(1)
    i = torch.arange(L, device=cmask.device)
    width = i[None, :] - i[:, None]                      # j - i
    tri = (width >= 0) & (width < max_span)
    return tri[None] & cmask[:, :, None] & cmask[:, None, :]





class TokenHead(nn.Module):
    def __init__(self, H, K, dropout):
        super().__init__(); self.drop = nn.Dropout(dropout); self.proj = nn.Linear(H, K)
    def forward(self, h): return self.proj(self.drop(h))

def bio_transition_mask():
    # start_ok stays all-True: training windows can legitimately begin in the middle of an entity (I- tag)
    K = len(TAGS); allowed = torch.ones(K, K, dtype=torch.bool); start_ok = torch.ones(K, dtype=torch.bool)
    for j, tj in enumerate(TAGS):
        if tj.startswith("I-"):
            for i, ti in enumerate(TAGS):
                if ti[2:] != tj[2:]: allowed[i, j] = False      # O->I-x, B-y->I-x, I-y->I-x forbidden
    return allowed, start_ok

class CRF(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(K, K)); self.start = nn.Parameter(torch.zeros(K)); self.end = nn.Parameter(torch.zeros(K))
        allowed, start_ok = bio_transition_mask()
        self.register_buffer("allowed", allowed, persistent=False); self.register_buffer("start_ok", start_ok, persistent=False)
    def T(self): return self.trans.masked_fill(~self.allowed, -1e4)
    def S(self): return self.start.masked_fill(~self.start_ok, -1e4)

    def nll(self, em, tags, mask):
        """em [B,L,K] (float32), tags [B,L], mask [B,L] prefix-mask. Returns mean NLL per token."""
        em = em.float(); T, S = self.T(), self.S()
        tags = tags.clamp(min=0); m = mask.float()
        e_score = em.gather(2, tags.unsqueeze(-1)).squeeze(-1)
        t_score = T[tags[:, :-1], tags[:, 1:]]
        lengths = mask.sum(1)
        last = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        gold = S[tags[:, 0]] + (e_score * m).sum(1) + (t_score * m[:, 1:]).sum(1) + self.end[last]
        alpha = S + em[:, 0]
        for t in range(1, em.size(1)):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + T.unsqueeze(0), dim=1) + em[:, t]
            alpha = torch.where(mask[:, t:t + 1], nxt, alpha)
        logZ = torch.logsumexp(alpha + self.end, dim=1)
        return (logZ - gold).sum() / m.sum()

    def viterbi(self, em):
        """em: numpy [L,K] for one whole document."""
        T, S, E = self.T().detach().cpu().numpy(), self.S().detach().cpu().numpy(), self.end.detach().cpu().numpy()
        L = em.shape[0]; score = S + em[0]; back = np.zeros((L, em.shape[1]), dtype=np.int64)
        for t in range(1, L):
            cand = score[:, None] + T
            back[t] = cand.argmax(0); score = cand.max(0) + em[t]
        best = [int((score + E).argmax())]
        for t in range(L - 1, 0, -1): best.append(int(back[t, best[-1]]))
        return best[::-1]

class BiaffineSpanHead(nn.Module):
    def __init__(self, H, C, d=128, dropout=0.1):
        super().__init__()
        self.s_mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(H, d), nn.GELU())
        self.e_mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(H, d), nn.GELU())
        self.W = nn.Parameter(torch.empty(d + 1, C + 1, d + 1)); nn.init.xavier_uniform_(self.W)
    def forward(self, h):                                  # -> [B, L, L, C+1]
        one = h.new_ones(h.shape[:-1] + (1,))
        s = torch.cat([self.s_mlp(h), one], -1); e = torch.cat([self.e_mlp(h), one], -1)
        x = torch.einsum("bid,dce->bice", s, self.W.to(s.dtype))
        return torch.einsum("bice,bje->bijc", x, e)





def is_linear_like(m): return isinstance(m, nn.Linear) or type(m).__name__ == "Conv1D"   # HF GPT-2 uses Conv1D

def io_dims(m):
    if isinstance(m, nn.Linear): return m.in_features, m.out_features
    return m.weight.shape[0], m.weight.shape[1]                                          # Conv1D: [in, out]

def get_parent(root, dotted):
    parts = dotted.split("."); parent = root
    for p in parts[:-1]: parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    return parent, parts[-1]

def set_module(root, dotted, new):
    parent, last = get_parent(root, dotted)
    if last.isdigit(): parent[int(last)] = new
    else: setattr(parent, last, new)

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__(); self.base = base; i, o = io_dims(base)
        self.A = nn.Parameter(torch.empty(r, i)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(o, r)); self.scale = alpha / r; self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t().to(x.dtype) @ self.B.t().to(x.dtype)) * self.scale

class LoRMALinear(nn.Module):
    """LoRMA (Bihany, Patel & Modi, ACL Findings 2025): multiplicative low-rank adaptation, W' = I(BA) · W.
    variant 'plus': I(BA) = I + (alpha/r)·BA, B=0 at init. Re-ordered so W' is never materialised:
                    y = h + (alpha/r)·B(A h) + bias,  with h = W x (bias excluded).
    variant 'pi'  : I(BA) = rotate row i of BA cyclically by i. B[:,0]=1 (rest random), A=e_00 at init, so I(BA)=I.
                    Needs the d×d matrix (costlier); y = I(BA) h + bias."""
    def __init__(self, base, r, alpha, dropout, variant="plus"):
        super().__init__(); self.base = base; self.variant = variant; _, o = io_dims(base)
        self.drop = nn.Dropout(dropout); self.scale = alpha / r
        if variant == "plus":
            self.A = nn.Parameter(torch.empty(r, o)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.B = nn.Parameter(torch.zeros(o, r))
        else:
            B = torch.randn(o, r) * 0.02; B[:, 0] = 1.0
            A = torch.zeros(r, o); A[0, 0] = 1.0
            self.A, self.B = nn.Parameter(A), nn.Parameter(B)
            idx = (torch.arange(o)[None, :] - torch.arange(o)[:, None]) % o     # out[i,j] = M[i, (j-i) mod d]
            self.register_buffer("rot_idx", idx, persistent=False)
    def forward(self, x):
        y = self.base(x)
        bias = getattr(self.base, "bias", None)
        h = y - bias if bias is not None else y
        if self.variant == "plus":
            return y + (self.drop(h) @ self.A.t().to(h.dtype) @ self.B.t().to(h.dtype)) * self.scale
        M = (self.B @ self.A).gather(1, self.rot_idx)                           # I_pi(BA), [o, o]
        out = self.drop(h) @ M.t().to(h.dtype)
        return out + bias if bias is not None else out

class AdapterAfter(nn.Module):
    def __init__(self, base, bottleneck, dropout):
        super().__init__(); self.base = base; _, o = io_dims(base)
        self.down = nn.Linear(o, bottleneck); self.up = nn.Linear(bottleneck, o); self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self, x):
        y = self.base(x); return y + self.up(self.drop(F.gelu(self.down(y))))

def block_linears(bb):
    """(name, module) of all linear layers inside the transformer blocks (not embeddings / LM head)."""
    prefix = bb.blocks_attr + "."
    mha = {n for n, m in bb.model.named_modules() if isinstance(m, nn.MultiheadAttention)}
    # nn.MultiheadAttention reads out_proj.weight directly (never calls the module), so it can't be wrapped
    return [(n, m) for n, m in bb.model.named_modules()
            if n.startswith(prefix) and is_linear_like(m) and n.rsplit(".", 1)[0] not in mha]

def apply_method(bb, cfg):
    model = bb.model
    if cfg.method in ("full", "kd"):
        for p in model.parameters(): p.requires_grad = True
        return
    for p in model.parameters(): p.requires_grad = False
    if cfg.method in ("lora", "lorma"):
        targets = block_linears(bb)
        if cfg.peft_targets:
            targets = [(n, m) for n, m in targets if any(n.endswith("." + t) for t in cfg.peft_targets)]
        for n, m in targets:
            new = LoRALinear(m, cfg.r, cfg.alpha, cfg.peft_dropout) if cfg.method == "lora" else \
                  LoRMALinear(m, cfg.r, cfg.alpha, cfg.peft_dropout, cfg.lorma_variant)
            set_module(model, n, new)
        print(f"  {cfg.method}: wrapped {len(targets)} linear layers")
    elif cfg.method == "adapter":
        targets = [(n, m) for n, m in block_linears(bb) if any(n.endswith("." + t) for t in cfg.adapter_targets)]
        assert targets, f"no adapter targets matched {cfg.adapter_targets}; names: {[n for n,_ in block_linears(bb)][:12]}"
        for n, m in targets: set_module(model, n, AdapterAfter(m, cfg.adapter_dim, cfg.peft_dropout))
        print(f"  adapter: inserted after {len(targets)} output projections")
    elif cfg.method == "bitfit":
        cnt = 0
        for n, p in model.named_parameters():
            if n.endswith("bias"): p.requires_grad = True; cnt += 1
        if cnt == 0:
            print("  bitfit: model has no bias terms -> training normalization weights instead")
            for n, p in model.named_parameters():
                if "norm" in n.lower() or re.search(r"(^|\.)ln", n): p.requires_grad = True
    elif cfg.method == "partial":
        blocks = model.get_submodule(bb.blocks_attr)
        for blk in list(blocks)[-cfg.partial_k:]:
            for p in blk.parameters(): p.requires_grad = True
        for n, p in model.named_parameters():       # final norm(s) after the blocks
            if not n.startswith(bb.blocks_attr + ".") and ("norm" in n.lower() or re.search(r"(^|\.)ln", n)):
                p.requires_grad = True
    else:
        raise ValueError(cfg.method)

def truncate_backbone(bb, keep_layers):
    """Student initialisation: keep a subset of the Phase-1 transformer blocks (e.g. every other one)."""
    blocks = bb.model.get_submodule(bb.blocks_attr)
    set_module(bb.model, bb.blocks_attr, nn.ModuleList([blocks[i] for i in keep_layers]))
    c = getattr(bb.model, "config", None)
    for k in ("n_layer", "num_hidden_layers", "num_layers"):
        if c is not None and hasattr(c, k): setattr(c, k, len(keep_layers))
    return bb





@dataclass
class ExpConfig:
    name: str
    model: str                       # key of PHASE1_MODELS
    method: str                      # full | partial | lora | lorma | adapter | bitfit | kd
    head: str = "crf"                # softmax | crf | span
    window: int = 512
    stride: int = 384
    echo: bool = False
    add_bos: bool = False
    epochs: int = 10
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 5e-5                 # backbone / adapter params
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup: float = 0.06
    dropout: float = 0.1
    patience: int = 3
    class_weighting: str = "none"    # none | sqrt_inv  (token heads)
    o_weight: float = 1.0            # extra weight on the O tag (<1 = favour recall)
    span_max_len: int = 0            # 0 = auto (p99 of train entity lengths, capped at 64)
    span_none_weight: float = 1.0
    r: int = 16
    alpha: float = 32
    peft_dropout: float = 0.05
    peft_targets: tuple = ()         # e.g. ("c_attn","q_proj","v_proj"); empty = all block linears
    lorma_variant: str = "plus"      # plus | pi
    adapter_dim: int = 64
    adapter_targets: tuple = ("c_proj", "o_proj", "out_proj", "fc2", "down_proj", "wo", "w2", "dense_4h_to_h",
                              "output.dense")   # dotted suffixes of attention-output / FFN-output projections
    partial_k: int = 4
    kd_teacher: str = ""             # name of a finished experiment (its checkpoint is loaded)
    kd_student_model: str = ""       # "" -> layer-pruned copy of the teacher's Phase-1 model
    kd_keep: str = "alternate"       # alternate | first_half | comma list e.g. "0,3,6,9"
    kd_alpha: float = 0.5            # weight of the gold loss; (1-alpha) on the distillation loss
    kd_T: float = 2.0
    kd_hidden_weight: float = 0.0    # optional MSE between final hidden states (same width only)
    seed: int = 42
    note: str = ""

class NERModel(nn.Module):
    def __init__(self, bb, cfg, max_span):
        super().__init__(); self.bb, self.cfg, self.max_span = bb, cfg, max_span
        H = bb.hidden_size
        if cfg.head in ("softmax", "crf"):
            self.head = TokenHead(H, len(TAGS), cfg.dropout)
            self.crf = CRF(len(TAGS)) if cfg.head == "crf" else None
        else:
            self.head = BiaffineSpanHead(H, len(LABELS), dropout=cfg.dropout); self.crf = None
    def forward(self, input_ids, attention_mask, content_pos, return_hidden=False):
        h = self.bb(input_ids, attention_mask)
        h = h.gather(1, content_pos.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        out = self.head(h)
        return (out, h) if return_hidden else out

def build_model(cfg, max_span, backbone=None):
    bb = backbone or load_backbone(cfg.model)
    apply_method(bb, cfg)
    return NERModel(bb, cfg, max_span)

def task_loss(model, logits, batch, tag_w):
    cfg = model.cfg
    if cfg.head == "softmax":
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), batch["tags"].reshape(-1),
                               weight=tag_w, ignore_index=-100)
    if cfg.head == "crf":
        return model.crf.nll(logits, batch["tags"], batch["content_mask"])
    C = len(LABELS) + 1
    w = torch.ones(C, device=logits.device); w[0] = cfg.span_none_weight
    return F.cross_entropy(logits.float().reshape(-1, C), batch["span_tags"].reshape(-1), weight=w, ignore_index=-100)

def kd_loss(s_logits, t_logits, batch, model, T):
    """Hinton et al. (2015): KL(teacher_T || student_T) * T^2 over valid positions (tokens or spans)."""
    if model.cfg.head == "span":
        valid = span_valid_mask(batch["content_mask"], model.max_span)
    else:
        valid = batch["content_mask"]
    s = F.log_softmax(s_logits.float()[valid] / T, -1); t = F.softmax(t_logits.float()[valid] / T, -1)
    return F.kl_div(s, t, reduction="batchmean") * T * T

def tag_weights(encs, cfg):
    if cfg.head != "softmax" or (cfg.class_weighting == "none" and cfg.o_weight == 1.0): return None
    cnt = np.bincount(np.concatenate([d.tags for d in encs]), minlength=len(TAGS)).astype(float) + 1
    w = np.ones(len(TAGS)) if cfg.class_weighting == "none" else np.sqrt(cnt.sum() / cnt)
    w = w / w.mean(); w[0] *= cfg.o_weight
    return torch.tensor(w, dtype=torch.float, device=DEVICE)

def to_dev(batch): return {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}

def amp_ctx():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if DEVICE == "cuda" else torch.autocast("cpu", enabled=False)

@torch.no_grad()
def predict_docs(model, encs, tok, trim, batch_size=16):
    """Sliding-window inference, doc-level aggregation and decoding -> {doc_id: [(s,e,label)]}."""
    model.eval(); cfg = model.cfg
    exs = make_examples(encs, cfg, model.bb, tok, with_labels=False)
    exs.sort(key=lambda x: len(x["input_ids"]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    by_id = {d.doc_id: d for d in encs}
    acc = {d.doc_id: np.zeros((len(d.keep), len(TAGS)), dtype=np.float32) for d in encs} if cfg.head != "span" else None
    cnt = {d.doc_id: np.zeros(len(d.keep), dtype=np.float32) for d in encs}
    span_best = defaultdict(dict)
    for k in range(0, len(exs), batch_size):
        chunk = exs[k:k + batch_size]
        b = to_dev(collate(chunk, pad, with_labels=False))
        with amp_ctx():
            out = model(b["input_ids"], b["attention_mask"], b["content_pos"])
        out = out.float()
        if cfg.head == "softmax": out = F.log_softmax(out, -1)
        if cfg.head != "span":
            out = out.cpu().numpy()
            for x, o in zip(chunk, out):
                m = x["hi"] - x["lo"]; acc[x["doc"]][x["lo"]:x["hi"]] += o[:m]; cnt[x["doc"]][x["lo"]:x["hi"]] += 1
        else:
            probs = out.softmax(-1)
            valid = span_valid_mask(b["content_mask"], model.max_span)
            p_none = probs[..., 0]
            best_p, best_c = probs[..., 1:].max(-1)
            cand = valid & (best_p > p_none)
            idx = cand.nonzero().cpu().numpy(); scs = best_p[cand].cpu().numpy(); labs = best_c[cand].cpu().numpy()
            if len(idx) > 400 * len(chunk):                      # untrained model safeguard: keep the top candidates
                top = np.argsort(-scs)[:400 * len(chunk)]; idx, scs, labs = idx[top], scs[top], labs[top]
            for (bi, i, j), sc, c in zip(idx.tolist(), scs.tolist(), labs.tolist()):
                x = chunk[bi]; key = (x["lo"] + i, x["lo"] + j)
                if sc > span_best[x["doc"]].get(key, (0, -1))[0]: span_best[x["doc"]][key] = (sc, c)
    preds = {}
    for did, d in by_id.items():
        if cfg.head == "span":
            chosen, used = [], np.zeros(len(d.keep), dtype=bool)
            for (i, j), (sc, c) in sorted(span_best[did].items(), key=lambda kv: -kv[1][0]):
                if not used[i:j + 1].any(): chosen.append((i, j, c)); used[i:j + 1] = True
            tspans = sorted(chosen)
        else:
            em = acc[did] / np.maximum(cnt[did], 1)[:, None]
            if len(em) == 0: preds[did] = []; continue
            tags = model.crf.viterbi(em) if cfg.head == "crf" else em.argmax(-1).tolist()
            tspans = tags_to_token_spans(tags)
        preds[did] = token_spans_to_chars(d, tspans, trim)
    return preds

def evaluate(model, dev_encs, tok, trim):
    pred = predict_docs(model, dev_encs, tok, trim)
    return strict_scores({d.doc_id: TRAIN_GOLD[d.doc_id] for d in dev_encs}, pred), pred

def train_model(model, cfg, trn_encs, dev_encs, tok, trim, teacher=None):
    seed_all(cfg.seed)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    train_ex = make_examples(trn_encs, cfg, model.bb, tok, with_labels=True)
    max_span = model.max_span if cfg.head == "span" else None
    tag_w = tag_weights(trn_encs, cfg)
    head_params = list(model.head.parameters()) + (list(model.crf.parameters()) if model.crf is not None else [])
    head_ids = {id(p) for p in head_params}
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    decay = lambda ps: [p for p in ps if p.ndim >= 2]; no_decay = lambda ps: [p for p in ps if p.ndim < 2]
    opt = torch.optim.AdamW([
        {"params": decay(body), "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": no_decay(body), "lr": cfg.lr, "weight_decay": 0.0},
        {"params": head_params, "lr": cfg.head_lr, "weight_decay": 0.0}])
    steps = math.ceil(len(train_ex) / (cfg.batch_size * cfg.grad_accum)) * cfg.epochs
    warm = int(cfg.warmup * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda" and AMP_DTYPE == torch.float16))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    print(f"  windows: {len(train_ex)} | optimizer steps: {steps} | trainable params: {n_params(model, True)/1e6:.2f}M")
    best_f1, best_state, bad, history = -1, None, 0, []
    if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train(); random.shuffle(train_ex); tot, nb = 0.0, 0
        for k in range(0, len(train_ex), cfg.batch_size):
            b = to_dev(collate(train_ex[k:k + cfg.batch_size], pad, max_span))
            with amp_ctx():
                logits, h = model(b["input_ids"], b["attention_mask"], b["content_pos"], return_hidden=True)
                loss = task_loss(model, logits, b, tag_w)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits, t_h = teacher(b["input_ids"], b["attention_mask"], b["content_pos"], return_hidden=True)
                    loss = cfg.kd_alpha * loss + (1 - cfg.kd_alpha) * kd_loss(logits, t_logits, b, model, cfg.kd_T)
                    if cfg.kd_hidden_weight > 0 and h.size(-1) == t_h.size(-1):
                        m = b["content_mask"]
                        loss = loss + cfg.kd_hidden_weight * F.mse_loss(h.float()[m], t_h.float()[m])
            scaler.scale(loss / cfg.grad_accum).backward()
            nb += 1; tot += loss.item()
            if nb % cfg.grad_accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        sc, _ = evaluate(model, dev_encs, tok, trim)
        history.append(dict(epoch=ep + 1, loss=tot / max(1, nb), dev_macro_F1=sc["macro_F1"], dev_micro_F1=sc["micro_F1"]))
        print(f"  ep {ep+1:2d} | loss {tot/max(1,nb):.4f} | dev macro-F1 {sc['macro_F1']:.4f} micro-F1 {sc['micro_F1']:.4f} | {time.time()-t0:.0f}s")
        if sc["macro_F1"] > best_f1:
            best_f1, bad = sc["macro_F1"], 0
            best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items() if n in trainable}
        else:
            bad += 1
            if bad >= cfg.patience: print("  early stop"); break
    model.load_state_dict(best_state, strict=False)
    stats = dict(train_time_s=time.time() - t0,
                 peak_mem_GB=(torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0))
    return history, stats, best_state





def load_experiment_model(name):
    ck = torch.load(os.path.join(OUT_DIR, name, "model.pt"), map_location="cpu")
    cfg = ExpConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})
    if cfg.method == "kd":
        bb = load_backbone(cfg.kd_student_model or cfg.model)
        if not cfg.kd_student_model: truncate_backbone(bb, ck["keep_layers"])
        model = build_model(cfg, ck["max_span"], backbone=bb)
    else:
        model = build_model(cfg, ck["max_span"])
    model.load_state_dict(ck["state"], strict=False)
    return model.to(DEVICE).eval(), cfg

def student_keep_layers(n, mode):
    if mode == "alternate": return list(range(0, n, 2))
    if mode == "first_half": return list(range(n // 2))
    return [int(x) for x in mode.split(",")]





def write_submission(preds, name):
    out = {}
    for did, text in TEST_TEXT.items():                          # every official ID, nothing extra
        ents = []
        for s, e, l in sorted(set(preds.get(did, []))):
            assert 0 <= s < e <= len(text) and l in LABELS, (did, s, e, l)
            ents.append([int(s), int(e) - (1 if END_INCLUSIVE else 0), l])
        out[did] = ents
    assert set(out) == set(TEST_TEXT)
    d = os.path.join(OUT_DIR, name); os.makedirs(d, exist_ok=True)
    pj = os.path.join(d, "prediction.json")
    json.dump(out, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    zp = os.path.join(d, "submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")                  # at the archive ROOT
    with zipfile.ZipFile(zp) as z: assert z.namelist() == ["prediction.json"]
    print(f"  wrote {zp} | {sum(map(len, out.values()))} entities over {len(out)} docs")
    return zp





def oracle_selftest(tok_path=TOKENIZER_PATH):
    train, test, trim = get_encodings(tok_path)
    dev = [train[d] for d in DEV_IDS]
    pred = {d.doc_id: token_spans_to_chars(d, tags_to_token_spans(d.tags), trim) for d in dev}
    sc = strict_scores({d: TRAIN_GOLD[d] for d in DEV_IDS}, pred)
    print(f"oracle dev macro-F1 {sc['macro_F1']:.4f} | micro-F1 {sc['micro_F1']:.4f}  (= alignment ceiling)")
    print("oracle errors:", error_analysis({d: TRAIN_GOLD[d] for d in DEV_IDS}, pred, TRAIN_TEXT, 3)["examples"])
    s = json.loads(json.dumps({k: [[a, b, c] for a, b, c in v] for k, v in pred.items()}))
    assert all(tuple(x) in set(pred[k]) for k, v in s.items() for x in v), "JSON round-trip changed offsets"
oracle_selftest()





RESULTS_PATH = os.path.join(OUT_DIR, "results.jsonl")

def run_experiment(cfg, make_submission=True):
    print(f"\n=== {cfg.name} | model={cfg.model} method={cfg.method} head={cfg.head} ===")
    seed_all(cfg.seed)
    tok_path = tokenizer_path_for(cfg.kd_student_model or cfg.model)
    tok = load_tokenizer(tok_path)
    train, test, trim = get_encodings(tok_path)
    trn, dev = [train[d] for d in TRN_IDS], [train[d] for d in DEV_IDS]
    lens = [j - i + 1 for d in trn for i, j, _ in d.spans]
    max_span = cfg.span_max_len or int(min(64, np.percentile(lens, 99) + 1))
    teacher, keep = None, None
    if cfg.method == "kd":
        teacher, tcfg = load_experiment_model(cfg.kd_teacher)
        for p in teacher.parameters(): p.requires_grad = False
        assert tcfg.head == cfg.head, "teacher and student must use the same head"
        assert tokenizer_path_for(tcfg.model) == tok_path, "teacher and student must share the tokenizer"
        cfg.model = tcfg.model if not cfg.kd_student_model else cfg.model
        cfg.window, cfg.stride, cfg.echo, cfg.add_bos = tcfg.window, tcfg.stride, tcfg.echo, tcfg.add_bos
        max_span = teacher.max_span
        bb = load_backbone(cfg.kd_student_model or tcfg.model)
        if not cfg.kd_student_model:
            keep = student_keep_layers(len(bb.model.get_submodule(bb.blocks_attr)), cfg.kd_keep)
            truncate_backbone(bb, keep)
        model = build_model(cfg, max_span, backbone=bb).to(DEVICE)
        print(f"  teacher {cfg.kd_teacher}: {n_params(teacher)/1e6:.1f}M params -> student {n_params(model)/1e6:.1f}M (layers {keep})")
    else:
        model = build_model(cfg, max_span).to(DEVICE)
    if cfg.echo and not model.bb.is_causal: print("  note: echo on a bidirectional backbone has no benefit")
    total, trainable = n_params(model), n_params(model, True)

    history, stats, state = train_model(model, cfg, trn, dev, tok, trim, teacher)
    t0 = time.time(); dev_sc, dev_pred = evaluate(model, dev, tok, trim); dev_time = time.time() - t0
    err = error_analysis({d: TRAIN_GOLD[d] for d in DEV_IDS}, dev_pred, TRAIN_TEXT, n_examples=3)

    d = os.path.join(OUT_DIR, cfg.name); os.makedirs(d, exist_ok=True)
    torch.save(dict(cfg=asdict(cfg), state=state, max_span=max_span, keep_layers=keep), os.path.join(d, "model.pt"))
    if make_submission:
        t0 = time.time(); test_pred = predict_docs(model, [test[k] for k in TEST_TEXT], tok, trim)
        stats["test_infer_s"] = time.time() - t0
        write_submission(test_pred, cfg.name)

    row = dict(name=cfg.name, model=cfg.model, method=cfg.method, head=cfg.head, echo=cfg.echo,
               total_params_M=total / 1e6, trainable_params_M=trainable / 1e6, trainable_pct=100 * trainable / total,
               dev_macro_F1=dev_sc["macro_F1"], dev_macro_P=dev_sc["macro_P"], dev_macro_R=dev_sc["macro_R"],
               dev_micro_F1=dev_sc["micro_F1"], dev_infer_s_per_doc=dev_time / len(dev), **stats,
               per_label_F1={l: round(v["F1"], 4) for l, v in dev_sc["per_label"].items()},
               errors=err["categories"], confusions=err["label_confusion"], history=history, cfg=asdict(cfg))
    with open(RESULTS_PATH, "a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")
    print(f"  DEV macro-F1 {dev_sc['macro_F1']:.4f} (P {dev_sc['macro_P']:.4f} R {dev_sc['macro_R']:.4f}) | "
          f"params {total/1e6:.1f}M, trainable {trainable/1e6:.2f}M ({100*trainable/total:.2f}%) | "
          f"peak mem {stats['peak_mem_GB']:.1f}GB | train {stats['train_time_s']/60:.1f} min")
    print("  errors:", err["categories"]); print("  top confusions:", err["label_confusion"][:5])
    del model, teacher; gc.collect(); torch.cuda.empty_cache()
    return row





MODEL_KEYS = list(PHASE1_MODELS)                 # e.g. ['M1_decoder', 'M2_encdec', 'M3_hybrid', 'M4_efficient']
M1, M2, M3, M4 = (MODEL_KEYS + MODEL_KEYS * 4)[:4]   # falls back to repeats if <4 are registered (fix that!)

EXPERIMENTS = {
    "E1_full":    ExpConfig("E1_full_" + M1,    M1, "full",    head="crf", lr=3e-5, epochs=10),
    "E2_lorma":   ExpConfig("E2_lorma_" + M1,   M1, "lorma",   head="crf", lr=1e-3, r=16, alpha=32, epochs=12),
    "E3_lora":    ExpConfig("E3_lora_" + M2,    M2, "lora",    head="crf", lr=5e-4, r=16, alpha=32, epochs=12),
    "E4_adapter": ExpConfig("E4_adapter_" + M3, M3, "adapter", head="crf", lr=1e-3, adapter_dim=64, epochs=12),
    "E5_partial": ExpConfig("E5_partial_" + M4, M4, "partial", head="crf", lr=1e-4, partial_k=4, epochs=12),
    "E6_kd":      ExpConfig("E6_kd_" + M1,      M1, "kd",      head="crf", lr=5e-5, epochs=12,
                            kd_teacher="E1_full_" + M1, kd_keep="alternate", kd_alpha=0.5, kd_T=2.0),
    # ---- optional extra submissions / ablations ----
    "E7_bitfit":  ExpConfig("E7_bitfit_" + M2,  M2, "bitfit",  head="crf", lr=1e-3, epochs=12),
    "E8_lormapi": ExpConfig("E8_lormapi_" + M3, M3, "lorma",   head="crf", lr=5e-4, r=8, lorma_variant="pi", epochs=12),
    "E9_span":    ExpConfig("E9_span_" + M1,    M1, "full",    head="span", lr=3e-5, batch_size=4, grad_accum=2, epochs=10),
    "E10_echo":   ExpConfig("E10_echo_" + M1,   M1, "lorma",   head="crf", lr=1e-3, echo=True, window=256, stride=192, epochs=12),
    "E11_lorma_M4": ExpConfig("E11_lorma_" + M4, M4, "lorma",  head="crf", lr=1e-3, epochs=12),
}


def time_baseline(cfg, n_warm=2, n_steps=6, n_eval_docs=3):
    seed_all(cfg.seed)
    tok_path = tokenizer_path_for(cfg.model); tok = load_tokenizer(tok_path)
    train, _, trim = get_encodings(tok_path)
    trn, dev = [train[d] for d in TRN_IDS], [train[d] for d in DEV_IDS]
    lens = [j - i + 1 for d in trn for i, j, _ in d.spans]
    max_span = cfg.span_max_len or int(min(64, np.percentile(lens, 99) + 1))
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    model = build_model(cfg, max_span).to(DEVICE).train()
    ex = make_examples(trn, cfg, model.bb, tok, with_labels=True)
    random.Random(0).shuffle(ex)
    mspan = max_span if cfg.head == "span" else None
    tag_w = tag_weights(trn, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda" and AMP_DTYPE == torch.float16))
    steps_per_epoch = math.ceil(len(ex) / (cfg.batch_size * cfg.grad_accum))
    if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()

    times, k = [], 0
    for step in range(n_warm + n_steps):
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.time(); opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            chunk = [ex[(k + i) % len(ex)] for i in range(cfg.batch_size)]; k += cfg.batch_size
            b = to_dev(collate(chunk, pad, mspan))
            with amp_ctx():
                logits = model(b["input_ids"], b["attention_mask"], b["content_pos"])
                loss = task_loss(model, logits, b, tag_w)
            scaler.scale(loss / cfg.grad_accum).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()
        if DEVICE == "cuda": torch.cuda.synchronize()
        if step >= n_warm: times.append(time.time() - t0)
    sec_step = float(np.mean(times))

    # dev evaluation cost (runs after every epoch), scaled from a few docs
    sub = dev[:n_eval_docs]; t0 = time.time(); predict_docs(model, sub, tok, trim)
    dev_epoch = (time.time() - t0) / len(sub) * len(dev)

    epoch = steps_per_epoch * sec_step + dev_epoch
    mem = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0
    print(f"{cfg.name:<28} windows {len(ex):>5} | {steps_per_epoch:>4} steps/epoch | {sec_step:5.2f} s/step | "
          f"dev eval {dev_epoch/60:4.1f} min | EPOCH ~{epoch/60:5.1f} min | "
          f"full run ({cfg.epochs} ep) ~{epoch*cfg.epochs/3600:4.1f} h | peak {mem:.1f} GB")
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return epoch

for key in ["E1_full", "E2_lorma", "E5_partial", "E9_noecho"]:
    time_baseline(EXPERIMENTS[key])


# Quick dry run (catches bugs in minutes): 1 epoch on a few docs, no submission.
DRY_RUN = False
if DRY_RUN:
    _TRN_BAK = TRN_IDS; TRN_IDS = TRN_IDS[:4]
    run_experiment(ExpConfig("dry_" + M1, M1, "lorma", 
                             epochs=1), make_submission=False)
    TRN_IDS = _TRN_BAK


row = run_experiment(EXPERIMENTS["E1_full"])          # -> upload OUT_DIR/E1_full_*/submission.zip


row = run_experiment(EXPERIMENTS["E2_lorma"])


row = run_experiment(EXPERIMENTS["E3_lora"])


row = run_experiment(EXPERIMENTS["E4_adapter"])


row = run_experiment(EXPERIMENTS["E5_partial"])


row = run_experiment(EXPERIMENTS["E6_kd"])            # needs E1 finished (loads its checkpoint)


# Optional extras, e.g.:
# for k in ["E7_bitfit", "E8_lormapi", "E9_span", "E10_echo", "E11_lorma_M4"]: run_experiment(EXPERIMENTS[k])





import pandas as pd
rows = [json.loads(l) for l in open(RESULTS_PATH, encoding="utf-8")]
df = pd.DataFrame(rows).drop_duplicates("name", keep="last")
cols = ["name", "model", "method", "head", "dev_macro_F1", "dev_macro_P", "dev_macro_R", "dev_micro_F1",
        "total_params_M", "trainable_params_M", "trainable_pct", "peak_mem_GB", "train_time_s", "dev_infer_s_per_doc"]
display(df[cols].sort_values("dev_macro_F1", ascending=False).round(4))

# per-label F1 across experiments: which entity types stay hard?
display(pd.DataFrame({r["name"]: r["per_label_F1"] for r in df.to_dict("records")}).round(3))

# error categories: boundary vs label vs spurious vs missed
display(pd.DataFrame({r["name"]: r["errors"] for r in df.to_dict("records")}).fillna(0).astype(int))


# Detailed error analysis for one experiment (examples of boundary / label errors, confusions)
def inspect(name, n=5):
    model, cfg = load_experiment_model(name)
    tok_path = tokenizer_path_for(cfg.kd_student_model or cfg.model)
    tok = load_tokenizer(tok_path); train, _, trim = get_encodings(tok_path)
    sc, pred = evaluate(model, [train[d] for d in DEV_IDS], tok, trim)
    ea = error_analysis({d: TRAIN_GOLD[d] for d in DEV_IDS}, pred, TRAIN_TEXT, n_examples=n)
    print(json.dumps({k: v for k, v in ea.items() if k != "examples"}, indent=1))
    for k, ex in ea["examples"].items():
        print(f"\n--- {k} ---")
        for e in ex: print("  pred:", e[0], repr(e[1]), "| gold:", e[2])
    del model; gc.collect(); torch.cuda.empty_cache()

# inspect(EXPERIMENTS["E1_full"].name)








M1 = "M1_lexweave"
PEFT_T = ("attn.q", "attn.kv", "attn.out", "up", "down")
COMMON = dict(head="crf", echo=True, window=512, stride=384, batch_size=4, grad_accum=2)

cfg_e3_long = ExpConfig(
    "E3_lora_long_" + M1, M1, "lora",
    lr=5e-4, r=16, alpha=32, peft_targets=PEFT_T,
    epochs=25,        # change this to the number of epochs you want
    patience=999,     # larger than epochs, so early stopping never triggers
    **COMMON,
)

row = run_experiment(cfg_e3_long)   #  writes OUT_DIR/E3_lora_long_M1_lexweave/submission.zip


import glob, os
name = "E3_lora_long_M1_lexweave"
d = os.path.join(OUT_DIR, name)
print("OUT_DIR:", OUT_DIR)
print("files on disk:", sorted(glob.glob(f"{OUT_DIR}/**/*", recursive=True))[:40])

zp, mp = os.path.join(d, "submission.zip"), os.path.join(d, "model.pt")

if os.path.exists(zp):
    from IPython.display import FileLink, display
    print("submission.zip already exists, nothing to rebuild:")
    display(FileLink(zp))

elif os.path.exists(mp):
    print("model.pt found (training had finished), rebuilding the submission from it...")
    model, cfg = load_experiment_model(name)
    tok_path = tokenizer_path_for(cfg.kd_student_model or cfg.model)
    tok = load_tokenizer(tok_path)
    train, test, trim = get_encodings(tok_path)
    test_pred = predict_docs(model, [test[k] for k in TEST_TEXT], tok, trim)
    zp = write_submission(test_pred, name)
    from IPython.display import FileLink, display
    display(FileLink(zp))

else:
    print("No model.pt or submission.zip for this run. Training never finished, so the weights were lost with the kernel.")


import dataclasses
SRC          = "E3_lora_M1_lexweave"        # saved run to continue from (folder name in OUT_DIR)
NEW          = "E3_lora_cont_M1_lexweave"   # name for the continued run
EXTRA_EPOCHS = 20
NEW_LR       = 2e-4                          # lower than the original 5e-4, since this is a continuation

# --- load the saved run (rebuilds LoRA wrappers + head/CRF and loads the trained weights) ---
model, cfg0 = load_experiment_model(SRC)
meta = torch.load(os.path.join(OUT_DIR, SRC, "model.pt"), map_location="cpu")
max_span, keep = meta["max_span"], meta["keep_layers"]; del meta
cfg = dataclasses.replace(cfg0, name=NEW, epochs=EXTRA_EPOCHS, patience=999, lr=NEW_LR, warmup=0.02)

tok_path = tokenizer_path_for(cfg.model); tok = load_tokenizer(tok_path)
train, test, trim = get_encodings(tok_path)
trn, dev = [train[d] for d in TRN_IDS], [train[d] for d in DEV_IDS]

# --- remember the starting point so the continuation can never end up worse than it ---
tnames = {n for n, p in model.named_parameters() if p.requires_grad}
start_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items() if n in tnames}
start_sc, _ = evaluate(model, dev, tok, trim)
print(f"starting dev macro-F1 (from {SRC}): {start_sc['macro_F1']:.4f}")

# --- continue training ---
history, stats, best_state = train_model(model, cfg, trn, dev, tok, trim)

# train_model has reloaded its best epoch; compare that with where we started
final_sc, _ = evaluate(model, dev, tok, trim)
if final_sc["macro_F1"] < start_sc["macro_F1"]:
    print(f"continuation did not beat the start ({final_sc['macro_F1']:.4f} < {start_sc['macro_F1']:.4f}); restoring the original weights")
    model.load_state_dict(start_state, strict=False); best_state = start_state

# --- evaluate, save, submit (same as run_experiment) ---
t0 = time.time(); dev_sc, dev_pred = evaluate(model, dev, tok, trim); dev_time = time.time() - t0
err = error_analysis({d: TRAIN_GOLD[d] for d in DEV_IDS}, dev_pred, TRAIN_TEXT, n_examples=3)

d = os.path.join(OUT_DIR, NEW); os.makedirs(d, exist_ok=True)
torch.save(dict(cfg=asdict(cfg), state=best_state, max_span=max_span, keep_layers=keep), os.path.join(d, "model.pt"))

t0 = time.time(); test_pred = predict_docs(model, [test[k] for k in TEST_TEXT], tok, trim)
stats["test_infer_s"] = time.time() - t0
zp = write_submission(test_pred, NEW)

total, trainable = n_params(model), n_params(model, True)
row = dict(name=NEW, model=cfg.model, method=cfg.method, head=cfg.head, echo=cfg.echo,
           total_params_M=total / 1e6, trainable_params_M=trainable / 1e6, trainable_pct=100 * trainable / total,
           dev_macro_F1=dev_sc["macro_F1"], dev_macro_P=dev_sc["macro_P"], dev_macro_R=dev_sc["macro_R"],
           dev_micro_F1=dev_sc["micro_F1"], dev_infer_s_per_doc=dev_time / len(dev), **stats,
           per_label_F1={l: round(v["F1"], 4) for l, v in dev_sc["per_label"].items()},
           errors=err["categories"], confusions=err["label_confusion"], history=history, cfg=asdict(cfg))
with open(RESULTS_PATH, "a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")
print(f"DONE | dev macro-F1 {dev_sc['macro_F1']:.4f} (start was {start_sc['macro_F1']:.4f}) | {zp}")

from IPython.display import FileLink, display
display(FileLink(zp))


import sys, time, os, json, torch
from dataclasses import asdict

tb, fr = sys.last_traceback, {}
assert tb is not None, "No traceback in memory, so this run can't be recovered."
while tb is not None:
    fr[tb.tb_frame.f_code.co_name] = tb.tb_frame
    tb = tb.tb_next
tm = fr["train_model"].f_locals
best_state, history = tm["best_state"], tm["history"]
assert best_state is not None, "Interrupted before the first epoch finished, so nothing to recover."
print("epochs finished:", len(history), "| best dev macro-F1 in this run:",
      round(max(h["dev_macro_F1"] for h in history), 4))

# restore the best finished epoch, then make sure it is not worse than where this continuation started
model.load_state_dict(best_state, strict=False)
final_sc, _ = evaluate(model, dev, tok, trim)
if final_sc["macro_F1"] < start_sc["macro_F1"]:
    print(f"did not beat the start ({final_sc['macro_F1']:.4f} < {start_sc['macro_F1']:.4f}); restoring original weights")
    model.load_state_dict(start_state, strict=False); best_state = start_state

stats = dict(train_time_s=time.time() - tm["t0"], peak_mem_GB=torch.cuda.max_memory_allocated() / 1e9)

t0 = time.time(); dev_sc, dev_pred = evaluate(model, dev, tok, trim); dev_time = time.time() - t0
err = error_analysis({d: TRAIN_GOLD[d] for d in DEV_IDS}, dev_pred, TRAIN_TEXT, n_examples=3)

d = os.path.join(OUT_DIR, NEW); os.makedirs(d, exist_ok=True)
torch.save(dict(cfg=asdict(cfg), state=best_state, max_span=max_span, keep_layers=keep), os.path.join(d, "model.pt"))

t0 = time.time(); test_pred = predict_docs(model, [test[k] for k in TEST_TEXT], tok, trim)
stats["test_infer_s"] = time.time() - t0
zp = write_submission(test_pred, NEW)

total, trainable = n_params(model), n_params(model, True)
row = dict(name=NEW, model=cfg.model, method=cfg.method, head=cfg.head, echo=cfg.echo,
           total_params_M=total / 1e6, trainable_params_M=trainable / 1e6, trainable_pct=100 * trainable / total,
           dev_macro_F1=dev_sc["macro_F1"], dev_macro_P=dev_sc["macro_P"], dev_macro_R=dev_sc["macro_R"],
           dev_micro_F1=dev_sc["micro_F1"], dev_infer_s_per_doc=dev_time / len(dev), **stats,
           per_label_F1={l: round(v["F1"], 4) for l, v in dev_sc["per_label"].items()},
           errors=err["categories"], confusions=err["label_confusion"], history=history, cfg=asdict(cfg))
with open(RESULTS_PATH, "a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")
print(f"DONE | dev macro-F1 {dev_sc['macro_F1']:.4f} (start was {start_sc['macro_F1']:.4f}) | {zp}")

from IPython.display import FileLink, display
display(FileLink(zp))


from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
from IPython.display import FileLink, display
import os
import shutil

phase = "phase2"
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
