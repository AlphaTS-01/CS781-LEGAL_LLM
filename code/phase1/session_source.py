


get_ipython().run_line_magic("pip", " install \"ijson>=3.2,<4\" \"tokenizers>=0.20,<1\" \"huggingface_hub>=0.26,<2\" \"requests>=2.31,<3\"")





from pathlib import Path
import os
ARCH='relay_lstm'
MODEL_SIZE='200m'          # small, 150m, 200m. Only ONE model is trained.
ABLATION='full'
RUN_TAG='v1'                  # Same tag restores/resumes; new experiment => new tag.
WORK=Path('/kaggle/working/cs781_remaining') if Path('/kaggle').exists() else Path.home()/'cs781_remaining'
# A COMPLETE earlier shared_stream_v2 cache (tokenizer, packed shards, manifests, dev set) can be attached here.
READY_CACHE=None              # e.g. Path('/kaggle/input/my-cache/shared_stream_v2')
CACHE=Path(READY_CACHE) if READY_CACHE else WORK/'shared_quick_v1'
DATA_MODE='stream'            # HF private official data. Optional 'local' with DATA_ROOT.
DATA_ROOT=None
REPO='Exploration-Lab/CS-781-Capstone'
HF_REVISION=None              # pinned automatically; restored run reuses original commit.
FOUNDATION_PATH=None          # Optional ZIP/folder: reuse ONLY your tokenizer, never its model weights.
TOKENIZER_PATH=None           # Or explicit path to your existing tokenizer.json.
RECONSTRUCT_OLD_SELECTION=False
EXCLUDE_OLD_TRAIN=False       # Scratch architecture comparison; reuse of old TRAIN text is allowed here.
OLD_CACHE=None
OLD_CHECKPOINT=None
START_MODE='scratch'
SEED=42
CONTEXT=1024
SOURCE_CONTEXT=768            # Unused by these decoder-only models; kept for shared cache compatibility.
TARGET_CONTEXT=256
VOCAB=16384
BPE_CHARS=30_000_000
VAL_FRACTION=.01
# Read bounded prefixes, NOT entire 4.5B-token corpus. Set None for full scan in a NEW cache.
SCAN_CHAR_LIMITS={'train_acts':100_000_000,'train_high_courts':400_000_000,
                  'train_supreme_court':250_000_000,'train_tribunals':250_000_000}
CHAR_BUDGETS={s:10**15 for s in SCAN_CHAR_LIMITS or
              ['train_acts','train_high_courts','train_supreme_court','train_tribunals']}
SOURCE_WEIGHTS={'train_acts':.10,'train_high_courts':.40,'train_supreme_court':.25,'train_tribunals':.25}
SHARD_DOCS=1000
DEV_PER_SOURCE=32
MICRO_BATCH=2                 # Start conservatively, benchmark; lower to 1 for OOM.
ACCUM=8
GRAD_CHECKPOINT=True
TRAIN_HOURS=6.0               # Used ONCE to derive token/LR budget from your measured throughput.
MAX_HOURS=6.0                 # Per invocation stopping time; does not finish an incomplete LR schedule.
TRAIN_TARGET_TOKENS=None      # None => auto budget; or set an explicit token count BEFORE first run.
WARMUP_TOKENS=0               # Set after benchmark to 2% of fixed token budget.
LR=3e-4
MIN_LR=3e-5
EVAL_EVERY=100
EVAL_BATCHES=12
SAVE_EVERY=50
LOCAL_EVERY_SECONDS=300
REMOTE_EVERY_SECONDS=1200     # Remote checkpoint at next local save after this interval.
# IMPORTANT INPUT: private repo you own; code creates it if absent.
BACKUP_MODE='local_only'              # 'local_only' explicitly opts out of durable remote backup.
BACKUP_REPO=None              # SET e.g. 'yourname/cs781-private-checkpoints'
RESTORE_REMOTE=True
os.environ['TOKENIZERS_PARALLELISM']='true'
os.environ['RAYON_NUM_THREADS']=str(min(8,os.cpu_count() or 1))
WORK.mkdir(parents=True,exist_ok=True)
RUN=WORK/'runs'/f'{ARCH}_{MODEL_SIZE}_{ABLATION}_{RUN_TAG}'
RUN.mkdir(parents=True,exist_ok=True)
print('RUN:',RUN,'\nCACHE:',CACHE)






import os, json, math, time, random, hashlib, sqlite3, re, gc, zipfile, shutil
from pathlib import Path
from collections import Counter, defaultdict
from contextlib import nullcontext
import numpy as np
import ijson
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SOURCES = ['train_acts','train_high_courts','train_supreme_court','train_tribunals']

def atomic_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()

def fingerprint(path):
    if isinstance(path,RemoteJSON):
        return {'name':path.name,'bytes':path.size,'repo':REPO,'path':path.path,'revision':path.revision}
    path = Path(path)
    with path.open('rb') as f:
        a=f.read(1<<20); f.seek(max(0,path.stat().st_size-(1<<20))); b=f.read()
    return {'name':path.name,'bytes':path.stat().st_size,'edge_sha':hashlib.sha256(a+b).hexdigest()}

def records(path, require_id=False):
    with open_binary(path) as f:
        if path.suffix == '.jsonl':
            for i,line in enumerate(f):
                if line.strip():
                    row=json.loads(line)
                    if require_id and (not isinstance(row,dict) or 'id' not in row):
                        raise ValueError('An explicit ID is required')
                    yield str(row.get('id',i)) if isinstance(row,dict) else str(i),row
            return
        first=f.peek(4096).lstrip()
        if first.startswith(b'['):
            for i,row in enumerate(ijson.items(f,'item')):
                if require_id and (not isinstance(row,dict) or 'id' not in row):
                    raise ValueError('Official array records must carry IDs')
                yield str(row.get('id',i)) if isinstance(row,dict) else str(i),row
        elif first.startswith(b'{'):
            for key,row in ijson.kvitems(f,''): yield str(key),row
        else: raise ValueError(f'Expected JSON object/array or .jsonl: {path}')

def text_of(row):
    if isinstance(row,str): return row
    if isinstance(row,dict):
        for k in ('text','document','content','case_text','judgment','prompt'):
            if isinstance(row.get(k),str): return row[k]
    raise ValueError('Unrecognized text schema; inspect the record rather than silently skipping it')

def clean(s):
    # Preserve paragraphs, spelling, dates, citations, tables and short orders.
    return s.replace('\r\n','\n').replace('\r','\n').replace('\x00','').strip()

def canonical(s): return ' '.join(clean(s).split())
def digest(s): return hashlib.sha256(canonical(s).encode()).hexdigest()
def uniform(key, salt):
    return int(hashlib.sha256((salt+key).encode()).hexdigest()[:16],16)/2**64

def prepare_data():
    CACHE.mkdir(parents=True, exist_ok=True)
    paths=resolve_sources()
    reconstructed=reconstruct_old_selection(paths)
    old_paths={}
    if OLD_CACHE is not None:
        for split in ('train','val'):
            p=OLD_CACHE/(split+'.jsonl')
            if p.exists(): old_paths[split]=p
    if EXCLUDE_OLD_TRAIN and 'train' not in old_paths and reconstructed is None:
        raise ValueError('Strict exclusion requires OLD_CACHE/train.jsonl. Token .bin files are insufficient.')
    if START_MODE=='continue' and 'val' not in old_paths and reconstructed is None:
        raise ValueError('Continuation requires old val.jsonl or verified manifest reconstruction to protect validation.')
    cfg=dict(version=5,scan_char_limits=SCAN_CHAR_LIMITS,files=[fingerprint(p) for p in paths],budgets=CHAR_BUDGETS,
             seed=SEED,val_fraction=VAL_FRACTION,exclude_old_train=EXCLUDE_OLD_TRAIN,
             old={k:sha(p) for k,p in old_paths.items()},
             reconstruction_sha=sha(CACHE/'old_selection_report.json') if reconstructed else None)
    manifest=CACHE/'data_config.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())==cfg,'Data config changed: choose a new CACHE_DIR.'
    else: atomic_json(cfg,manifest)
    db=sqlite3.connect(CACHE/'corpus.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS old (h TEXT PRIMARY KEY, split TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS seen (h TEXT PRIMARY KEY)')
    db.execute('CREATE TABLE IF NOT EXISTS docs (h TEXT PRIMARY KEY, source TEXT, id TEXT, split TEXT, text TEXT, meta TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS progress (source TEXT PRIMARY KEY, state TEXT)')
    def state(k):
        r=db.execute('SELECT state FROM progress WHERE source=?',(k,)).fetchone()
        return json.loads(r[0]) if r else None
    def save(k,v): db.execute('INSERT OR REPLACE INTO progress VALUES (?,?)',(k,json.dumps(v)))
    try:
        if state('_old') is None:
            if reconstructed is not None:
                old_db=sqlite3.connect(reconstructed)
                try:
                    db.executemany('INSERT INTO old VALUES (?,?)',old_db.execute('SELECT h,split FROM membership'))
                finally:old_db.close()
            for split,p in old_paths.items():
                for key,row in records(p):
                    h=digest(text_of(row))
                    found=db.execute('SELECT split FROM old WHERE h=?',(h,)).fetchone()
                    if found and found[0]!=split: raise ValueError('Old train/val overlap detected')
                    db.execute('INSERT OR IGNORE INTO old VALUES (?,?)',(h,split))
            save('_old',{'done':True});db.commit()
        for source,path in zip(SOURCES,paths):
            st=state(source) or {'index':0,'done':False,'counts':{}}
            if st['done']: print(source,'already complete');continue
            counts=Counter(st['counts']);skip=st['index']; start=time.monotonic();last=start
            probability=min(1.,CHAR_BUDGETS[source]/max(1,path.stat().st_size))
            print(source,'selection probability',round(probability,4),'resume record',skip,flush=True)
            # Resume reparses the prefix of monolithic JSON but does not redo its cleaning/hashing.
            index=skip
            for index,(key,row) in enumerate(records(path),1):
                if index<=skip:continue
                if SCAN_CHAR_LIMITS and counts['chars_scanned']>=SCAN_CHAR_LIMITS[source]:
                    index-=1;break
                s=clean(text_of(row));counts['records']+=1;counts['chars_scanned']+=len(s)
                if not s:
                    counts['empty']+=1
                else:
                    h=digest(s)
                    fresh=db.execute('INSERT OR IGNORE INTO seen VALUES (?)',(h,)).rowcount
                    if not fresh:
                        counts['duplicates']+=1;counts['duplicate_chars']+=len(s)
                    else:
                        old=db.execute('SELECT split FROM old WHERE h=?',(h,)).fetchone()
                        old=old[0] if old else None
                        if old:counts['old_'+old+'_overlap']+=1
                        # Old validation always stays held out. Old training never becomes new validation.
                        split=('val' if old=='val' else 'train' if old=='train' else
                               'val' if uniform(h,'split'+str(SEED))<VAL_FRACTION else 'train')
                        chosen=uniform(h,'select'+str(SEED))<probability
                        if old=='conflict':
                            counts['ambiguous_old_split_excluded']+=1
                        elif old=='train' and EXCLUDE_OLD_TRAIN:
                            counts['old_train_excluded']+=1
                        elif chosen:
                            # Only obvious content-free attestations are removed; no length cutoff.
                            junk=len(s)<300 and re.fullmatch(
                                r'[\s\S]*I attest to the accuracy and integrity of this document[\s.]*',s,re.I)
                            if junk:counts['attestation_only']+=1
                            else:
                                meta={k:str(row[k]) for k in ('court','court_id','year','tribunal','title','date')
                                      if isinstance(row,dict) and k in row}
                                db.execute('INSERT INTO docs VALUES (?,?,?,?,?,?)',
                                           (h,source,key,split,s,json.dumps(meta)))
                                counts[split+'_docs']+=1;counts[split+'_chars']+=len(s)
                if index%2000==0:
                    save(source,{'index':index,'done':False,'counts':dict(counts)});db.commit()
                if time.monotonic()-last>20:
                    print(source,index,'records;',dict(counts),'elapsed min',round((time.monotonic()-start)/60,1),flush=True)
                    last=time.monotonic()
            save(source,{'index':index,'done':True,'counts':dict(counts)});db.commit()
        report={s:state(s) for s in SOURCES}
        report['overlap_claim']=('Exact canonical-text exclusion against old selection verified (supplied text or manifest reconstruction); near duplicates not excluded.'
                                 if EXCLUDE_OLD_TRAIN else 'Old training text may be reused. No claim of an unseen-only corpus.')
        atomic_json(report,CACHE/'data_report.json')
        print(json.dumps(report,indent=2))
    except BaseException:
        db.rollback()  # stats/data since the last committed boundary are replayed together
        print('Stopped. Committed progress is safe; rerun this cell to resume.',flush=True)
        raise
    finally:db.close()
    return paths

def prepare_tokenizer():
    dest = CACHE / 'tokenizer.json'
    cfg = {
        'data_config': sha(CACHE / 'data_config.json'),
        'external': sha(TOKENIZER_PATH) if TOKENIZER_PATH else None,
        'vocab': VOCAB,
        'chars': BPE_CHARS,
        'version': 1
    }
    meta = CACHE / 'tokenizer_config.json'

    if meta.exists():
        assert json.loads(meta.read_text()) == cfg, 'Tokenizer config changed; use a new CACHE_DIR'
        return Tokenizer.from_file(str(dest))

    if TOKENIZER_PATH:
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    else:
        assert START_MODE == 'scratch', 'Do not change tokenizer when continuing a checkpoint'
        tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=VOCAB, min_frequency=2,
            special_tokens=['<pad>', '<bos>', '<eos>', '<unk>'],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

        # Read everything on THIS thread, close SQLite, then train.
        db = sqlite3.connect(CACHE / 'corpus.sqlite')
        texts = []
        try:
            used = [0] * len(SOURCES)
            cap = BPE_CHARS // len(SOURCES)
            for i, source in enumerate(SOURCES):
                cur = db.execute("SELECT text FROM docs WHERE source=? AND split='train' ORDER BY h", (source,))
                while used[i] < cap:
                    row = cur.fetchone()
                    if row is None: break
                    s = row[0][:cap - used[i]]
                    used[i] += len(s)
                    for j in range(0, len(s), 16000):
                        texts.append(s[j:j + 16000])
        finally:
            db.close()

        print(f"Tokenizer training data: {len(texts):,} pieces, {sum(len(x) for x in texts):,} characters", flush=True)
        tokenizer.train_from_iterator(texts, trainer=trainer)
        del texts
        gc.collect()

    assert tokenizer.get_vocab_size() <= 65535
    assert all(tokenizer.token_to_id(s) is not None for s in ['<pad>', '<bos>', '<eos>', '<unk>'])

    if TOKENIZER_PATH:
        shutil.copyfile(TOKENIZER_PATH, str(dest) + '.tmp')
    else:
        tokenizer.save(str(dest) + '.tmp')
    os.replace(str(dest) + '.tmp', str(dest))
    atomic_json(cfg, meta)
    return tokenizer

def pack_data(tok):
    # Each shard is atomic. Resume skips committed shards; no repeated full-corpus tokenization.
    pack=CACHE/'packed';pack.mkdir(exist_ok=True)
    cfg={'version':2,'tokenizer':sha(CACHE/'tokenizer.json'),'data':sha(CACHE/'data_config.json'),'shard_docs':SHARD_DOCS}
    cp=pack/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==cfg
    else:atomic_json(cfg,cp)
    if (pack/'report.json').exists() and (CACHE/'continuation_dev.json').exists():
        report=json.loads((pack/'report.json').read_text())
        metas=list(pack.glob('train_*.json'))+list(pack.glob('val_*.json'))
        valid=len(metas)==sum(v['shards'] for v in report.values())
        for p in metas:
            m=json.loads(p.read_text())
            valid=valid and p.with_suffix('.bin').exists() and p.with_suffix('.idx').exists()
            if valid:
                valid=(p.with_suffix('.bin').stat().st_size==2*m['tokens'] and
                       p.with_suffix('.idx').stat().st_size==8*(m['docs']+1))
        if valid:
            print('Complete token cache reused without rereading selected text.')
            return pack
    db=sqlite3.connect(CACHE/'corpus.sqlite'); stats={};dev=[]
    try:
        for split in ('train','val'):
            for source in SOURCES:
                cur=db.execute('SELECT h,id,text FROM docs WHERE source=? AND split=? ORDER BY h',(source,split))
                shard=0;total=0;docs=0;dev_n=0
                while True:
                    rows=cur.fetchmany(SHARD_DOCS)
                    if not rows:break
                    stem=f'{split}_{source}_{shard:05d}';done=pack/(stem+'.json')
                    if not done.exists():
                        offsets=[0];ids_meta=[];temp=pack/(stem+'.bin.tmp')
                        with temp.open('wb') as f:
                            # Threaded Rust BPE batching; cap batch characters to bound RAM.
                            batch=[];chars=0
                            def flush(batch):
                                if not batch:return
                                for (h,key,s),enc in zip(batch,tok.encode_batch([r[2] for r in batch])):
                                    ids=enc.ids+[tok.token_to_id('<eos>')]
                                    np.asarray(ids,dtype=np.uint16).tofile(f)
                                    offsets.append(offsets[-1]+len(ids));ids_meta.append([h,key])
                            for r in rows:
                                batch.append(r);chars+=len(r[2])
                                if len(batch)>=32 or chars>=1_000_000:flush(batch);batch=[];chars=0
                            flush(batch)
                        os.replace(temp,pack/(stem+'.bin'))
                        with (pack/(stem+'.idx.tmp')).open('wb') as f:np.asarray(offsets,dtype=np.int64).tofile(f)
                        os.replace(pack/(stem+'.idx.tmp'),pack/(stem+'.idx'))
                        atomic_json({'tokens':offsets[-1],'docs':len(rows),'ids':ids_meta},done)
                        print(stem,offsets[-1],'tokens',flush=True)
                    m=json.loads(done.read_text());total+=m['tokens'];docs+=m['docs'];shard+=1
                    if split=='val' and dev_n<DEV_PER_SOURCE:
                        for h,key,s in rows:
                            # Limit splitting memory for exceptionally large documents.
                            words=s.split(None,641)
                            if len(words)>=640 and dev_n<DEV_PER_SOURCE:
                                dev.append({'id':source+':'+key,'source':source,'hash':h,
                                            'prompt':' '.join(words[:512]),'reference':' '.join(words[512:640])})
                                dev_n+=1
                stats[split+'_'+source]={'tokens':total,'docs':docs,'shards':shard}
    finally:db.close()
    atomic_json(stats,pack/'report.json');atomic_json(dev,CACHE/'continuation_dev.json')
    print(json.dumps(stats,indent=2));print('Held-out continuations:',len(dev))
    return pack

class Corpus:
    def __init__(self,root,split):
        self.data=[];self.offsets=[];self.groups={s:[] for s in SOURCES};self.mass=[]
        for p in sorted(Path(root).glob(split+'_*.bin')):
            x=np.memmap(p,dtype=np.uint16,mode='r');off=np.memmap(p.with_suffix('.idx'),dtype=np.int64,mode='r')
            source=next(s for s in SOURCES if p.name.startswith(split+'_'+s+'_'))
            self.groups[source].append(len(self.data));self.data.append(x);self.offsets.append(off);self.mass.append(len(x))
        self.sources=[s for s in SOURCES if self.groups[s]]
        if len(self.sources)!=4:raise ValueError('Every split must contain all four sources; increase data budgets if needed')
        self.prob=np.array([SOURCE_WEIGHTS[s] for s in self.sources],float);self.prob/=self.prob.sum()
    def sample(self,rng,max_len):
        s=rng.choice(self.sources,p=self.prob); inds=self.groups[s]
        weights=np.array([self.mass[i] for i in inds],float);weights/=weights.sum()
        k=int(rng.choice(inds,p=weights));off=self.offsets[k];x=self.data[k]
        # Token-proportional document sampling, but never cross a document boundary.
        pos=int(rng.integers(len(x)));j=int(np.searchsorted(off,pos,side='right')-1)
        lo,hi=int(off[j]),int(off[j+1]);n=min(max_len,hi-lo)
        start=int(rng.integers(lo,hi-n+1))
        return np.array(x[start:start+n],dtype=np.int64)

def batch(corpus,rng,arch,micro=None):
    b=MICRO_BATCH if micro is None else micro
    if arch!='bridge':
        x=np.full((b,CONTEXT),PAD,np.int64);y=np.full((b,CONTEXT),-100,np.int64)
        for i in range(b):
            seq=corpus.sample(rng,CONTEXT+1)
            if len(seq)<2:seq=np.array([BOS,EOS])
            n=len(seq)-1;x[i,:n]=seq[:-1];y[i,:n]=seq[1:]
        return torch.as_tensor(x,device=DEVICE),torch.as_tensor(y,device=DEVICE),None
    src=np.full((b,SOURCE_CONTEXT),PAD,np.int64)
    x=np.full((b,TARGET_CONTEXT),PAD,np.int64);y=np.full_like(x,-100)
    for i in range(b):
        seq=corpus.sample(rng,SOURCE_CONTEXT+TARGET_CONTEXT)
        if len(seq)<2:seq=np.array([BOS,EOS])
        # Random prefix length. The encoder NEVER sees continuation target tokens.
        hi=min(SOURCE_CONTEXT,len(seq)-1);lo=max(1,min(128,hi))
        cut=int(rng.integers(lo,hi+1));target=seq[cut:cut+TARGET_CONTEXT]
        src[i,:cut]=seq[:cut];x[i,0]=BOS
        if len(target)>1:x[i,1:len(target)]=target[:-1]
        y[i,:len(target)]=target
    return (torch.as_tensor(x,device=DEVICE),torch.as_tensor(y,device=DEVICE),torch.as_tensor(src,device=DEVICE))

import io
import getpass
from dataclasses import dataclass
from types import SimpleNamespace
from contextlib import contextmanager

_HF_TOKEN = None

@dataclass(frozen=True)
class RemoteJSON:
    path: str
    size: int
    revision: str
    @property
    def name(self): return Path(self.path).name
    @property
    def suffix(self): return Path(self.path).suffix
    @property
    def stem(self): return Path(self.path).stem
    def stat(self): return SimpleNamespace(st_size=self.size)

class HTTPReader(io.RawIOBase):
    def __init__(self,raw):super().__init__();self.raw=raw
    def readable(self):return True
    def readinto(self,b):
        data=self.raw.read(len(b),decode_content=True);n=len(data);b[:n]=data;return n

def hf_token():
    global _HF_TOKEN
    if _HF_TOKEN:return _HF_TOKEN
    from huggingface_hub import get_token
    _HF_TOKEN=os.environ.get('HF_TOKEN') or get_token() or getpass.getpass('Hugging Face READ token (hidden): ').strip()
    if not _HF_TOKEN:raise ValueError('A read token with access to the private assignment is required.')
    return _HF_TOKEN

@contextmanager
def open_binary(path):
    if not isinstance(path,RemoteJSON):
        with Path(path).open('rb') as f:yield f
        return
    import requests
    from huggingface_hub import hf_hub_url
    url=hf_hub_url(REPO,path.path,repo_type='dataset',revision=path.revision)
    try:
        # requests strips Authorization on cross-host redirects; signed storage URLs need no token.
        with requests.get(url,headers={'Authorization':'Bearer '+hf_token()},stream=True,timeout=(30,180)) as r:
            if r.status_code in (401,403):raise RuntimeError('HF access denied. Check account access and token read scope.')
            if r.status_code!=200:raise RuntimeError(f'HF returned HTTP {r.status_code} for {path.name}.')
            with io.BufferedReader(HTTPReader(r.raw),buffer_size=65536) as f:yield f
    except requests.RequestException:
        raise RuntimeError(f'Network read failed for {path.name}. Rerun to resume committed progress; no raw file was cached.') from None


def initialize_foundation():
    global OLD_CACHE,TOKENIZER_PATH,OLD_CHECKPOINT,RAW_MANIFEST_PATH
    RAW_MANIFEST_PATH=None
    if FOUNDATION_PATH is None:
        if START_MODE=='continue':raise ValueError('Continuation needs FOUNDATION_PATH or an explicitly configured checkpoint bundle.')
        print('No foundation bundle: scratch training; no old-data exclusion claim.')
        return
    p=Path(FOUNDATION_PATH)
    if not p.exists():raise FileNotFoundError(f'Upload foundation_model.zip to {p}, or set FOUNDATION_PATH to its extracted folder.')
    if p.is_file():
        stamp=sha(p);dest=WORK/'foundation_import'/stamp[:16];dest.mkdir(parents=True,exist_ok=True)
        allowed={'model.pt','tokenizer.json','raw_manifest.json','packed_manifest.json','tokenizer_manifest.json',
                 'requested_config.json','environment.json','metrics.json','README.txt'}
        with zipfile.ZipFile(p) as z:
            for name in allowed:
                members=[m for m in z.infolist() if not m.is_dir() and Path(m.filename).name==name]
                if len(members)>1:raise ValueError(f'Ambiguous {name} in ZIP')
                if not members:continue
                target=dest/name;m=members[0]
                if not target.exists() or target.stat().st_size!=m.file_size:
                    tmp=target.with_suffix(target.suffix+'.tmp')
                    with z.open(m) as src,tmp.open('wb') as out:shutil.copyfileobj(src,out,1<<20)
                    os.replace(tmp,target)
        bundle=dest
    else:
        hits=list(p.rglob('tokenizer.json'))
        if len(hits)!=1:raise ValueError('Expected one tokenizer.json in foundation folder')
        bundle=hits[0].parent
    if not (bundle/'tokenizer.json').exists():raise FileNotFoundError('tokenizer.json missing from foundation bundle')
    TOKENIZER_PATH=bundle/'tokenizer.json'
    if (bundle/'model.pt').exists():OLD_CHECKPOINT=bundle/'model.pt'
    if (bundle/'raw_manifest.json').exists():RAW_MANIFEST_PATH=bundle/'raw_manifest.json'
    if START_MODE=='continue' and OLD_CHECKPOINT is None:raise FileNotFoundError('model.pt missing')
    if EXCLUDE_OLD_TRAIN and not RECONSTRUCT_OLD_SELECTION and OLD_CACHE is None:
        raise ValueError('Strict exclusion requires reconstruction or the original text cache')
    if RECONSTRUCT_OLD_SELECTION:
        if RAW_MANIFEST_PATH is None:raise FileNotFoundError('raw_manifest.json is required for reconstruction')
        tm=bundle/'tokenizer_manifest.json'
        if tm.exists():
            expected=json.loads(tm.read_text()).get('raw_sha')
            if expected and expected!=sha(RAW_MANIFEST_PATH):raise ValueError('Foundation manifest integrity mismatch')
    if START_MODE=='continue':
        ck=torch.load(OLD_CHECKPOINT,map_location='cpu',weights_only=False)
        if not isinstance(ck,dict) or not {'model_config','model'}<=set(ck):
            raise ValueError('Unexpected model.pt schema; need a weights/config checkpoint from the old export.')
        if ck['model_config'].get('arch')!='lexweave':raise ValueError('This is not the expected LEXWeave checkpoint')
        if MODEL_SIZE!='95m':raise ValueError('Continuing the old model requires MODEL_SIZE=95m. Larger models start from scratch.')
        expected=ck.get('tokenizer_sha',ck.get('train_config',{}).get('tokenizer_sha'))
        if expected!=sha(TOKENIZER_PATH):raise ValueError('model.pt and tokenizer.json do not have matching checksums')
        print('Exported checkpoint step:',ck.get('step'),'state tensor elements:',sum(v.numel() for v in ck['model'].values()))
        del ck;gc.collect()
    print('Foundation imported:',bundle)
    print('Tokenizer reused. Model weights', 'will be continued with a NEW optimizer.' if START_MODE=='continue' else 'are NOT loaded (scratch run).')
    print('No old train/val texts or optimizer state inferred merely from the ZIP filenames.')


def resolve_sources():
    CACHE.mkdir(parents=True,exist_ok=True)
    if DATA_MODE=='local':
        if DATA_ROOT is None:raise ValueError('Set DATA_ROOT for local mode')
        paths=[]
        for s in SOURCES:
            hits=list(DATA_ROOT.rglob(s+'.json'))
            if len(hits)!=1:raise ValueError(f'Expected exactly one {s}.json')
            paths.append(hits[0])
        return paths
    assert DATA_MODE=='stream'
    cache=CACHE/'hf_sources.json'
    old_revision=None
    if RECONSTRUCT_OLD_SELECTION and RAW_MANIFEST_PATH is not None:
        old_revision=json.loads(RAW_MANIFEST_PATH.read_text())['config'].get('revision')
        if not isinstance(old_revision,str) or not re.fullmatch(r'[0-9a-fA-F]{40}',old_revision):old_revision=None
    requested=HF_REVISION or old_revision
    if HF_REVISION and old_revision and HF_REVISION!=old_revision:
        raise ValueError('Reconstruction requires the old pinned revision; choose it for this experiment.')
    if cache.exists():
        info=json.loads(cache.read_text())
        assert info['repo']==REPO
        if requested and info['revision']!=requested:raise ValueError('Dataset revision changed: use a new CACHE')
    else:
        from huggingface_hub import HfApi
        try:remote=HfApi().dataset_info(REPO,revision=requested,token=hf_token(),files_metadata=True)
        except Exception:
            raise RuntimeError('Cannot list the private HF dataset. Check token read access and SageMaker internet access.') from None
        sizes={s.rfilename:s.size for s in remote.siblings}
        def pick(name,split):
            matches=[p for p in sizes if p.startswith('Phase_1_Pretraining/'+split+'/') and Path(p).name==name]
            if len(matches)!=1:raise ValueError(f'Expected one official {split}/{name}; got {matches}')
            p=matches[0]
            if not isinstance(sizes[p],int) or sizes[p]<=0:raise ValueError(f'Missing byte size for {p}')
            return {'path':p,'size':sizes[p],'revision':remote.sha}
        info={'repo':REPO,'revision':remote.sha,'train':[pick(s+'.json','Train') for s in SOURCES],
              'test':pick('test.json','Test')}
        atomic_json(info,cache)
    print('Official dataset revision:',info['revision'])
    print('Streaming JSON: no snapshot_download, no full raw training files stored on disk.')
    return [RemoteJSON(**r) for r in info['train']]


def official_test_path():
    if DATA_MODE=='local':
        hits=list(DATA_ROOT.rglob('test.json'))
        if len(hits)!=1:raise ValueError(f'Expected one test.json; got {hits}')
        return hits[0]
    # Called only during submission generation, never during corpus construction.
    info=json.loads((CACHE/'hf_sources.json').read_text());remote=RemoteJSON(**info['test'])
    dest=RUN/'official_test'/remote.revision/'test.json';dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists() or dest.stat().st_size!=remote.size:
        tmp=dest.with_suffix('.partial')
        try:
            with open_binary(remote) as src,tmp.open('wb') as out:shutil.copyfileobj(src,out,65536)
            if tmp.stat().st_size!=remote.size:raise IOError('Incomplete test download')
            os.replace(tmp,dest)
        finally:tmp.unlink(missing_ok=True)
    return dest


def verify_old_sources(paths,manifest):
    cfg=manifest['config']
    if cfg.get('version')!=1:raise ValueError('Unsupported old preprocessing version; cannot reconstruct safely')
    expected=cfg.get('files',[])
    if [r.get('name') for r in expected]!=[p.name for p in paths]:raise ValueError('Old source order differs; reconstruction stopped')
    for p,e in zip(paths,expected):
        if p.stat().st_size!=e['size']:raise ValueError(f'Old source size mismatch: {p.name}')
        if isinstance(p,RemoteJSON):
            if e.get('revision')==p.revision and e.get('repo')==REPO and e.get('path')==p.path:continue
            # A local-source manifest does not pin immutable remote bytes. Fail closed rather than guessing.
            raise ValueError('Old manifest does not pin these HF files. Use original train/val JSONL, or scratch mode without old exclusion.')
        current=fingerprint(p)
        if e.get('edge_sha256')!=current['edge_sha']:raise ValueError(f'Old source fingerprint mismatch: {p.name}')


def reconstruct_old_selection(paths):
    """Replay the original version-1 round-robin selection, NOT the optimizer's random windows.
    Produces hashes only; excludes the entire old selected corpus (conservative superset of actual training exposure).
    """
    if not RECONSTRUCT_OLD_SELECTION or RAW_MANIFEST_PATH is None:return None
    manifest=json.loads(RAW_MANIFEST_PATH.read_text());verify_old_sources(paths,manifest)
    signature={'version':1,'raw_manifest_sha':sha(RAW_MANIFEST_PATH)}
    out=CACHE/'old_selection.sqlite';report_path=CACHE/'old_selection_report.json'
    if report_path.exists() and out.exists():
        report=json.loads(report_path.read_text())
        assert report['config']==signature
        print('Verified old-selection reconstruction reused:',report['documents'])
        return out
    out.unlink(missing_ok=True)
    db=sqlite3.connect(out)
    db.execute('CREATE TABLE membership (h TEXT PRIMARY KEY, split TEXT)')
    db.execute('CREATE TABLE raw_seen (h TEXT PRIMARY KEY)')
    cfg=manifest['config'];limits={'train':cfg['train_chars'],'val':cfg['val_chars']}
    sizes={'train':0,'val':0};counts={'train':Counter(),'val':Counter()};duplicates=0;conflicts=0;kept=0
    streams=[(p.stem,iter(records(p))) for p in paths];all_streams=[s for _,s in streams]
    started=last=time.monotonic()
    print('Reconstructing old selected documents from the pinned official streams (~old corpus size, not the full corpus).')
    try:
        while streams and any(sizes[k]<limits[k] for k in sizes):
            remaining=[]
            for source,stream in streams:
                try:key,row=next(stream)
                except StopIteration:continue
                remaining.append((source,stream));s=clean(text_of(row))
                if len(s.split())<32:continue
                raw_h=hashlib.sha256(s.encode()).hexdigest()
                split_hash=hashlib.sha256((str(cfg['seed'])+raw_h).encode()).hexdigest()
                split='val' if int(split_hash[:16],16)/2**64<cfg['val_fraction'] else 'train'
                if sizes[split]>=limits[split]:continue
                if not db.execute('INSERT OR IGNORE INTO raw_seen VALUES (?)',(raw_h,)).rowcount:
                    duplicates+=1;continue
                h=digest(s);existing=db.execute('SELECT split FROM membership WHERE h=?',(h,)).fetchone()
                if existing and existing[0]!=split:
                    db.execute("UPDATE membership SET split='conflict' WHERE h=?",(h,));conflicts+=1
                else:db.execute('INSERT OR IGNORE INTO membership VALUES (?,?)',(h,split))
                sizes[split]+=len(s);counts[split][source]+=1;kept+=1
                if time.monotonic()-last>20:
                    print('Old reconstruction:',kept,'documents;',sizes,'chars;',round((time.monotonic()-started)/60,1),'min',flush=True)
                    last=time.monotonic()
            streams=remaining
        actual={k:dict(v) for k,v in counts.items()}
        if sizes!=manifest['chars'] or actual!=manifest['documents'] or duplicates!=manifest['duplicates_removed']:
            raise ValueError('Reconstructed old selection DOES NOT match manifest counts/characters/duplicates. No exclusion claim is valid.')
        db.commit()
        atomic_json({'config':signature,'documents':actual,'chars':sizes,'duplicates':duplicates,
                     'canonical_conflicts':conflicts,'verified':True,
                     'scope':'Entire old raw selected corpus; not a record of optimizer-sampled windows. No near-duplicate guarantee.'},report_path)
        print('Old selection matches manifest exactly:',sizes)
        return out
    except BaseException:
        db.rollback();print('Old reconstruction incomplete; rerun this stage. New-data progress is unaffected.');raise
    finally:
        for stream in all_streams:stream.close()
        db.close()

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

_backup_api=None
_backup_token=None
_last_remote_time=0.

def setup_backup():
    global _backup_api,_backup_token
    if BACKUP_MODE=='local_only':
        print('LOCAL ONLY: a lost session can erase all progress. Download checkpoints manually.')
        return
    if BACKUP_MODE!='hf' or not BACKUP_REPO or '/' not in BACKUP_REPO:
        raise ValueError("Set BACKUP_REPO='your-HF-name/private-cs781-runs'. A write token is needed. "
                         "Only deliberately choose BACKUP_MODE='local_only' if handling backups yourself.")
    _backup_token=os.environ.get('HF_BACKUP_TOKEN') or getpass.getpass('HF token with WRITE access to your private backup repo: ')
    _backup_api=HfApi(token=_backup_token)
    _backup_api.create_repo(repo_id=BACKUP_REPO,repo_type='model',private=True,exist_ok=True)
    if not _backup_api.repo_info(BACKUP_REPO,repo_type='model').private:
        raise ValueError('Backup repo is public. Select a PRIVATE repo; no upload was made.')
    if RESTORE_REMOTE and not (RUN/'last.pt').exists():
        files=_backup_api.list_repo_files(BACKUP_REPO,repo_type='model')
        prefix=RUN.name+'/'
        if prefix+'checkpoint_receipt.json' in files:
            # Pin one committed revision; never combine files from concurrent uploads.
            revision=_backup_api.repo_info(BACKUP_REPO,repo_type='model').sha
            folder=WORK/'restore_download'
            snapshot_download(BACKUP_REPO,repo_type='model',revision=revision,token=_backup_token,
                              allow_patterns=[prefix+'*'],local_dir=folder)
            source=folder/RUN.name
            receipt=json.loads((source/'checkpoint_receipt.json').read_text())
            for name,expected in receipt['files'].items():
                if sha(source/name)!=expected:raise RuntimeError('Remote checkpoint integrity mismatch: '+name)
            for p in source.rglob('*'):
                if p.is_file():
                    dest=RUN/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(p,dest)
            for name,expected in receipt['files'].items():
                if sha(RUN/name)!=expected:raise RuntimeError('Restored local copy failed verification: '+name)
            shutil.rmtree(source) # Only the temporary downloaded duplicate; keep RUN and remote commit.
            print('RESTORED verified remote checkpoint:',receipt['step'],'from commit',revision)
        else:print('No previous checkpoint for this RUN_TAG; starting a new run.')
    # Probe write/read before preprocessing or training consumes hours.
    probe=RUN/'backup_probe.json';atomic_json({'run':RUN.name,'time':time.time()},probe)
    commit=_backup_api.upload_file(path_or_fileobj=str(probe),path_in_repo=RUN.name+'/'+probe.name,
                                 repo_id=BACKUP_REPO,repo_type='model')
    fetched=hf_hub_download(BACKUP_REPO,RUN.name+'/'+probe.name,revision=commit.oid,token=_backup_token)
    assert sha(fetched)==sha(probe),'Backup read-back mismatch'
    print('REMOTE BACKUP write/read verified:',BACKUP_REPO,'commit',commit.oid)

def backup_run(checkpoints=True,force=False):
    global _last_remote_time
    if BACKUP_MODE=='local_only':return
    if _backup_api is None:raise RuntimeError('Run setup_backup first.')
    if not force and time.monotonic()-_last_remote_time<REMOTE_EVERY_SECONDS:return
    if checkpoints:
        names=[p.name for p in RUN.glob('*.pt')]+['tokenizer.json','model_config.json','modeling.py','experiment.json']
        present=[name for name in names if (RUN/name).exists()]
        # Receipt is committed in the same atomic Hub commit as its checkpoint files.
        atomic_json({'step':json.loads((RUN/'progress.json').read_text()).get('step',0) if (RUN/'progress.json').exists() else 0,
                     'files':{name:sha(RUN/name) for name in present}},RUN/'checkpoint_receipt.json')
    allowed=['*.json','*.jsonl','*.py','submission_*/*.json','*.zip']
    ignored=['*.tmp','backup_probe.json']
    if checkpoints:allowed+=['*.pt']
    else:ignored+=['checkpoint_receipt.json','*.pt']
    # Synchronous: do not announce success or resume compute before upload completes.
    commit=_backup_api.upload_folder(repo_id=BACKUP_REPO,repo_type='model',folder_path=str(RUN),
                                    path_in_repo=RUN.name,allow_patterns=allowed,ignore_patterns=ignored,
                                    commit_message='CS781 checkpoint' if checkpoints else 'CS781 predictions')
    _last_remote_time=time.monotonic()
    print('REMOTE BACKUP CONFIRMED:',commit.oid,flush=True)

def preserve_metadata():
    for name in ('tokenizer.json','data_config.json','data_report.json','hf_sources.json','continuation_dev.json'):
        if (CACHE/name).exists():shutil.copy2(CACHE/name,RUN/name)
    shutil.copy2(PACK/'report.json',RUN/'packed_report.json')
    atomic_json(CFG,RUN/'model_config.json')
    (RUN/'modeling.py').write_text(MODEL_SOURCE,encoding='utf-8')
    atomic_json({'torch':torch.__version__,'cuda':torch.version.cuda},RUN/'environment.json')






setup_backup()  # Restores a committed run and verifies remote write/read before expensive work.
# On recovery, preserve the exact tokenizer and pinned official source revision.
if (CACHE/'tokenizer_config.json').exists():
    previous=json.loads((CACHE/'tokenizer_config.json').read_text())
    if TOKENIZER_PATH is None and previous.get('external'):TOKENIZER_PATH=CACHE/'tokenizer.json'
elif (RUN/'tokenizer.json').exists():TOKENIZER_PATH=RUN/'tokenizer.json'
if (RUN/'hf_sources.json').exists() and HF_REVISION is None:
    saved=json.loads((RUN/'hf_sources.json').read_text())
    HF_REVISION=saved['revision']
if READY_CACHE is None:
    initialize_foundation()
    paths=prepare_data()
else:
    required=['tokenizer.json','data_config.json','data_report.json','packed/config.json','packed/report.json','continuation_dev.json']
    if DATA_MODE=='stream':required.append('hf_sources.json')
    for name in required:
        if not (CACHE/name).exists():raise FileNotFoundError(CACHE/name)
    print('Reusing attached cache. Existing split is authoritative; source stream will not be read.')






if READY_CACHE is None:
    tok=prepare_tokenizer();PACK=pack_data(tok)
else:
    tok=Tokenizer.from_file(str(CACHE/'tokenizer.json'));PACK=CACHE/'packed'
    pc=json.loads((PACK/'config.json').read_text())
    assert pc['tokenizer']==sha(CACHE/'tokenizer.json') and pc['data']==sha(CACHE/'data_config.json')
    report=json.loads((PACK/'report.json').read_text())
    metas=list(PACK.glob('train_*.json'))+list(PACK.glob('val_*.json'))
    assert len(metas)==sum(v['shards'] for v in report.values())
    for p in metas:
        m=json.loads(p.read_text())
        assert p.with_suffix('.bin').stat().st_size==2*m['tokens']
        assert p.with_suffix('.idx').stat().st_size==8*(m['docs']+1)
PAD,BOS,EOS,UNK=[tok.token_to_id(s) for s in ['<pad>','<bos>','<eos>','<unk>']]
assert all(i is not None for i in (PAD,BOS,EOS,UNK))
V=tok.get_vocab_size()
TRAIN=Corpus(PACK,'train');VAL=Corpus(PACK,'val')
report=json.loads((PACK/'report.json').read_text())
print('Unique selected train tokens:',sum(v['tokens'] for k,v in report.items() if k.startswith('train_')))
print('Held-out continuations:',len(json.loads((CACHE/'continuation_dev.json').read_text())))






"""Project experiments, initialized from scratch. No complete pretrained architecture."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        z = x.float()
        return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight


class CausalAttention(nn.Module):
    def __init__(self, d, heads, rope=False):
        super().__init__(); self.h=heads; self.dh=d//heads; self.rope=rope
        assert d % heads == 0 and self.dh % 2 == 0
        self.qkv=nn.Linear(d,3*d,bias=False); self.out=nn.Linear(d,d,bias=False)
        self.register_buffer('freq',10000**(-torch.arange(0,self.dh,2).float()/self.dh),persistent=False)
    def rotate(self,x,start=0):
        a=torch.outer(torch.arange(start,start+x.shape[-2],device=x.device).float(),self.freq.float())
        c,s=a.cos().to(x.dtype)[None,None],a.sin().to(x.dtype)[None,None]
        u,v=x[...,::2],x[...,1::2]
        return torch.stack((u*c-v*s,u*s+v*c),-1).flatten(-2)
    def forward(self,x,cache=None,use_cache=False):
        b,t,d=x.shape
        q,k,v=self.qkv(x).reshape(b,t,3,self.h,self.dh).permute(2,0,3,1,4).unbind(0)
        past=0 if cache is None else cache[0].shape[-2]
        if self.rope:q,k=self.rotate(q,past),self.rotate(k,past)
        if cache is not None:k,v=torch.cat((cache[0],k),-2),torch.cat((cache[1],v),-2)
        mask=None
        if past and t>1:
            mask=torch.arange(k.shape[-2],device=x.device)[None,:]<=torch.arange(past,past+t,device=x.device)[:,None]
        z=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=(past==0),dropout_p=0.)
        out=self.out(z.transpose(1,2).contiguous().view(b,t,d))
        return (out,(k,v)) if use_cache else out


class AnchorBlock(nn.Module):
    """GPT block with token-dependent, low-rank reinjection of a group-entry state."""
    def __init__(self,c):
        super().__init__(); d=c['dim']; self.abl=c['ablation']
        self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d)
        self.attn=CausalAttention(d,c['heads'])
        self.ff=nn.Sequential(nn.Linear(d,c['ff_dim']),nn.GELU(),nn.Linear(c['ff_dim'],d))
        if self.abl!='no_anchor':
            self.an=nn.LayerNorm(d)
            self.anchor_down=nn.Linear(d,c['anchor_rank'],bias=False)
            self.anchor_up=nn.Linear(c['anchor_rank'],d,bias=False)
            if self.abl=='full':self.gate=nn.Linear(2*d,1)
    def forward(self,x,anchor,cache=None,use_cache=False):
        if self.abl!='no_anchor':
            a=self.an(anchor)
            g=torch.sigmoid(self.gate(torch.cat((self.n1(x),a),-1))) if self.abl=='full' else 0.12
            x=x+g*self.anchor_up(F.gelu(self.anchor_down(a)))
        a=self.attn(self.n1(x),cache,use_cache)
        if use_cache:a,cache=a
        x=x+a;x=x+self.ff(self.n2(x))
        return (x,cache) if use_cache else x


class RecurrentBlock(nn.Module):
    """Gated LSTM residual, periodic causal attention, then a SwiGLU FFN."""
    def __init__(self,c,index):
        super().__init__();d=c['dim'];self.abl=c['ablation']
        self.n1=RMSNorm(d);self.n2=RMSNorm(d)
        if self.abl!='no_recurrence':
            # nn.LSTM is a primitive recurrent kernel, not an imported complete LM.
            self.rnn=nn.LSTM(d,c['recurrent_dim'],batch_first=True)
            self.proj=nn.Linear(c['recurrent_dim'],d,bias=False)
            if self.abl!='fixed_gate':self.gate=nn.Linear(2*d,1)
        self.attn=None
        if (index+1)%c['attention_every']==0 and self.abl!='no_attention':
            self.na=RMSNorm(d);self.attn=CausalAttention(d,c['heads'],rope=True)
        self.up=nn.Linear(d,2*c['ff_dim'],bias=False);self.down=nn.Linear(c['ff_dim'],d,bias=False)
    def forward(self,x,unused=None,cache=None,use_cache=False):
        rc,ac=(None,None) if cache is None else cache
        if self.abl!='no_recurrence':
            z=self.n1(x)
            # Keep fused cuDNN LSTM in FP32: portable across T4 and BF16 GPUs.
            with torch.autocast(device_type=x.device.type,enabled=False):
                r,rc=self.rnn(z.float(),rc)
            r=self.proj(r.to(z.dtype))
            g=torch.sigmoid(self.gate(torch.cat((z,r),-1))) if self.abl!='fixed_gate' else 0.5
            x=x+g*r
        if self.attn is not None:
            a=self.attn(self.na(x),ac,use_cache)
            if use_cache:a,ac=a
            x=x+a
        a,b=self.up(self.n2(x)).chunk(2,-1)
        x=x+self.down(F.silu(a)*b)
        return (x,(rc,ac)) if use_cache else x


class ExperimentalLM(nn.Module):
    def __init__(self,c):
        super().__init__();self.cfg=dict(c);self.kind=c['arch'];self.grad_ckpt=c['grad_checkpoint']
        d=c['dim'];self.emb=nn.Embedding(c['vocab'],d)
        self.pos=nn.Embedding(c['context'],d) if self.kind=='anchor_gpt' else None
        self.blocks=nn.ModuleList([AnchorBlock(c) if self.kind=='anchor_gpt' else RecurrentBlock(c,i) for i in range(c['layers'])])
        self.norm=nn.LayerNorm(d) if self.kind=='anchor_gpt' else RMSNorm(d)
        self.apply(self.init_weights)
        for b in self.blocks:
            if hasattr(b,'gate'):
                nn.init.zeros_(b.gate.weight);nn.init.constant_(b.gate.bias,-2. if self.kind=='anchor_gpt' else 0.)
            if hasattr(b,'rnn'):
                for name,p in b.rnn.named_parameters():
                    if 'weight' in name:nn.init.normal_(p,std=.02)
                    else:nn.init.zeros_(p)
                # Combined forget bias +1; input/hidden biases otherwise zero.
                r=c['recurrent_dim']
                with torch.no_grad():b.rnn.bias_ih_l0[r:2*r].fill_(1.)
        # Scale residual output projections for depth.
        for name,p in self.named_parameters():
            if name.endswith(('attn.out.weight','ff.2.weight','down.weight','anchor_up.weight','proj.weight')):
                nn.init.normal_(p,std=.02/math.sqrt(2*c['layers']))
    @staticmethod
    def init_weights(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,std=.02)
            if isinstance(m,nn.Linear) and m.bias is not None:nn.init.zeros_(m.bias)
    def hidden(self,ids,src=None,memory=None):
        assert 0 < ids.shape[1] <= self.cfg['context']
        x=self.emb(ids)
        if self.pos is not None:x=x+self.pos(torch.arange(ids.shape[1],device=ids.device))[None]
        anchor=x
        for i,b in enumerate(self.blocks):
            if self.kind=='anchor_gpt' and i%self.cfg['anchor_every']==0:anchor=x
            x=checkpoint(b,x,anchor,use_reentrant=False) if self.training and self.grad_ckpt else b(x,anchor)
        return self.norm(x)
    def forward(self,ids,src=None,targets=None,memory=None):
        x=self.hidden(ids)
        if targets is None:return F.linear(x[:,-1],self.emb.weight).float()
        def ce(h,y):
            return F.cross_entropy(F.linear(h,self.emb.weight).float().reshape(-1,self.cfg['vocab']),y.reshape(-1),reduction='sum')
        total=x.new_zeros((),dtype=torch.float32)
        for j in range(0,x.shape[1],128):
            h,y=x[:,j:j+128],targets[:,j:j+128]
            total=total+(checkpoint(ce,h,y,use_reentrant=False) if self.training and self.grad_ckpt else ce(h,y))
        return total/targets.ne(-100).sum().clamp_min(1)

    @torch.inference_mode()
    def decode(self,ids,state=None):
        """Exact incremental inference inside one context window; no cache during training."""
        start=0 if state is None else state['length']
        assert start+ids.shape[1]<=self.cfg['context']
        x=self.emb(ids)
        if self.pos is not None:x=x+self.pos(torch.arange(start,start+ids.shape[1],device=ids.device))[None]
        anchor=x; caches=[]
        for i,b in enumerate(self.blocks):
            if self.kind=='anchor_gpt' and i%self.cfg['anchor_every']==0:anchor=x
            old=None if state is None else state['blocks'][i]
            x,cache=b(x,anchor,old,True);caches.append(cache)
        return F.linear(self.norm(x[:,-1]),self.emb.weight).float(),dict(length=start+ids.shape[1],blocks=caches)


def model_config(arch,vocab,context,size,ablation='full',grad_checkpoint=True):
    if arch=='anchor_gpt':
        presets={'small':(640,10,12,2560),'150m':(768,12,19,3072),'200m':(896,14,19,3584)}
        assert ablation in ('full','no_anchor','fixed_gate')
    else:
        assert arch=='relay_lstm'
        presets={'small':(640,10,12,1728),'150m':(768,12,16,2048),'200m':(768,12,22,2048)}
        assert ablation in ('full','no_recurrence','no_attention','fixed_gate')
    d,h,n,ff=presets[size]
    return dict(arch=arch,vocab=vocab,context=context,dim=d,heads=h,layers=n,ff_dim=ff,
                anchor_rank=64,anchor_every=4,recurrent_dim=512,attention_every=3,
                ablation=ablation,grad_checkpoint=grad_checkpoint)



CFG=model_config(ARCH,V,CONTEXT,MODEL_SIZE,ABLATION,GRAD_CHECKPOINT)
MODEL_SOURCE='"""Project experiments, initialized from scratch. No complete pretrained architecture."""\nimport math\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\nfrom torch.utils.checkpoint import checkpoint\n\n\nclass RMSNorm(nn.Module):\n    def __init__(self, d):\n        super().__init__(); self.weight = nn.Parameter(torch.ones(d))\n    def forward(self, x):\n        z = x.float()\n        return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight\n\n\nclass CausalAttention(nn.Module):\n    def __init__(self, d, heads, rope=False):\n        super().__init__(); self.h=heads; self.dh=d//heads; self.rope=rope\n        assert d % heads == 0 and self.dh % 2 == 0\n        self.qkv=nn.Linear(d,3*d,bias=False); self.out=nn.Linear(d,d,bias=False)\n        self.register_buffer(\'freq\',10000**(-torch.arange(0,self.dh,2).float()/self.dh),persistent=False)\n    def rotate(self,x,start=0):\n        a=torch.outer(torch.arange(start,start+x.shape[-2],device=x.device).float(),self.freq.float())\n        c,s=a.cos().to(x.dtype)[None,None],a.sin().to(x.dtype)[None,None]\n        u,v=x[...,::2],x[...,1::2]\n        return torch.stack((u*c-v*s,u*s+v*c),-1).flatten(-2)\n    def forward(self,x,cache=None,use_cache=False):\n        b,t,d=x.shape\n        q,k,v=self.qkv(x).reshape(b,t,3,self.h,self.dh).permute(2,0,3,1,4).unbind(0)\n        past=0 if cache is None else cache[0].shape[-2]\n        if self.rope:q,k=self.rotate(q,past),self.rotate(k,past)\n        if cache is not None:k,v=torch.cat((cache[0],k),-2),torch.cat((cache[1],v),-2)\n        mask=None\n        if past and t>1:\n            mask=torch.arange(k.shape[-2],device=x.device)[None,:]<=torch.arange(past,past+t,device=x.device)[:,None]\n        z=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=(past==0),dropout_p=0.)\n        out=self.out(z.transpose(1,2).contiguous().view(b,t,d))\n        return (out,(k,v)) if use_cache else out\n\n\nclass AnchorBlock(nn.Module):\n    """GPT block with token-dependent, low-rank reinjection of a group-entry state."""\n    def __init__(self,c):\n        super().__init__(); d=c[\'dim\']; self.abl=c[\'ablation\']\n        self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d)\n        self.attn=CausalAttention(d,c[\'heads\'])\n        self.ff=nn.Sequential(nn.Linear(d,c[\'ff_dim\']),nn.GELU(),nn.Linear(c[\'ff_dim\'],d))\n        if self.abl!=\'no_anchor\':\n            self.an=nn.LayerNorm(d)\n            self.anchor_down=nn.Linear(d,c[\'anchor_rank\'],bias=False)\n            self.anchor_up=nn.Linear(c[\'anchor_rank\'],d,bias=False)\n            if self.abl==\'full\':self.gate=nn.Linear(2*d,1)\n    def forward(self,x,anchor,cache=None,use_cache=False):\n        if self.abl!=\'no_anchor\':\n            a=self.an(anchor)\n            g=torch.sigmoid(self.gate(torch.cat((self.n1(x),a),-1))) if self.abl==\'full\' else 0.12\n            x=x+g*self.anchor_up(F.gelu(self.anchor_down(a)))\n        a=self.attn(self.n1(x),cache,use_cache)\n        if use_cache:a,cache=a\n        x=x+a;x=x+self.ff(self.n2(x))\n        return (x,cache) if use_cache else x\n\n\nclass RecurrentBlock(nn.Module):\n    """Gated LSTM residual, periodic causal attention, then a SwiGLU FFN."""\n    def __init__(self,c,index):\n        super().__init__();d=c[\'dim\'];self.abl=c[\'ablation\']\n        self.n1=RMSNorm(d);self.n2=RMSNorm(d)\n        if self.abl!=\'no_recurrence\':\n            # nn.LSTM is a primitive recurrent kernel, not an imported complete LM.\n            self.rnn=nn.LSTM(d,c[\'recurrent_dim\'],batch_first=True)\n            self.proj=nn.Linear(c[\'recurrent_dim\'],d,bias=False)\n            if self.abl!=\'fixed_gate\':self.gate=nn.Linear(2*d,1)\n        self.attn=None\n        if (index+1)%c[\'attention_every\']==0 and self.abl!=\'no_attention\':\n            self.na=RMSNorm(d);self.attn=CausalAttention(d,c[\'heads\'],rope=True)\n        self.up=nn.Linear(d,2*c[\'ff_dim\'],bias=False);self.down=nn.Linear(c[\'ff_dim\'],d,bias=False)\n    def forward(self,x,unused=None,cache=None,use_cache=False):\n        rc,ac=(None,None) if cache is None else cache\n        if self.abl!=\'no_recurrence\':\n            z=self.n1(x)\n            # Keep fused cuDNN LSTM in FP32: portable across T4 and BF16 GPUs.\n            with torch.autocast(device_type=x.device.type,enabled=False):\n                r,rc=self.rnn(z.float(),rc)\n            r=self.proj(r.to(z.dtype))\n            g=torch.sigmoid(self.gate(torch.cat((z,r),-1))) if self.abl!=\'fixed_gate\' else 0.5\n            x=x+g*r\n        if self.attn is not None:\n            a=self.attn(self.na(x),ac,use_cache)\n            if use_cache:a,ac=a\n            x=x+a\n        a,b=self.up(self.n2(x)).chunk(2,-1)\n        x=x+self.down(F.silu(a)*b)\n        return (x,(rc,ac)) if use_cache else x\n\n\nclass ExperimentalLM(nn.Module):\n    def __init__(self,c):\n        super().__init__();self.cfg=dict(c);self.kind=c[\'arch\'];self.grad_ckpt=c[\'grad_checkpoint\']\n        d=c[\'dim\'];self.emb=nn.Embedding(c[\'vocab\'],d)\n        self.pos=nn.Embedding(c[\'context\'],d) if self.kind==\'anchor_gpt\' else None\n        self.blocks=nn.ModuleList([AnchorBlock(c) if self.kind==\'anchor_gpt\' else RecurrentBlock(c,i) for i in range(c[\'layers\'])])\n        self.norm=nn.LayerNorm(d) if self.kind==\'anchor_gpt\' else RMSNorm(d)\n        self.apply(self.init_weights)\n        for b in self.blocks:\n            if hasattr(b,\'gate\'):\n                nn.init.zeros_(b.gate.weight);nn.init.constant_(b.gate.bias,-2. if self.kind==\'anchor_gpt\' else 0.)\n            if hasattr(b,\'rnn\'):\n                for name,p in b.rnn.named_parameters():\n                    if \'weight\' in name:nn.init.normal_(p,std=.02)\n                    else:nn.init.zeros_(p)\n                # Combined forget bias +1; input/hidden biases otherwise zero.\n                r=c[\'recurrent_dim\']\n                with torch.no_grad():b.rnn.bias_ih_l0[r:2*r].fill_(1.)\n        # Scale residual output projections for depth.\n        for name,p in self.named_parameters():\n            if name.endswith((\'attn.out.weight\',\'ff.2.weight\',\'down.weight\',\'anchor_up.weight\',\'proj.weight\')):\n                nn.init.normal_(p,std=.02/math.sqrt(2*c[\'layers\']))\n    @staticmethod\n    def init_weights(m):\n        if isinstance(m,(nn.Linear,nn.Embedding)):\n            nn.init.normal_(m.weight,std=.02)\n            if isinstance(m,nn.Linear) and m.bias is not None:nn.init.zeros_(m.bias)\n    def hidden(self,ids,src=None,memory=None):\n        assert 0 < ids.shape[1] <= self.cfg[\'context\']\n        x=self.emb(ids)\n        if self.pos is not None:x=x+self.pos(torch.arange(ids.shape[1],device=ids.device))[None]\n        anchor=x\n        for i,b in enumerate(self.blocks):\n            if self.kind==\'anchor_gpt\' and i%self.cfg[\'anchor_every\']==0:anchor=x\n            x=checkpoint(b,x,anchor,use_reentrant=False) if self.training and self.grad_ckpt else b(x,anchor)\n        return self.norm(x)\n    def forward(self,ids,src=None,targets=None,memory=None):\n        x=self.hidden(ids)\n        if targets is None:return F.linear(x[:,-1],self.emb.weight).float()\n        def ce(h,y):\n            return F.cross_entropy(F.linear(h,self.emb.weight).float().reshape(-1,self.cfg[\'vocab\']),y.reshape(-1),reduction=\'sum\')\n        total=x.new_zeros((),dtype=torch.float32)\n        for j in range(0,x.shape[1],128):\n            h,y=x[:,j:j+128],targets[:,j:j+128]\n            total=total+(checkpoint(ce,h,y,use_reentrant=False) if self.training and self.grad_ckpt else ce(h,y))\n        return total/targets.ne(-100).sum().clamp_min(1)\n\n    @torch.inference_mode()\n    def decode(self,ids,state=None):\n        """Exact incremental inference inside one context window; no cache during training."""\n        start=0 if state is None else state[\'length\']\n        assert start+ids.shape[1]<=self.cfg[\'context\']\n        x=self.emb(ids)\n        if self.pos is not None:x=x+self.pos(torch.arange(start,start+ids.shape[1],device=ids.device))[None]\n        anchor=x; caches=[]\n        for i,b in enumerate(self.blocks):\n            if self.kind==\'anchor_gpt\' and i%self.cfg[\'anchor_every\']==0:anchor=x\n            old=None if state is None else state[\'blocks\'][i]\n            x,cache=b(x,anchor,old,True);caches.append(cache)\n        return F.linear(self.norm(x[:,-1]),self.emb.weight).float(),dict(length=start+ids.shape[1],blocks=caches)\n\n\ndef model_config(arch,vocab,context,size,ablation=\'full\',grad_checkpoint=True):\n    if arch==\'anchor_gpt\':\n        presets={\'small\':(640,10,12,2560),\'150m\':(768,12,19,3072),\'200m\':(896,14,19,3584)}\n        assert ablation in (\'full\',\'no_anchor\',\'fixed_gate\')\n    else:\n        assert arch==\'relay_lstm\'\n        presets={\'small\':(640,10,12,1728),\'150m\':(768,12,16,2048),\'200m\':(768,12,22,2048)}\n        assert ablation in (\'full\',\'no_recurrence\',\'no_attention\',\'fixed_gate\')\n    d,h,n,ff=presets[size]\n    return dict(arch=arch,vocab=vocab,context=context,dim=d,heads=h,layers=n,ff_dim=ff,\n                anchor_rank=64,anchor_every=4,recurrent_dim=512,attention_every=3,\n                ablation=ablation,grad_checkpoint=grad_checkpoint)\n'





DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
AMP_DTYPE=torch.bfloat16 if DEVICE=='cuda' and torch.cuda.is_bf16_supported() else torch.float16

def amp():return torch.autocast('cuda',dtype=AMP_DTYPE) if DEVICE=='cuda' else nullcontext()
def sync():
    if DEVICE=='cuda':torch.cuda.synchronize()
def atomic_torch(obj,path):
    p=Path(path);torch.save(obj,str(p)+'.tmp');os.replace(str(p)+'.tmp',p)

def make_model():return ExperimentalLM(CFG)

def initial_model():return make_model() # New architecture: always random initialization.

@torch.no_grad()
def validate(m):
    m.eval();rng=np.random.default_rng(SEED+9001);total=0.;nt=0
    for _ in range(EVAL_BATCHES):
        x,y,src=batch(VAL,rng,ARCH)
        with amp():loss=m(x,src=src,targets=y)
        n=int(y.ne(-100).sum());total+=float(loss)*n;nt+=n
    m.train();return total/max(1,nt)

def experiment_config():
    config=dict(model=CFG,arch=ARCH,seed=SEED,start_mode=START_MODE,
                initial_checkpoint_sha=sha(OLD_CHECKPOINT) if START_MODE=='continue' else None,
                tokenizer_sha=sha(CACHE/'tokenizer.json'),data_sha=sha(CACHE/'data_config.json'),
                packed_report_sha=sha(PACK/'report.json'),target_tokens=TRAIN_TARGET_TOKENS,
                micro=MICRO_BATCH,accum=ACCUM,lr=LR,min_lr=MIN_LR,warmup_tokens=WARMUP_TOKENS,
                weights=SOURCE_WEIGHTS,source_context=SOURCE_CONTEXT,target_context=TARGET_CONTEXT,
                eval_batches=EVAL_BATCHES)
    return json.loads(json.dumps(config))

def benchmark(updates=3):
    assert DEVICE=='cuda','Run preprocessing on CPU; switch to a GPU for this cell.'
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        m=make_model().to(DEVICE).train();opt=torch.optim.AdamW(m.parameters(),lr=LR,betas=(.9,.95),weight_decay=.1)
        scaler=torch.amp.GradScaler('cuda',enabled=DEVICE=='cuda' and AMP_DTYPE==torch.float16,init_scale=256.)
        rng=np.random.default_rng(SEED+123);dur=[];tokens=[]
        torch.cuda.reset_peak_memory_stats()
        for i in range(updates+1):
            sync();t=time.monotonic();opt.zero_grad(set_to_none=True);nt=0
            batches=[batch(TRAIN,rng,ARCH) for _ in range(ACCUM)]
            nt=sum(int(y.ne(-100).sum()) for _,y,_ in batches)
            for x,y,src in batches:
                count=int(y.ne(-100).sum())
                with amp():loss=m(x,src=src,targets=y)*(count/max(1,nt))
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(m.parameters(),1.);scaler.step(opt);scaler.update();sync()
            if i:dur.append(time.monotonic()-t);tokens.append(nt)
        tps=sum(tokens)/sum(dur)
        result=dict(gpu=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in m.parameters()),
                    seconds_per_update=float(np.mean(dur)),loss_tokens_per_second=tps,
                    estimated_training_hours=(TRAIN_TARGET_TOKENS/tps/3600 if TRAIN_TARGET_TOKENS else None),
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    note='Short benchmark; excludes validation, generation and checkpoints. One GPU only.')
        atomic_json(result,RUN/'benchmark.json');print(result)
        del m,opt,scaler;gc.collect();torch.cuda.empty_cache()
    return result

def train_model():
    assert DEVICE=='cuda','Enable a CUDA GPU.'
    assert TRAIN_TARGET_TOKENS is not None, 'Run benchmark/budget cell first.'
    if BACKUP_MODE=='hf' and _backup_api is None:raise RuntimeError('Configure and verify backup first.')
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    config=experiment_config();config_path=RUN/'experiment.json'
    if config_path.exists():assert json.loads(config_path.read_text())==config,'Config changed: restore settings or choose new RUN_TAG.'
    else:atomic_json(config,config_path)
    preserve_metadata()
    last=RUN/'last.pt'
    m=(make_model() if last.exists() else initial_model()).to(DEVICE)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,betas=(.9,.95),weight_decay=.1)
    scaler=torch.amp.GradScaler('cuda',enabled=DEVICE=='cuda' and AMP_DTYPE==torch.float16,init_scale=256.)
    rng=np.random.default_rng(SEED);step=ntokens=0;best=float('inf');elapsed_before=0.
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False)
        assert ck['experiment']==config
        m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        step,ntokens,best,elapsed_before=ck['step'],ck['target_tokens'],ck['best'],ck['elapsed']
        rng.bit_generator.state=ck['numpy_rng'];random.setstate(ck['python_rng']);torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng']);del ck
        print('Resumed',step,'updates;',ntokens,'loss tokens')
    start=time.monotonic();m.train();replay=None;phase="idle"
    last_save_time=time.monotonic()
    def save(force_remote=False):
        nonlocal last_save_time
        atomic_torch(dict(model_config=CFG,model=m.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
            experiment=config,step=step,target_tokens=ntokens,best=best,elapsed=elapsed_before+time.monotonic()-start,
            numpy_rng=rng.bit_generator.state,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()),last)
        atomic_json(dict(step=step,target_tokens=ntokens),RUN/'progress.json')
        last_save_time=time.monotonic()
        print('LOCAL checkpoint saved:',last,'step',step,flush=True)
        backup_run(checkpoints=True,force=force_remote)
    def assess():
        nonlocal best
        val=validate(m)
        if not math.isfinite(val):raise FloatingPointError('Nonfinite validation loss')
        row=dict(step=step,target_tokens=ntokens,val_loss=val,elapsed_seconds=elapsed_before+time.monotonic()-start)
        with (RUN/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(row,flush=True)
        if val<best:
            best=val;atomic_torch(dict(model_config=CFG,model=m.state_dict(),tokenizer_sha=sha(CACHE/'tokenizer.json'),
                step=step,target_tokens=ntokens,val_loss=val,experiment=config),RUN/'best.pt')
    save(force_remote=True) # Recoverable starting state before training.
    try:
        while ntokens<TRAIN_TARGET_TOKENS and (time.monotonic()-start)/3600<MAX_HOURS:
            p=max(0.,min(1.,(ntokens-WARMUP_TOKENS)/max(1,TRAIN_TARGET_TOKENS-WARMUP_TOKENS)))
            lr=LR*(ntokens+1)/max(1,WARMUP_TOKENS) if ntokens<WARMUP_TOKENS else MIN_LR+.5*(LR-MIN_LR)*(1+math.cos(math.pi*p))
            for group in opt.param_groups:group['lr']=lr
            # Snapshot RNG before accumulation, so interrupted updates can be replayed exactly.
            phase="accumulation"
            replay=(json.loads(json.dumps(rng.bit_generator.state)),torch.get_rng_state(),torch.cuda.get_rng_state_all())
            opt.zero_grad(set_to_none=True);n=0;loss_sum=0
            batches=[batch(TRAIN,rng,ARCH) for _ in range(ACCUM)]
            n=sum(int(y.ne(-100).sum()) for _,y,_ in batches)
            for x,y,src in batches:
                count=int(y.ne(-100).sum())
                with amp():loss=m(x,src=src,targets=y)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                scaler.scale(loss*(count/max(1,n))).backward();loss_sum+=float(loss.detach())*count
            scaler.unscale_(opt)
            norm=nn.utils.clip_grad_norm_(m.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient; rerun with BF16 or investigate LR')
            phase="optimizer"
            scaler.step(opt);scaler.update();step+=1;ntokens+=n;replay=None;phase="idle"
            if step%20==0:print('update',step,'target tokens',ntokens,'loss',round(loss_sum/max(1,n),4),'lr',lr,flush=True)
            if step==20 or step%EVAL_EVERY==0:assess()
            if step%SAVE_EVERY==0 or time.monotonic()-last_save_time>=LOCAL_EVERY_SECONDS:save()
    except KeyboardInterrupt:
        if phase=="optimizer":
            raise RuntimeError("Interrupted during optimizer mutation; rerun from last.pt. No partial state saved.")
        if replay is not None:
            rng.bit_generator.state=replay[0];torch.set_rng_state(replay[1]);torch.cuda.set_rng_state_all(replay[2])
        print('Interrupted; saving completed updates. Rerun to resume.')
    except BaseException:
        print('Error: resume from the last committed checkpoint, not a partially updated optimizer.')
        raise
    else:
        if step:assess()
    save(force_remote=True)
    del m,opt,scaler;gc.collect();torch.cuda.empty_cache()
    print('Run saved:',RUN)



with torch.device('meta'):probe=make_model()
n=sum(p.numel() for p in probe.parameters());del probe
assert 50_000_000 <= n < 300_000_000
print(ARCH,MODEL_SIZE,f'{n:,} parameters; GPU:',torch.cuda.get_device_name() if DEVICE=='cuda' else 'none')
preserve_metadata()





def smoke():
    base=dict(CFG,dim=48,heads=6,layers=4,ff_dim=128,recurrent_dim=24,anchor_rank=12,
              vocab=128,context=96,anchor_every=2,attention_every=2)
    variants=['full','no_anchor','fixed_gate'] if ARCH=='anchor_gpt' else ['full','no_recurrence','no_attention','fixed_gate']
    for ab in variants:
        c=dict(base,ablation=ab);torch.manual_seed(SEED);m=ExperimentalLM(c).cpu().eval()
        x=torch.randint(4,128,(2,40));changed=x.clone();changed[:,24:]=torch.randint(4,128,changed[:,24:].shape)
        with torch.no_grad():
            h=m.hidden(x);h2=m.hidden(changed)
            assert torch.allclose(h[:,:24],h2[:,:24],atol=3e-5),'Future-token leak'
            assert torch.allclose(h[:,:24],m.hidden(x[:,:24]),atol=3e-5),'Prefix mismatch'
            out,state=m.decode(x[:,:20])
            for i in range(20,40):
                out,state=m.decode(x[:,i:i+1],state)
                assert torch.allclose(out,m(x[:,:i+1]),atol=3e-5,rtol=3e-5),'Incremental cache mismatch'
        m.train();y=torch.randint(4,128,x.shape);y[:,-5:]=-100
        loss=m(x,targets=y);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        print(ab,'causality, cached inference and gradient checks passed')
smoke()






# Close any old inference model before benchmarking a resumed session.
if 'model' in globals():del model;gc.collect();torch.cuda.empty_cache()
benchmark_result=benchmark(updates=3)
budget_path=RUN/'training_budget.json'
if budget_path.exists():
    budget=json.loads(budget_path.read_text())
    if TRAIN_TARGET_TOKENS is not None:assert TRAIN_TARGET_TOKENS==budget['target_tokens'],'Do not change a resumed schedule.'
    TRAIN_TARGET_TOKENS=budget['target_tokens'];WARMUP_TOKENS=budget['warmup_tokens']
else:
    if TRAIN_TARGET_TOKENS is None:
        TRAIN_TARGET_TOKENS=max(10000,int(benchmark_result['loss_tokens_per_second']*TRAIN_HOURS*3600*.75))
    WARMUP_TOKENS=max(1,int(TRAIN_TARGET_TOKENS*.02))
    atomic_json(dict(target_tokens=TRAIN_TARGET_TOKENS,warmup_tokens=WARMUP_TOKENS),budget_path)
print('Fixed token budget:',TRAIN_TARGET_TOKENS,'warmup:',WARMUP_TOKENS)
print('25% time margin is an estimate; slow uploads/evaluation can exceed it. Generation time is separate.')
backup_run(checkpoints=False,force=True)






train_model()





# Run only after training or restoring a checkpoint from this architecture.
best_path=RUN/'best.pt'
if not best_path.exists():raise FileNotFoundError('No best.pt: train through at least one validation first.')
ck=torch.load(best_path,map_location='cpu',weights_only=False)
assert ck['model_config']==CFG and ck['tokenizer_sha']==sha(CACHE/'tokenizer.json')
model=make_model();model.load_state_dict(ck['model']);model=model.to(DEVICE).eval()
print('Loaded best checkpoint:',ck['step'],'updates, val loss',ck['val_loss']);del ck

GEN_TEMPERATURE=.7
GEN_TOP_P=.9
GEN_REPETITION_PENALTY=1.15
MAX_NEW_TOKENS=768
NO_REPEAT_NGRAM=4
PUNCT_REPEAT_LIMIT=6
DECODER_VERSION='remaining_v1_cached_retry'
RETRY_TEMPERATURES=[GEN_TEMPERATURE,.5,.85,0.]

class DecodeFailure(RuntimeError):pass

def corrupt(text):
    if any((ord(c)<32 and c not in '\n\r\t') or 127<=ord(c)<=159 for c in text):return True
    boundary=max(text.rfind(' '),text.rfind('\n'),text.rfind('\t'))
    # A final incomplete UTF-8 byte sequence can complete on the next token.
    return boundary>=0 and '\ufffd' in text[:boundary]

punct=torch.tensor([i for i in range(tok.get_vocab_size())
                    if (p:=tok.decode([i],skip_special_tokens=False)) and not any(c.isalnum() for c in p)],
                   device=DEVICE,dtype=torch.long)

@torch.inference_mode()
def continue_text(prompt,seed=42):
    model.eval();prefix=tok.encode(prompt).ids or [BOS];failures=[]
    for attempt,temp in enumerate(RETRY_TEMPERATURES):
        trial_seed=seed+100003*attempt
        rng=torch.Generator(device=DEVICE).manual_seed(trial_seed)
        current=prefix[-CONTEXT:];gen=[];state=None;ngrams=defaultdict(set);text='';stalled=0;wc=0;resets=0
        try:
            for _ in range(MAX_NEW_TOKENS):
                if state is None:feed=current
                elif state['length']>=CONTEXT:
                    # Restart positions/recurrent states on the recent half-window.
                    # This is an explicit inference policy, not unlimited trained context.
                    state=None;feed=current[-max(1,CONTEXT//2):];resets+=1
                else:feed=[current[-1]]
                with amp():logits,state=model.decode(torch.tensor([feed],device=DEVICE),state)
                logits=logits[0].float().clone()
                if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite model logits; inspect checkpoint/precision.')
                logits[[PAD,BOS,EOS,UNK]]=-float('inf')
                if gen:
                    seen=list(set(gen[-256:]));z=logits[seen]
                    logits[seen]=torch.where(z>=0,z/GEN_REPETITION_PENALTY,z*GEN_REPETITION_PENALTY)
                if len(gen)>=NO_REPEAT_NGRAM-1:
                    banned=ngrams.get(tuple(gen[-(NO_REPEAT_NGRAM-1):]),())
                    if banned:logits[list(banned)]=-float('inf')
                if re.search(r'([^\w\s])(?:\s*\1){5,}$',text) or len(text)-len(text.rstrip())>=16:
                    logits[punct]=-float('inf')
                if not torch.isfinite(logits).any():raise DecodeFailure('All tokens blocked')
                if temp==0:nxt=int(logits.argmax())
                else:
                    scores,order=torch.sort(logits/temp,descending=True);p=scores.softmax(-1)
                    scores[(p.cumsum(-1)-p)>GEN_TOP_P]=-float('inf')
                    nxt=int(order[torch.multinomial(scores.softmax(-1),1,generator=rng)])
                gen.append(nxt);current.append(nxt)
                if len(gen)>=NO_REPEAT_NGRAM:ngrams[tuple(gen[-NO_REPEAT_NGRAM:-1])].add(nxt)
                text=tok.decode(gen,skip_special_tokens=True)
                if corrupt(text):raise DecodeFailure('Control characters or invalid completed UTF-8')
                words=text.split();stalled=stalled+1 if len(words)<=wc else 0;wc=len(words)
                if stalled>=96:raise DecodeFailure('96 tokens without another word')
                if len(words)>=129:
                    result=' '.join(words[:128])
                    if '\ufffd' in result:raise DecodeFailure('Invalid decoded text')
                    return result,dict(tokens=len(gen),attempt=attempt,seed=trial_seed,temperature=temp,
                                       context_resets=resets,previous_failures=failures)
            raise DecodeFailure(f'Only {wc} words at token cap')
        except DecodeFailure as e:
            failures.append(dict(attempt=attempt,reason=str(e)))
            print('Retrying generation:',failures[-1],flush=True)
        finally:state=None
    raise DecodeFailure('All bounded retries failed: '+json.dumps(failures))

def rouge_l(a,b):
    a=a.lower().split();b=b.lower().split();dp=[0]*(len(b)+1)
    for x in a:
        old=dp[:]
        for j,y in enumerate(b,1):dp[j]=old[j-1]+1 if x==y else max(old[j],dp[j-1])
    return 2*dp[-1]/max(1,len(a)+len(b))

def evaluate_continuations(n=16):
    rows=json.loads((CACHE/'continuation_dev.json').read_text())
    groups=[[r for r in rows if r['source']==s] for s in SOURCES]
    rows=[g[j] for j in range(max(map(len,groups),default=0)) for g in groups if j<len(g)][:n]
    if not rows:raise ValueError('No held-out continuations: increase the corpus sample.')
    result=[];start=time.monotonic()
    for i,r in enumerate(rows):
        try:
            text,info=continue_text(r['prompt'],SEED+i)
            result.append({**r,'prediction':text,'rouge_l':rouge_l(text,r['reference']),**info})
        except DecodeFailure as e:result.append({**r,'error':str(e),'rouge_l':0.})
        print(i+1,'/',len(rows),'ROUGE-L',result[-1]['rouge_l'],flush=True)
    atomic_json(result,RUN/'continuation_eval.json')
    print('Mean local ROUGE-L:',np.mean([r['rouge_l'] for r in result]))
    print('Measured seconds/prompt:',(time.monotonic()-start)/len(rows),'; allow time for 1000 official prompts.')
    return result

def valid_prediction(v):
    return isinstance(v,dict) and set(v)=={'text'} and isinstance(v['text'],str) and len(v['text'].split())==128 and not corrupt(v['text']) and '\ufffd' not in v['text']

def generate_submission():
    path=official_test_path();rows=[(key,text_of(row)) for key,row in records(path,require_id=True)]
    ids=[key for key,_ in rows]
    assert len(ids)==len(set(ids)) and len(ids)>0
    cfg=dict(checkpoint_sha=sha(best_path),test_sha=sha(path),tokenizer_sha=sha(CACHE/'tokenizer.json'),
             decoder=DECODER_VERSION,temperatures=RETRY_TEMPERATURES,top_p=GEN_TOP_P,
             penalty=GEN_REPETITION_PENALTY,max_tokens=MAX_NEW_TOKENS,ngram=NO_REPEAT_NGRAM,
             punct=PUNCT_REPEAT_LIMIT,seed=SEED,context=CONTEXT)
    tag=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:12]
    dest=RUN/('submission_'+tag);dest.mkdir(exist_ok=True)
    atomic_json(cfg,dest/'manifest.json');pred_path=dest/'prediction.json'
    preds=json.loads(pred_path.read_text()) if pred_path.exists() else {}
    assert set(preds)<=set(ids)
    failed={};new=0
    info_path=dest/'generation_info.json'
    details=json.loads(info_path.read_text()) if info_path.exists() else {}
    for i,(key,prompt) in enumerate(rows):
        if key in preds:
            assert valid_prediction(preds[key]);continue
        try:text,info=continue_text(prompt,SEED+i)
        except DecodeFailure as e:
            failed[key]=str(e);atomic_json(failed,dest/'failures.json')
            print('Failed ID',key,'— continuing other prompts, no filler');continue
        preds[key]={'text':text};details[key]=info;new+=1
        atomic_json(preds,pred_path);atomic_json(details,info_path)
        print(f'Test {i+1}/{len(rows)} LOCAL saved; total={len(preds)}, tokens={info["tokens"]}',flush=True)
        if new%10==0:backup_run(checkpoints=False,force=True)
    backup_run(checkpoints=False,force=True)
    missing=set(ids)-set(preds)
    if missing:raise RuntimeError(f'{len(missing)} IDs remain. Saved successes are resumable. Inspect failures.json; no incomplete ZIP emitted.')
    assert all(valid_prediction(v) for v in preds.values())
    atomic_json({},dest/'failures.json')
    archive=RUN/(RUN.name+'_'+tag+'.zip')
    with zipfile.ZipFile(str(archive)+'.tmp','w',zipfile.ZIP_DEFLATED) as z:z.write(pred_path,'prediction.json')
    os.replace(str(archive)+'.tmp',archive)
    with zipfile.ZipFile(archive) as z:
        assert z.namelist()==['prediction.json'] and json.loads(z.read('prediction.json'))==preds
    backup_run(checkpoints=False,force=True)
    print('Validated submission:',archive)
    return archive






dev_results=evaluate_continuations(n=16)
for row in dev_results[:2]:
    print('PREDICTION:',row.get('prediction',row.get('error')))
    print('REFERENCE:',row['reference'])





archive=generate_submission()
from IPython.display import FileLink,display
display(FileLink(str(archive)))





export=RUN/(RUN.name+'_foundation.zip')
with zipfile.ZipFile(str(export)+'.tmp','w',zipfile.ZIP_STORED) as z:
    for name in ('best.pt','model_config.json','tokenizer.json','modeling.py','data_config.json','data_report.json','experiment.json','training_budget.json'):
        z.write(RUN/name,'model.pt' if name=='best.pt' else name)
os.replace(str(export)+'.tmp',export)
backup_run(checkpoints=True,force=True)
from IPython.display import FileLink,display
display(FileLink(str(export)))





print("done")


print(RUN)
for z in sorted(Path('/kaggle/working').rglob('*.zip')): print(z)
for p in sorted(RUN.glob('submission_*/prediction.json')): print(p, len(json.loads(p.read_text())), 'predictions')

from IPython.display import FileLink, display
zs = sorted(RUN.glob('*.zip'))
if zs: display(FileLink(str(zs[-1])))


from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
from IPython.display import FileLink, display
import os
import shutil

phase = "relay_phase1"
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













































