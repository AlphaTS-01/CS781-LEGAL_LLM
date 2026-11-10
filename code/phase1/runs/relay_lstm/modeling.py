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
