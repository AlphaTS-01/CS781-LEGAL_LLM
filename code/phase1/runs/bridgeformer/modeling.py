"""BridgeFormer: a project-specific encoder-decoder LM, not a pretrained model.

Bidirectional prompt encoder + causal decoder with a gated mix of token-level and
latent-resampled cross-attention. No claim of literature novelty.
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


class BridgeAttention(nn.Module):
    def __init__(self,d,h,rope=False):
        super().__init__();self.h=h;self.dh=d//h;self.rope=Rotary(self.dh) if rope else None
        self.q=nn.Linear(d,d,bias=False);self.kv=nn.Linear(d,2*d,bias=False);self.out=nn.Linear(d,d,bias=False)
    def forward(self,x,memory=None,valid=None,causal=False):
        m=x if memory is None else memory;b,t,d=x.shape
        q=self.q(x).view(b,t,self.h,self.dh).transpose(1,2)
        k,v=self.kv(m).view(b,m.shape[1],2,self.h,self.dh).unbind(2)
        k=k.transpose(1,2);v=v.transpose(1,2)
        if self.rope is not None:
            q=self.rope(q,torch.arange(t,device=x.device));k=self.rope(k,torch.arange(m.shape[1],device=x.device))
        mask=valid[:,None,None,:] if valid is not None else None
        if causal and mask is not None:raise ValueError('Use right padding for causal decoder self-attention')
        z=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=causal,dropout_p=0.)
        return self.out(z.transpose(1,2).contiguous().view(b,t,d))

class BridgeFFN(nn.Module):
    def __init__(self,d,inner):
        super().__init__();self.up=nn.Linear(d,2*inner,bias=False);self.down=nn.Linear(inner,d,bias=False)
    def forward(self,x):
        a,b=self.up(x).chunk(2,-1);return self.down(F.silu(a)*b)

class EncoderBlock(nn.Module):
    def __init__(self,c):
        super().__init__();d=c['dim'];self.n1=RMSNorm(d);self.n2=RMSNorm(d)
        self.attn=BridgeAttention(d,c['heads'],True);self.ff=BridgeFFN(d,c['ff_dim'])
    def forward(self,x,valid):
        x=x+self.attn(self.n1(x),valid=valid)
        return x+self.ff(self.n2(x))

class BridgeDecoderBlock(nn.Module):
    def __init__(self,c):
        super().__init__();d=c['dim'];self.ablation=c['ablation']
        self.n1=RMSNorm(d);self.n2=RMSNorm(d);self.n3=RMSNorm(d)
        self.self_attn=BridgeAttention(d,c['heads'],True)
        self.detail=BridgeAttention(d,c['heads'])
        self.summary=BridgeAttention(d,c['heads']) if self.ablation!='token_only' else None
        self.gate=nn.Linear(d,1,bias=True) if self.ablation=='full' else None
        self.ff=BridgeFFN(d,c['ff_dim'])
    def forward(self,x,tokens,latent,valid):
        x=x+self.self_attn(self.n1(x),causal=True);z=self.n2(x)
        detail=self.detail(z,tokens,valid)
        if self.summary is not None:
            summary=self.summary(z,latent)
            g=torch.sigmoid(self.gate(z)) if self.gate is not None else 0.5
            detail=g*detail+(1-g)*summary
        x=x+detail;return x+self.ff(self.n3(x))

class BridgeLM(nn.Module):
    """Bidirectional prompt encoder; causal decoder with gated token/latent cross attention.
    Latents are global learned queries, not causal chunk memory. Encoder sees prompt only.
    """
    def __init__(self,c):
        super().__init__();self.cfg=dict(c);self.kind='bridge';self.grad_ckpt=c['grad_checkpoint']
        d=c['dim'];self.emb=nn.Embedding(c['vocab'],d)
        self.encoder=nn.ModuleList([EncoderBlock(c) for _ in range(c['encoder_layers'])])
        self.enc_norm=RMSNorm(d)
        if c['ablation']!='token_only':
            self.latents=nn.Parameter(torch.empty(c['latent_slots'],d))
            self.resample=BridgeAttention(d,c['heads']);self.latent_norm=RMSNorm(d)
        self.decoder=nn.ModuleList([BridgeDecoderBlock(c) for _ in range(c['decoder_layers'])])
        self.norm=RMSNorm(d);self.apply(self.init_weights)
        if hasattr(self,'latents'):nn.init.normal_(self.latents,std=.02)
        for b in self.decoder:
            if b.gate is not None:nn.init.zeros_(b.gate.weight);nn.init.constant_(b.gate.bias,1.)
    @staticmethod
    def init_weights(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,std=.02)
            if isinstance(m,nn.Linear) and m.bias is not None:nn.init.zeros_(m.bias)
    def encode(self,src):
        if src.shape[1]>self.cfg['source_context']:raise ValueError('Prompt too long')
        valid=src.ne(self.cfg['pad'])
        if not valid.any(-1).all():raise ValueError('Empty encoder prompt')
        x=self.emb(src)
        for b in self.encoder:
            x=checkpoint(b,x,valid,use_reentrant=False) if self.training and self.grad_ckpt else b(x,valid)
        x=self.enc_norm(x)
        latent=x[:,:1]  # unused in token_only ablation
        if hasattr(self,'latents'):
            q=self.latents[None].expand(src.shape[0],-1,-1)
            latent=self.latent_norm(q+self.resample(q,x,valid))
        return x,latent,valid
    def hidden(self,ids,src=None,memory=None):
        if ids.shape[1]>self.cfg['context']:raise ValueError('Decoder context exceeded')
        if memory is None:memory=self.encode(src)
        x=self.emb(ids)
        for b in self.decoder:
            x=checkpoint(b,x,*memory,use_reentrant=False) if self.training and self.grad_ckpt else b(x,*memory)
        return self.norm(x)
    def forward(self,ids,src=None,targets=None,memory=None):
        x=self.hidden(ids,src,memory)
        if targets is None:return F.linear(x[:,-1],self.emb.weight).float()
        total=x.new_zeros((),dtype=torch.float32)
        def ce(h,y):
            return F.cross_entropy(F.linear(h,self.emb.weight).float().reshape(-1,self.cfg['vocab']),y.reshape(-1),reduction='sum')
        for j in range(0,x.shape[1],128):
            h,y=x[:,j:j+128],targets[:,j:j+128]
            total+=checkpoint(ce,h,y,use_reentrant=False) if self.training and self.grad_ckpt else ce(h,y)
        return total/targets.ne(-100).sum().clamp_min(1)

def bridge_config(vocab,size='200m',ablation='full',grad_checkpoint=True):
    sizes={'small':(512,8,4,8,1376),'200m':(768,12,6,12,2048),'240m':(768,12,8,14,2048)}
    d,h,e,n,ff=sizes[size]
    assert ablation in ('full','token_only','fixed_gate')
    return dict(arch='bridge',dim=d,heads=h,encoder_layers=e,decoder_layers=n,ff_dim=ff,
                vocab=vocab,pad=PAD,context=CONTEXT,source_context=SOURCE_CONTEXT,
                latent_slots=32,ablation=ablation,grad_checkpoint=grad_checkpoint)
