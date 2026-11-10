"""LEXWeave: a project-specific experimental causal LM, not a pretrained model.

Parallel token attention, strictly completed-chunk memory, and routed dilated
convolution, combined by a token-dependent gate. Copied verbatim from the Phase-1 notebook (section 5).
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
