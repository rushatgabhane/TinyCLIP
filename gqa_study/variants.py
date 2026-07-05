import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_heads(t, n_heads, head_dim, batch_first):
    if batch_first:
        B, S, _ = t.shape
        return t.view(B, S, n_heads, head_dim).transpose(1, 2)
    S, B, _ = t.shape
    return t.view(S, B, n_heads, head_dim).permute(1, 2, 0, 3)


def _merge_heads(o, embed_dim, batch_first):
    B, _, S, _ = o.shape
    if batch_first:
        return o.transpose(1, 2).reshape(B, S, embed_dim)
    return o.permute(2, 0, 1, 3).reshape(S, B, embed_dim)


class GroupedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads, sdpa_gqa=False, batch_first=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.sdpa_gqa = sdpa_gqa
        self.batch_first = batch_first
        kv_dim = num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, kv_dim)
        self.v_proj = nn.Linear(embed_dim, kv_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, need_weights=False, attn_mask=None):
        bf = self.batch_first
        q = _split_heads(self.q_proj(query), self.num_heads, self.head_dim, bf)
        k = _split_heads(self.k_proj(key), self.num_kv_heads, self.head_dim, bf)
        v = _split_heads(self.v_proj(value), self.num_kv_heads, self.head_dim, bf)
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            if self.sdpa_gqa:
                try:
                    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
                except TypeError:
                    o = F.scaled_dot_product_attention(
                        q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1),
                        attn_mask=attn_mask)
            else:
                o = F.scaled_dot_product_attention(
                    q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1),
                    attn_mask=attn_mask)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out_proj(_merge_heads(o, self.embed_dim, bf)), None

    @classmethod
    def from_mha(cls, mha, num_kv_heads, sdpa_gqa=False):
        d = mha.embed_dim
        h = mha.num_heads
        hd = d // h
        g = h // num_kv_heads
        mod = cls(d, h, num_kv_heads, sdpa_gqa=sdpa_gqa,
                  batch_first=getattr(mha, "batch_first", False))
        W = mha.in_proj_weight.detach()
        b = mha.in_proj_bias.detach()
        Wq, Wk, Wv = W[:d], W[d:2 * d], W[2 * d:]
        bq, bk, bv = b[:d], b[d:2 * d], b[2 * d:]

        def pool_w(M):
            return M.view(h, hd, d).reshape(num_kv_heads, g, hd, d).mean(1).reshape(num_kv_heads * hd, d)

        def pool_b(vec):
            return vec.view(h, hd).reshape(num_kv_heads, g, hd).mean(1).reshape(num_kv_heads * hd)

        with torch.no_grad():
            mod.q_proj.weight.copy_(Wq)
            mod.q_proj.bias.copy_(bq)
            mod.k_proj.weight.copy_(pool_w(Wk))
            mod.k_proj.bias.copy_(pool_b(bk))
            mod.v_proj.weight.copy_(pool_w(Wv))
            mod.v_proj.bias.copy_(pool_b(bv))
            mod.out_proj.weight.copy_(mha.out_proj.weight.detach())
            mod.out_proj.bias.copy_(mha.out_proj.bias.detach())
        return mod


def encoder_blocks(model):
    encoders = [model.visual.transformer.resblocks]
    text = getattr(model, "transformer", None)
    if text is not None and hasattr(text, "resblocks"):
        encoders.append(text.resblocks)
    return encoders


def convert(model, num_kv_heads, sdpa_gqa=False):
    for blocks in encoder_blocks(model):
        for blk in blocks:
            blk.attn = GroupedAttention.from_mha(blk.attn, num_kv_heads, sdpa_gqa=sdpa_gqa)
    return model


def param_report(model):
    total = sum(p.numel() for p in model.parameters())
    attn = kv = 0
    for m in model.modules():
        if isinstance(m, GroupedAttention):
            attn += sum(p.numel() for p in m.parameters())
            kv += sum(p.numel() for p in list(m.k_proj.parameters()) + list(m.v_proj.parameters()))
        elif isinstance(m, nn.MultiheadAttention):
            attn += sum(p.numel() for p in m.parameters())
            d = m.embed_dim
            kv += 2 * (d * d + d)
    return {"total": total, "attn": attn, "kv": kv}
