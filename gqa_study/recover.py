import torch
import torch.nn as nn
import torch.nn.functional as F

from gqa_study.variants import encoder_blocks


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.t()) @ self.B.t()


def add_lora(module, r=8, alpha=16):
    wrapped = []
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        sub = getattr(module, name, None)
        if isinstance(sub, nn.Linear):
            setattr(module, name, LoRALinear(sub, r, alpha))
            wrapped.append(name)
    assert wrapped
    return module


def distill_loss(student, teacher, kind="cosine"):
    if kind == "mse":
        return F.mse_loss(student, teacher)
    return (1 - F.cosine_similarity(student, teacher, dim=-1)).mean()


def attn_modules(model):
    return [blk.attn for blocks in encoder_blocks(model) for blk in blocks]


def light_tune(student, teacher, loader, device="cpu", use_lora=False, steps=100, lr=1e-4):
    student = student.to(device)
    if use_lora:
        for a in attn_modules(student):
            add_lora(a)
        student = student.to(device)
    params = [p for a in attn_modules(student) for p in a.parameters() if p.requires_grad]
    assert params
    opt = torch.optim.AdamW(params, lr=lr)
    student.train()
    it = iter(loader)
    done = 0
    while done < steps:
        try:
            imgs, _ = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, _ = next(it)
        imgs = imgs.to(device)
        with torch.no_grad():
            tfeat = teacher.encode_image(imgs)
        sfeat = student.encode_image(imgs)
        loss = distill_loss(sfeat, tfeat)
        opt.zero_grad()
        loss.backward()
        opt.step()
        done += 1
    return student.eval()
