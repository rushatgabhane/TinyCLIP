import torch
import torch.nn.functional as F

from gqa_study.variants import encoder_blocks


def distill_loss(student, teacher):
    return (1 - F.cosine_similarity(student, teacher, dim=-1)).mean()


def attn_params(model):
    return [p for blocks in encoder_blocks(model) for blk in blocks
            for p in blk.attn.parameters() if p.requires_grad]


def uptrain(student, teacher, image_loader, texts, device="cpu", steps=300, lr=1e-4):
    student = student.to(device).train()
    texts = texts.to(device)
    opt = torch.optim.AdamW(attn_params(student), lr=lr)
    it = iter(image_loader)
    for step in range(steps):
        try:
            imgs, _ = next(it)
        except StopIteration:
            it = iter(image_loader)
            imgs, _ = next(it)
        imgs = imgs.to(device)
        with torch.no_grad():
            ti = teacher.encode_image(imgs)
            tt = teacher.encode_text(texts)
        si = student.encode_image(imgs)
        st = student.encode_text(texts)
        loss = distill_loss(si, ti) + distill_loss(st, tt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"  step {step + 1}/{steps} loss {loss.item():.4f}")
    return student.eval()
