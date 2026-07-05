import torch
import torch.nn.functional as F


def accuracy(logits, target, topk=(1, 5)):
    maxk = min(max(topk), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(target.view(-1, 1))
    return [int(correct[:, :min(k, maxk)].any(1).sum().item()) for k in topk]


@torch.no_grad()
def build_text_classifier(model, tokenizer, classnames, templates, device="cpu"):
    weights = []
    for cn in classnames:
        texts = tokenizer([t(cn) for t in templates]).to(device)
        emb = F.normalize(model.encode_text(texts), dim=-1)
        proto = F.normalize(emb.mean(0), dim=0)
        weights.append(proto)
    return torch.stack(weights, dim=1)


@torch.no_grad()
def evaluate_classifier(model, classifier, loader, device="cpu", topk=(1, 5)):
    classifier = classifier.to(device)
    model = model.to(device).eval()
    n = c1 = c5 = 0
    for images, target in loader:
        images = images.to(device)
        target = target.to(device)
        feats = F.normalize(model.encode_image(images), dim=-1)
        logits = feats @ classifier
        a1, a5 = accuracy(logits, target, topk=topk)
        c1 += a1
        c5 += a5
        n += images.shape[0]
    return {"top1": c1 / n, "top5": c5 / n, "n": n}
