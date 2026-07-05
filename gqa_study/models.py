from dataclasses import dataclass

SEED = 0


@dataclass(frozen=True)
class ModelSpec:
    key: str
    open_clip_id: str
    layers: int
    width: int
    n_heads: int
    head_dim: int
    patch: int
    image_size: int


MODELS = {
    "tinyclip-39m": ModelSpec(
        "tinyclip-39m",
        "hf-hub:timm/vit_medium_patch16_clip_224.tinyclip_yfcc15m",
        layers=12, width=512, n_heads=8, head_dim=64, patch=16, image_size=224),
    "tinyclip-8m": ModelSpec(
        "tinyclip-8m",
        "hf-hub:timm/vit_xsmall_patch16_clip_224.tinyclip_yfcc15m",
        layers=10, width=256, n_heads=4, head_dim=64, patch=16, image_size=224),
}


def method_grid(n_heads):
    names = {n_heads: "MHA", 1: "MQA"}
    grid = []
    for kv in sorted({n_heads, n_heads // 2, 1}, reverse=True):
        if kv < 1 or n_heads % kv != 0:
            continue
        grid.append((names.get(kv, f"GQA-{kv}"), kv))
    return grid


def load_tinyclip(model_key, device="cpu"):
    spec = MODELS[model_key]
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(spec.open_clip_id)
    tokenizer = open_clip.get_tokenizer(spec.open_clip_id)
    return model.to(device).eval(), preprocess, tokenizer


def describe_attention(model):
    blocks = model.visual.transformer.resblocks
    a0 = blocks[0].attn
    info = {
        "num_blocks": len(blocks),
        "attn_type": type(a0).__module__ + "." + type(a0).__name__,
    }
    if hasattr(a0, "in_proj_weight"):
        info["in_proj_shape"] = tuple(a0.in_proj_weight.shape)
        info["num_heads"] = a0.num_heads
    return info
