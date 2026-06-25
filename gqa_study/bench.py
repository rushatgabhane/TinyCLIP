import time

import torch


def _percentile(sorted_vals, q):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = (n - 1) * (q / 100.0)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize(times_ms, batch_size):
    s = sorted(times_ms)
    mean = sum(s) / len(s)
    return {
        "mean_ms": mean,
        "p50_ms": _percentile(s, 50),
        "p95_ms": _percentile(s, 95),
        "throughput_img_s": batch_size * 1000.0 / mean,
    }


@torch.no_grad()
def benchmark(model, input_shape, device="cpu", warmup=30, iters=100):
    model = model.to(device).eval()
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = torch.float32
    x = torch.randn(*input_shape, device=device, dtype=dtype)
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        model(x)
    if cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if cuda:
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            model(x)
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b))
        else:
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    out = summarize(times, input_shape[0])
    out["peak_mem_mb"] = (torch.cuda.max_memory_allocated() / 1024 ** 2) if cuda else None
    return out


class ImageTower(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, x):
        return self.clip.encode_image(x)
