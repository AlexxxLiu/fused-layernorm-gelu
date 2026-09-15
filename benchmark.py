"""Benchmark: fused LayerNorm+GELU vs PyTorch eager vs torch.compile.

Run:
    python setup.py build_ext --inplace
    python benchmark.py --out results.csv
"""

import argparse
import csv
import statistics
import sys

import torch
import torch.nn.functional as F

try:
    import fused_ln_gelu
except ImportError:
    sys.exit("build the extension first:  python setup.py build_ext --inplace")


# Shapes chosen to sweep from "launch-overhead dominated" to "bandwidth bound".
# (N, H) where N = batch*seq tokens, H = hidden dim.
DEFAULT_SHAPES = [
    (128, 256),
    (512, 512),
    (2048, 768),
    (4096, 1024),
    (8192, 1024),
    (8192, 2048),
    (16384, 2048),
    (16384, 4096),
]


def baseline(x, gamma, beta, eps):
    y = F.layer_norm(x, (x.shape[-1],), weight=gamma, bias=beta, eps=eps)
    return F.gelu(y, approximate="tanh")


def fused(x, gamma, beta, eps):
    return fused_ln_gelu.fused_layernorm_gelu(x, gamma, beta, eps)


def time_fn(fn, args, warmup=25, iters=100):
    """Returns (median_ms, p90_ms). CUDA events, one measurement per iter."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn(*args)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return statistics.median(samples), samples[int(0.9 * len(samples))]


def check_correctness(x, gamma, beta, eps, tol=2e-3):
    ref = baseline(x, gamma, beta, eps)
    got = fused(x, gamma, beta, eps)
    max_abs = (ref - got).abs().max().item()
    rel = max_abs / max(ref.abs().max().item(), 1e-6)
    ok = rel < tol
    return ok, max_abs, rel


def bytes_moved(n, h, fused_path):
    """Minimum HBM traffic in bytes, fp32."""
    elem = 4
    if fused_path:
        return n * h * elem * 2           # read x, write z
    return n * h * elem * 4              # read x, write y, read y, write z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--compile", action="store_true",
                    help="also benchmark torch.compile'd baseline")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible")

    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}\n")

    compiled = torch.compile(baseline) if args.compile else None

    rows = []
    header = f"{'N':>7} {'H':>6} {'eager(ms)':>10} {'fused(ms)':>10} {'speedup':>8} {'GB/s':>8}"
    if compiled:
        header += f" {'compiled':>10}"
    print(header)
    print("-" * len(header))

    for n, h in DEFAULT_SHAPES:
        x = torch.randn(n, h, device=dev, dtype=torch.float32)
        gamma = torch.randn(h, device=dev, dtype=torch.float32)
        beta = torch.randn(h, device=dev, dtype=torch.float32)

        ok, max_abs, rel = check_correctness(x, gamma, beta, args.eps)
        if not ok:
            print(f"  !! correctness FAILED at ({n},{h}): max_abs={max_abs:.3e} rel={rel:.3e}")

        t_base, _ = time_fn(baseline, (x, gamma, beta, args.eps), iters=args.iters)
        t_fused, _ = time_fn(fused, (x, gamma, beta, args.eps), iters=args.iters)
        t_comp = None
        if compiled:
            t_comp, _ = time_fn(compiled, (x, gamma, beta, args.eps), iters=args.iters)

        speedup = t_base / t_fused
        gbps = bytes_moved(n, h, True) / (t_fused * 1e-3) / 1e9

        line = f"{n:>7} {h:>6} {t_base:>10.4f} {t_fused:>10.4f} {speedup:>7.2f}x {gbps:>8.1f}"
        if t_comp is not None:
            line += f" {t_comp:>10.4f}"
        print(line)

        rows.append({
            "N": n, "H": h,
            "eager_ms": round(t_base, 5),
            "fused_ms": round(t_fused, 5),
            "compiled_ms": round(t_comp, 5) if t_comp is not None else "",
            "speedup": round(speedup, 3),
            "fused_gbps": round(gbps, 2),
            "max_abs_err": f"{max_abs:.3e}",
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
