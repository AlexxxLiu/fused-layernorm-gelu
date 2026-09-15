# Fused LayerNorm + GELU

LayerNorm followed by GELU is everywhere in a transformer. Run as two ops it
costs four passes over HBM:

```
read x -> write y        (layer_norm)
read y -> write z        (gelu)
```

The arithmetic is trivial, so the kernel is bandwidth bound and those extra two
passes *are* the runtime. Fusing gets it to read-once, write-once, and drops a
kernel launch.

```
read x -> write z        (fused)
```

Ceiling is therefore ~2x on large shapes, less on small ones where launch
overhead rather than bandwidth dominates.

## Build and run

```bash
python setup.py build_ext --inplace
python benchmark.py --out results.csv
python plot_results.py --csv results.csv --out speedup.png
```

Add `--compile` to also time `torch.compile`, which fuses these itself — a
fairer opponent than eager, and the one worth beating.

## What it does

`fused_layernorm_gelu.cu` — one block per row. Warp-shuffle reduction for mean
and variance, block-level reduction across warps, then a second pass that
normalizes, applies the affine transform, runs GELU, and writes out. Block
size adapts to hidden dim.

`benchmark.py` — sweeps shapes from launch-bound to bandwidth-bound. Uses CUDA
events, warms up before timing, reports median over 100 iterations, and
verifies against the PyTorch reference before timing anything. Also reports
achieved GB/s so you can see how close to the memory roofline you are.

Correctness tolerance is 2e-3 relative. `--use_fast_math` changes `tanhf`
slightly, so bit-exactness is not the goal.

## Where to take it next

- **`float4` vectorized loads** when `H % 4 == 0`. The scalar loop is the
  obvious first thing to fix; on a memory-bound kernel this is usually the
  single biggest remaining win.
- **fp16 / bf16**, accumulating statistics in fp32. This is what you actually
  run in production, and it changes the bandwidth math by 2x.
- **Welford** for the variance. The `E[x²] − E[x]²` form is fine in fp32 but
  loses precision once inputs get large or you accumulate in half precision.
- **Backward pass**, if you want it usable for training rather than inference.
- **Profile with `ncu`**: `ncu --set full python benchmark.py`. The build
  already passes `-lineinfo`, so stalls map back to source lines. Check
  `dram__bytes.sum` against the theoretical minimum — if the fused kernel is
  moving more than `2 * N * H * 4` bytes, something is reading twice.
- **Compare against `torch.compile`** on your actual shapes. If Inductor wins,
  read its generated Triton and find out why.
