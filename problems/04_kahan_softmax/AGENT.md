# Problem 04 — Kahan-corrected Softmax

## Task

Write `solution.py` implementing a numerically stable softmax over the last
dimension:

```
y[b, i] = exp(x[b, i] - max_j x[b, j]) / sum_k exp(x[b, k] - max_j x[b, j])
```

The `Model` class must define the same `__init__(batch, vocab)` signature as
`reference.py` and accept a single fp32 input `x` of shape `(batch, vocab)`.

The reference is computed in **fp64** so the tolerance (atol = rtol = 1e-5)
is tight enough that naive fp16 accumulation across a 256K-element row
drifts past it. You must use either fp32 accumulation throughout or — if
you want to stage exponentials in lower precision for bandwidth — apply a
Kahan / Neumaier compensated summation to recover the lost low bits.

## Shapes

`shapes.py` covers small, GPT-2 vocab, Llama3 (128K), DeepSeek-class (256K),
and an extreme-logit edge case where a few entries per row are 30.0 — if
your kernel forgets the max-subtract step before `exp`, that shape will
overflow fp32 and you will see NaN.

## Success criterion

Score is **achieved GB/s / peak DRAM bandwidth** on RTX PRO 6000 (1.8 TB/s
GDDR7). Softmax is memory-bound; aim for >50% of peak.

## Rules

- **Forbidden:** `torch.nn.functional.softmax`, `torch.softmax`, `F.softmax`,
  any `liger_kernel.softmax` import, and the bare method `.softmax(`.
- You must write the reduction yourself (Triton, CUDA, or pure PyTorch
  primitives — `exp`, `sum`, `max`, `div`).
- A single-pass online softmax (Milakov & Gimelshein 2018) is the fastest
  formulation; a two-pass with explicit max + Kahan-corrected sum is the
  most accurate. Either is acceptable as long as the tolerance is met.

## References

- **Liger-Kernel softmax (Triton, fp32-acc)** — https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/softmax.py
- **Kahan summation algorithm** — https://en.wikipedia.org/wiki/Kahan_summation_algorithm
- **Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)** — https://arxiv.org/abs/1805.02867

## Budget

30 minutes wall clock.
