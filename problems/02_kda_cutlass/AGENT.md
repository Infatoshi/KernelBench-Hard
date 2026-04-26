# Problem 02 -- Kimi Delta Attention (chunk forward) via CUTLASS CuTe

## Task

Write `solution.py` implementing the Kimi Delta Attention (KDA) forward pass in
chunk form. KDA is the per-channel-decay variant of the Delta Rule used in
Moonshot's Kimi Linear architecture: it sits between Mamba-style state-space
recurrence and softmax attention, with linear-time complexity in sequence length.

`Model` must take the same `__init__(B, T, H, K, V, chunk_size)` signature as
`reference.py` and accept the five activations `(q, k, v, g, beta)` exactly as
produced by `reference.get_inputs()`. No learned parameters. Output is `o` of
shape `(B, T, H, V)` in bf16.

## Shapes

See `shapes.py`. Mid-context bf16 with K=V=128 per head, chunk_size=64 (the
Kimi Linear default). Geomean over four shapes.

## Success criterion

Score is **achieved bf16 TFLOPS / peak bf16 TFLOPS** on RTX PRO 6000 (SM120,
~200 TFLOPS dense bf16). Tolerance is `atol=rtol=5e-2` to absorb the
chunked recurrence's accumulated error.

## Rules

- **Forbidden:** any import or symbol from `fla.ops.kda`, including
  `chunk_kda`, `fused_recurrent_kda`, `naive_chunk_kda`,
  `naive_recurrent_kda`. The whole point is to write the kernel.
- You MUST implement the math yourself via CUTLASS CuTe, raw CUDA / PTX,
  Triton, or `torch.utils.cpp_extension.load_inline`. SM120 means
  register-based MMA accumulation (no `tcgen05`, no TMEM, no 2-CTA).
- Compile flags for nvcc paths: `-arch=sm_120a -std=c++17 -O3
  -I/opt/cutlass/include`.

## References (navigate yourself)

- KDA paper / Kimi Linear architecture --
  https://arxiv.org/abs/2510.26692 ("Kimi Linear: An Expressive, Efficient
  Attention Architecture")
- FLA reference (Triton, the SOTA we benchmark against) --
  https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda
- Songlin Yang's chunk-parallel linear-attention writeup --
  https://sustcsonglin.github.io/blog/2024/deltanet-2/
- CUTLASS CuTe DSL on SM120 -- `/opt/cutlass/include/cute/` and the
  `examples/7*_blackwell_*` directory; only the `Sm120a` variants apply.
- Mirror-naive PyTorch (the correctness oracle) -- `reference.py` in this
  directory matches `fla/ops/kda/naive.py::naive_chunk_kda`.

## Budget

45 minutes wall clock.
