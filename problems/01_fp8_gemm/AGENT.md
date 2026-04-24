# Problem 01 — FP8 e4m3 GEMM

## Task

Write `solution.py` implementing an FP8 (e4m3) general matrix multiply:

```
C_bf16[M, N] = A_fp8[M, K] @ B_fp8[N, K].T
```

The `Model` class must define the same `__init__(M, N, K)` signature as `reference.py`, expose a `weight` parameter of shape `(N, K)` in bf16 (which will be cast to fp8 at call time), and accept a single `x` input of shape `(M, K)` in `torch.float8_e4m3fn`.

## Shapes (evaluated per-shape, geomean-scored)

See `shapes.py`. Mix of square aligned, K-misaligned, skinny, and Llama up-proj rectangular.

## Success criterion

Score is **achieved TFLOPS / peak FP8 TFLOPS** on RTX PRO 6000 (SM120), peak ≈ 400 TFLOPS dense.

Reference ceiling for context (H100, from the flashinfer docs): ~550 TFLOPS at the square shape. SM120 peak is lower; target >50% of peak to be competitive.

## Rules

- **Forbidden:** `torch._scaled_mm`, `torch.ops.aten._scaled_mm`. Using them fails correctness post-hoc. This problem is about *writing* the GEMM, not dispatching to cuBLAS.
- You MUST use one of: CUDA C++ via `torch.utils.cpp_extension.load_inline`, Triton `@triton.jit`, inline PTX, or CUTLASS templates.
- Correctness tolerance for fp8: `atol=0.15, rtol=0.15`. Strict enough that a degenerate solution won't pass; loose enough that fp8 quantization noise won't falsely fail.

## References (navigate these yourself)

- **FlashInfer FP8 GEMM** — https://github.com/flashinfer-ai/flashinfer (CUDA + Triton)
- **CUTLASS 4.x Blackwell examples** — `/opt/cutlass/include/../examples/7*_blackwell_*` on disk; look at ones targeting `Sm120` (not `Sm100`).
- **Blackwell PTX ISA (mma.sync kind f8f6f4)** — https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-multiply-and-accumulate-operation
- **Nvidia cuBLASLt FP8 guide** — https://docs.nvidia.com/cuda/cublas/#fp8-gemm-workflow
- **Tri Dao's QuACK (CuTeDSL grouped GEMM)** — https://github.com/Dao-AILab/quack

## Hardware note

- You are on SM120, NOT SM100. Do not emit `tcgen05`, TMEM, or 2-CTA MMA. Use `mma.sync.aligned.m16n16k32.row.col.kind.f8f6f4` or equivalent, with register accumulation.
- CUTLASS: `cutlass::arch::Sm120a` collective builder, `OpClassTensorOp` for fp8, `kind::f8f6f4` on MMA atoms.
- Compile with `-arch=sm_120a -std=c++17 -O3 -I/opt/cutlass/include`.

## Budget

45 minutes wall clock.
