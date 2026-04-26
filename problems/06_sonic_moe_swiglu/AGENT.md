# Problem 06 — Sonic-MoE up-projection: Grouped GEMM + Fused SwiGLU

## Task

Write `solution.py` implementing the up-projection of a top-K MoE FFN: a variable-length **grouped GEMM** with a **fused SwiGLU** activation. Per expert e with token slice `x_e = hidden_states[offsets[e]:offsets[e+1]]`:

```
gated_up_e = silu(x_e @ W_gate[e]) * (x_e @ W_up[e])
```

The `Model` class must expose the same `__init__(T_total, H, I, E, K)` signature as `reference.py` and own two `(E, H, I)` bf16 weight tensors named `W_gate` and `W_up` so `state_dict` matches strictly. `forward(hidden_states, expert_offsets)` takes a permuted `(T_total*K, H)` bf16 tensor plus an `(E+1,)` int32 offsets prefix-sum and returns `(T_total*K, I)` bf16.

## Shapes (geomean-scored)

See `shapes.py`. Headline shape is sonic-moe's: `T_total=32768, H=4096, I=1536, E=128, K=8`. Smaller iteration shape and a wider-intermediate aspect-ratio shape included.

## Success criterion

Achieved BF16 TFLOPS as a fraction of SM120 dense peak (~200 BF16 TFLOPS). Sonic-MoE reports 1.87-4.04x over ScatterMoE / MoMoE on H100 (paper's headline number ~480 BF16 TFLOPS at this shape) — that is your ceiling for context, not a live measurement.

## Rules

- **Forbidden:** `torch.matmul`, `torch.bmm`, `torch.nn.functional.linear` (any of these = vendor cuBLAS dispatch, defeats the point), and `from sonic_moe ...` / `import sonic_moe` (calling the SOTA is cheating; SOTA is graded separately).
- You MUST write the grouped GEMM + fused SwiGLU yourself: CUTLASS 4.x grouped-GEMM collective, Triton `@triton.jit`, raw CUDA via `torch.utils.cpp_extension.load_inline`, or inline PTX. Fuse SwiGLU into the epilogue — do not write back two `(T_perm, I)` tensors and elementwise-merge.
- Tolerance: bf16 `atol=rtol=0.02`. Tight enough to catch precision downcast; loose enough for legitimate accumulator variance.

## Hardware note

- RTX PRO 6000 Blackwell Workstation, **SM120**, not SM100. No `tcgen05`, no TMEM, no 2-CTA MMA. Use `mma.sync.aligned.m16n16k16` BF16 tensor-core ops with **register accumulation**, or CUTLASS `cutlass::arch::Sm120a` collective builder.
- Sonic-MoE installs (PyPI `sonic-moe>=0.1.2`, requires Python>=3.12) but its CuTeDSL grouped-GEMM kernels target Sm90/Sm100 in the public release; SM120 support is in-progress upstream. `sota.py.is_available()` may return False at runtime — the benchmark will then score against PyTorch eager + the documented H100 paper ceiling.
- Consider: persistent CTAs that pull (expert_id, tile) work items off a queue, TMA / `cp.async.bulk` for weight loads, single epilogue write that fuses `silu(gate)*up` so you visit the I-axis once.

## References (navigate yourself)

- **Sonic-MoE (Tri Dao)** — https://github.com/Dao-AILab/sonic-moe
- **QuACK (CuTeDSL grouped GEMM)** — https://github.com/Dao-AILab/quack
- **Sonic-MoE paper** — https://arxiv.org/abs/2512.14080
- **ScatterMoE** — https://github.com/shawntan/scattermoe
- **MegaBlocks** — https://github.com/databricks/megablocks
- **CUTLASS 4.x grouped GEMM examples** — `/opt/cutlass/include/../examples/` (search for `grouped` and `Sm120`).
- **Blackwell PTX ISA (mma.sync)** — https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-multiply-and-accumulate-operation

## Scope

Forward only. Routing metadata (`expert_offsets`) is provided — do not implement top-k / softmax routing or token permutation. 45 minutes wall clock.
