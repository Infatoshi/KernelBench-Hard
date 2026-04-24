# Problem 06 — Sonic-MoE up-projection: Grouped GEMM + Fused SwiGLU (STUB)

**Status: partially specified. Shape and dtype declared; kernel-writing brief needs expansion.**

## Task

Implement the fused up-projection of an MoE layer: variable-length grouped GEMM with SwiGLU activation, on SM120. No backward. No routing logic — routing metadata (token-to-expert assignment) is provided as input.

## Shape (fixed)

- `T_total = 32,768` tokens
- `H = 4096` (hidden dim, K in GEMM)
- `I = 1536` (expert intermediate, N in GEMM; doubled for SwiGLU gate+up)
- `E = 128` experts
- `K = 8` (top-k routing)
- `dtype = bfloat16` input, `bfloat16` output

## Success criterion

Achieved BF16 TFLOPS as a fraction of SM120 peak (~200 BF16 TFLOPS dense). Reference ceiling: sonic-moe reports 1.87-4.04× speedup over ScatterMoE/MoMoE on H100 — approaching sonic-moe's per-shape throughput (scaled to SM120 peak) is the goal.

## References

- **Sonic-MoE (Tri Dao)** — https://github.com/Dao-AILab/sonic-moe (Apache 2.0)
- **QuACK** — https://github.com/Dao-AILab/quack (the CuTeDSL grouped GEMM kernels sonic-moe calls)
- **Paper** — https://arxiv.org/abs/2512.14080
- **ScatterMoE** — https://github.com/shawntan/scattermoe
- **MegaBlocks** — https://github.com/databricks/megablocks (for comparison)

## Hardware note

- **Sonic-MoE SM120 support is in-progress as of 2026-04 and may not install or run on RTX PRO 6000.** You can still use it as a *reference* (read the source, understand the algorithm) even if the pip-installed version doesn't execute on your GPU. Your baseline for benchmark purposes is PyTorch eager; your ceiling is the sonic-moe H100 number (reference only, not measured live).
- Consider: token permutation to contiguous layout per expert, persistent kernel tiles with dynamic block assignment, TMA for weight loads, register accumulation.
- No TMEM, no tcgen05 — register-based accumulation as on Hopper.

## Scope

45 minutes. Forward only. Given routing metadata — do not implement top-k/softmax routing.
