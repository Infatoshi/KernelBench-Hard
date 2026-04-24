# KernelBench-Hard

Surgical GPU kernel benchmark. 7 carefully-chosen problems, frontier coding agents, roofline-based metric (achieved TFLOPS or GB/s vs hardware peak). Link-don't-spoil problem briefs: agents receive repo/paper URLs, not source snippets.

Sibling project to [KernelBench-v3](https://github.com/Infatoshi/KernelBench-v3) (volume-oriented; local open-weight models). Hard is for frontier-model harnesses on a small, high-signal deck.

## Problem deck

| # | Problem | Hardware | What it tests |
|---|---------|----------|---------------|
| 01 | FP8 e4m3 GEMM (off-alignment shapes) | RTX PRO 6000 (SM120) | Tensor-core GEMM, epilogue fusion |
| 02 | KDA (Kimi Delta Attention) via CUTLASS CuTe | RTX PRO 6000 | Novel attention from paper, CUTLASS 4.x |
| 03 | Paged Attention decode | RTX PRO 6000 | Indirect indexing, pointer chasing |
| 04 | Kahan-corrected Softmax | RTX PRO 6000 | Floating-point error awareness |
| 05 | TopK with bitonic sort | RTX PRO 6000 | Small-output, comparator networks |
| 06 | Sonic-MoE up-projection: grouped GEMM + fused SwiGLU | RTX PRO 6000 | Megakernel, load balancing, variable-length |
| 07 | W4A16 weight-only GEMM (AWQ/GPTQ-style) | RTX PRO 6000 | Bit unpack, quantization, memory-bound decode |
| 08 | Lightning Attention step (decode) | M4 Max | Metal kernel, novel attention on Apple |

## Hardware

- **RTX PRO 6000 Blackwell Workstation** (SM120, 96GB GDDR7, 1.8 TB/s, ~200 BF16 / ~400 FP8 / ~800 FP4 TFLOPS dense)
- **M4 Max** (Metal 3, unified memory) — for problem 08 only

Required: CUDA 13.x (symlink `/usr/local/cuda-13`), torch 2.11+cu130, Python 3.11+.

## Active model matrix

One harness per model, each pinned to the highest-fidelity native endpoint.

| Model | Harness | Route |
|-------|---------|-------|
| Claude Opus 4.7 | `claude` | Anthropic direct |
| GPT-5.5 xhigh | `codex` (`-c model_reasoning_effort="xhigh"`) | OpenAI direct |
| Kimi K2.6 | `kimi` | Moonshot direct (api.moonshot.cn) |
| GLM-5.1 | `claude` + [ccr-rust](https://github.com/RESMP-DEV/ccr-rust) | Z.AI direct (api.z.ai) |
| Minimax M2.7 | `claude` + ccr-rust | Minimax direct (api.minimaxi.com) |
| DeepSeek V4 Pro Max | `claude` + ccr-rust | DeepSeek direct (api.deepseek.com) |
| DeepSeek V4 Flash | `claude` + ccr-rust | DeepSeek direct (api.deepseek.com) |

7 models × 7 problems = **49 agent-runs per sweep** (~37 GPU-hours at 45min/run).

## Deferred / upcoming

- **Gemini 3.1 Pro** (via `gemini-cli`) — low community interest; adding once bandwidth clears.
- Other models: _to be populated — Grok, Qwen, Sonnet 4.6, etc._

## Quick start

```bash
# Install (uv only)
uv sync

# Apply torch inductor CSE typing hotfix (required for torch 2.11.0 compile baseline)
./scripts/patch_torch.sh

# Run a single problem through a single harness
./scripts/run_hard.sh claude claude-opus-4-7 problems/01_fp8_gemm

# Full sweep (active matrix × all 7 CUDA problems)
./scripts/sweep.sh

# Plot roofline for a completed run
uv run python scripts/roofline_plot.py outputs/runs/<run_dir>
```

## Design principles

See [SPEC.md](./SPEC.md). The short version:

- **Roofline, not speedup ratio.** Score is achieved throughput as a fraction of hardware peak. PyTorch eager and the SOTA reference (sonic-moe, flashinfer, marlin, cudnn) are reference lines on the plot, not the grading denominator.
- **Per-dtype tolerance.** fp32 atol/rtol 1e-4; fp16/bf16 1e-2. Closes the "identity kernel passes" class of cheats.
- **Multi-shape eval.** Each problem has 3-5 canonical shapes. Score is geomean over shapes. Rewards general kernels, penalizes hyperspecialization.
- **Link, don't spoil.** `AGENT.md` per problem gives repo URLs and paper links. Agents navigate, grep, and read source themselves. Navigation skill is part of the evaluation.
- **Algorithmic FLOPS.** For sparse/conditional kernels (MoE, paged attn), FLOPS is the dense-equivalent; agents can't skip work and call it optimization.
- **No custom tools, no MCP.** Each harness uses its native shell/read/write. We measure the harness as shipped.

## License

TBD (probably Apache 2.0 to match kernel-space conventions).
