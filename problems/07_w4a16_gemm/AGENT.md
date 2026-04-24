# Problem 07 — W4A16 Weight-only Quantized GEMM (STUB)

**Status: partially specified.**

## Task

AWQ/GPTQ-style weight-only quantized GEMM: bf16 activations × int4 packed weights → bf16 output, with per-group dequantization.

```
x:      (M, K)                bf16
w_q:    (K // 2, N)           uint8  (two int4 weights packed per byte)
scales: (K // group, N)       bf16
zeros:  (K // group, N)       bf16   (asymmetric quant)
out:    (M, N)                bf16
```

`group_size = 128`. Dequant: `w_bf = (w_q - zeros) * scales`.

## Shapes

- Decode: `M=1, K=4096, N=12288` — memory-bound, scored on GB/s
- Small prefill: `M=32, K=4096, N=12288` — mixed regime
- Larger prefill: `M=256, K=4096, N=12288` — approaching compute-bound

## Regime

Memory-bound at M=1 (classic LLM decode). Score is achieved GB/s vs peak 1.8 TB/s GDDR7.

## References

- **Marlin** — https://github.com/IST-DASLab/marlin (the SOTA W4A16 kernel)
- **GPTQ-Triton** — https://github.com/fpgaminer/GPTQ-triton
- **AWQ** — https://github.com/mit-han-lab/llm-awq
- **bitsandbytes NF4** — https://github.com/TimDettmers/bitsandbytes

## What to investigate

- Where do you dequantize? Shared memory, registers, or on-the-fly during MMA?
- Can you use mma.sync's int4 kind directly, or must you dequant to bf16 first?
- Coalesced loads of packed int4 weights require careful striding.
- Per-group scale broadcast patterns.

## Scope

45 minutes.
