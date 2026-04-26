# Problem 07 -- W4A16 Weight-only Quantized GEMM

## Task

Implement an AWQ/GPTQ-style weight-only quantized GEMM in `solution.py`:

```
x:      (M, K)            bf16
w_q:    (K // 2, N)       uint8   -- two int4 weights packed per byte
                                    (low nibble = even-K, high nibble = odd-K)
scales: (K // 128, N)     bf16    -- per-group scale, group_size = 128
zeros:  (K // 128, N)     bf16    -- asymmetric zero-point
out:    (M, N)            bf16
```

Per-group dequant along K: `w_bf[k, n] = (w_q[k, n] - zeros[k // 128, n]) * scales[k // 128, n]`. Then `out = x @ w_bf`.

`Model.__init__(M, N, K)` matches `reference.py`. State_dict carries `w_q`, `scales`, `zeros` as buffers; the agent loads them with `strict=True` and must not redeclare the data layout.

## Regime

Memory-bound, scored on achieved GB/s vs RTX PRO 6000 peak GDDR7 (~1.8 TB/s). The M=1 decode case (12 KB activation, 24 MB packed weight per layer) is what every LLM inference engine optimizes; beating Marlin/AWQ on M=1 is the bar. M=32 and M=256 prefill shapes drag the regime toward compute and exercise the unpack-throughput ceiling.

## Forbidden

- `bitsandbytes.functional.dequantize_4bit`
- `bitsandbytes.functional.gemv_4bit`
- `marlin_kernel.gemm`
- `torch.nn.functional.linear` after explicit dequant

You write the unpack and the GEMM; you do not call a vendor library that does both for you.

## Where to dequantize

Three places, three tradeoffs:

1. **Shared memory.** Load packed int4 -> shmem -> per-thread unpack to bf16 staging tile -> MMA reads bf16. Easiest. Costs an extra shmem hop and doubles the staging footprint vs A16/W16.
2. **Registers.** Load packed int4 directly to registers, unpack with `prmt` / shifts, materialize bf16 fragments fed to MMA. Lower shmem pressure; harder to coalesce the packed loads.
3. **On-the-fly during MMA.** Only viable if you can convince `mma.sync` to consume int4 operands, which on SM120 means `kind::s4` -- but that's INT4 weights with INT8 activations, not bf16 activations. So in practice you dequantize first; the question is whether you keep the bf16 fragments in registers across multiple `mma.sync` issues to amortize the unpack.

## Per-group scale broadcast

Scales/zeros are `(K/128, N)`. Each warp's K-tile spans some integer number of groups. Strategies:

- Cooperatively load the relevant scale rows once per K-block, broadcast within a warp via shuffles.
- Or rely on L1 to absorb the redundant loads -- usually fine because (K/128)*N is small (4096/128 * 12288 = 384 KB of scales+zeros per layer).

## References

- **Marlin** -- https://github.com/IST-DASLab/marlin (the reference W4A16 kernel; Ampere/Hopper only, no SM120 path yet)
- **GPTQ-Triton** -- https://github.com/fpgaminer/GPTQ-triton (pure Triton; portable to SM120)
- **AWQ** -- https://github.com/mit-han-lab/llm-awq (CUDA kernels; not built for SM120 in the wheel)
- **bitsandbytes** -- https://github.com/TimDettmers/bitsandbytes (NF4, runs on SM120; used as our SOTA line, different scheme)

## Hardware note

You are on SM120 (Blackwell consumer). `mma.sync.aligned.m16n8k16` with bf16 operands is your friend. No tcgen05, no TMEM, no 2-CTA. Triton is fine; CUDA via `torch.utils.cpp_extension.load_inline` with `-arch=sm_120a` is fine. CUTLASS 4.x has SM120 collectives.

## Budget

45 minutes wall clock.
