# Problem 03 — Paged Attention Decode

## Task

Write `solution.py` implementing single-query decode attention over a paged KV cache. The reference module signature is `Model(batch, num_heads, num_kv_heads, head_dim, seq_len, page_size)` and forward takes `(query, kv_cache, block_table, seq_lens)`. The KV cache is laid out as `(num_blocks, page_size, num_kv_heads, 2*head_dim)` with `[K | V]` packed on the last dim. `block_table[b]` lists the page indices owned by batch element `b`; `seq_lens[b]` is its true length. GQA is in scope (`num_heads % num_kv_heads == 0`).

The whole point is fusing the page gather into the attention tile load -- a separate gather + matmul will not hit the bandwidth target.

## Shapes

See `shapes.py`. Five shapes covering small batch / long context, large batch / medium context, wide GQA (Llama-3 70B), non-power-of-2 `seq_len` (forces predicated tail handling), and small `head_dim=64`.

## Success criterion

Score is **achieved GB/s / peak DRAM bandwidth** on RTX PRO 6000 Blackwell (peak ~1.8 TB/s GDDR7). Decode is memory-bound; the KV cache must be streamed exactly once. A well-tuned kernel reaches 70-85% of peak HBM bandwidth on H100; SM120 GDDR7 should be similar.

## Rules

- **Forbidden:** `vllm.attention`, `flashinfer.batch_decode_with_paged_kv_cache`, `flashinfer.decode`, and `torch.nn.functional.scaled_dot_product_attention`. Using any of these fails correctness post-hoc. This problem is about *writing* paged attention, not dispatching to a vendor library.
- You MUST use one of: CUDA C++ via `torch.utils.cpp_extension.load_inline`, Triton `@triton.jit`, inline PTX, or CUTLASS / CuTeDSL templates.
- Tolerance: `atol=0.02, rtol=0.02` for bf16/fp16 outputs.

## References (navigate these yourself)

- **vLLM PagedAttention v2** — https://github.com/vllm-project/vllm/blob/main/csrc/attention/paged_attention_v2.cu  (the canonical CUDA implementation; study the partition / reduce split)
- **vLLM PagedAttention v1** — https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cuh  (single-stage; simpler to read first)
- **FlashInfer paged decode** — https://github.com/flashinfer-ai/flashinfer/tree/main/include/flashinfer/attention  (`decode.cuh`, `BatchDecodeWithPagedKVCacheWrapper`)
- **Triton paged attention example** — https://github.com/triton-lang/triton/blob/main/python/tutorials/  (and vLLM's Triton backend at `vllm/attention/ops/triton_unified_attention.py`)
- **PagedAttention paper (Efficient Memory Management for LLM Serving)** — https://arxiv.org/abs/2309.06180

## Hardware note

You are on RTX PRO 6000 Blackwell (SM120). No `tcgen05`, no TMEM, no 2-CTA MMA. Use `mma.sync` with register accumulation, or Triton (recommended for first-pass: it handles the predicated tail correctly out of the box). Compile CUDA with `-arch=sm_120a -std=c++17 -O3`. The KV cache is bf16; do attention math in fp32 accumulator and write back bf16.

## Budget

45 minutes wall clock.
