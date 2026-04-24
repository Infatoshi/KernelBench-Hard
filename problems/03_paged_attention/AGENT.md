# Problem 03 — Paged Attention Decode (STUB)

**Status: not yet specified.**

## Task (draft)

Single-query decode attention with KV cache stored in fixed-size pages indexed via a block table. Fuse the page-gather into the attention tile load.

## References

- **vLLM PagedAttention** — https://github.com/vllm-project/vllm (csrc/attention/paged_attention_v2.cu)
- **FlashInfer paged attention** — https://github.com/flashinfer-ai/flashinfer
- **Paper** — https://arxiv.org/abs/2309.06180

## Scope

To be defined. Budget 45 minutes.
