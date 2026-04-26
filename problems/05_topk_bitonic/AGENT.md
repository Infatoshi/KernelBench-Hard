# Problem 05 — TopK via Bitonic Sort

## Task

Write `solution.py` implementing top-k over the last dim of a 2D fp32 tensor:

```
values[batch, k], indices[batch, k] = topk(x[batch, n], k, dim=-1, largest=True, sorted=True)
```

Same `Model(batch, n, k)` signature as `reference.py`. `forward(x)` returns a `(values, indices)` tuple, both sorted descending. Indices are int64.

## Why this is hard

Top-k is the awkward middle child of GPU primitives. The output is tiny (`batch * k`), so there is no reduction to amortize compute across — the kernel is **bandwidth-bound on the input read** and the comparator network is essentially free *if you implement it well*. But the comparator network is also where you lose: a naive per-thread heap serializes; `torch.sort` does O(n log n) work when you only need the top k; `torch.topk` itself dispatches to a CUB radix-select that is already tuned.

There is no clean library shortcut here. Faiss `BlockSelect` is designed for `k >= 32` and very large `n` (vector search regime). CUB `DeviceSegmentedRadixSort` sorts the full row. cuCollections has no top-k primitive. You must build the selector yourself.

The intended solution is a warp-or-block-level bitonic sort network in shared memory (or registers, for small `k`), fed by a coalesced strided read of the input. For the `(1, 131072, 64)` decoder shape, the score is purely how close you get to DRAM peak (~1.8 TB/s on RTX PRO 6000).

## Forbidden

`torch.topk`, `torch.kthvalue`, `torch.sort`, `torch.argsort` (and their `Tensor.` and `torch.ops.aten.` variants). Solution must be CUDA C++ via `torch.utils.cpp_extension.load_inline`, Triton, inline PTX, or CUTLASS.

## Score

Memory regime: `achieved_gbps / peak_dram_gbps`, geomean over shapes.

## References (navigate yourself)

- **Faiss BlockSelect** — https://github.com/facebookresearch/faiss/blob/main/faiss/gpu/utils/Select.cuh and `BlockSelectKernel.cuh` (warp-level merge networks, k up to 1024).
- **NVIDIA bitonic sort sample** — https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/sortingNetworks (classic reference, full row sort but the comparator pattern is what you want).
- **PyTorch's topk kernel** — https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/TensorTopK.cu (the bar to beat).
- **CUB warp / block primitives** — https://nvlabs.github.io/cub/ (`WarpMergeSort`, `BlockRadixSort` if you go the radix-select route).
- **Alabi et al. "Fast K-Selection on the GPU"** — radix-select algorithm, the algorithm `torch.topk` actually uses for moderate k.

## Hardware note

RTX PRO 6000 SM120, 96 GB, ~1.8 TB/s DRAM. No tcgen05/TMEM. Compile with `-arch=sm_120a -O3`.

## Budget

30 minutes wall clock.
