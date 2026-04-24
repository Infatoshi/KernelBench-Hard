"""Per-dtype correctness tolerance.

Stanford's KernelBench uses 1e-4 atol/rtol for fp32 and 1e-2 for fp16/bf16.
We extend to fp8 (0.1, permissive) and int (exact).
"""
from __future__ import annotations

import torch

DEFAULT_TOLERANCE = {
    torch.float32: {"atol": 1e-4, "rtol": 1e-4},
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
    torch.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
    torch.float8_e4m3fn: {"atol": 1e-1, "rtol": 1e-1},
    torch.float8_e5m2: {"atol": 1e-1, "rtol": 1e-1},
    torch.int8: {"atol": 0, "rtol": 0},
    torch.int32: {"atol": 0, "rtol": 0},
    torch.int64: {"atol": 0, "rtol": 0},
}


def tolerance_for_dtype(dtype: torch.dtype, override: dict | None = None) -> dict:
    """Lookup atol/rtol for a given dtype, with optional per-problem override."""
    if override is not None and str(dtype) in override:
        v = override[str(dtype)]
        return {"atol": v, "rtol": v} if isinstance(v, (int, float)) else v
    if dtype not in DEFAULT_TOLERANCE:
        # Fall back to fp32 precision for unknown types.
        return DEFAULT_TOLERANCE[torch.float32]
    return DEFAULT_TOLERANCE[dtype]


def check_correctness(
    reference_out: torch.Tensor,
    solution_out: torch.Tensor,
    dtype: torch.dtype | None = None,
    override: dict | None = None,
) -> tuple[bool, str]:
    """Return (passed, message). Integer comparisons are bitwise; floats use atol/rtol."""
    if reference_out.shape != solution_out.shape:
        return False, f"shape mismatch: ref={tuple(reference_out.shape)} sol={tuple(solution_out.shape)}"

    if torch.isnan(solution_out).any():
        return False, "solution contains NaN"
    if torch.isinf(solution_out).any():
        return False, "solution contains Inf"

    dtype = dtype or reference_out.dtype
    tol = tolerance_for_dtype(dtype, override)

    # Cast both to fp32 for the comparison to avoid dtype-specific allclose quirks
    ref_f = reference_out.float()
    sol_f = solution_out.float()

    if tol["atol"] == 0 and tol["rtol"] == 0:
        if torch.equal(ref_f, sol_f):
            return True, "ok (exact)"
        n_diff = (ref_f != sol_f).sum().item()
        return False, f"exact match required; {n_diff} elements differ"

    if torch.allclose(ref_f, sol_f, atol=tol["atol"], rtol=tol["rtol"]):
        return True, f"ok (atol={tol['atol']}, rtol={tol['rtol']})"

    max_diff = (ref_f - sol_f).abs().max().item()
    return False, f"tolerance exceeded: max_abs_diff={max_diff:.6g} (atol={tol['atol']}, rtol={tol['rtol']})"
