"""Unit tests for src/eval helpers. These run without a GPU."""
import pytest
import torch

from src.eval.correctness import check_correctness, tolerance_for_dtype
from src.eval.roofline import compute_gbps, compute_tflops, peak_fraction
from src.hardware import ENV_VAR, get, select


def test_tolerance_defaults():
    assert tolerance_for_dtype(torch.float32)["atol"] == 1e-4
    assert tolerance_for_dtype(torch.bfloat16)["atol"] == 1e-2
    assert tolerance_for_dtype(torch.int8)["atol"] == 0


def test_tolerance_override_scalar():
    tol = tolerance_for_dtype(torch.float32, override={"torch.float32": 5e-4})
    assert tol == {"atol": 5e-4, "rtol": 5e-4}


def test_correctness_shape_mismatch():
    a = torch.zeros(4, 4)
    b = torch.zeros(4, 5)
    ok, msg = check_correctness(a, b)
    assert not ok
    assert "shape mismatch" in msg


def test_correctness_nan_fails():
    a = torch.zeros(4, 4)
    b = torch.zeros(4, 4)
    b[0, 0] = float("nan")
    ok, _ = check_correctness(a, b)
    assert not ok


def test_correctness_pass_within_tol():
    torch.manual_seed(0)
    a = torch.randn(128, 128, dtype=torch.bfloat16)
    b = a + torch.randn_like(a) * 1e-3  # well within 1e-2 atol
    ok, _ = check_correctness(a, b)
    assert ok


def test_correctness_int_exact():
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)
    b = a.clone()
    assert check_correctness(a, b)[0]
    b[0, 0] = 99
    assert not check_correctness(a, b)[0]


def test_tflops_math():
    # 1 TFLOPS for 1 TFLOP in 1 ms -> 1000 TFLOPS
    assert compute_tflops(1e12, 1.0) == pytest.approx(1000.0)
    # 100 GFLOPs in 0.1ms => 1 TFLOPS
    assert compute_tflops(1e11, 0.1) == pytest.approx(1000.0)


def test_gbps_math():
    # 1 GB in 1 ms -> 1000 GB/s
    assert compute_gbps(1e9, 1.0) == pytest.approx(1000.0)


def test_peak_fraction():
    assert peak_fraction(100, 400) == pytest.approx(0.25)
    assert peak_fraction(0, 400) == 0.0
    assert peak_fraction(100, 0) == 0.0


def test_hardware_lookup():
    hw = get("RTX_PRO_6000")
    assert hw.sm == "sm_120a"
    assert hw.peak_bandwidth_gb_s == pytest.approx(1800.0)
    assert hw.peak_tflops_dense["fp8"] == pytest.approx(400.0)


def test_hardware_lookup_5080():
    hw = get("RTX_5080")
    assert hw.sm == "sm_120a"
    assert hw.vram_gb == 16
    # GDDR7 256-bit @ 30 Gbps
    assert hw.peak_bandwidth_gb_s == pytest.approx(960.0)
    # Lower than PRO 6000 across the board.
    pro = get("RTX_PRO_6000")
    for dtype in ("fp4", "fp8", "bf16"):
        assert hw.peak_tflops_dense[dtype] < pro.peak_tflops_dense[dtype]


def test_select_default_uses_first_entry(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    meta = {"hardware": ["RTX_PRO_6000", "RTX_5080"]}
    assert select(meta).name == get("RTX_PRO_6000").name


def test_select_env_override(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "RTX_5080")
    meta = {"hardware": ["RTX_PRO_6000", "RTX_5080"]}
    assert select(meta).name == get("RTX_5080").name


def test_select_env_override_unknown(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "BOGUS_GPU")
    meta = {"hardware": ["RTX_PRO_6000", "RTX_5080"]}
    with pytest.raises(ValueError, match="not a known hardware target"):
        select(meta)


def test_select_env_override_unsupported_by_problem(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "M4_MAX")
    meta = {"hardware": ["RTX_PRO_6000", "RTX_5080"]}
    with pytest.raises(ValueError, match="not in this problem's supported"):
        select(meta)


def test_select_missing_hardware_list():
    with pytest.raises(ValueError, match="missing a non-empty"):
        select({})
