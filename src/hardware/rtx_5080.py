"""RTX 5080 — SM120, consumer Blackwell (GB203).

Same SM ISA as the RTX PRO 6000 Workstation (sm_120a, mma.sync.kind::f8f6f4,
block-scaled MMAs, etc.) but a smaller die: ~84 SMs vs ~188 on GB202, with
GDDR7 on a 256-bit bus.

Peak numbers below are dense (sparsity disabled). The BF16 entry was anchored
empirically by a cuBLAS BF16 4096^3 GEMM on the actual card (median 1.227 ms,
112 TFLOPS achieved) and back-solved at ~80% efficiency, giving ~140 TFLOPS
dense peak. Other dtypes are scaled from BF16 by the standard tensor-core
ratios (FP8 = 2x BF16, FP4 = 4x BF16, etc.) which match PRO 6000's published
ratios. Bandwidth is the spec value: 256-bit GDDR7 @ 30 Gbps.

Recalibrate any specific dtype with its own microbenchmark before publishing a
5080 leaderboard column for problems gated on that dtype.
"""
from src.hardware.rtx_pro_6000 import HardwareTarget


RTX_5080 = HardwareTarget(
    name="NVIDIA GeForce RTX 5080",
    sm="sm_120a",
    vram_gb=16,
    peak_bandwidth_gb_s=960.0,   # 256-bit GDDR7 @ 30 Gbps
    peak_tflops_dense={
        "fp4":   560.0,           # 4x bf16
        "nvfp4": 560.0,
        "mxfp4": 560.0,
        "fp6":   560.0,
        "fp8":   280.0,           # 2x bf16
        "bf16":  140.0,           # empirical: cuBLAS 112 TFLOPS / 0.80 efficiency
        "fp16":  140.0,
        "tf32":   70.0,           # 0.5x bf16
        "fp32":    5.4,           # SIMT, not tensor core
        "int8":  280.0,
        "int4":  560.0,
    },
)
