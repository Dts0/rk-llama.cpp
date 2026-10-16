# RK35xx quantized matmul pipelines

This note records the measured pipeline behavior for the RK3576 and RK3588
quantized matmul paths. Measurements use Qwen35-2B-SMI-v6 Q4_K_M, context
32000, F16 KV, flash attention on, parallel 1, 4 CPU threads, and the quick7
suite (7 scenarios, 12 independent cases, 25 requests).

Test date: 2026-10-16
Source commit: 5290990ef32a9b56a455a35b6e983cc11b8f5ccf

## RK3576

Runtime: librknnrt 2.3.2, RKNPU driver 0.9.8.

The RK3576 runtime supports FP16 activations with INT8 and INT4 weights.

| Pipeline | Mapping | Quick7 score | Wall | Prefill avg | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| `W8A16_STANDARD` | FP16 x INT8, full K | 93/100 | 787.2 s | 41.81 tok/s | 5,467,020 KiB |
| `W4A16_Q4K_FFN_INT4_HYBRID` | FFN `W4A16_HADAMARD`; other tensors `W8A16_STANDARD` | 100/100 | 817.4 s | 40.49 tok/s | 5,123,004 KiB |
| `W4A16_Q4K_STAGED_INT8` | Keep native Q4_K; stage INT8 per execution | 93/100 | 925.9 s | 34.38 tok/s | 4,590,676 KiB |
| `W4A16_Q4K_HYBRID` | Exact attention/SSM; W8A16 FFN | 93/100 | 2152.2 s | 13.32 tok/s | about 5,493,536 KiB |
| `W4A16_STANDARD` | Exact Q4_K per-32 asymmetric INT4 | 93/100 | 5226.7 s | 5.24 tok/s | 5,124,780 KiB |

Notes:

- The RK3576 default for Q4_K is `W4A16_Q4K_FFN_INT4_HYBRID`.
- This hybrid keeps INT8 for attention/SSM and uses INT4 plus Hadamard for
  Q4_K FFN tensors. It matches or improves the quick7 score and saves about
  344 MiB RSS against W8A16, with about 3% lower prefill throughput.
- `W4A16_Q4K_STAGED_INT8` saves more memory, about 856 MiB RSS, but is about
  17.6% slower on the same quick7.
- `W4A16_Q4K_K256` is numerically correct on the tested tensors but is slower
  and uses more memory than the exact per-32 path. It is not recommended.
- FP16 compute was rejected. On RK3576, FP16 matmul was slower than INT8/INT4,
  and tiled FP16 conversion could not beat a full-K INT8 matmul.

## RK3588

Runtime: librknnrt 2.3.2 (`429f97ae6b@2025-04-09T09:09:27`), RKNPU driver 0.9.8.

The RK3588 runtime rejects mixed activation/weight types:

| Matmul type | Result |
| --- | --- |
| `RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32` | supported |
| `RKNN_FLOAT16_MM_INT8_TO_FLOAT32` | unsupported, ret=-5 |
| `RKNN_FLOAT16_MM_INT4_TO_FLOAT32` | unsupported, ret=-5 |

Because of this, W8A16 and W4A16 cannot run on this RK3588 runtime. The
supported comparison uses W8A8 and W4A4.

| Pipeline | Mapping | Result |
| --- | --- | --- |
| `W8A8_STANDARD` | INT8 x INT8 | 93/100 quick7, 324.8 s, 109.93 tok/s prefill, 5,035,284 KiB peak RSS |
| `W4A4_HADAMARD` | INT4 x INT4 | Weather passed 100/100, but was about 6.3x slower than W8A8 and was not competitive |
| `W4A4_Q4K_FFN_HYBRID` | FFN `W4A4_HADAMARD`; other tensors `W8A8_STANDARD` | 3-case subset 57/100; weather was about 3.3x slower than W8A8 |

Notes:

- The RK3588 default for Q4_K remains `W8A8_STANDARD`.
- `W4A4_Q4K_FFN_HYBRID` is kept as an opt-in experiment, but it is both
  slower and lower quality than W8A8 on this runtime.
- No direct RK3588 W8A16 comparison is possible with this runtime.

## Selecting a pipeline

Set `RKNPU_HYBRID` to one of the pipeline names above. Omit it to use the
platform default mapping.

Examples:

```sh
# RK3576 default Q4_K hybrid
RKNPU_HYBRID=W4A16_Q4K_FFN_INT4_HYBRID

# RK3576 memory-first alternative
RKNPU_HYBRID=W4A16_Q4K_STAGED_INT8

# RK3588 supported baseline
RKNPU_HYBRID=W8A8_STANDARD

# RK3588 experimental INT4 FFN hybrid
RKNPU_HYBRID=W4A4_Q4K_FFN_HYBRID
```

## Evidence

The detailed device logs and result JSON files are kept outside this Git
repository under the workspace benchmark directory:

- `benchmark/rk3576-ffn-int4-quick7-20261016/`
- `benchmark/rk3576-staged-int8-fused-quick7-20261016/`
- `benchmark/rk3588-comparison-20261016/`
- `benchmark/rk3576-w8a16-quick7-20261016/`
- `benchmark/rk3576-fp16-vs-int-microbench-20261016/`
