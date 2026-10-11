# rk-llama.cpp

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

The [llama.cpp](https://github.com/ggml-org/llama.cpp) fork with the Rockchip NPU integration as a GGML backend.

## Description

This project integrates Rockchip's Neural Processing Unit as a computational backend, enabling energy-efficient, high-performance inference for Large Language Models.

The primary motivation is to leverage the NPU as a powerful accelerator. This approach aims to achieve speedups over CPU-only inference while maintaining comparable accuracy with reduced power consumption.

The backend has configurations for **RK3588** and **RK3576**, including FP16 activation pipelines with INT8 or INT4 weights on RK3576. This integration branch still requires numerical and performance validation on Rockchip hardware.

## Quick start

1. Clone the repository

```sh
git clone https://github.com/Dts0/rk-llama.cpp
cd rk-llama.cpp
```

2. Build the project

```sh
cmake -S . -B build -DGGML_RKNPU2=ON
cmake --build build --parallel 4
```

3. Run inference

```sh
# Set new max limit for open files
ulimit -n 65536

# Run from the repository root
./build/bin/llama completion -m ./gemma-3-1b-it-Q8_0.gguf
```

## Benchmarks

The following tables retain measurements from the [upstream RK3588 backend](https://github.com/invisiofficial/rk-llama.cpp/tree/rknpu2), comparing NPU inference against the CPU (NEON) backend.

> **Note:** These are upstream reference measurements, not results from `rknpu2_merge_2610`. The pipeline precision tables in **Chipsets** are also upstream RK3588 reference data. They do not establish the accuracy, speed, or power consumption of this branch or RK3576.

### Dense Models

| Model | Type | Backend | Perplexity | PP (tok/s) | TG (tok/s) | Power (W) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Granite4.0 350M** | F16 | CPU | 🟢 20.73±0.74 | 🔴 154.1±0.1 | 🟢 25.8±0.1 | 🔴 6.4±0.4 |
| | | NPU | 🟢 20.74±0.74 | 🟢 432.3±0.6 | 🟡 20.2±0.4 | 🟢 3.2±0.2 |
| | Q8_0 | CPU | 🟢 20.71±0.74 | 🔴 163.4±0.2 | 🟢 40.6±0.1 | 🔴 6.4±0.4 |
| | | NPU | 🟡 22.68±0.82 | 🟢 311.8±2.1 | 🔴 25.4±0.4 | 🟢 3.6±0.2 |
| | Q6_K | CPU | 🟢 21.55±0.77 | 🔴 78.7±0.1 | 🟢 42.5±0.1 | 🔴 6.8±0.4 |
| | | NPU | 🔴 29.52±1.07 | 🟢 142.4±1.3 | 🔴 25.8±1.2 | 🟢 3.8±0.2 |
| | Q4_0 | CPU | 🟢 24.46±0.88 | 🟢 340.4±0.9 | 🟢 55.2±0.1 | 🔴 6.2±0.4 |
| | | NPU | 🔴 74.09±2.87 | 🔴 163.6±0.2 | 🔴 26.7±0.5 | 🟢 4.0±0.2 |
| **Gemma3 1B** | F16 | CPU | 🟢 26.20±1.08 | 🔴 68.5±0.1 | 🟢 11.1±0.1 | 🔴 6.8±0.4 |
| | | NPU | 🟢 26.18±1.07 | 🟢 249.6±0.2 | 🟢 10.8±0.2 | 🟢 2.8±0.2 |
| | Q8_0 | CPU | 🟢 26.08±1.07 | 🔴 73.3±0.1 | 🟢 19.5±0.1 | 🔴 7.4±0.4 |
| | | NPU | 🟡 29.15±1.22 | 🟢 378.6±0.4 | 🟡 16.5±0.3 | 🟢 3.0±0.2 |
| | Q6_K | CPU | 🟢 25.94±1.06 | 🔴 51.9±0.1 | 🟢 18.7±0.1 | 🔴 7.2±0.4 |
| | | NPU | 🟡 38.46±1.68 | 🟢 209.4±3.2 | 🟡 16.6±0.2 | 🟢 3.0±0.2 |
| | Q4_0 | CPU | 🟢 30.77±1.31 | 🟢 164.7±0.2 | 🟢 28.3±0.1 | 🔴 7.0±0.4 |
| | | NPU | 🔴 55.53±2.30 | 🔴 51.4±0.1 | 🔴 16.7±0.3 | 🟢 3.0±0.2 |
| **Ministral3 3B** | Q8_0 | CPU | 🟢 7.67±0.21 | 🔴 17.5±0.1 | 🟡 6.1±0.1 | 🔴 7.2±0.6 |
| | | NPU | 🟢 7.84±0.22 | 🟢 120.9±0.6 | 🟢 6.9±0.1 | 🟢 3.2±0.2 |
| | Q4_0 | CPU | 🟢 7.93±0.22 | 🟢 15.1±0.1 | 🟢 8.3±0.2 | 🔴 7.8±0.6 |
| | | NPU | 🔴 32.37±1.16 | 🟡 13.8±0.1 | 🟡 7.2±0.1 | 🟢 3.2±0.2 |
| **Qwen3.5 9B** | Q8_0 | CPU | 🟢 6.82±0.19 | 🔴 7.2±0.1 | 🟡 2.5±0.1 | 🔴 7.4±0.6 |
| | | NPU | 🟢 7.14±0.20 | 🟢 44.6±0.1 | 🟢 3.2±0.1 | 🟢 3.2±0.2 |
| | Q4_0 | CPU | 🟢 7.10±0.20 | 🟡 6.5±0.1 | 🟡 3.6±0.1 | 🔴 8.0±0.6 |
| | | NPU | 🟡 14.67±0.48 | 🟢 7.3±0.1 | 🟢 4.4±0.1 | 🟢 4.0±0.2 |

### MoE Models

| Model | Type | Backend | Perplexity | PP (tok/s) | TG (tok/s) | Power (W) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LFM2 8B A1B** | F16 | CPU | 🟢 15.79±0.58 | 🟡 31.1±2.9 | 🟢 6.8±0.2 | 🟡 7.0±0.6 |
| | | NPU | 🟢 15.82±0.58 | 🟢 38.3±3.2 | 🟢 6.3±0.4 | 🟢 5.8±0.4 |
| | Q8_0 | CPU | 🟢 15.92±0.59 | 🟡 31.7±0.1 | 🟢 12.9±0.1 | 🟡 7.4±0.6 |
| | | NPU | 🟡 16.76±0.62 | 🟢 40.7±0.7 | 🟢 12.5±0.3 | 🟢 5.8±0.4 |
| | Q6_K | CPU | 🟢 15.91±0.58 | 🟡 16.4±0.1 | 🟢 12.9±0.1 | 🟡 7.4±0.6 |
| | | NPU | 🟡 21.16±0.83 | 🟢 21.4±0.1 | 🟢 13.7±0.1 | 🟢 5.8±0.4 |
| | Q4_0 | CPU | 🟢 18.24±0.53 | 🟢 62.2±0.1 | 🟢 22.7±0.1 | 🟡 7.4±0.6 |
| | | NPU | 🟡 26.09±1.06 | 🟡 47.5±0.1 | 🟡 19.0±0.1 | 🟢 5.8±0.4 |
| **Gemma4 26B A4B** | Q8_0 | CPU | 🟢 169.08±10.50 | 🟡 9.8±0.4 | 🟢 4.1±0.2 | 🔴 7.2±0.6 |
| | | NPU | 🟢 215.18±13.84 | 🟢 13.4±0.4 | 🟡 2.9±0.1 | 🟢 4.2±0.4 |
| | Q4_0 | CPU | 🟢 201.44±12.77 | 🟢 11.3±0.4 | 🟢 6.1±0.1 | 🔴 7.4±0.6 |
| | | NPU | 🟡 635.77±44.36 | 🟢 12.9±0.5 | 🟡 5.6±0.1 | 🟢 4.4±0.4 |

**Legend**: 🟢 Excellent | 🟡 Acceptable | 🔴 Poor

### Methodology

*   **Perplexity:** Evaluates accuracy. Measured over 32 chunks of 512 tokens using `wiki.test.raw`. Lower is better.

    ```sh
    taskset -c 4-7 ./build/bin/llama-perplexity -m ./model.gguf -f ./wiki.test.raw -t 4 -b 512 --chunks 32
    ```

*   **PP / TG (tok/s):** Prompt Processing and Text Generation speeds. Evaluated using standard pp512 and tg128. Higher is better.

    ```sh
    taskset -c 4-7 ./build/bin/llama-bench -m ./model.gguf -t 4
    ```

*   **Power (W):** Represents relative active power consumption. Calculated as `Power(TextGeneration) - Power(Idle)`. Lower is better.

## Hybrid Quantization

Hybrid quantization is a technique designed to strike an optimal balance between memory consumption, inference speed, and generation quality.

The backend assigns hardware pipelines cyclically to supported weight tensors. Supported GGUF weight types are `F16`, `Q8_0`, `Q6_K`, `Q4_0`, and `Q4_K`. You can find the default pipeline sequences for each chip in the **Chipsets** section below.

### Custom Patterns

You can override the default tensor-to-pipeline mapping to experiment with quantization strategies.

By setting the `RKNPU_HYBRID` environment variable, you define a custom sequence of hardware pipelines for supported weight types. The backend cycles through the sequence as it first encounters each distinct weight tensor; several tensors can belong to one model layer. Pipeline names must be available on the selected chip.

```sh
# Alternate tensors between FP16, INT8 and INT4 NPU pipelines on RK3588
RKNPU_HYBRID="W16A16_STANDARD,W8A8_STANDARD,W4A4_HADAMARD" ./build/bin/llama completion -m ./model.gguf

# A pipeline name absent from the chip configuration selects CPU fallback
RKNPU_HYBRID="CPU_STANDARD,W8A8_STANDARD" ./build/bin/llama completion -m ./model.gguf

# Enable optional low-rank scale reconstruction on RK3588
RKNPU_HYBRID="W8A8_LOWRANK,W4A4_HADAMARD_LOWRANK" ./build/bin/llama completion -m ./model.gguf

# Use FP16 activations with INT4 weights on RK3576
RKNPU_DEVICE="RK3576" RKNPU_HYBRID="W4A16_STANDARD" ./build/bin/llama completion -m ./model.gguf
```

### Weights Requantizations

The backend does not execute GGUF formats (like `Q4_0` or `Q8_0`) natively. Instead, input weights are first dequantized to `FP32` on the CPU, calibrated, and then requantized into the NPU's native hardware formats.

Standard and Hadamard integer weight pipelines use the upstream per-channel absolute-maximum calibration: each output row in each K segment has its own scale, `amax / 127` for INT8 or `amax / 7` for INT4. This replaces the earlier block scale and entropy calibration, including for `W4A16_STANDARD`. Optional `*_LOWRANK` pipelines use reconstructed per-element scales from the segment's `u` and `v` factors instead. LOWRANK pipelines are available on RK3588 but are not selected by the defaults in this branch.

Unsupported tensor shapes and operations fall back to the CPU. `Q4_K` matrix operations with at most four activation rows also use the CPU, while eligible larger batches can use the NPU. The backend retains the original host weights for CPU fallback.

Because of this double conversion process, you must carefully pair your input GGUF model with your target NPU pipelines:

1. **Ideal Case (GGUF Precision > NPU Precision)**<br>
   Using an `F16` or `Q6_K` model to run `INT8` or `INT4` NPU pipelines. The NPU quantization algorithm receives highly accurate FP32 reference data, allowing it to calculate optimal scales and minimize the final quantization error.

2. **Recommended Case (GGUF Precision ≈ NPU Precision)**<br>
   Using `Q8_0` for an `INT8` pipeline, or `Q4_0` for an `INT4` pipeline. The precision levels match closely, keeping the initial GGUF file size small on disk while minimizing further information loss during NPU requantization.

3. **Terrible Case (GGUF Precision < NPU Precision)**<br>
   Upscaling a `Q4_0` model to run on an `FP16` or `INT8` NPU pipeline. Information was irreversibly lost when the model was originally compressed to `Q4_0`. Requantizing it to a higher bit-depth cannot restore the lost accuracy, meaning you will get the poor quality of a 4-bit model while wasting the memory bandwidth and compute resources of an 8-bit or 16-bit model.

### Native Weight Cache

The native weight cache is disabled by default. Set `RKNPU_WEIGHT_CACHE=1` and `RKNPU_WEIGHT_CACHE_DIR` to store the packed result and reuse it on later starts. This avoids repeating the GGUF dequantization, NPU requantization, and native packing steps:

```sh
RKNPU_DEVICE="RK3576" RKNPU_WEIGHT_CACHE=1 RKNPU_WEIGHT_CACHE_DIR="/path/to/cache" ./build/bin/llama-server -m ./model.gguf
```

Allow enough free disk space for the packed model weights in the selected NPU pipeline format plus scale metadata. Cache format **v2** stores the packed weights, per-channel scales or low-rank `u`/`v` factors, and Hadamard signs. Cache identity includes the source tensor content, SoC configuration, active cores, pipeline layout, scale scheme, Hadamard seed scheme and sign hash, and RKNN runtime version.

Older cache formats are not reused. A cache miss, incompatible descriptor, or failed integrity check causes rebuilding from the original GGUF weights; there is no need to convert old cache files manually.

Set `RKNPU_WEIGHT_CACHE_REBUILD=1` for one run to replace all entries used by that process. Set `RKNPU_TRACE=1` to print cache hits, misses, invalid entries, and stores.

### Execution Plan Cache

With `--no-mmap`, the loader writes directly to host buffers. Each NPU weight tensor is prepared on its first NPU operation, using the same quantization and weight-cache path as regular loading.

The execution plan cache is disabled by default. Set `RKNPU_EXECUTION_PLAN_CACHE=1` to reuse the prepared combination of cached RKNN contexts and buffers, IO binding state, matrix segmentation, scales, and low-rank factors for each weight tensor and operation batch size. A plan is invalidated when the weight tensor version or source buffer instance changes. The cache uses process memory and is released when the backend is destroyed.

## Chipsets

The backend configures operations based on hardware pipelines-specific hardware-accelerated paths mapping mathematical operations to native NPU types. Each supported chipset defines its own set of pipelines and default quantization behaviors.

The tables below list the configured hardware pipelines and default mappings. All numerical accuracy values in this section are retained from upstream RK3588 measurements on `Gemma-3-1b-it`; neither this branch nor RK3576 has been measured here.

Upstream used the command below and obtained baseline logits from F16 weights on the CPU backend. Its CPU precision measurements are retained for comparison.

```sh
taskset -c 4-7 ./build/bin/llama-perplexity -m ./model.gguf -f ./wiki.test.raw -t 4 -b 512 --chunks 32 --kl-divergence-base ./model-logits.kld --kl-divergence
```

| Name | Operation | Perplexity | KL-Divergence | Top-P | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `F16` | - | 26.20 ± 1.08 | 0.000000 ± 0.000000 | 100.00 ± 0.00 % | - |
| `Q8_0` | - | 26.08 ± 1.07 | 0.004442 ± 0.000289 | 97.30 ± 0.18 % | - |
| `Q4_0` | - | 30.76 ± 1.31 | 0.188193 ± 0.005248 | 80.97 ± 0.44 %  | - |

### RK3588

The default configuration uses NPU cores 0, 1, and 2 and a maximum K segment size of 8192.

#### Available Pipelines (Upstream Reference Measurements)

| Name | Operation | Perplexity | KL-Divergence | Top-P | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `W16A16_HADAMARD` | FP16xFP16 | 26.17 ± 1.07 | 0.000158 ± 0.000011 | 99.49 ± 0.08 % | Hadamard Transform\* |
| `W16A16_STANDARD` | FP16xFP16 | 26.17 ± 1.07 | 0.000156 ± 0.000011 | 99.46 ± 0.08 % | - |
| `W8A8_HADAMARD_LOWRANK` | INT8xINT8 | 26.08 ± 1.07 | 0.002805 ± 0.000202 | 97.79 ± 0.16 % | Hadamard Transform\*, Low-Rank Approximation\*\* |
| `W8A8_HADAMARD` | INT8xINT8 | 26.15 ± 1.07 | 0.003031 ± 0.000143 | 97.64 ± 0.17 % | Hadamard Transform\* |
| `W8A8_LOWRANK` | INT8xINT8 | 26.30 ± 1.07 | 0.072085 ± 0.002560 | 88.49 ± 0.35 % | Low-Rank Approximation\*\* |
| `W8A8_STANDARD` | INT8xINT8 | 26.87 ± 1.10 | 0.074325 ± 0.002338 | 87.88 ± 0.36 % | - |
| `W4A4_HADAMARD_LOWRANK` | INT4xINT4 | 36.54 ± 1.50 | 0.701652 ± 0.012223 | 64.38 ± 0.53 % | Hadamard Transform\*, Low-Rank Approximation\*\*|
| `W4A4_HADAMARD` | INT4xINT4 | 40.93 ± 1.71 | 0.817203 ± 0.013658 | 62.13 ± 0.54 % | Hadamard Transform\*|
| `W4A4_LOWRANK` | INT4xINT4 | 485565.45 ± 31080.30 | 10.918173 ± 0.053307 | 1.56 ± 0.14 % | Low-Rank Approximation\*\* |
| `W4A4_STANDARD` | INT4xINT4 | 88738.26 ± 4853.41 | 9.126788 ± 0.046108 | 1.45 ± 0.13 % | - |

#### Default Mappings

| Input Weight Type | Default Hardware Pipeline | Bits Per Weight |
| :--- | :--- | :--- |
| `F16` | [`W16A16_STANDARD`] | 16 |
| `Q8_0` | [`W8A8_STANDARD`] | 8 |
| `Q6_K` | [`W8A8_STANDARD`, `W4A4_HADAMARD`] | 6 |
| `Q4_0` | [`W4A4_HADAMARD`] | 4 |
| `Q4_K` | [`W4A4_HADAMARD`] | 4 |

\* **Hadamard Transform:** Applies a randomized Fast Walsh-Hadamard Transform to smooth out activation outliers before quantization (see [QuaRot](https://arxiv.org/abs/2404.00456)).</br>
\*\* **Low-Rank Approximation:** Reconstructs per-element weight scales with rank-1 factors and compensates the activation and output matrices to preserve GEMM compatibility (see [LRSR](https://doi.org/10.5281/zenodo.22976487)). It can be selected with `RKNPU_HYBRID`.

### RK3576

The default configuration uses NPU cores 0 and 1 and a maximum K segment size of 4096. The mixed pipelines use FP16 activations with INT8 or INT4 weights and FP32 output. No RK3576 accuracy or performance measurements are supplied here.

#### Available Pipelines

Operation order is activations (A) x weights (B) -> output (C).

| Name | Operation | Notes |
| :--- | :--- | :--- |
| `W16A16_STANDARD` | FP16 x FP16 -> FP32 | - |
| `W16A16_HADAMARD` | FP16 x FP16 -> FP32 | Hadamard transform |
| `W8A8_STANDARD` | INT8 x INT8 -> INT32 | - |
| `W8A8_HADAMARD` | INT8 x INT8 -> INT32 | Hadamard transform |
| `W8A16_STANDARD` | FP16 x INT8 -> FP32 | - |
| `W8A16_HADAMARD` | FP16 x INT8 -> FP32 | Hadamard transform |
| `W4A4_STANDARD` | INT4 x INT4 -> INT16 | - |
| `W4A4_HADAMARD` | INT4 x INT4 -> INT16 | Hadamard transform |
| `W4A16_STANDARD` | FP16 x INT4 -> FP32 | - |
| `W4A16_HADAMARD` | FP16 x INT4 -> FP32 | Hadamard transform |

#### Default Mappings

| Input Weight Type | Default Hardware Pipeline | Bits Per Weight |
| :--- | :--- | :--- |
| `F16` | [`W16A16_STANDARD`] | 16 |
| `Q8_0` | [`W8A16_STANDARD`] | 8 |
| `Q6_K` | [`W8A16_STANDARD`, `W4A16_HADAMARD`] | 6 |
| `Q4_0` | [`W4A16_HADAMARD`] | 4 |
| `Q4_K` | [`W4A16_STANDARD`] | 4 |

## FAQ

### Q: How do I select a specific Rockchip SoC?
By default, the backend targets the **RK3588**. If you are using a different chip, specify it using the `RKNPU_DEVICE` environment variable. You can find the list of all available chips in **Chipsets** section above.

```sh
# Select RK3576 configuration
RKNPU_DEVICE="RK3576" ./build/bin/llama completion -m ./model.gguf
```

### Q: Can I limit inference to specific NPU cores?
The backend tries to utilize all available NPU cores. You can restrict execution to specific cores using the `RKNPU_CORES` variable. This is useful for saving power or leaving cores free for other tasks.

```sh
# Use only the first core
RKNPU_CORES="0" ./build/bin/llama completion -m ./model.gguf

# Use cores 0 and 2
RKNPU_CORES="0,2" ./build/bin/llama completion -m ./model.gguf
```

### Q: How can I run multiple model instances without memory conflicts?
The Rockchip NPU has a strict 4GB limit per IOMMU domain. To run multiple instances, you can manually partition the available domains using the `RKNPU_DOMAINS` variable.

```sh
# Use domains 0, 1, 2 and 3 for the model
RKNPU_DOMAINS="0-3" ./build/bin/llama completion -m ./model.gguf

# Use domains 4, 5, 6, 7 and 9 for the model
RKNPU_DOMAINS="4-7,9" ./build/bin/llama completion -m ./model.gguf
```

> **Warning:** Due to library and driver limitations, assigning custom domains while running multiple independent processes will cause a **System Kernel Panic** if they attempt to access the NPU hardware simultaneously. This leads to a total **Operating System Freeze**. To run concurrent models, ensure that inference in independent processes is executed **sequentially**. Loaded models can safely idle for an indefinite amount of time.

## Contributing

Feel free to open an issue to discuss a bug or feature, or submit a pull request with your improvements. Please refer to CONTRIBUTING.md for a breakdown of the project structure, current development goals, and workflow guidelines.
