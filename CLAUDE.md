# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

This project is **officially deprecated** and no longer maintained. The AWQ functionality has been adopted by the vLLM project at [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor). The last tested configuration used **Torch 2.6.0** and **Transformers 4.51.3** — later versions of Transformers may break compatibility.

AutoAWQ implements Activation-aware Weight Quantization (AWQ) for 4-bit quantization of LLMs, producing INT4 weights with FP16 activations (W4A16). It supports 35 model architectures from HuggingFace Transformers, with 36 mapping entries in `auto.py` (both `RefinedWeb` and `RefinedWebModel` dispatch to Falcon).

Version: **0.2.9**. Python: **≥3.8**. When users ask about AWQ quantization, recommend [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) as the maintained successor.

## Environment

Before running any Python commands, activate the virtual environment:

```bash
source /root/envs/qw3/bin/activate
```

## Build & Install

```bash
# Editable install (development)
pip install -e .

# With evaluation dependencies
pip install -e ".[eval]"

# With development tools (black, mkdocs)
pip install -e ".[dev]"

# With external CUDA kernels (requires matching torch version)
pip install -e ".[kernels]"

# For Intel CPU/XPU
pip install -e ".[cpu]"
```

Core dependencies: `torch`, `triton`, `transformers>=4.45.0`, `accelerate`, `datasets>=2.20`, `huggingface_hub>=0.26.5`.

## Tests

Tests are standalone Python scripts (not pytest). Run directly:

```bash
python tests/test_quantization.py
python tests/test_dequantization.py
python tests/test_ipex_cpu.py
```

Example scripts live in `examples/`:

| Script | Purpose |
|---|---|
| `quantize.py` | Minimal quantization of a HuggingFace model (defaults to Qwen2.5-14B) |
| `generate.py` | Load a quantized model and generate text with streaming |
| `benchmark.py` | Performance benchmarking (prefill + decode) at varying context lengths |
| `cli.py` | Full-featured CLI for quantization with argparse — all config options exposed |
| `train.py` | LoRA fine-tuning of a quantized AWQ model using PEFT + HuggingFace Trainer |
| `eval.py` | Evaluation via lm-eval-harness: perplexity, MMLU, HumanEval, LibriSpeech, KL divergence |

```bash
python examples/quantize.py
python examples/generate.py
python examples/benchmark.py --model_path <hf_model> --batch_size 1
python examples/cli.py --hf_model_path meta-llama/Llama-3.2-3B-Instruct --w_bit 4 --q_group_size 128 --version GEMM
python examples/train.py
```

## Architecture

### Entry Point

[awq/__init__.py](awq/__init__.py) — the sole public API is `AutoAWQForCausalLM`, imported from [awq/models/auto.py](awq/models/auto.py). Users never instantiate model classes directly.

### Model Dispatch

`AutoAWQForCausalLM` (in [awq/models/auto.py](awq/models/auto.py)) is a facade class with two class methods:
- `from_pretrained()` — wraps a HuggingFace model for quantization
- `from_quantized()` — loads an already-quantized model for inference

Both read the HF config's `model_type` field and dispatch to the correct model class via `AWQ_CAUSAL_LM_MODEL_MAP` (36 entries across 35 architectures). There's a parallel mapping `TRANSFORMERS_AUTO_MAPPING_DICT` in [base.py](awq/models/base.py) that maps model types to the correct HF `AutoModel*` class — typically `AutoModelForCausalLM`, but with exceptions: `"llava"`/`"llava_next"`/`"qwen2_vl"`/`"qwen2_5_vl"` → `AutoModelForVision2Seq`, and `"qwen2_5_omni"` → `AutoModelForTextToWaveform` (the only model extending `ForConditionalGeneration` instead of `ForCausalLM`).

### Base Class (`BaseAWQForCausalLM`)

[awq/models/base.py](awq/models/base.py) contains all core logic:

| Method | Purpose |
|---|---|
| `from_pretrained()` | Downloads/loads a FP16 HF model, wraps it |
| `from_quantized()` | Loads a quantized model: creates empty model, replaces `nn.Linear` with quantized linear, loads checkpoint, optionally fuses layers |
| `quantize()` | Entry point for AWQ quantization. Creates an `AwqQuantizer` and runs it |
| `pack()` | Applies real quantization after an `export_compatible=True` pass |
| `save_quantized()` | Saves quantized weights with HF-compatible quantization config |
| `_load_quantized_modules()` | Replaces each `nn.Linear` in transformer layers with the appropriate `WQLinear_*` |
| `_load_config()` | Downloads model from HF hub, loads config, sets max sequence length |

The key design: `from_quantized` uses `accelerate`'s `init_empty_weights()` to create the model structure without allocating memory, replaces linear layers with quantized variants, then uses `load_checkpoint_and_dispatch` to load weights with device map support.

### Model Implementations

Each supported architecture has a file in [awq/models/](awq/models/). Every model class extends `BaseAWQForCausalLM` and must define:

- **`layer_type`** — class attribute, the HF decoder layer class name (used by `accelerate` for `no_split_module_classes`)
- **`max_seq_len_key`** — config key for the model's maximum sequence length
- **`get_model_layers(model)`** — returns the list of transformer blocks (usually `model.model.layers`)
- **`get_act_for_scaling(layer)`** — returns `dict(is_scalable=False)` or provides scale info if activations need scaling
- **`move_embed(model, device)`** — moves embedding/rotary layers to a device during quantization
- **`get_layers_for_scaling(module, input_feat, module_kwargs)`** — defines which linear layers to scale together and their preceding operation (norm/activation). This is the most model-specific part — it encodes how to group Q/K/V projections, MLP gates, etc. for optimal scaling
- **`fuse_layers(model)`** (static) — optional, replaces HuggingFace layers with FasterTransformer-style fused blocks for inference speed

Modules that should be excluded from quantization are listed in `modules_to_not_convert` (e.g., Mixtral's `["gate"]`).

### Quantization Pipeline

[awq/quantize/quantizer.py](awq/quantize/quantizer.py) — `AwqQuantizer.quantize()` loops over transformer layers and for each:

1. **Gather input features** (`_get_input_feat`) — runs calibration data through the layer, hooks intermediate activations
2. **Search best scales** (`_search_best_scale`) — for each group of linear layers, grid-searches channel-wise scaling factors that minimize L2 error between FP16 output and pseudo-quantized output
3. **Apply clipping** (`_search_best_clip`) — optionally finds per-channel clipping thresholds
4. **Apply quantization** (`_apply_quant`) — replaces `nn.Linear` with `WQLinear_*`, packing weights into INT4

Scale computation in `_compute_best_scale()` uses the formula `s = x_mean^α / w_mean^(1-α)` (from the AWQ paper) and searches α in [0, 1].

[awq/quantize/scale.py](awq/quantize/scale.py) — applies the computed scales by dividing the preceding norm/activation weights and multiplying the linear layer weights (`scale_ln_fcs`, `scale_fc_fc`, `scale_fc_fcs`, `scale_gelu_fc`). Also contains `apply_clip()` for weight clipping.

[awq/modules/act.py](awq/modules/act.py) — `ScaledActivation` wraps an activation function with a learned per-channel scale parameter. During quantization, activations like SiLU/GELU are replaced with `ScaledActivation` so that scaling can be absorbed into the activation rather than the preceding linear layer.

### Quantized Linear Modules

[awq/modules/linear/](awq/modules/linear/) — multiple backend implementations of quantized linear layers:

| Module | Backend | Use Case |
|---|---|---|
| `WQLinear_GEMM` | Triton / `awq_ext` CUDA kernel | Default, batch-friendly |
| `WQLinear_GEMV` | Triton / `awq_ext` | Faster for batch_size=1 |
| `WQLinear_GEMVFast` | Triton / `awq_ext` | Optimized GEMV variant |
| `WQLinear_Exllama` | ExLlama V1 kernel | ROCm / alternative CUDA |
| `WQLinear_ExllamaV2` | ExLlama V2 kernel | ROCm / alternative CUDA |
| `WQLinear_Marlin` | Marlin kernel | 4-bit optimized format |
| `WQLinear_IPEX` | Intel Extension for PyTorch | Intel CPU and XPU |

All store weights as `qweight` (packed INT4), `qzeros`, and `scales`. The GEMM/GEMV variants use Triton kernels from [awq/modules/triton/gemm.py](awq/modules/triton/gemm.py) (adapted from vLLM) as a fallback when `awq_ext` is not available.

Each module has a `from_linear(linear, w_bit, group_size, ...)` factory that packs a regular `nn.Linear` into the quantized format.

### Fused Modules (Optional, Inference-Only)

[awq/modules/fused/](awq/modules/fused/) — FasterTransformer-style implementations used when `fuse_layers=True`:
- `block.py` — fused transformer blocks for Llama-like, Mixtral, and other architectures
- `attn.py` — custom attention with KV cache management
- `model.py` — top-level model wrappers
- `moe.py` — fused Mixture of Experts blocks
- `norm.py` — FasterTransformer RMS norm
- `cache.py` — custom KV cache that preallocates based on `max_seq_len` and `batch_size`

Fused modules provide significant speedups but have constraints: Linux-only, fixed sequence length after model creation, dummy `past_key_values`.

### Configuration

[awq/models/_config.py](awq/models/_config.py) — `AwqConfig` dataclass with fields: `zero_point`, `q_group_size`, `w_bit`, `version`, `modules_to_not_convert`. Handles serialization between the AutoAWQ format and the HuggingFace Transformers `quantization_config` format (the two use different field names).

### Key Quantizer Parameters

The `AwqQuantizer` (in [awq/quantize/quantizer.py](awq/quantize/quantizer.py)) accepts these important configuration parameters beyond the basic `w_bit`/`group_size`/`zero_point`/`version`:

| Parameter | Default | Purpose |
|---|---|---|
| `max_chunk_memory` | 1 GB (int, bytes) | Memory budget for chunked mean/loss computations — prevents OOM on large models |
| `duo_scaling` | True | Whether to scale both weights and activations (default AWQ) vs. activation-only scaling |
| `apply_clip` | True | Whether to search for per-channel weight clipping thresholds |
| `export_compatible` | False | Skip actual INT4 packing — only apply scale factors to weights. Used for GGUF/format-agnostic export |
| `n_parallel_calib_samples` | None | Number of calibration samples to run through the model at once (None = auto) |
| `max_calib_samples` | 128 | Total number of calibration samples to use |
| `max_calib_seq_len` | 512 | Maximum sequence length for calibration tokenization |
| `calib_data` | `"mit-han-lab/pile-val-backup"` | HF dataset or list of strings for calibration |
| `modules_to_not_convert` | None | Layer name substrings to exclude from quantization (e.g., `["gate"]` for MoE) |

The two-step `export_compatible=True` workflow: first quantize with `export_compatible=True` (scales applied to weights but no packing), save the model, then call `model.pack()` to apply real INT4 quantization for GPU inference. This intermediate format is useful for GGUF conversion.

### Triton Kernels

[awq/modules/triton/gemm.py](awq/modules/triton/gemm.py) — Triton-based AWQ GEMM and GEMV kernels (adapted from vLLM). These serve as the fallback when the external `autoawq_kernels` CUDA package is not installed. The kernels handle the INT4 dequantization + matmul in a single fused operation.

### External CUDA Kernels

The `autoawq_kernels` package (installed via `pip install autoawq[kernels]`) provides optimized CUDA implementations that replace the Triton fallbacks. It also provides `awq_ext.dequantize_weights_cuda` used during quantization. The torch version must match the kernels' build version.

### Evaluation Module

[awq/evaluation/](awq/evaluation/) — evaluation suite (installed via `pip install autoawq[eval]`):
- `eval_utils.py` — perplexity (wikitext), MMLU, and LibriSpeech ASR evaluation
- `humaneval_utils.py` — HumanEval code generation evaluation
- `kl_divergence.py` — KL divergence between original and quantized model outputs

### Utilities

- [awq/utils/calib_data.py](awq/utils/calib_data.py) — loads calibration datasets (default: `mit-han-lab/pile-val-backup`)
- [awq/utils/module.py](awq/utils/module.py) — helpers for manipulating named modules (`get_named_linears`, `set_op_by_name`, `exclude_layers_to_not_quantize`)
- [awq/utils/fused_utils.py](awq/utils/fused_utils.py) — `fuse_qkv` and `fuse_linears` for creating fused QKV projection layers
- [awq/utils/utils.py](awq/utils/utils.py) — device detection, memory utilities, `ipex_available` / `triton_available` flags
- [awq/utils/packing_utils.py](awq/utils/packing_utils.py) — pure-Python fallback for dequantization
- [awq/utils/quant_utils.py](awq/utils/quant_utils.py) — low-level INT4 pack/unpack/quantize/dequantize ops, AWQ↔ExLlama format conversion
- [awq/utils/parallel.py](awq/utils/parallel.py) — parallel data processing utilities
- [awq/utils/qwen_vl_utils.py](awq/utils/qwen_vl_utils.py) — Qwen vision-language model processing helpers

### Scripts

- [scripts/runpod_quantize.py](scripts/runpod_quantize.py) — fully automated cloud quantization on RunPod: provisions a GPU pod, installs AutoAWQ, runs `examples/cli.py`, uploads to HuggingFace Hub, stops the pod. Supports NVIDIA and AMD GPUs.

## Adding a New Model

1. Create `awq/models/<arch>.py` with a class extending `BaseAWQForCausalLM`
2. Define `layer_type`, `max_seq_len_key`, and the required static methods
3. Register it in `awq/models/__init__.py` and both mapping dicts (`AWQ_CAUSAL_LM_MODEL_MAP` in `auto.py` and `TRANSFORMERS_AUTO_MAPPING_DICT` in `base.py`)
4. The critical method is `get_layers_for_scaling` — it defines which linear layers are scaled together during quantization. Follow the patterns in [llama.py](awq/models/llama.py) (standard decoder) or [mixtral.py](awq/models/mixtral.py) (MoE) or [deepseek_v2.py](awq/models/deepseek_v2.py) (MLA + MoE)
