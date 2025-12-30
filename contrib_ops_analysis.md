# ONNX Runtime contrib_ops Size Analysis for fastembed-rs

This document analyzes the potential binary size savings from pruning the contrib_ops directory in ONNX Runtime for embedding model use cases.

## Source Code Size Analysis

| Directory | Size | Purpose |
|-----------|------|---------|
| `contrib_ops/cuda/` | 198 MB | CUDA operators |
| ├── `cuda/bert/` | 175 MB | 159 MB is pre-compiled .cubin.cc (fused attention) |
| ├── `cuda/sparse/` | 22 MB | Sparse CUDA kernels |
| └── others | ~1 MB | LLM, MoE, diffusion, etc. |
| `contrib_ops/cpu/` | 1.5 MB | CPU operators |
| ├── `cpu/transformers/` | 414 KB | beam_search, greedy_search, sampling |
| ├── `cpu/bert/` | 282 KB | attention, EmbedLayerNorm, BiasGelu |
| ├── `cpu/quantization/` | 268 KB | Quantization support |
| ├── `cpu/moe/` | 102 KB | Mixture of Experts |
| └── others | ~400 KB | attnlstm, sparse, utils, tensor, math |
| `contrib_ops/webgpu/` | 449 KB | WebGPU operators |
| `contrib_ops/js/` | 45 KB | JavaScript operators |
| **Total** | **~200 MB** | |

For reference, the core `onnxruntime/core/` directory is about **33 MB**.

## What Embedding Models Need

### REQUIRED for BERT/transformer embeddings:
- `embed_layer_norm` - Fused embedding + layer normalization
- `attention`, `multihead_attention` - Attention operations
- `bias_gelu`, `fast_gelu` - Activation functions
- `quantization/*` - If using quantized models (e.g., BGE-small-en-v1.5-quantized)

### NOT REQUIRED for embeddings:
- `cpu/transformers/*` - All autoregressive generation ops:
  - `beam_search`, `greedy_search`, `sampling`
  - Subgraphs for GPT, T5, Whisper
- `cpu/moe/*` - Mixture of Experts architectures
- `cpu/attnlstm/*` - LSTM attention variants
- `cpu/aten_ops/*` - PyTorch ATen compatibility

## Binary Size Reference

**ONNX Runtime v1.22.0 official release (CPU-only, Linux x64):**
- `libonnxruntime.so.1.22.0`: **21 MB** (uncompressed shared library)
- Compressed tarball: 7.4 MB

**fastembed-rs build artifacts (debug build):**
- `libort_sys-*.rlib`: 95 MB (Rust static library with ONNX Runtime)
- `libfastembed.rlib`: 28 MB

## Estimated Savings

### CPU-only builds (fastembed-rs default):

**Prunable contrib_ops source code:**
- `cpu/transformers/`: 414 KB (generation ops)
- `cpu/moe/`: 102 KB
- `cpu/attnlstm/`: 78 KB
- `cpu/aten_ops/`: 12 KB
- **Total prunable: ~606 KB** of 1.5 MB (~40% of CPU contrib_ops source)

**Binary size estimate:**
- CPU contrib_ops source is ~4.3% of total source (1.5 MB / 34.5 MB)
- 40% of that is prunable for embeddings (~1.7% of total)
- **Estimated savings: ~0.4 MB** (~1.7% of 21 MB binary)

Note: Source code size to binary size ratio is not 1:1. Actual savings require measurement via custom build.

### CUDA builds:

**Prunable CUDA contrib_ops:**
- `cuda/llm/`: 792 KB
- `cuda/diffusion/`: 72 KB
- `cuda/moe/`: 582 KB
- Parts of `cuda/transformers/`: ~100 KB (generation-specific code)

The 159 MB of pre-compiled .cubin.cc files in cuda/bert/ are fused attention kernels for GPU architectures (sm70, sm75, sm80, sm86, sm89). These are used for BERT inference, so they would need to be kept for embedding models using GPU.

**Estimated CUDA prunable: ~1.5 MB** (excluding the cubin files needed for embeddings)

## Implementation Notes

1. **Cannot use `--disable_contrib_ops`**: Embedding models use some contrib ops like `EmbedLayerNorm` and fused attention. Disabling all contrib ops would break optimized embedding models.

2. **Use `--include_ops_by_config`**: The ONNX Runtime build system supports selective operator inclusion via a config file generated from your target models.

3. **Current fastembed-rs setup**: The `ort` crate downloads prebuilt ONNX Runtime binaries from pykeio's CDN which include full contrib_ops.

## How to Measure Precisely

To get exact binary size savings:

1. Generate operator config from embedding models:
   ```bash
   python onnxruntime/tools/python/create_reduced_build_config.py \
       --model_path path/to/embedding_model.onnx \
       --output_path reduced_ops.config
   ```

2. Build ONNX Runtime with reduced operators:
   ```bash
   ./build.sh --config MinSizeRel --include_ops_by_config reduced_ops.config
   ```

3. Compare resulting `libonnxruntime.so` sizes with and without the config.

## References

- [ONNX Runtime Reduced Operator Kernel build](https://github.com/microsoft/onnxruntime/blob/main/docs/Reduced_Operator_Kernel_build.md)
- [ONNX Runtime Custom build](https://onnxruntime.ai/docs/build/custom.html)
- [Contrib Ops Wiki](https://github.com/microsoft/onnxruntime/wiki/Contrib-Ops)
- [ort crate](https://github.com/pykeio/ort)
- [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
