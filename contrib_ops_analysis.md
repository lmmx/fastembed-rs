# ONNX Runtime contrib_ops Analysis for fastembed-rs

This document analyzes the potential space savings from pruning the contrib_ops directory in ONNX Runtime for embedding model use cases.

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

## Binary Size Context

ONNX Runtime Python wheel sizes (CPU-only, v1.23.2):
- Linux x86-64: 17.4 MB
- Windows x86-64: 13.5 MB
- macOS x86-64: 19.2 MB
- macOS ARM64: 17.2 MB

## Estimated Savings

### CPU-only builds (fastembed-rs default):

**Prunable contrib_ops:**
- `cpu/transformers/`: 414 KB (generation ops)
- `cpu/moe/`: 102 KB
- `cpu/attnlstm/`: 78 KB
- `cpu/aten_ops/`: 12 KB
- **Total prunable: ~606 KB** of 1.5 MB (~40% of CPU contrib_ops)

**Binary size impact:**
- CPU contrib_ops ≈ 4-5% of total binary
- **Estimated savings: 0.6-1.2 MB** (3-7% of ~17 MB binary)

### CUDA builds:

The 159 MB of pre-compiled .cubin.cc files in cuda/bert/ are fused attention kernels for different GPU architectures (sm70, sm75, sm80, sm86, sm89). These provide significant performance benefits for BERT inference on GPU.

**Note:** Removing these would impact GPU inference performance, not just binary size.

## Important Caveats

1. **Cannot use `--disable_contrib_ops`**: Embedding models DO use some contrib ops like `EmbedLayerNorm` and fused attention. Disabling all contrib ops would break optimized embedding models.

2. **Correct approach**: Use `--include_ops_by_config <config_file>` with a config listing only operators your models use.

3. **fastembed-rs uses prebuilt binaries**: The `ort` crate downloads prebuilt ONNX Runtime binaries from pykeio's CDN, which include full contrib_ops. Custom-building would require significant effort.

## Recommendations

### For minimal impact on binary size:

The savings from pruning contrib_ops (~0.6-1.2 MB) are relatively small compared to:
- The complexity of maintaining custom ONNX Runtime builds
- Model file sizes (typically 20-400 MB per model)
- The `ort` crate's reliance on prebuilt binaries

### If you want to pursue this:

1. Use ONNX Runtime's `create_reduced_build_config.py` script with your target models
2. Build with `--include_ops_by_config <generated_config>`
3. Fork/modify the `ort` crate to use your custom-built libraries

### Alternative approaches for reducing overall footprint:

1. Use quantized models (already supported in fastembed-rs)
2. Use smaller embedding models (e.g., all-MiniLM-L6-v2 at 22MB vs BGE-large at 1.3GB)
3. Consider alternative ONNX Runtime backends (ort-candle, ort-tract) for pure-Rust solutions

## References

- [ONNX Runtime Reduced Operator Kernel build](https://github.com/microsoft/onnxruntime/blob/main/docs/Reduced_Operator_Kernel_build.md)
- [ONNX Runtime Custom build](https://onnxruntime.ai/docs/build/custom.html)
- [Contrib Ops Wiki](https://github.com/microsoft/onnxruntime/wiki/Contrib-Ops)
- [ort crate](https://github.com/pykeio/ort)
