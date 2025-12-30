# ONNX Runtime contrib_ops Size Analysis for fastembed-rs

This document analyzes the binary size savings from disabling contrib_ops in ONNX Runtime for embedding model use cases.

## Measured Binary Sizes

Built from ONNX Runtime v1.24.0 (main branch) with `--config MinSizeRel --build_shared_lib`:

| Build | Size (bytes) | Size (MiB) |
|-------|-------------|------------|
| WITH contrib_ops | 19,328,992 | 18.43 |
| WITHOUT contrib_ops (`--disable_contrib_ops`) | 16,251,688 | 15.50 |
| **Difference** | **3,077,304** | **2.93** |
| **Reduction** | **15.9%** | |

## Important Caveat

**`--disable_contrib_ops` removes ALL contrib operators**, including those needed for optimized BERT inference:
- `EmbedLayerNorm` - fused embedding + layer normalization
- `Attention`, `MultiHeadAttention` - optimized attention operations
- `BiasGelu`, `FastGelu` - fused activation functions
- Quantization operators

Embedding models exported with these fused operators will **fail to run** without contrib_ops.

For selective pruning, use `--include_ops_by_config` with a config file listing only the operators your models need.

## Source Code Breakdown

| Directory | Size | Purpose |
|-----------|------|---------|
| `contrib_ops/cpu/` | 1.5 MB | CPU operators |
| ├── `cpu/transformers/` | 414 KB | beam_search, greedy_search, sampling |
| ├── `cpu/bert/` | 282 KB | attention, EmbedLayerNorm, BiasGelu |
| ├── `cpu/quantization/` | 268 KB | Quantization support |
| ├── `cpu/moe/` | 102 KB | Mixture of Experts |
| └── others | ~400 KB | attnlstm, sparse, utils, tensor, math |

### What embeddings DON'T need:
- `cpu/transformers/*` - autoregressive generation ops (beam_search, greedy_search, sampling)
- `cpu/moe/*` - Mixture of Experts
- `cpu/attnlstm/*` - LSTM attention variants
- `cpu/aten_ops/*` - PyTorch ATen compatibility

## How to Use with fastembed-rs

fastembed-rs supports dynamic ONNX Runtime loading via the `ort-load-dynamic` feature:

```toml
[dependencies]
fastembed = { version = "5.5", default-features = false, features = ["ort-load-dynamic", "hf-hub-native-tls"] }
```

Then build a custom ONNX Runtime and set `ORT_DYLIB_PATH` to point to it.

## Build Commands

```bash
# Clone ONNX Runtime
git clone --depth 1 https://github.com/microsoft/onnxruntime.git

# Build with contrib_ops (default)
./build.sh --config MinSizeRel --build_shared_lib --parallel --skip_tests --allow_running_as_root

# Build without contrib_ops
./build.sh --config MinSizeRel --build_shared_lib --parallel --skip_tests --allow_running_as_root --disable_contrib_ops

# Build with selective ops (recommended for embeddings)
python tools/python/create_reduced_build_config.py --model_path model.onnx --output_path ops.config
./build.sh --config MinSizeRel --build_shared_lib --parallel --skip_tests --allow_running_as_root --include_ops_by_config ops.config
```

## References

- [ONNX Runtime Reduced Operator Kernel build](https://github.com/microsoft/onnxruntime/blob/main/docs/Reduced_Operator_Kernel_build.md)
- [ONNX Runtime Custom build](https://onnxruntime.ai/docs/build/custom.html)
- [ort crate](https://github.com/pykeio/ort)
