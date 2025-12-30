# ONNX Runtime contrib_ops Size Analysis for fastembed-rs

This document analyzes the binary size savings from pruning operators in ONNX Runtime for embedding model use cases.

## Measured Binary Sizes

Built from ONNX Runtime v1.24.0 (main branch) with `--config MinSizeRel --build_shared_lib`:

| Build | Size (bytes) | Size (MiB) | vs Full |
|-------|-------------|------------|---------|
| Full (with all contrib_ops) | 19,328,992 | 18.43 | baseline |
| `--disable_contrib_ops` | 16,251,688 | 15.50 | -15.9% |
| **`--include_ops_by_config` (embedding ops only)** | **10,475,624** | **9.99** | **-45.8%** |

### Summary

- Disabling all contrib_ops saves **~3 MB (16%)**
- Selective pruning to embedding-only ops saves **~9 MB (46%)**

## Operator Config for Embedding Models

Generated from `Qdrant/bge-small-en-v1.5-onnx-Q` optimized model:

```
ai.onnx;11;Add,Cast,Constant,Gather,LayerNormalization,MatMul,ReduceSum,Shape,Slice,Unsqueeze
com.microsoft;1;Attention,FastGelu,SkipLayerNormalization
```

Only 13 operators needed:
- **10 standard ONNX ops**: Add, Cast, Constant, Gather, LayerNormalization, MatMul, ReduceSum, Shape, Slice, Unsqueeze
- **3 contrib ops**: Attention, FastGelu, SkipLayerNormalization

## What Gets Pruned

With selective pruning, these are excluded:

**Contrib ops not needed for embeddings:**
- `cpu/transformers/*` - beam_search, greedy_search, sampling (414 KB source)
- `cpu/moe/*` - Mixture of Experts (102 KB source)
- `cpu/attnlstm/*` - LSTM attention (78 KB source)
- `cpu/aten_ops/*` - PyTorch ATen compatibility (12 KB source)
- Most of `cpu/bert/*` except Attention
- All quantization ops (unless using quantized models)

**Standard ONNX ops not needed:**
- Convolution operators
- RNN/LSTM operators
- Many math/tensor operations unused by transformers

## How to Build

```bash
# 1. Clone ONNX Runtime
git clone --depth 1 https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# 2. Generate config from your embedding models
pip install onnx flatbuffers
python tools/python/create_reduced_build_config.py /path/to/models/ ops.config

# 3. Build with selective ops
./build.sh --config MinSizeRel --build_shared_lib --parallel \
    --skip_tests --include_ops_by_config ops.config
```

## Using with fastembed-rs

fastembed-rs supports dynamic ONNX Runtime loading:

```toml
[dependencies]
fastembed = { version = "5.5", default-features = false, features = ["ort-load-dynamic", "hf-hub-native-tls"] }
```

Then set `ORT_DYLIB_PATH` to point to your custom-built library.

## Caveats

1. The config must include all operators used by your models
2. Different embedding models may use different operators
3. Quantized models need additional quantization operators
4. ONNX Runtime optimizations may add new operators at runtime

To ensure compatibility, generate the config from all models you plan to use:

```bash
# Put all your ONNX models in a directory
python tools/python/create_reduced_build_config.py /path/to/all/models/ ops.config
```

## References

- [ONNX Runtime Reduced Operator Kernel build](https://github.com/microsoft/onnxruntime/blob/main/docs/Reduced_Operator_Kernel_build.md)
- [ONNX Runtime Custom build](https://onnxruntime.ai/docs/build/custom.html)
- [ort crate](https://github.com/pykeio/ort)
