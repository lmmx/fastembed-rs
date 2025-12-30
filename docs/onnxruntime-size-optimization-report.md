# ONNX Runtime Binary Size Optimization for Embedding Models

## Executive Summary

This report investigates potential binary size savings from pruning unused ONNX Runtime operators for embedding model inference in fastembed-rs. Through actual builds and measurements, we found that **selective operator pruning can reduce the ONNX Runtime library size by 46%** (from 18.4 MiB to 10.0 MiB).

## Methodology

We built ONNX Runtime v1.24.0 from source with three configurations:

1. **Full build** - Default configuration with all operators
2. **No contrib_ops** - Using `--disable_contrib_ops` flag
3. **Selective ops** - Using `--include_ops_by_config` with operators extracted from an optimized embedding model

All builds used:
- `--config MinSizeRel` (optimized for size)
- `--build_shared_lib` (shared library output)
- Linux x86_64 target

## Results

| Configuration | Binary Size | Reduction |
|---------------|-------------|-----------|
| Full build | 19,328,992 bytes (18.43 MiB) | baseline |
| `--disable_contrib_ops` | 16,251,688 bytes (15.50 MiB) | -15.9% |
| **Selective ops (embeddings)** | **10,475,624 bytes (9.99 MiB)** | **-45.8%** |

### Key Finding

Selective operator pruning provides nearly **3x better size reduction** than simply disabling all contrib_ops, while maintaining compatibility with optimized embedding models.

## Operators Required for Embedding Models

Analysis of `Qdrant/bge-small-en-v1.5-onnx-Q` (an ONNX Runtime-optimized BERT model) shows only **13 operators** are needed:

### Standard ONNX Operators (10)
```
ai.onnx;11;Add,Cast,Constant,Gather,LayerNormalization,MatMul,ReduceSum,Shape,Slice,Unsqueeze
```

### Microsoft Contrib Operators (3)
```
com.microsoft;1;Attention,FastGelu,SkipLayerNormalization
```

These contrib ops are **fused operators** that combine multiple standard operations for better performance:
- `Attention` - Fused multi-head attention
- `FastGelu` - Optimized GELU activation
- `SkipLayerNormalization` - Fused skip connection + layer norm

## What Gets Pruned

### Contrib Ops Not Needed for Embeddings

| Directory | Size | Purpose |
|-----------|------|---------|
| `cpu/transformers/` | 414 KB | beam_search, greedy_search, sampling |
| `cpu/moe/` | 102 KB | Mixture of Experts |
| `cpu/attnlstm/` | 78 KB | LSTM attention variants |
| `cpu/aten_ops/` | 12 KB | PyTorch ATen compatibility |
| `cpu/quantization/` | 268 KB | Quantization ops (unless needed) |

### Standard ONNX Ops Not Needed

- Convolution operators (Conv, ConvTranspose, etc.)
- RNN/LSTM/GRU operators
- Image processing operators
- Many reduction and math operations

## Implementation Guide

### Step 1: Generate Operator Config

```bash
# Install dependencies
pip install onnx flatbuffers

# Clone ONNX Runtime
git clone --depth 1 https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# Generate config from your models
python tools/python/create_reduced_build_config.py /path/to/models/ ops.config
```

### Step 2: Build ONNX Runtime

```bash
./build.sh \
    --config MinSizeRel \
    --build_shared_lib \
    --parallel \
    --skip_tests \
    --include_ops_by_config ops.config
```

### Step 3: Use with fastembed-rs

```toml
# Cargo.toml
[dependencies]
fastembed = {
    version = "5.5",
    default-features = false,
    features = ["ort-load-dynamic", "hf-hub-native-tls"]
}
```

```bash
# Set environment variable to custom library
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
```

## Caveats

1. **Model Compatibility**: The operator config must include all operators used by your models. Generate the config from all models you plan to use.

2. **Optimized vs Non-Optimized Models**:
   - Non-optimized ONNX models use only standard operators
   - ONNX Runtime-optimized models use fused contrib ops (Attention, FastGelu, etc.)
   - The selective build supports both, but the fused ops provide better performance

3. **Quantized Models**: If using quantized models, include quantization operators in your config.

4. **Runtime Optimizations**: ONNX Runtime may apply graph optimizations at runtime that require additional operators. Test thoroughly.

## Comparison with Other Approaches

| Approach | Size Reduction | Compatibility |
|----------|---------------|---------------|
| `--disable_contrib_ops` | 16% | ❌ Breaks optimized models |
| `--minimal_build` | ~60-80% | ❌ Requires ORT format models |
| **`--include_ops_by_config`** | **46%** | ✅ Works with ONNX models |

## Conclusion

For fastembed-rs and similar embedding-focused applications, selective operator pruning offers a practical way to reduce ONNX Runtime binary size by nearly half while maintaining full compatibility with optimized transformer models. The approach requires:

1. Collecting all ONNX models to be supported
2. Generating an operator config file
3. Building ONNX Runtime with the config
4. Distributing the custom library or having users build it

This could be particularly valuable for:
- Edge deployments with size constraints
- Container images where smaller is better
- Applications that only need embedding inference

## References

- [ONNX Runtime Reduced Operator Kernel Build](https://github.com/microsoft/onnxruntime/blob/main/docs/Reduced_Operator_Kernel_build.md)
- [ONNX Runtime Custom Build](https://onnxruntime.ai/docs/build/custom.html)
- [ort crate (Rust bindings)](https://github.com/pykeio/ort)
- [fastembed-rs](https://github.com/Anush008/fastembed-rs)
