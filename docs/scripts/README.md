# ONNX Runtime Size Optimization Scripts

Scripts to reproduce the binary size measurements from the optimization report.

## Quick Start

```bash
# Run all builds and compare (takes ~30 minutes)
./compare-builds.sh
```

## Scripts

### `compare-builds.sh`

Runs all three builds and produces a comparison table:

```
| Configuration | Size (bytes) | Size (MiB) | Reduction |
|---------------|--------------|------------|-----------|
| Full          |     19328992 |      18.43 | baseline  |
| No contrib    |     16251688 |      15.50 | -15.9%    |
| Selective     |     10475624 |       9.99 | -45.8%    |
```

### `build-onnxruntime.sh`

Build ONNX Runtime with a specific configuration:

```bash
./build-onnxruntime.sh full        # All operators (default)
./build-onnxruntime.sh no-contrib  # --disable_contrib_ops
./build-onnxruntime.sh selective   # --include_ops_by_config
```

### `generate-ops-config.sh`

Generate operator config from embedding models:

```bash
# Use sample models (downloads from HuggingFace)
./generate-ops-config.sh

# Use your own models
./generate-ops-config.sh /path/to/your/models/
```

## Pre-generated Config

`embedding_ops.config` contains the operators needed for optimized BERT embedding models:

```
ai.onnx;11;Add,Cast,Constant,Gather,LayerNormalization,MatMul,ReduceSum,Shape,Slice,Unsqueeze
com.microsoft;1;Attention,FastGelu,SkipLayerNormalization
```

## Requirements

- **OS**: Linux (tested on Ubuntu 22.04)
- **CMake**: 3.26+
- **Python**: 3.10+
- **Compiler**: GCC 11+ or Clang 14+
- **Disk space**: ~30GB for all three builds
- **RAM**: 8GB+ recommended
- **Time**: ~30 minutes total on 16-core machine

### Python packages

```bash
pip install onnx flatbuffers huggingface_hub
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORK_DIR` | `/tmp/onnxruntime-build` | Build directory |
| `ORT_VERSION` | `main` | ONNX Runtime git branch/tag |

## Reproduction Notes

The measurements in the report were produced on:

- **Date**: 2024-12-30
- **ONNX Runtime**: v1.24.0 (main branch, commit from that date)
- **OS**: Linux (Ubuntu-based, x86_64)
- **Build flags**: `--config MinSizeRel --build_shared_lib`

Results may vary slightly with different:
- ONNX Runtime versions
- Compiler versions
- Build configurations

## Troubleshooting

### "Running as root is not allowed"

The scripts automatically add `--allow_running_as_root` when needed. If you still see this error, add it manually to the build command.

### "flatbuffers module is not installed"

```bash
pip install flatbuffers
```

### Build takes too long

- Use `--parallel N` to limit parallelism (default uses all cores)
- Ensure sufficient RAM (builds can use 1-2GB per core)

### "Cannot find operator X"

Your models may use operators not in the config. Generate a new config:

```bash
./generate-ops-config.sh /path/to/all/your/models/
```
