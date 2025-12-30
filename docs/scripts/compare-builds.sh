#!/bin/bash
# Build all three ONNX Runtime configurations and compare sizes
#
# Usage:
#   ./compare-builds.sh
#
# This script will:
#   1. Generate operator config from sample embedding models
#   2. Build ONNX Runtime with full operators
#   3. Build ONNX Runtime without contrib ops
#   4. Build ONNX Runtime with selective ops
#   5. Compare and report sizes
#
# Requirements:
#   - CMake 3.26+
#   - Python 3.10+
#   - GCC/Clang
#   - ~30GB disk space
#   - pip packages: onnx, flatbuffers, huggingface_hub

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-/tmp/onnxruntime-build}"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "ONNX Runtime Size Optimization Comparison"
echo "=========================================="
echo ""
echo "Work directory: $WORK_DIR"
echo "Results directory: $RESULTS_DIR"
echo ""

# Step 1: Generate ops config
echo "=== Step 1: Generate Operator Config ==="
"$SCRIPT_DIR/generate-ops-config.sh"
echo ""

# Step 2: Build full
echo "=== Step 2: Build Full (with all contrib ops) ==="
"$SCRIPT_DIR/build-onnxruntime.sh" full
cp "$WORK_DIR/build/Linux/MinSizeRel/libonnxruntime.so.1."* \
    "$RESULTS_DIR/libonnxruntime-full.so"
FULL_SIZE=$(stat -c%s "$RESULTS_DIR/libonnxruntime-full.so" 2>/dev/null || \
            stat -f%z "$RESULTS_DIR/libonnxruntime-full.so")
echo ""

# Step 3: Build no-contrib
echo "=== Step 3: Build No Contrib Ops ==="
"$SCRIPT_DIR/build-onnxruntime.sh" no-contrib
cp "$WORK_DIR/build/Linux/MinSizeRel/libonnxruntime.so.1."* \
    "$RESULTS_DIR/libonnxruntime-no-contrib.so"
NO_CONTRIB_SIZE=$(stat -c%s "$RESULTS_DIR/libonnxruntime-no-contrib.so" 2>/dev/null || \
                  stat -f%z "$RESULTS_DIR/libonnxruntime-no-contrib.so")
echo ""

# Step 4: Build selective
echo "=== Step 4: Build Selective (embedding ops only) ==="
"$SCRIPT_DIR/build-onnxruntime.sh" selective
cp "$WORK_DIR/build/Linux/MinSizeRel/libonnxruntime.so.1."* \
    "$RESULTS_DIR/libonnxruntime-selective.so"
SELECTIVE_SIZE=$(stat -c%s "$RESULTS_DIR/libonnxruntime-selective.so" 2>/dev/null || \
                 stat -f%z "$RESULTS_DIR/libonnxruntime-selective.so")
echo ""

# Calculate reductions
NO_CONTRIB_REDUCTION=$(echo "scale=1; (1 - $NO_CONTRIB_SIZE / $FULL_SIZE) * 100" | bc)
SELECTIVE_REDUCTION=$(echo "scale=1; (1 - $SELECTIVE_SIZE / $FULL_SIZE) * 100" | bc)

FULL_MB=$(echo "scale=2; $FULL_SIZE / 1048576" | bc)
NO_CONTRIB_MB=$(echo "scale=2; $NO_CONTRIB_SIZE / 1048576" | bc)
SELECTIVE_MB=$(echo "scale=2; $SELECTIVE_SIZE / 1048576" | bc)

# Report
echo "=========================================="
echo "           RESULTS SUMMARY"
echo "=========================================="
echo ""
echo "| Configuration | Size (bytes) | Size (MiB) | Reduction |"
echo "|---------------|--------------|------------|-----------|"
printf "| Full          | %12d | %10s | baseline  |\n" "$FULL_SIZE" "$FULL_MB"
printf "| No contrib    | %12d | %10s | -%s%%     |\n" "$NO_CONTRIB_SIZE" "$NO_CONTRIB_MB" "$NO_CONTRIB_REDUCTION"
printf "| Selective     | %12d | %10s | -%s%%     |\n" "$SELECTIVE_SIZE" "$SELECTIVE_MB" "$SELECTIVE_REDUCTION"
echo ""
echo "Libraries saved to: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR/"*.so
echo ""
echo "Operator config: $SCRIPT_DIR/embedding_ops.config"
