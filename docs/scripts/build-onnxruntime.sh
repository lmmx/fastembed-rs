#!/bin/bash
# Build ONNX Runtime with different configurations for size comparison
#
# Usage:
#   ./build-onnxruntime.sh [full|no-contrib|selective]
#
# Requirements:
#   - CMake 3.26+
#   - Python 3.10+
#   - GCC/Clang
#   - ~10GB disk space for build
#   - pip packages: onnx, flatbuffers

set -e

BUILD_TYPE="${1:-full}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-/tmp/onnxruntime-build}"
ORT_VERSION="${ORT_VERSION:-main}"

echo "=== ONNX Runtime Size Optimization Build ==="
echo "Build type: $BUILD_TYPE"
echo "Work directory: $WORK_DIR"
echo ""

# Clone if not exists
if [ ! -d "$WORK_DIR" ]; then
    echo "Cloning ONNX Runtime..."
    git clone --depth 1 --branch "$ORT_VERSION" \
        https://github.com/microsoft/onnxruntime.git "$WORK_DIR"
fi

cd "$WORK_DIR"

# Clean previous build
rm -rf build/Linux

# Common build flags
BUILD_FLAGS=(
    --config MinSizeRel
    --build_shared_lib
    --parallel
    --skip_tests
    --update
    --build
)

# Add --allow_running_as_root if running as root
if [ "$(id -u)" -eq 0 ]; then
    BUILD_FLAGS+=(--allow_running_as_root)
fi

case "$BUILD_TYPE" in
    full)
        echo "Building with all operators (full build)..."
        ./build.sh "${BUILD_FLAGS[@]}"
        ;;

    no-contrib)
        echo "Building without contrib ops..."
        ./build.sh "${BUILD_FLAGS[@]}" --disable_contrib_ops
        ;;

    selective)
        if [ ! -f "$SCRIPT_DIR/embedding_ops.config" ]; then
            echo "Error: embedding_ops.config not found in $SCRIPT_DIR"
            echo "Run generate-ops-config.sh first to create it"
            exit 1
        fi
        echo "Building with selective ops from embedding_ops.config..."
        ./build.sh "${BUILD_FLAGS[@]}" \
            --include_ops_by_config "$SCRIPT_DIR/embedding_ops.config"
        ;;

    *)
        echo "Unknown build type: $BUILD_TYPE"
        echo "Usage: $0 [full|no-contrib|selective]"
        exit 1
        ;;
esac

# Report results
LIBRARY="$WORK_DIR/build/Linux/MinSizeRel/libonnxruntime.so.1.*"
if ls $LIBRARY 1>/dev/null 2>&1; then
    echo ""
    echo "=== Build Complete ==="
    ls -lh $LIBRARY
    SIZE_BYTES=$(stat -c%s $LIBRARY 2>/dev/null || stat -f%z $LIBRARY)
    SIZE_MB=$(echo "scale=2; $SIZE_BYTES / 1048576" | bc)
    echo "Size: $SIZE_BYTES bytes ($SIZE_MB MiB)"
else
    echo "Error: Library not found"
    exit 1
fi
