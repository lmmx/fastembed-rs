#!/bin/bash
# Generate ONNX Runtime operator config from embedding models
#
# Usage:
#   ./generate-ops-config.sh [model_dir]
#
# If no model_dir is provided, downloads sample embedding models
#
# Requirements:
#   - Python 3.10+
#   - pip packages: onnx, flatbuffers, huggingface_hub

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${1:-$SCRIPT_DIR/models}"
OUTPUT_FILE="$SCRIPT_DIR/embedding_ops.config"
WORK_DIR="${WORK_DIR:-/tmp/onnxruntime-build}"

echo "=== Generate Operator Config for Embedding Models ==="
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --quiet onnx flatbuffers huggingface_hub

# Download sample models if no model dir provided
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "Downloading sample embedding models..."
    mkdir -p "$MODEL_DIR"

    python3 << 'EOF'
import os
from huggingface_hub import hf_hub_download

model_dir = os.environ.get('MODEL_DIR', './models')
os.makedirs(model_dir, exist_ok=True)

models = [
    # Optimized models with fused ops
    ("Qdrant/bge-small-en-v1.5-onnx-Q", "model_optimized.onnx"),
    ("Qdrant/all-MiniLM-L6-v2-onnx", "model.onnx"),
]

for repo_id, filename in models:
    try:
        print(f"Downloading {repo_id}/{filename}...")
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        # Copy to model_dir
        import shutil
        dest = os.path.join(model_dir, f"{repo_id.replace('/', '_')}_{filename}")
        shutil.copy(path, dest)
        print(f"  -> {dest}")
    except Exception as e:
        print(f"  Warning: Could not download {repo_id}: {e}")
EOF
fi

# Clone ONNX Runtime if needed (for the config generation script)
if [ ! -d "$WORK_DIR" ]; then
    echo "Cloning ONNX Runtime for config generation script..."
    git clone --depth 1 https://github.com/microsoft/onnxruntime.git "$WORK_DIR"
fi

# Generate config
echo ""
echo "Generating operator config from models in $MODEL_DIR..."
python3 "$WORK_DIR/tools/python/create_reduced_build_config.py" \
    "$MODEL_DIR" "$OUTPUT_FILE"

echo ""
echo "=== Generated Config ==="
cat "$OUTPUT_FILE"
echo ""
echo "Config written to: $OUTPUT_FILE"
