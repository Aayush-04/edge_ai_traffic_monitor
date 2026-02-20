#!/bin/bash
# setup_board.sh — One-time ZCU104 board setup.
# Run after cloning the project to the board.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

echo "=== Edge AI Traffic Monitor — Board Setup ==="

# 1. Install model files
MODEL_SRC="$PROJECT_ROOT/compiled_model"
MODEL_DST="/usr/share/vitis_ai_library/models/yolov3_quant"

echo "[1/3] Installing model files..."
mkdir -p "$MODEL_DST"
cp "$MODEL_SRC/yolov3_quant.xmodel" "$MODEL_DST/"
cp "$MODEL_SRC/yolov3_quant.prototxt" "$MODEL_DST/"
cp "$MODEL_SRC/meta.json" "$MODEL_DST/"
cp "$MODEL_SRC/md5sum.txt" "$MODEL_DST/" 2>/dev/null || true
echo "  Model installed to $MODEL_DST"

# 2. Build C++ applications
echo "[2/3] Building C++ applications..."
cd "$PROJECT_ROOT/deploy/cpp"
chmod +x build.sh
./build.sh

# 3. Verify
echo ""
echo "[3/3] Verification..."
echo "  Model files:"
ls -la "$MODEL_DST/"
echo ""
echo "  Executables:"
ls -la "$PROJECT_ROOT/deploy/cpp/detect_image" \
       "$PROJECT_ROOT/deploy/cpp/test_video_perf" 2>/dev/null
echo ""
echo "  Camera:"
ls /dev/video* 2>/dev/null || echo "  WARNING: No camera detected"

echo ""
echo "=== Setup complete ==="
echo "Run:  python3 deploy/python/stream_server.py"
echo "Open: http://192.168.137.96:8080"