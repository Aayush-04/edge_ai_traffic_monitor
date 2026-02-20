#!/bin/bash
# Quick-launch FPS benchmark
cd "$(dirname "$0")/../deploy/cpp"
echo "Running DPU benchmark (Ctrl+C to stop)..."
./test_video_perf yolov3_quant