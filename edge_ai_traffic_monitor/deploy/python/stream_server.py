#!/usr/bin/env python3
"""
stream_server.py
Main entry point: real-time YOLO detection with MJPEG browser streaming.
DPU inference runs in main thread (thread-safe), HTTP server in background.

Usage:
    python3 stream_server.py
    → Open http://192.168.137.96:8080 in browser

"""
import cv2
import time
import sys
import os

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from config.model_config import DEFAULT_XMODEL_PATH, STREAM_PORT
from dpu_runner import DpuRunner
from postprocess import decode_all_outputs
from camera import open_camera
from drawing import draw_detections
from mjpeg_server import start_server, update_frame


def main():
    print("=" * 55)
    print("  YOLO Real-Time Detection — ZCU104 FPGA")
    print("=" * 55)
    print(f"  Browser: http://192.168.137.96:{STREAM_PORT}")
    print(f"  Press Ctrl+C to stop")
    print("=" * 55)

    # Start HTTP server in background thread
    start_server(STREAM_PORT)

    # Open camera
    cap = open_camera()
    print("Camera opened.")

    # Initialize DPU runner
    detector = DpuRunner(DEFAULT_XMODEL_PATH)

    # Warmup — catches errors early
    print("Warmup...")
    ret, frame = cap.read()
    if ret:
        outputs = detector.run(frame)
        boxes = decode_all_outputs(outputs, detector.model_w)
        print(f"  Warmup OK: {len(boxes)} detections")

    print("\nDetection loop running...\n")

    frame_count = 0
    t_start = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # DPU inference — main thread only (VART not thread-safe)
        float_outputs = detector.run(frame)

        # Decode + NMS
        boxes = decode_all_outputs(float_outputs, detector.model_w)

        # Draw detections
        display = draw_detections(frame, boxes, fps)

        # FPS measurement
        frame_count += 1
        elapsed = time.time() - t_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            t_start = time.time()
            print(f"FPS: {fps:.1f} | Detections: {len(boxes)}")

        # Encode and push to MJPEG server
        _, jpeg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        update_frame(jpeg.tobytes())

    cap.release()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")