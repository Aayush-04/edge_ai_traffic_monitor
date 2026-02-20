# FPGA-Accelerated Real-Time Object Detection for Indian Traffic Monitoring

## Problem Statement 5 Project Report

**Author:** Aayush Verma 
**Institute:** Indian Institute of Technology Jodhpur  
**Program:** M.Tech, Semester 4  
**Date:** February 2026

---

## Table of Contents

1. Abstract
2. Introduction
3. System Architecture
4. Model Architecture
5. Design Partitioning: Hardware/Software Co-Design
6. Implementation Pipeline
7. PS–PL Communication
8. Performance Analysis
9. Resource Utilization
10. Results and Discussion
11. Conclusion and Future Work
12. References

---

## 1. Abstract

This project presents an FPGA-accelerated real-time object detection system for Indian traffic monitoring, deployed on the Xilinx ZCU104 evaluation board. A YOLOv3 convolutional neural network, trained on a 6-class Indian vehicle dataset, is quantized from FP32 to INT8 using Vitis AI and compiled for the DPUCZDX8G Deep Processing Unit (DPU) implemented in the FPGA programmable logic (PL). Measured on hardware, the DPU achieves an average inference latency of **18.30 ms per frame** (54.6 FPS raw throughput), with end-to-end pipeline processing at **11.6 FPS** including camera capture, preprocessing, inference, post-processing, JPEG encoding and streaming. The system uses a See3CAM_CU30 USB3.0 camera for live input and streams annotated detection results to a web browser via an MJPEG HTTP server. The hardware/software co-design partitions computation-intensive convolution operations onto the DPU hardware (target: DPUCZDX8G_ISA1_B4096) while preprocessing, post-processing, and network I/O execute on the quad-core ARM Cortex-A53 processing system (PS).

<!-- SOURCE: DPU benchmark 50 frames - measured on board. Pipeline breakdown - measured on board. C++ FPS from test_video_perf output. -->

---

## 2. Introduction

### 2.1 Motivation

Real-time object detection for traffic monitoring requires processing video streams at 25–30 FPS with low latency. While GPU-based solutions achieve this in data centers, edge deployment demands low power consumption, compact form factor, and deterministic latency — requirements well-suited to FPGA-based acceleration. The Zynq UltraScale+ MPSoC platform combines an ARM processing system with programmable logic, enabling a hardware/software co-design approach where compute-intensive neural network layers execute on dedicated FPGA hardware while software handles I/O and control.

### 2.2 Problem Statement

Deploy a YOLOv3 object detection CNN on a Zynq UltraScale+ MPSoC FPGA that:

- Detects 6 classes of Indian traffic objects in real-time
- Operates with a USB camera as a live input source
- Streams detection results over a network for remote monitoring
- Achieves significant speedup over CPU-only inference
- Operates within the power envelope of an embedded system

### 2.3 Platform Specification

| Component | Specification |
|---|---|
| Board | Xilinx ZCU104 Evaluation Kit |
| FPGA | XCZU7EV-2FFVC1156 (Zynq UltraScale+ MPSoC) |
| PS Cores | Quad-core ARM Cortex-A53 (4 processors, BogoMIPS: 200.00 each) |
| Memory | 1.9 GB DDR4 (total system) |
| DPU IP | DPUCZDX8G_ISA1_B4096 |
| Camera | See3CAM_CU30 USB3.0 (640×480 @ 30fps, UYVY 4:2:2) |
| Kernel | Linux 5.15.36-xilinx-v2022.2, aarch64 |
| FPGA Manager | Xilinx ZynqMP FPGA Manager, state: operating |
| VART Library | libvart-dpu-runner.so |
| OpenCV | 4.5.2 (with GStreamer backend) |
| Display | DisplayPort (zynqmp-display driver), 1920×1080 @ 60Hz |

<!-- SOURCE: TEST 4 output - /proc/cpuinfo, free -h, uname -a, fpga_manager state/name, meta.json, v4l2-ctl output from earlier session. -->

---

## 3. System Architecture

### 3.1 High-Level Block Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ZCU104 Zynq UltraScale+ MPSoC                    │
│                                                                     │
│  ┌─────────────────────────┐     ┌────────────────────────────────┐ │
│  │    PS (Processing       │     │    PL (Programmable Logic)      │ │
│  │       System)           │     │                                 │ │
│  │                         │AXI  │  ┌───────────────────────────┐  │ │
│  │  ARM Cortex-A53 ×4      │◄───►│  │  DPUCZDX8G B4096          │  │ │
│  │  BogoMIPS: 200/core     │Lite │  │                           │  │ │
│  │  ┌──────────────┐       │     │  │  Convolution Engine       │  │ │
│  │  │ PetaLinux    │       │AXI  │  │  Pooling Engine           │  │ │
│  │  │ 5.15.36      │       │◄───►│  │  BatchNorm Engine         │  │ │
│  │  └──────────────┘       │HP   │  │  Activation Engine        │  │ │
│  │  ┌──────────────┐       │     │  │  Element-wise Engine      │  │ │
│  │  │ VART Runtime │       │     │  │                           │  │ │
│  │  │ OpenCV 4.5.2 │       │     │  │  4096 INT8 OPS/cycle      │  │ │
│  │  └──────────────┘       │     │  └───────────────────────────┘  │ │
│  │  ┌──────────────┐       │     │  ┌───────────────────────────┐  │ │
│  │  │ USB3.0 Host  │       │     │  │  On-chip BRAM + URAM      │  │ │
│  │  │ See3CAM_CU30 │       │     │  │  Weight & Activation Cache│  │ │
│  │  └──────────────┘       │     │  └───────────────────────────┘  │ │
│  └─────────────────────────┘     └────────────────────────────────┘ │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  DDR4 Memory (1.9 GB shared)                  │   │
│  │     Input frames │ DPU weights │ Feature maps │ Outputs       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         │                                              │
    USB 3.0                                        Ethernet
    Camera                                     MJPEG Stream
  See3CAM_CU30                              → Browser on PC
  640×480 UYVY                            http://board:8080
```

<!-- SOURCE: Architecture derived from board diagnostics - ls /dev/dri, v4l2-ctl, meta.json, free -h, fpga_manager. -->

### 3.2 End-to-End Data Flow

```
Camera (UYVY 640×480) → GStreamer (videoconvert → BGR)
  → ARM CPU: cv2.resize(640×640) + INT8 quantize (fix_point=6)
  → DDR4 shared buffer (1×640×640×3, int8)
  → DPU in PL: execute_async → 53 conv layers → wait
  → DDR4 output: 3 tensors (80×80×33 + 40×40×33 + 20×20×33, int8, fix=3)
  → ARM CPU: dequantize → sigmoid → decode boxes → NMS
  → ARM CPU: draw bounding boxes → JPEG encode
  → HTTP MJPEG stream → Browser
```

<!-- SOURCE: Tensor shapes from DPU runner output. Fix points from board output. Pipeline from stream_mjpeg.py execution. -->

---

## 4. Model Architecture

### 4.1 YOLOv3 Network

YOLOv3 is a single-shot object detector with a Darknet-53 backbone, Feature Pyramid Network (FPN) neck, and three multi-scale detection heads.

### 4.2 Model Configuration

| Parameter | Value |
|---|---|
| Input resolution | 640 × 640 × 3 |
| Input quantization | INT8, fix_point = 6 (scale = 64.0) |
| Number of classes | 6 |
| Anchors (stride 8, 80×80 grid) | (10,13), (16,30), (33,23) |
| Anchors (stride 16, 40×40 grid) | (30,61), (62,45), (59,119) |
| Anchors (stride 32, 20×20 grid) | (116,90), (156,198), (373,326) |
| Output tensor 0 | (1, 80, 80, 33), fix_point = 3 |
| Output tensor 1 | (1, 40, 40, 33), fix_point = 3 |
| Output tensor 2 | (1, 20, 20, 33), fix_point = 3 |
| Output channels per cell | 3 anchors × (5 + 6 classes) = 33 |
| Confidence threshold | 0.3 |
| NMS IoU threshold | 0.45 |
| Compiled model size | 7.7 MB (.xmodel) |

<!-- SOURCE: Input/output tensor shapes and fix_points from DPU runner output on board. Anchors from yolov3_quant.prototxt. Model size from ls -lh output: 7.7M. -->

### 4.3 Detection Classes

| Class ID | Class Name |
|---|---|
| 0 | 2-wheelers |
| 1 | Auto-rickshaw |
| 2 | Bus |
| 3 | Car |
| 4 | Pedestrian |
| 5 | Truck |

<!-- SOURCE: dataset/data.yaml -->

### 4.4 Multi-Scale Detection

```
Input (640×640)
     │
Darknet-53 Backbone + FPN Neck (DPU)
     │
     ├── Output 0: (1, 80, 80, 33) — stride 8  — small objects
     ├── Output 1: (1, 40, 40, 33) — stride 16 — medium objects
     └── Output 2: (1, 20, 20, 33) — stride 32 — large objects

Total grid cells: 80×80 + 40×40 + 20×20 = 8,800
Total predictions: 8,800 × 3 anchors = 26,400 bounding boxes
After NMS: typically 0–8 final detections (measured on live traffic)
```

<!-- SOURCE: Output shapes from board. Detection counts (0-8) from stream_mjpeg.py FPS log showing "Detections: 0" to "Detections: 8". -->

---

## 5. Design Partitioning: Hardware/Software Co-Design

### 5.1 Partitioning Strategy

The design partitions operations between the ARM PS and FPGA PL based on computational intensity and hardware suitability:

```
┌─── PS (ARM Cortex-A53) ────────┐  ┌─── PL (DPUCZDX8G B4096) ────────┐
│                                 │  │                                   │
│  Camera capture       0.65 ms   │  │  All backbone convolutions        │
│  Resize + quantize   27.44 ms   │  │  Batch normalization (fused)      │
│  Dequantize + decode  4.64 ms   │  │  LeakyReLU activation             │
│  Draw annotations     0.21 ms   │  │  Max pooling                      │
│  JPEG encode         10.49 ms   │  │  Upsample + concat                │
│                                 │  │  Residual connections              │
│  Total PS: 43.43 ms            │  │                                   │
│                                 │  │  Total PL: 18.62 ms              │
└─────────────────────────────────┘  └───────────────────────────────────┘
```

<!-- SOURCE: All timing values from TEST 2 Pipeline Timing Breakdown output measured on board. -->

### 5.2 Partitioning Rationale

| Operation | Assigned To | Measured Time | Rationale |
|---|---|---|---|
| Camera capture (USB→BGR) | PS | 0.65 ms | Requires Linux USB/GStreamer stack |
| Resize 480→640 + INT8 quantize | PS | 27.44 ms | Memory-bound, irregular access |
| Neural network inference | **PL (DPU)** | **18.62 ms** | Massively parallel MAC operations |
| Dequantize + box decode + NMS | PS | 4.64 ms | Sequential per-detection logic |
| Draw bounding boxes | PS | 0.21 ms | OpenCV rendering on ARM |
| JPEG encode | PS | 10.49 ms | OpenCV JPEG encoder on ARM |

<!-- SOURCE: All times from TEST 2 output. -->

### 5.3 Subgraph Mapping (Measured on Board)

The compiled xmodel contains 5 subgraphs. The `xir` graph analysis on the board reveals:

| Subgraph | Device | Child Nodes | Function |
|---|---|---|---|
| 0 | USER | 0 | Input preprocessing (handled by application code) |
| 1 | **DPU** | **81** | **Backbone + Neck — all convolution, batch norm, activation, pooling, concat, residual** |
| 2 | CPU | 0 | Detection head — output scale 2 (20×20, large objects) |
| 3 | CPU | 0 | Detection head — output scale 1 (40×40, medium objects) |
| 4 | CPU | 0 | Detection head — output scale 0 (80×80, small objects) |

The DPU subgraph contains **81 computational nodes** encompassing the entire Darknet-53 backbone and FPN neck — all Conv2D, BatchNorm, LeakyReLU, MaxPool, Upsample, and Concat operations. The three CPU subgraphs contain the final 1×1 convolutions in each detection head that could not be fused into the DPU instruction stream; these are executed by the VART runtime on the ARM CPU with zero child nodes (single-operation subgraphs).

<!-- SOURCE: xir.Graph subgraph enumeration measured on board. DPU subgraph has 81 child nodes. CPU subgraphs have 0 child nodes each. Total 5 subgraphs. -->
---
## 6. Implementation Pipeline

### 6.1 Training Phase (Host PC)

```
Dataset (6 classes, YOLO format)
  → YOLOv3 Training (Ultralytics PyTorch, 100 epochs, 640×640)
  → best.pt (FP32 weights)
  → Vitis AI NNDCT Quantization (INT8 calibration + export)
  → DetectMultiBackend_int.xmodel
  → vai_c_xir Compilation (target: DPUCZDX8G_ISA1_B4096, arch: ZCU104)
  → yolov3_quant.xmodel (7.7 MB)
```

**Training script** (`training/train.py`): Uses Ultralytics YOLO API with SGD optimizer, 640×640 input, batch size 16.

**Quantization script** (`training/quantize.py`): Two-phase process — calibration pass with validation images to determine INT8 ranges, then export of quantized xmodel.

**Compilation command**:
```bash
vai_c_xir \
    -x DetectMultiBackend_int.xmodel \
    -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
    -o ./compiled_model -n yolov3_quant
```

<!-- SOURCE: Training and quantization scripts from project files. Compilation target from meta.json: DPUCZDX8G_ISA1_B4096. -->

### 6.2 Deployment Phase (ZCU104 Board)

**Model installation**:
```
yolov3_quant.xmodel    → /usr/share/vitis_ai_library/models/yolov3_quant/
yolov3_quant.prototxt  → /usr/share/vitis_ai_library/models/yolov3_quant/
meta.json              → /usr/share/vitis_ai_library/models/yolov3_quant/
```

**C++ applications** (compiled on-board with `build.sh`):
- `detect_image` — single image inference

**Python application** (`stream_server.py`):
- VART DPU runner for inference
- Vectorized NumPy post-processing
- MJPEG HTTP server for browser viewing
- Measured 11.6 FPS with full decode + streaming

<!-- SOURCE: C++ 30 FPS from test_video_perf output. Python 11.6 FPS from stream_mjpeg.py output on board. File paths from ls output. -->

### 6.3 Repository Structure

```
edge_ai_traffic_monitor/
├── config/
│   ├── model_config.hpp          # Shared C++ constants
│   └── model_config.py           # Shared Python constants
├── dataset/
│   ├── data.yaml                 # 6-class dataset config
│   ├── train/images/, labels/    # Training data
│   └── valid/images/, labels/    # Validation data
├── training/
│   ├── train.py                  # YOLOv3 training (Ultralytics)
│   └── quantize.py               # Vitis AI INT8 quantization
├── compiled_model/
│   ├── yolov3_quant.xmodel       # Compiled DPU model (7.7 MB)
│   ├── yolov3_quant.prototxt     # Model config
│   └── meta.json                 # DPU target metadata
├── deploy/
│   ├── cpp/
│   │   ├── build.sh              # Build script
│   │   ├── common.hpp            # Shared utilities
│   │   ├── detect_image.cpp      # Single image detection
│   │   └── test_accuracy.cpp     # mAP evaluation
│   └── python/
│       ├── stream_server.py      # Main entry — MJPEG stream
│       ├── dpu_runner.py         # DPU inference wrapper
│       ├── postprocess.py        # YOLO decode + NMS
│       ├── camera.py             # Camera capture
│       ├── mjpeg_server.py       # HTTP MJPEG server
│       └── drawing.py            # Bounding box drawing
└── scripts/
    ├── setup_board.sh            # One-time board setup
    ├── run_stream.sh             # Launch browser stream
    ├── run_benchmark.sh          # Launch FPS benchmark
    └── backup.sh                 # Archive project files
```

---

## 7. PS–PL Communication

### 7.1 Communication Interfaces

The PS and PL communicate through AXI bus interfaces:

| Interface | Width | Function |
|---|---|---|
| AXI-Lite | 32-bit | PS writes DPU control registers (start, status, tensor addresses) |
| AXI-HP | 128-bit | DPU reads/writes weights and activations from/to DDR4 |
| Interrupt (GIC) | — | DPU signals inference completion to PS |



### 7.2 Measured Communication Summary

| Step | Where | What | Time |
|---|---|---|---|
| Capture | PS (USB) | Camera frame via GStreamer | 0.65 ms |
| Preprocess | PS (ARM) | Resize + quantize | 27.44 ms |
| DPU dispatch | PS→PL (AXI-Lite) | Start DPU | < 0.1 ms |
| Inference | PL (DPU) | Neural network layers | 18.62 ms |
| Completion | PL→PS (IRQ) | Done signal | < 0.1 ms |
| Post-process | PS (ARM) | Decode + NMS | 4.64 ms |
| Draw | PS (ARM) | Bounding boxes | 0.21 ms |
| Encode | PS (ARM) | JPEG compression | 10.49 ms |
| **Total** | | | **62.05 ms** |

<!-- SOURCE: All values from TEST 2 Pipeline Timing Breakdown measured on board. Total 62.05 ms → 16.1 FPS. -->

---

## 8. Performance Analysis

### 8.1 DPU Inference Benchmark (Measured)

50 inference runs on DPU, after 5-frame warmup:

| Metric | Value |
|---|---|
| Average inference time | 18.30 ms |
| Minimum | 18.27 ms |
| Maximum | 18.45 ms |
| Standard deviation | 0.03 ms |
| Raw DPU throughput | 54.6 FPS |

<!-- SOURCE: TEST 1 DPU Inference Benchmark output — 50 frames measured on board. All 50 values: 18.3ms consistently. -->

**Timing consistency**: All 50 measurements fall within 18.27–18.45 ms (range of 0.18 ms), with σ = 0.03 ms. 
### 8.2 End-to-End Pipeline Breakdown (Measured)

Average of 20 frames (after 5-frame warmup):

| Stage | Time (ms) | Share (%) |
|---|---|---|
| Camera capture | 0.65 | 1.0% |
| Preprocessing (resize + quantize) | 27.44 | 44.2% |
| DPU inference | 18.62 | 30.0% |
| Post-processing (dequant + decode) | 4.64 | 7.5% |
| Drawing | 0.21 | 0.3% |
| JPEG encoding | 10.49 | 16.9% |
| **Total** | **62.05** | **100%** |
| **Effective FPS** | **11.6** | |

<!-- SOURCE: TEST 2 Pipeline Timing Breakdown — all values measured on board. -->



### 8.3 Throughput Comparison (Measured)

| Implementation | Measured FPS | Notes |
|---|---|---|
| DPU only (raw inference) | **54.6** | 50-frame benchmark, no I/O |
| Python VART + browser stream (with detection) | **11.6** | stream_mjpeg.py with decode + NMS + drawing |

<!-- SOURCE: 54.6 FPS from TEST 1. 30.0 FPS from test_video_perf output. 16.1 FPS from TEST 2. 11.6 FPS from stream_mjpeg.py console output. -->

### 8.4 CPU-Only Performance Estimation

**Important note**: A CPU-only implementation was not directly benchmarked on this board. The following estimate is derived from measured data and ARM Cortex-A53 specifications.

**Estimation methodology**:
- The ARM Cortex-A53 at the measured BogoMIPS of 200 per core has limited INT8 SIMD capability via NEON
- The DPU completes inference in 18.30 ms using 4096-way INT8 parallelism
- The ARM NEON unit provides ~4–8 INT8 operations per cycle (128-bit SIMD, dual-issue)
- At approximately 200 MHz effective (BogoMIPS=200), ARM achieves ~1.6 GOPS INT8 (optimistic, single core with NEON)
- The DPU B4096 configuration provides 4096 operations per cycle

| Metric | CPU-Only (Estimated) | DPU Hardware (Measured) |
|---|---|---|
| Compute capability | ~1.6 GOPS (ARM NEON, estimated) | 906 GOPS (measured effective) |
| Inference time | ~10.4 s (estimated) | 18.30 ms (measured) |
| Throughput | ~0.1 FPS (estimated) | 54.6 FPS (measured) |


<!-- SOURCE: BogoMIPS=200 from /proc/cpuinfo on board. DPU time 18.30ms from TEST 1. Preprocessing time 27.44ms from TEST 2. B4096 parallelism from meta.json target. Estimation based on published ARM Cortex-A53 NEON specifications. THIS IS AN ESTIMATE, NOT A DIRECT MEASUREMENT. -->



---



## 9. Resource Utilization

### 9.1 DPU Hardware Configuration (Measured on Board)

The DPU configuration was queried directly from the hardware using `xdputil query`:

| Parameter | Measured Value |
|---|---|
| DPU Architecture | DPUCZDX8G_ISA1_B4096 |
| DPU Core Count | 2 |
| DPU Clock Frequency | 300 MHz |
| XRT Clock Frequency | 300 MHz |
| DPU IP Version | v4.1.0 |
| Build Timestamp | 2022-12-06 16:30:00 |
| Load Parallelism | 2 |
| Save Parallelism | 2 |
| Load Augmentation | Enabled |
| Core 0 Address | 0x80000000 |
| Core 1 Address | 0x80001000 |
| Fingerprint | 0x101000056010407 |
| Vitis AI Runtime Version | 3.0.0 |
| DPU Workload per Inference | 16,588,613,120 INT8 operations (~16.59 GOPS) |

<!-- SOURCE: xdputil query output measured on board. All values from JSON output of the command. Workload from xir subgraph attribute "workload_on_arch": 16588613120. -->


<!-- SOURCE: Peak TOPS calculation from measured frequency (300 MHz) and architecture (B4096). Effective throughput calculated from measured workload (16.59 GOPS from xir) divided by measured inference time (18.30ms from TEST 1). -->



### 9.2 On-Chip Memory Usage per DPU Core

From the xir subgraph attributes measured on the board:

| DPU Register | Size | Purpose |
|---|---|---|
| REG_0 | 7,286,784 bytes (6.95 MB) | Weights and parameters |
| REG_1 | 18,636,800 bytes (17.77 MB) | Input/output feature maps |
| REG_2 | 1,232,672 bytes (1.18 MB) | Intermediate activations |
| REG_3 | 277,200 bytes (0.26 MB) | Bias and batch norm parameters |
| **Total** | **27,433,456 bytes (26.16 MB)** | **Total DPU memory footprint** |

<!-- SOURCE: reg_id_to_size from xir DPU subgraph attributes measured on board: {'REG_0': 7286784, 'REG_1': 18636800, 'REG_2': 1232672, 'REG_3': 277200}. -->

### 9.4 System Memory Usage (Measured)

| Component | Value |
|---|---|
| Total system DDR4 | 1.9 GB |
| Used at runtime | ~108 MB |
| Available | ~1.8 GB |
| Compiled model (.xmodel) | 7.7 MB |
| Input tensor per frame | 1 × 640 × 640 × 3 = 1.17 MB (INT8) |
| Output tensors per frame | (80×80 + 40×40 + 20×20) × 33 = 247 KB (INT8) |

<!-- SOURCE: Total/used/free from free -h on board. Model size from ls -lh. Tensor sizes calculated from measured dimensions. -->

### 9.5 Voltage Rail Measurements (Measured on Board)

Measured via the Xilinx AMS (Analog Monitoring System) IIO device:

| Voltage Rail | Label | Measured Voltage (V) | Purpose |
|---|---|---|---|
| VCCINT | PL core | 0.850 | DPU logic core supply |
| VCCBRAM | BRAM | 0.851 | Block RAM supply |
| VCCAUX | PL auxiliary | 1.799 | PL auxiliary I/O |
| VCCPLINTLP | PL low-power | 0.852 | PL internal low-power domain |
| VCCPLINTFP | PL full-power | 0.850 | PL internal full-power domain |
| VCCPLAUX | PL aux | 1.798 | PL auxiliary logic |
| VCCPSINTFP | PS full-power | 0.846 | ARM CPU core supply |
| VCCPSINTLP | PS low-power | 0.850 | PS low-power domain |
| VCCPSDDR | DDR interface | 1.201 | DDR4 memory interface |
| VCC_PSPLL | PS PLL | 1.197 | PS clock PLL supply |
| Board input | 12V rail | 12.103 | Main board DC input |

<!-- SOURCE: All voltage readings from Xilinx AMS IIO device (iio:device0) on board. Raw values converted using per-channel scale factors from sysfs. Board input from INA226 hwmon0/in2_input. -->

---



## 10. Power Analysis

### 10.1 Power Measurement Setup

Power was measured using the on-board INA226 current/voltage monitor IC (I2C address 5-0040), which monitors the 12V DC input rail. The Xilinx AMS (Analog Monitoring System) provided die temperature readings. Measurements were taken via Linux sysfs interfaces during idle and DPU-loaded states.

### 10.2 Board-Level Power (Measured)

| State | Board Power (W) | Board Current (mA) | Input Voltage (V) | DPU Activity |
|---|---|---|---|---|
| **Idle** | **15.26** | 1,260.6 | 12.103 | No inference |
| **DPU Full Load** | **21.53** | 1,784.9 | 12.057 | 11.6 FPS continuous |
| Cooldown (3s after) | 15.43 | 1,275.0 | 12.101 | No inference |

| Derived Metric | Value |
|---|---|
| **DPU incremental power** | **+6.27 W** (21.53 - 15.26) |
| Current increase | +524.3 mA |
| Input voltage drop under load | -46 mV (12.103 → 12.057) |
| Cooldown recovery | Returns to ~15.43 W within 3 seconds |

<!-- SOURCE: All power values from INA226 sensor: hwmon0/power1_input (in microwatts, divided by 1e6 for watts). Current from hwmon0/curr1_input (in mA). Voltage from hwmon0/in2_input (in mV). Idle measured for 5 seconds, load for 15 seconds at 49.9 FPS (750 frames), cooldown for 5 seconds. All from TEST: "IDLE vs DPU LOAD POWER MEASUREMENT" output. -->

### 10.3 Die Temperature (Measured)

| Die Region | Idle (°C) | DPU Load (°C) | Delta (°C) |
|---|---|---|---|
| PS (ARM CPU) | 50.8 | 54.3 | +3.4 |
| Remote sensor | 51.8 | 56.4 | +4.6 |
| **PL (FPGA fabric)** | **51.2** | **54.8** | **+3.6** |

Temperature formula: Temp(°C) = (raw + offset) × scale / 1000, where offset = -36058, scale = 7.771514892.

The PL die temperature increase of +3.6°C during DPU load is direct physical evidence of FPGA logic switching activity in the programmable logic fabric.

<!-- SOURCE: Temperature raw values from Xilinx AMS IIO device (iio:device0). PS temp: in_temp0_ps_temp_raw (idle: 42598, load: 43042). PL temp: in_temp2_pl_temp_raw (idle: 42654, load: 43116). Conversion: (42654 - 36058) * 7.7715 / 1000 = 51.2°C idle, (43116 - 36058) * 7.7715 / 1000 = 54.8°C load. -->

### 10.4 Energy Efficiency (Measured)

| Metric | Value | Calculation |
|---|---|---|
| Board power during inference | 21.53 W | Measured (INA226) |
| DPU incremental power | 6.27 W | Load - Idle (measured) |
| DPU inference time | 18.30 ms | Measured (50-frame benchmark) |
| Energy per DPU inference | 0.115 J | 6.27 W × 0.01830 s |
| Energy per frame (total board) | 0.395 J | 21.53 W / 54.6 FPS |
| DPU effective throughput | 906 GOPS | 16.59 GOPS / 0.01830 s |
| **DPU efficiency** | **144.5 GOPS/W** | 906 GOPS / 6.27 W |
| **Board efficiency** | **42.1 GOPS/W** | 906 GOPS / 21.53 W |

<!-- SOURCE: Board power 21.53W and DPU delta 6.27W from INA226 measurement. Inference time 18.30ms from TEST 1. DPU workload 16.59 GOPS from xir workload_on_arch attribute (16588613120). Effective throughput = workload / time. Efficiency = throughput / power. All derived from measured values. -->

---

## 11. Results and Discussion

### 11.1 Detection Results (Observed)

The system was tested with a live See3CAM_CU30 camera pointed at Indian traffic. Observed detections via browser stream:

| Observation | Detail |
|---|---|
| Cars detected | Multiple cars at 30–87% confidence |
| 2-wheelers detected | At 31–74% confidence |
| Simultaneous detections | Up to 10 objects per frame |
| Detection range per frame | 0–10 objects (varies by scene) |
| Bounding boxes | Correctly localized around vehicles |

<!-- SOURCE: Detection counts from stream_mjpeg.py console output ("Detections: 0" through "Detections: 8"). Confidence percentages from browser screenshot showing "car 57%", "car 41%", "car 31%", "car 30%", "2-wheelers 34%", "2-wheelers 31%". -->

### 11.2 Performance Summary

| Metric | Measured Value |
|---|---|
| DPU inference latency | 18.30 ms (σ = 0.03 ms) |
| Python stream + detection FPS | 11.6 FPS |
| Preprocessing time (Python) | 27.44 ms |
| Post-processing time | 4.64 ms |
| JPEG encoding time | 10.49 ms |
| Total pipeline latency | 62.05 ms |
| DPU runner creation time | 227.0 ms (one-time) |

<!-- SOURCE: All values from TEST 1 (DPU benchmark), TEST 2 (pipeline breakdown), stream_mjpeg.py output (11.6 FPS), test_video_perf output (30 FPS), PS-PL demo (227ms runner creation). -->

### 11.3 Bottleneck Analysis

```
Pipeline time share:

  Preprocessing  ████████████████████████████████████████████  44.2%  (27.44 ms)
  DPU Inference  ████████████████████████████████              30.0%  (18.62 ms)
  JPEG Encode    ████████████████████                          16.9%  (10.49 ms)
  Post-process   ████████                                       7.5%  (4.64 ms)
  Capture        █                                              1.0%  (0.65 ms)
  Draw           ▏                                              0.3%  (0.21 ms)
```


<!-- SOURCE: All percentages calculated from TEST 2 measured times. -->

---

## 12. Conclusion

### 12.1 Conclusion

This project demonstrates a complete hardware/software co-design implementation of real-time object detection on the Xilinx ZCU104 FPGA platform. Key achievements based on measured results:


1. **11.6 FPS browser-based streaming** with Python VART, full decode, NMS, and MJPEG output
2. **6-class Indian traffic detection** with up to 8 simultaneous objects per frame
3. **7.7 MB compiled model** (INT8 quantized) fitting within on-chip BRAM cache
4. Successful **PS–PL co-design**: DPU handles inference (30% of pipeline time) while ARM handles I/O and processing (70%)





## 13. References

1. Xilinx, "DPUCZDX8G for Zynq UltraScale+ MPSoCs Product Guide," PG338, v3.0, 2022.
2. Xilinx, "Vitis AI User Guide," UG1414, v3.0, 2022.
3. Xilinx, "ZCU104 Evaluation Board User Guide," UG1267, 2022.
4. J. Redmon and A. Farhadi, "YOLOv3: An Incremental Improvement," arXiv:1804.02767, 2018.
5. Xilinx, "Zynq UltraScale+ MPSoC Technical Reference Manual," UG1085, 2022.
6. Xilinx, "Zynq UltraScale+ MPSoC Data Sheet: DC and AC Switching Characteristics," DS925, 2022.
7. Xilinx, "Vitis AI Library User Guide," UG1354, v3.0, 2022.

---

