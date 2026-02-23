# Edge AI Traffic Monitor — FPGA-Accelerated Object Detection

Real-time YOLOv3 object detection for Indian traffic monitoring on Xilinx ZCU104.

## Platform
- **Board**: Xilinx ZCU104 (XCZU7EV Zynq UltraScale+ MPSoC)
- **DPU**: DPUCZDX8G_ISA1_B4096, 2 cores @ 300 MHz
- **Camera**: See3CAM_CU30 USB3.0
- **OS**: PetaLinux 5.15.36, Vitis AI 3.0

## Performance (Measured)
| Metric | Value |
|---|---|
| DPU inference latency | 18.30 ms (σ = 0.03 ms) |
| DPU raw throughput | 54.6 FPS |
| End-to-end browser stream | 11.6 FPS |
| Board power (idle) | 15.26 W |
| Board power (DPU load) | 21.53 W |

## Repository Structure

edge_ai_traffic_monitor/
│
├── README.md                         
│
├── config/
│   ├── model_config.hpp               
│   └── model_config.py                
│
├── dataset/
│   └── data.yaml                      
│  
│
├── training/
│   ├── train.py                       
│   └── quantize.py                    
│
├── compiled_model/
│   ├── yolov3_quant.prototxt         
│   └── meta.json                    
│   └── yolov3_quant.xmodel 
│
├── deploy/
│   ├── cpp/
│   │   ├── build.sh                  
│   │   ├── common.hpp               
│   │   ├── detect_image.cpp           
│   │   └── test_accuracy.cpp         
│   │
│   └── python/
│       ├── stream_server.py           
│       ├── dpu_runner.py             
│       ├── postprocess.py             
│       ├── camera.py                 
│       ├── mjpeg_server.py            
│       └── drawing.py               
│
└── scripts/
    ├── setup_board.sh                 
    ├── run_stream.sh                  
    ├── run_benchmark.sh             
    └── backup.sh                      

    
## Quick Start (on ZCU104 board)
```bash
# Clone to board
cd ~
git clone https://github.com/YOUR_USERNAME/edge_ai_traffic_monitor.git
cd edge_ai_traffic_monitor

# One-time setup
chmod +x scripts/setup_board.sh
./scripts/setup_board.sh

# Run browser stream
python3 deploy/python/stream_server.py
# Open http://BOARD_IP:8080 in browser
