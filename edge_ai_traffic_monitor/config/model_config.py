"""
model_config.py
"""

# Detection classes 
CLASS_NAMES = ['2-wheelers', 'auto', 'bus', 'car', 'pedestrian', 'truck']
NUM_CLASSES = len(CLASS_NAMES)

# Detection thresholds
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.45

# Model input dimensions
MODEL_INPUT_W = 640
MODEL_INPUT_H = 640


import numpy as np

ANCHORS_BY_GRID = {
    80: np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32),   # stride 8
    40: np.array([[30, 61], [62, 45], [59, 119]], dtype=np.float32),  # stride 16
    20: np.array([[116, 90], [156, 198], [373, 326]], dtype=np.float32),  # stride 32
}

STRIDES_BY_GRID = {80: 8, 40: 16, 20: 32}

# Colors for bounding boxes (BGR for OpenCV)
COLORS_BGR = [
    (0, 255, 0),      # green      - 2-wheelers
    (255, 128, 0),     # orange     - auto
    (0, 0, 255),       # red        - bus
    (255, 255, 0),     # cyan       - car
    (255, 0, 255),     # magenta    - pedestrian
    (0, 255, 255),     # yellow     - truck
]

# Camera GStreamer pipeline
CAMERA_PIPELINE = (
    "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, "
    "framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink"
)

# Default model path on board
DEFAULT_XMODEL_PATH = "/usr/share/vitis_ai_library/models/yolov3_quant/yolov3_quant.xmodel"

# HTTP stream port
STREAM_PORT = 8080