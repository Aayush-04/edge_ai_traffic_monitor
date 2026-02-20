"""
camera.py
Camera capture wrapper for See3CAM_CU30 on ZCU104.
"""
import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.model_config import CAMERA_PIPELINE


def open_camera(pipeline=None):
    """
    Open USB camera with GStreamer pipeline.
    Falls back to default device if GStreamer fails.

    Returns: cv2.VideoCapture object (opened).
    Raises: RuntimeError if camera cannot be opened.
    """
    if pipeline is None:
        pipeline = CAMERA_PIPELINE

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Warning: GStreamer pipeline failed, trying default device...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    return cap