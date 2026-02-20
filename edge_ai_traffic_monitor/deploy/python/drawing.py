"""
drawing.py
"""
import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.model_config import CLASS_NAMES, NUM_CLASSES, COLORS_BGR


def draw_detections(frame, boxes, fps=0):

    h, w = frame.shape[:2]

    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        px1 = max(0, int(x1 * w))
        py1 = max(0, int(y1 * h))
        px2 = min(w, int(x2 * w))
        py2 = min(h, int(y2 * h))

        ci = int(cls_id) % NUM_CLASSES
        color = COLORS_BGR[ci]
        label = f"{CLASS_NAMES[ci]} {int(conf * 100)}%"

        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (px1, py1 - th - 8), (px1 + tw + 4, py1), color, -1)
        cv2.putText(frame, label, (px1 + 2, py1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # FPS overlay
    cv2.putText(frame, f"FPS: {int(fps)} | Objects: {len(boxes)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame