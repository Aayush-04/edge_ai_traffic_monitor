"""
postprocess.py
YOLOv3 output decoding and NMS.
"""
import numpy as np
import sys
import os

# Import config from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.model_config import (
    ANCHORS_BY_GRID, STRIDES_BY_GRID, NUM_CLASSES,
    CONF_THRESHOLD, NMS_THRESHOLD
)


def sigmoid(x):
    """Numerically stable sigmoid. Preserves original clipping range."""
    return 1.0 / (1.0 + np.exp(np.clip(-x, -50, 50)))


def decode_single_scale(output, input_size, conf_thresh=CONF_THRESHOLD):

    if len(output.shape) == 4:
        output = output[0]
    gh, gw, _ = output.shape

    anchors = ANCHORS_BY_GRID.get(gh)
    stride = STRIDES_BY_GRID.get(gh)
    if anchors is None or stride is None:
        return np.zeros((0, 6), dtype=np.float32)

    # Reshape: (grid_h, grid_w, 3, 5 + num_classes)
    out = output.reshape(gh, gw, 3, 5 + NUM_CLASSES)

    # Objectness and class scores — vectorized
    obj = sigmoid(out[..., 4])
    cls = sigmoid(out[..., 5:5 + NUM_CLASSES])
    conf = obj * cls.max(axis=-1)

    # Early filter — skip cells below threshold
    mask = conf > conf_thresh
    if not mask.any():
        return np.zeros((0, 6), dtype=np.float32)

    yi, xi, ai = np.where(mask)

    # Extract raw predictions for surviving cells only
    tx = out[yi, xi, ai, 0]
    ty = out[yi, xi, ai, 1]
    tw = out[yi, xi, ai, 2]
    th = out[yi, xi, ai, 3]
    cf = conf[yi, xi, ai]
    cid = cls[yi, xi, ai].argmax(axis=-1).astype(np.float32)

    # Decode coordinates — preserves original formula exactly
    bx = (sigmoid(tx) + xi.astype(np.float32)) * stride
    by = (sigmoid(ty) + yi.astype(np.float32)) * stride
    bw = np.exp(np.clip(tw, -10, 10)) * anchors[ai, 0]
    bh = np.exp(np.clip(th, -10, 10)) * anchors[ai, 1]

    s = float(input_size)
    x1 = (bx - bw / 2) / s
    y1 = (by - bh / 2) / s
    x2 = (bx + bw / 2) / s
    y2 = (by + bh / 2) / s

    return np.stack([x1, y1, x2, y2, cf, cid], axis=-1)


def nms(boxes, nms_thresh=NMS_THRESHOLD):

    if len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    scores = boxes[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)

        order = order[np.where(iou <= nms_thresh)[0] + 1]

    return boxes[keep]


def decode_all_outputs(float_outputs, input_size, conf_thresh=CONF_THRESHOLD,
                       nms_thresh=NMS_THRESHOLD):

    all_boxes = []
    for output in float_outputs:
        boxes = decode_single_scale(output, input_size, conf_thresh)
        if len(boxes) > 0:
            all_boxes.append(boxes)

    if not all_boxes:
        return np.zeros((0, 6), dtype=np.float32)

    return nms(np.concatenate(all_boxes), nms_thresh)