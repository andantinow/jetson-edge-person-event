#!/usr/bin/env python3
import numpy as np


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def nms_numpy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float, topk: int = 300):
    if boxes_xyxy.size == 0:
        return np.array([], dtype=np.int32)

    order = scores.argsort()[::-1]
    if topk is not None:
        order = order[:topk]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[rest, 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[rest, 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[rest, 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[rest, 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        area_i = (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
        area_r = (boxes_xyxy[rest, 2] - boxes_xyxy[rest, 0]) * (boxes_xyxy[rest, 3] - boxes_xyxy[rest, 1])
        union = area_i + area_r - inter + 1e-9
        iou = inter / union

        order = rest[iou <= iou_thresh]

    return np.array(keep, dtype=np.int32)


def yolo_v8_decode_person(
    out0: np.ndarray,
    score_thresh: float,
    iou_thresh: float,
    imgsz: int = 640,
    class_id: int = 0,   # COCO: person=0
    max_dets: int = 50,
):
    """
    out0: (1,84,8400) float
      0:4    -> box (cx,cy,w,h) in input pixel space (imgsz)
      4:84   -> class scores (80 classes), already sigmoid-combined in export

    returns list of (box_xyxy_px, score) in input pixel coordinates (0..imgsz)
    """
    if out0.ndim != 3 or out0.shape[0] != 1 or out0.shape[1] != 84:
        raise ValueError(f"unexpected out0 shape: {out0.shape} (expected (1,84,8400))")

    pred = out0[0].transpose(1, 0)  # (8400,84)
    boxes = pred[:, 0:4]            # cx,cy,w,h
    cls = pred[:, 4:]               # (8400,80)

    scores = cls[:, class_id]
    idx = np.where(scores >= score_thresh)[0]
    if idx.size == 0:
        return []

    b = boxes[idx].astype(np.float32, copy=False)
    s = scores[idx].astype(np.float32, copy=False)

    cx = b[:, 0]
    cy = b[:, 1]
    w = b[:, 2]
    h = b[:, 3]

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    det = np.stack([x1, y1, x2, y2], axis=1)
    det[:, 0] = np.clip(det[:, 0], 0, imgsz)
    det[:, 1] = np.clip(det[:, 1], 0, imgsz)
    det[:, 2] = np.clip(det[:, 2], 0, imgsz)
    det[:, 3] = np.clip(det[:, 3], 0, imgsz)

    keep = nms_numpy(det, s, iou_thresh=iou_thresh, topk=300)
    det = det[keep]
    s = s[keep]

    # sort by score
    order = s.argsort()[::-1]
    det = det[order]
    s = s[order]

    if det.shape[0] > max_dets:
        det = det[:max_dets]
        s = s[:max_dets]

    return [(det[i], float(s[i])) for i in range(det.shape[0])]

