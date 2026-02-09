#!/usr/bin/env python3
import os
import time
import math
import json
import signal
import argparse
import subprocess
from dataclasses import dataclass
from io import BytesIO
from collections import deque
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


# -------------------------
# basic utils
# -------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_ms():
    return time.perf_counter() * 1000.0

def ts_utc_iso(ts: float) -> str:
    # UTC ISO with milliseconds
    # (avoid timezone module dependency; keep simple)
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

def atomic_save_jpg(img: Image.Image, out_path: str, quality: int = 85):
    tmp = out_path + ".tmp"
    img.save(tmp, format="JPEG", quality=quality)
    os.replace(tmp, out_path)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

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

def pct(arr, p):
    if not arr:
        return None
    a = np.array(arr, dtype=np.float32)
    return float(np.percentile(a, p))

def stats_line(name, arr):
    if not arr:
        return f"{name}: n=0"
    a = np.array(arr, dtype=np.float32)
    return (f"{name}: n={len(a)} "
            f"p50={np.percentile(a,50):.1f} "
            f"p90={np.percentile(a,90):.1f} "
            f"p99={np.percentile(a,99):.1f} "
            f"max={np.max(a):.1f}")


# -------------------------
# MJPEG pipe reader (ffmpeg image2pipe mjpeg)
# -------------------------

class MJPEGStreamReader:
    def __init__(self, cmd):
        self.cmd = cmd
        self.proc = None
        self.buf = bytearray()

    def start(self):
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )
        if self.proc.stdout is None:
            raise RuntimeError("ffmpeg stdout is None")

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=1.0)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None

    def read_frame(self, timeout_s=2.0) -> Image.Image:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("ffmpeg not started")

        t0 = time.perf_counter()
        soi = b"\xff\xd8"
        eoi = b"\xff\xd9"

        while True:
            s = self.buf.find(soi)
            if s != -1:
                e = self.buf.find(eoi, s + 2)
                if e != -1:
                    jpg = self.buf[s:e + 2]
                    del self.buf[:e + 2]
                    try:
                        im = Image.open(BytesIO(jpg))
                        im.load()
                        return im.convert("RGB")
                    except Exception:
                        continue

            if time.perf_counter() - t0 > timeout_s:
                raise TimeoutError("MJPEG read timeout")

            chunk = self.proc.stdout.read(4096)
            if not chunk:
                raise RuntimeError("ffmpeg stdout ended")
            self.buf.extend(chunk)


# -------------------------
# TensorRT infer (keep as-is: verified in your env)
# -------------------------

class TRTInfer:
    def __init__(self, engine_path: str, input_name: str, boxes_name: str, scores_name: str):
        self.engine_path = engine_path
        self.input_name = input_name
        self.boxes_name = boxes_name
        self.scores_name = scores_name

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            plan = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.stream = cuda.Stream()
        self._alloc()

    def _alloc(self):
        in_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        boxes_shape = tuple(self.engine.get_tensor_shape(self.boxes_name))
        scores_shape = tuple(self.engine.get_tensor_shape(self.scores_name))

        self.in_shape = in_shape
        self.boxes_shape = boxes_shape
        self.scores_shape = scores_shape

        def vol(shape):
            v = 1
            for s in shape:
                v *= int(s)
            return v

        in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        boxes_dtype = trt.nptype(self.engine.get_tensor_dtype(self.boxes_name))
        scores_dtype = trt.nptype(self.engine.get_tensor_dtype(self.scores_name))

        self.in_host = np.empty(vol(in_shape), dtype=in_dtype).reshape(in_shape)
        self.boxes_host = np.empty(vol(boxes_shape), dtype=boxes_dtype).reshape(boxes_shape)
        self.scores_host = np.empty(vol(scores_shape), dtype=scores_dtype).reshape(scores_shape)

        self.in_dev = cuda.mem_alloc(self.in_host.nbytes)
        self.boxes_dev = cuda.mem_alloc(self.boxes_host.nbytes)
        self.scores_dev = cuda.mem_alloc(self.scores_host.nbytes)

        self.context.set_tensor_address(self.input_name, int(self.in_dev))
        self.context.set_tensor_address(self.boxes_name, int(self.boxes_dev))
        self.context.set_tensor_address(self.scores_name, int(self.scores_dev))

    def infer(self, inp: np.ndarray):
        if inp.shape != self.in_shape:
            raise ValueError(f"bad input shape: {inp.shape} expected {self.in_shape}")

        if inp.dtype != self.in_host.dtype:
            inp = np.ascontiguousarray(inp, dtype=self.in_host.dtype)
        if not inp.flags["C_CONTIGUOUS"]:
            inp = np.ascontiguousarray(inp)

        t0 = now_ms()
        cuda.memcpy_htod_async(self.in_dev, inp, self.stream)
        t1 = now_ms()

        ok = self.context.execute_async_v3(stream_handle=int(self.stream.handle))
        if not ok:
            raise RuntimeError("execute_async_v3 failed")

        cuda.memcpy_dtoh_async(self.boxes_host, self.boxes_dev, self.stream)
        cuda.memcpy_dtoh_async(self.scores_host, self.scores_dev, self.stream)
        self.stream.synchronize()
        t2 = now_ms()

        return self.boxes_host.copy(), self.scores_host.copy(), (t1 - t0), (t2 - t1)


# -------------------------
# SSD priors / decode / nms (keep as-is)
# -------------------------

def generate_ssd_priors_1917():
    fm_sizes = [19, 10, 5, 3, 2, 1]
    min_scale = 0.2
    max_scale = 0.95
    num_layers = len(fm_sizes)

    scales = [min_scale + (max_scale - min_scale) * k / (num_layers - 1) for k in range(num_layers)]
    scales.append(1.0)

    priors = []
    for k, fm in enumerate(fm_sizes):
        s_k = scales[k]
        s_k1 = scales[k + 1]
        s_prime = math.sqrt(s_k * s_k1)

        if k == 0:
            aspect_ratios = [1.0, 2.0, 0.5]
            add_interpolated = False
        else:
            aspect_ratios = [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]
            add_interpolated = True

        for i in range(fm):
            for j in range(fm):
                cx = (j + 0.5) / fm
                cy = (i + 0.5) / fm

                priors.append([cy, cx, s_k, s_k])
                if add_interpolated:
                    priors.append([cy, cx, s_prime, s_prime])

                for ar in aspect_ratios:
                    if abs(ar - 1.0) < 1e-9:
                        continue
                    w = s_k * math.sqrt(ar)
                    h = s_k / math.sqrt(ar)
                    priors.append([cy, cx, h, w])

    priors = np.array(priors, dtype=np.float32)
    if priors.shape[0] != 1917:
        raise RuntimeError(f"priors count mismatch: {priors.shape[0]} (expected 1917)")

    priors[:, 0:2] = np.clip(priors[:, 0:2], 0.0, 1.0)
    priors[:, 2:4] = np.clip(priors[:, 2:4], 1e-6, 2.0)
    return priors  # (1917,4) [cy,cx,h,w]

def decode_boxes(loc: np.ndarray, priors: np.ndarray, variances=(0.1, 0.1, 0.2, 0.2), loc_order="tytxthtw"):
    if loc_order == "tytxthtw":
        ty, tx, th, tw = loc[:, 0], loc[:, 1], loc[:, 2], loc[:, 3]
    elif loc_order == "txtytwth":
        tx, ty, tw, th = loc[:, 0], loc[:, 1], loc[:, 2], loc[:, 3]
    else:
        raise ValueError("loc_order must be tytxthtw or txtytwth")

    cy = ty * variances[0] * priors[:, 2] + priors[:, 0]
    cx = tx * variances[1] * priors[:, 3] + priors[:, 1]
    h = np.exp(th * variances[2]) * priors[:, 2]
    w = np.exp(tw * variances[3]) * priors[:, 3]

    y1 = cy - 0.5 * h
    x1 = cx - 0.5 * w
    y2 = cy + 0.5 * h
    x2 = cx + 0.5 * w

    boxes = np.stack([x1, y1, x2, y2], axis=1)  # xyxy
    return np.clip(boxes, 0.0, 1.0)

def nms_numpy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float, topk: int = 200):
    if boxes_xyxy.size == 0:
        return np.array([], dtype=np.int32)

    order = scores.argsort()[::-1]
    if topk is not None:
        order = order[:topk]

    keep = []
    while order.size > 0:
        i = order[0]
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


# -------------------------
# preprocess (keep as-is)
# -------------------------

def preprocess_pil(im_rgb: Image.Image, mode: str, input_size=300, dtype=np.float32, skip_resize_if_already=True):
    if skip_resize_if_already and im_rgb.size == (input_size, input_size):
        im = im_rgb
    else:
        im = im_rgb.resize((input_size, input_size), Image.BILINEAR)

    arr = np.asarray(im, dtype=np.float32)  # HWC 0..255

    if mode == "tf":
        arr = (arr - 127.5) / 127.5
    elif mode == "raw01":
        arr = arr / 255.0
    elif mode == "meanstd":
        arr = arr / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        arr = (arr - mean) / std
    else:
        raise ValueError("mode must be tf/raw01/meanstd")

    chw = np.transpose(arr, (2, 0, 1))  # CHW
    chw = np.ascontiguousarray(chw, dtype=dtype)
    out = chw[None, :, :, :]
    return np.ascontiguousarray(out)


# -------------------------
# tracking (keep as-is)
# -------------------------

@dataclass
class TrackState:
    box_xyxy: np.ndarray
    score: float
    last_seen: int

def track_update(track: TrackState, dets, frame_idx: int, iou_gate=0.3, ema=0.7, max_miss=10):
    if track is None:
        if not dets:
            return None
        box, sc = dets[0]
        return TrackState(box_xyxy=box.copy(), score=float(sc), last_seen=frame_idx)

    if frame_idx - track.last_seen > max_miss:
        track = None
        if not dets:
            return None
        box, sc = dets[0]
        return TrackState(box_xyxy=box.copy(), score=float(sc), last_seen=frame_idx)

    if not dets:
        return track  # hold

    best_iou = -1.0
    best = None
    for box, sc in dets:
        v = iou_xyxy(track.box_xyxy, box)
        if v > best_iou:
            best_iou = v
            best = (box, sc)

    if best_iou >= iou_gate:
        box, sc = best
        track.box_xyxy = ema * track.box_xyxy + (1.0 - ema) * box
        track.score = float(sc)
        track.last_seen = frame_idx
        return track

    return track


# -------------------------
# ar filter (keep as-is)
# -------------------------

def ar_filter(det_boxes: np.ndarray, det_scores: np.ndarray, ar_min: float, ar_max: float):
    if det_scores.size == 0:
        return det_boxes, det_scores
    w = det_boxes[:, 2] - det_boxes[:, 0]
    h = det_boxes[:, 3] - det_boxes[:, 1]
    ar = w / (h + 1e-6)
    keep = (ar > ar_min) & (ar < ar_max)
    return det_boxes[keep], det_scores[keep]


# -------------------------
# NEW: state machine + event logging
# -------------------------

class PersonStateMachine:
    """
    States:
      NO_PERSON
      PERSON_PRESENT   (entered but not yet stay)
      PERSON_STAY      (present for >= stay_sec)

    Transitions use hysteresis:
      ENTER: NO_PERSON -> PERSON_PRESENT when detection appears
      STAY : PERSON_PRESENT -> PERSON_STAY when duration >= stay_sec
      LOST : PERSON_PRESENT/PERSON_STAY -> NO_PERSON when no detection for lost_sec
    """
    def __init__(self, stay_sec: float, lost_sec: float):
        self.stay_sec = float(stay_sec)
        self.lost_sec = float(lost_sec)

        self.state = "NO_PERSON"
        self.present_since_ts = None
        self.last_seen_ts = None
        self.stay_emitted = False

    def update(self, ts: float, has_det: bool) -> (str, float, str):
        """
        returns: (state, duration_sec, event_or_empty)
        """
        event = ""

        if has_det:
            if self.state == "NO_PERSON":
                self.state = "PERSON_PRESENT"
                self.present_since_ts = ts
                self.last_seen_ts = ts
                self.stay_emitted = False
                event = "ENTER"
            else:
                self.last_seen_ts = ts

            dur = (ts - self.present_since_ts) if self.present_since_ts is not None else 0.0
            if self.state == "PERSON_PRESENT" and (not self.stay_emitted) and dur >= self.stay_sec:
                self.state = "PERSON_STAY"
                self.stay_emitted = True
                event = "STAY"
            return self.state, dur, event

        # no det
        if self.state in ("PERSON_PRESENT", "PERSON_STAY"):
            if self.last_seen_ts is None:
                self.last_seen_ts = ts
            if (ts - self.last_seen_ts) >= self.lost_sec:
                event = "LOST"
                self.state = "NO_PERSON"
                self.present_since_ts = None
                self.last_seen_ts = None
                self.stay_emitted = False
                return self.state, 0.0, event

            dur = (ts - self.present_since_ts) if self.present_since_ts is not None else 0.0
            return self.state, dur, ""

        return self.state, 0.0, ""


class JsonlWriter:
    def __init__(self, path: str):
        ensure_dir(os.path.dirname(path))
        self.f = open(path, "a", buffering=1)  # line-buffered

    def write(self, obj: dict):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def make_events_path(log_dir: str) -> str:
    ensure_dir(log_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"events_{ts}.jsonl")


# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--engine", required=True)
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--video_size", default="640x480")
    ap.add_argument("--input_format", default="mjpeg")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mode", default="meanstd", choices=["tf", "raw01", "meanstd"])

    ap.add_argument("--input_name", default="Preprocessor/sub:0")
    ap.add_argument("--boxes_name", default="Squeeze:0")
    ap.add_argument("--scores_name", default="concat_1:0")

    ap.add_argument("--class_id", type=int, default=1)
    ap.add_argument("--score_thresh", type=float, default=0.55)
    ap.add_argument("--iou_thresh", type=float, default=0.30)
    ap.add_argument("--loc_order", default="tytxthtw", choices=["tytxthtw", "txtytwth"])

    ap.add_argument("--max_dets", type=int, default=3)
    ap.add_argument("--hold_frames", type=int, default=6)

    # legacy: save_every == vis save interval
    ap.add_argument("--save_every", type=int, default=1)

    # NEW: vis interval separated; 0 => use save_every
    ap.add_argument("--vis_every", type=int, default=0, help="vis.jpg write interval (frames). 0 => use --save_every")

    ap.add_argument("--track", action="store_true")

    ap.add_argument("--ar_filter", action="store_true")
    ap.add_argument("--ar_min", type=float, default=0.40)
    ap.add_argument("--ar_max", type=float, default=2.6)

    ap.add_argument("--out_dir", default="/workspace/outputs/live")
    ap.add_argument("--vis_name", default="vis.jpg")
    ap.add_argument("--jpeg_quality", type=int, default=85)

    ap.add_argument("--stats_every", type=float, default=5.0)
    ap.add_argument("--stats_window", type=int, default=300)
    ap.add_argument("--stats_json", action="store_true")
    ap.add_argument("--warmup_frames", type=int, default=60)

    # ffmpeg tuning
    ap.add_argument("--ffmpeg_lowlat", action="store_true", help="enable low-latency ffmpeg flags")
    ap.add_argument("--ffmpeg_scale300", action="store_true", help="scale to 300x300 in ffmpeg (reduces pre)")

    # NEW: events
    ap.add_argument("--event_log", action="store_true", help="enable JSONL event logging")
    ap.add_argument("--log_dir", default="/workspace/logs")
    ap.add_argument("--event_log_path", default="", help="explicit jsonl path (optional)")
    ap.add_argument("--stay_sec", type=float, default=2.0)
    ap.add_argument("--lost_sec", type=float, default=0.7)
    ap.add_argument("--event_snapshot", action="store_true", help="save snapshot only on events")
    ap.add_argument("--event_snapshot_dir", default="/workspace/outputs/live/events")
    ap.add_argument("--event_snapshot_quality", type=int, default=90)

    args = ap.parse_args()
    if args.save_every < 1:
        args.save_every = 1
    if args.vis_every is None or args.vis_every <= 0:
        args.vis_every = args.save_every

    ensure_dir(args.out_dir)
    out_vis = os.path.join(args.out_dir, args.vis_name)

    # event logger
    ev_writer = None
    ev_path = ""
    if args.event_log:
        ev_path = args.event_log_path.strip() or make_events_path(args.log_dir)
        ev_writer = JsonlWriter(ev_path)
        ensure_dir(args.log_dir)

    if args.event_snapshot:
        ensure_dir(args.event_snapshot_dir)

    # ffmpeg cmd
    vf = f"fps={args.fps}"
    if args.ffmpeg_scale300:
        vf = f"fps={args.fps},scale=300:300"

    cmd = ["ffmpeg", "-loglevel", "error"]
    if args.ffmpeg_lowlat:
        cmd += ["-fflags", "nobuffer", "-flags", "low_delay", "-analyzeduration", "0", "-probesize", "32"]
    cmd += [
        "-f", "v4l2",
        "-input_format", args.input_format,
        "-video_size", args.video_size,
        "-i", args.device,
        "-vf", vf,
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-q:v", "5",
        "-"
    ]

    reader = MJPEGStreamReader(cmd)
    reader.start()

    trt_runner = TRTInfer(args.engine, args.input_name, args.boxes_name, args.scores_name)
    priors = generate_ssd_priors_1917()

    # stats deques
    cap_q = deque(maxlen=args.stats_window)
    pre_q = deque(maxlen=args.stats_window)
    h2d_q = deque(maxlen=args.stats_window)
    infer_q = deque(maxlen=args.stats_window)
    post_q = deque(maxlen=args.stats_window)
    draw_q = deque(maxlen=args.stats_window)
    save_q = deque(maxlen=args.stats_window)
    e2e_q = deque(maxlen=args.stats_window)

    # for events: keep last computed fps & last latencies
    t_stats0 = time.perf_counter()
    fps_frames = 0
    fps_est = 0.0

    # state machine
    sm = PersonStateMachine(stay_sec=args.stay_sec, lost_sec=args.lost_sec)

    stop = False
    def _sig(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    frame = 0
    track_state = None
    last_draw_list = []
    last_seen_frame = -10**9

    print("[live] start loop. Ctrl+C to stop.", flush=True)
    print(f"[live] writing: {out_vis}", flush=True)
    if ev_writer is not None:
        print(f"[live] event_log: {ev_path}", flush=True)

    try:
        while not stop:
            t_e2e0 = now_ms()

            # capture
            t_cap0 = now_ms()
            im = reader.read_frame(timeout_s=2.0)
            t_cap1 = now_ms()

            # preprocess
            t_pre0 = now_ms()
            inp = preprocess_pil(
                im, mode=args.mode, input_size=300, dtype=trt_runner.in_host.dtype,
                skip_resize_if_already=True
            )
            t_pre1 = now_ms()

            # infer
            boxes_raw, scores_raw, t_h2d, t_infer = trt_runner.infer(inp)

            # post
            t_post0 = now_ms()
            loc = boxes_raw[0].astype(np.float32, copy=False)
            logits = scores_raw[0].astype(np.float32, copy=False)
            probs = softmax(logits, axis=1)
            person_scores = probs[:, args.class_id]

            idx = np.where(person_scores >= args.score_thresh)[0]
            det_boxes = np.zeros((0, 4), dtype=np.float32)
            det_scores = np.zeros((0,), dtype=np.float32)

            if idx.size > 0:
                boxes_xyxy = decode_boxes(loc, priors, loc_order=args.loc_order)
                det_boxes = boxes_xyxy[idx]
                det_scores = person_scores[idx]

                keep = nms_numpy(det_boxes, det_scores, iou_thresh=args.iou_thresh, topk=200)
                det_boxes = det_boxes[keep]
                det_scores = det_scores[keep]

                if det_scores.size > 0:
                    order = det_scores.argsort()[::-1]
                    det_boxes = det_boxes[order]
                    det_scores = det_scores[order]

                # 기존 정책 유지: ar_filter는 track이 꺼져 있을 때만 적용
                if args.ar_filter and (not args.track) and det_scores.size > 0:
                    b2, s2 = ar_filter(det_boxes, det_scores, args.ar_min, args.ar_max)
                    if s2.size > 0:
                        det_boxes, det_scores = b2, s2

                if det_scores.size > args.max_dets:
                    det_boxes = det_boxes[:args.max_dets]
                    det_scores = det_scores[:args.max_dets]

            # track / hold
            if args.track:
                dets = [(det_boxes[i], float(det_scores[i])) for i in range(det_scores.size)]
                track_state = track_update(track_state, dets, frame_idx=frame, iou_gate=0.3, ema=0.7, max_miss=10)
                draw_list = [(track_state.box_xyxy, track_state.score)] if track_state is not None else []
            else:
                draw_list = [(det_boxes[i], float(det_scores[i])) for i in range(det_scores.size)]
                if args.hold_frames > 0:
                    if len(draw_list) > 0:
                        last_draw_list = draw_list
                        last_seen_frame = frame
                    else:
                        if last_draw_list and (frame - last_seen_frame) <= args.hold_frames:
                            draw_list = last_draw_list

            t_post1 = now_ms()

            # state machine + event logging
            now_ts = time.time()
            has_det = (len(draw_list) > 0)
            state, dur_sec, event = sm.update(now_ts, has_det)

            # draw
            t_draw0 = now_ms()
            vis = im.copy()
            W, H = vis.size
            dr = ImageDraw.Draw(vis)

            best_score = None
            best_box = None
            if has_det:
                # single person policy: use draw_list[0]
                best_box = draw_list[0][0]
                best_score = float(draw_list[0][1])

            for b, sc in draw_list:
                x1 = int(b[0] * W)
                y1 = int(b[1] * H)
                x2 = int(b[2] * W)
                y2 = int(b[3] * H)
                dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                dr.text((x1, max(0, y1 - 16)), f"person {sc:.2f}", fill=(255, 0, 0))

            # overlay state/fps/dur (cheap)
            dr.text((6, 6), f"state={state}", fill=(255, 255, 0))
            dr.text((6, 22), f"dur={dur_sec:.2f}s", fill=(255, 255, 0))
            dr.text((6, 38), f"fps~{fps_est:.1f}", fill=(255, 255, 0))

            cap_ms = (t_cap1 - t_cap0)
            pre_ms = (t_pre1 - t_pre0)
            post_ms = (t_post1 - t_post0)
            draw_ms = (now_ms() - t_draw0)

            # vis save (separate interval)
            save_ms = 0.0
            if (frame % args.vis_every) == 0:
                t_save0 = now_ms()
                atomic_save_jpg(vis, out_vis, quality=args.jpeg_quality)
                save_ms = now_ms() - t_save0

            t_e2e1 = now_ms()
            e2e_ms = (t_e2e1 - t_e2e0)

            # stats collect AFTER warmup
            if frame >= args.warmup_frames:
                cap_q.append(cap_ms)
                pre_q.append(pre_ms)
                h2d_q.append(t_h2d)
                infer_q.append(t_infer)
                post_q.append(post_ms)
                draw_q.append(draw_ms)
                save_q.append(save_ms)
                e2e_q.append(e2e_ms)

            # fps update
            fps_frames += 1
            if (time.perf_counter() - t_stats0) >= args.stats_every and frame >= args.warmup_frames:
                dt = time.perf_counter() - t_stats0
                fps_est = fps_frames / dt if dt > 0 else 0.0

                print(f"[stats] fps~{fps_est:.1f}", flush=True)
                print("[stats] " + stats_line("cap", list(cap_q)), flush=True)
                print("[stats] " + stats_line("pre", list(pre_q)), flush=True)
                print("[stats] " + stats_line("h2d", list(h2d_q)), flush=True)
                print("[stats] " + stats_line("infer", list(infer_q)), flush=True)
                print("[stats] " + stats_line("post", list(post_q)), flush=True)
                print("[stats] " + stats_line("draw", list(draw_q)), flush=True)
                print("[stats] " + stats_line("save", list(save_q)), flush=True)
                print("[stats] " + stats_line("e2e", list(e2e_q)), flush=True)

                if args.stats_json:
                    out = {
                        "fps": fps_est,
                        "cap":  {"p50": pct(cap_q,50),  "p90": pct(cap_q,90),  "p99": pct(cap_q,99),  "max": (max(cap_q) if cap_q else None)},
                        "pre":  {"p50": pct(pre_q,50),  "p90": pct(pre_q,90),  "p99": pct(pre_q,99),  "max": (max(pre_q) if pre_q else None)},
                        "h2d":  {"p50": pct(h2d_q,50),  "p90": pct(h2d_q,90),  "p99": pct(h2d_q,99),  "max": (max(h2d_q) if h2d_q else None)},
                        "infer": {"p50": pct(infer_q,50),"p90": pct(infer_q,90),"p99": pct(infer_q,99),"max": (max(infer_q) if infer_q else None)},
                        "post": {"p50": pct(post_q,50), "p90": pct(post_q,90), "p99": pct(post_q,99), "max": (max(post_q) if post_q else None)},
                        "draw": {"p50": pct(draw_q,50), "p90": pct(draw_q,90), "p99": pct(draw_q,99), "max": (max(draw_q) if draw_q else None)},
                        "save": {"p50": pct(save_q,50), "p90": pct(save_q,90), "p99": pct(save_q,99), "max": (max(save_q) if save_q else None)},
                        "e2e":  {"p50": pct(e2e_q,50),  "p90": pct(e2e_q,90),  "p99": pct(e2e_q,99),  "max": (max(e2e_q) if e2e_q else None)},
                        "ts": time.time()
                    }
                    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
                        json.dump(out, f)

                t_stats0 = time.perf_counter()
                fps_frames = 0

            # EVENT emit AFTER fps/lat computed (use latest)
            if event and ev_writer is not None and frame >= args.warmup_frames:
                lat = {
                    "cap": cap_ms,
                    "pre": pre_ms,
                    "h2d": float(t_h2d),
                    "infer": float(t_infer),
                    "post": post_ms,
                    "draw": draw_ms,
                    "save": save_ms,
                    "e2e": e2e_ms,
                }
                ev_obj = {
                    "ts": ts_utc_iso(now_ts),
                    "frame_idx": frame,
                    "event": event,           # ENTER/STAY/LOST
                    "state": state,           # NO_PERSON/PERSON_PRESENT/PERSON_STAY
                    "duration": round(float(dur_sec), 3),
                    "fps": round(float(fps_est), 2),
                    "score": (round(float(best_score), 4) if best_score is not None else None),
                    "bbox": (best_box.tolist() if best_box is not None else None),  # xyxy norm
                    "lat_ms": {k: round(float(v), 3) for k, v in lat.items()},
                }
                ev_writer.write(ev_obj)

                if args.event_snapshot:
                    # event-only snapshot (annotated)
                    fn = f"{event}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{frame}.jpg"
                    outp = os.path.join(args.event_snapshot_dir, fn)
                    atomic_save_jpg(vis, outp, quality=args.event_snapshot_quality)

            if frame % 30 == 0:
                print(f"[{frame}] dets={len(draw_list)} state={state} cap={cap_ms:.1f} pre={pre_ms:.1f} infer={t_infer:.1f} post={post_ms:.1f}", flush=True)

            frame += 1

    finally:
        reader.close()
        if ev_writer is not None:
            ev_writer.close()
        print("\n[live] stopped.", flush=True)


if __name__ == "__main__":
    main()

