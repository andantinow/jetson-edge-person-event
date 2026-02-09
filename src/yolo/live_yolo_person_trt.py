#!/usr/bin/env python3
import os
import time
import json
import signal
import argparse
import subprocess
from dataclasses import dataclass
from io import BytesIO
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

from infer_yolo_trt import TRTInferYOLO
from yolo_post import yolo_v8_decode_person, iou_xyxy


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def now_ms():
    return time.perf_counter() * 1000.0


def utc_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_save_jpg(img: Image.Image, out_path: str, quality: int = 85):
    tmp = out_path + ".tmp"
    img.save(tmp, format="JPEG", quality=quality)
    os.replace(tmp, out_path)


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


def letterbox_pil(im: Image.Image, new_size=640):
    """
    Letterbox to (new_size,new_size) with gray padding.
    returns: lb_img, scale, pad_x, pad_y
    """
    w, h = im.size
    s = min(new_size / w, new_size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = im.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("RGB", (new_size, new_size), (114, 114, 114))
    pad_x = (new_size - nw) // 2
    pad_y = (new_size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, float(s), int(pad_x), int(pad_y)


def preprocess_yolo(im_rgb: Image.Image, imgsz=640, dtype=np.float32):
    lb, scale, pad_x, pad_y = letterbox_pil(im_rgb, new_size=imgsz)
    arr = np.asarray(lb, dtype=np.float32) / 255.0  # HWC, 0..1
    chw = np.transpose(arr, (2, 0, 1))              # CHW
    chw = np.ascontiguousarray(chw, dtype=dtype)
    inp = chw[None, :, :, :]                        # (1,3,H,W)
    return inp, scale, pad_x, pad_y


def map_box_back_xyxy_px(box_xyxy_lb, scale, pad_x, pad_y, orig_w, orig_h):
    """
    box_xyxy_lb: xyxy in letterboxed pixel space (0..imgsz)
    map back to original image pixel space.
    """
    x1, y1, x2, y2 = box_xyxy_lb
    x1 = (x1 - pad_x) / (scale + 1e-9)
    y1 = (y1 - pad_y) / (scale + 1e-9)
    x2 = (x2 - pad_x) / (scale + 1e-9)
    y2 = (y2 - pad_y) / (scale + 1e-9)

    x1 = float(np.clip(x1, 0, orig_w))
    y1 = float(np.clip(y1, 0, orig_h))
    x2 = float(np.clip(x2, 0, orig_w))
    y2 = float(np.clip(y2, 0, orig_h))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


@dataclass
class TrackState:
    box_xyxy_norm: np.ndarray
    score: float
    last_seen: int


def track_update(track: TrackState, dets_norm, frame_idx: int, iou_gate=0.3, ema=0.7, max_miss=10):
    """
    dets_norm: list[(box_norm_xyxy, score)]
    """
    if track is None:
        if not dets_norm:
            return None
        box, sc = dets_norm[0]
        return TrackState(box_xyxy_norm=box.copy(), score=float(sc), last_seen=frame_idx)

    if frame_idx - track.last_seen > max_miss:
        track = None
        if not dets_norm:
            return None
        box, sc = dets_norm[0]
        return TrackState(box_xyxy_norm=box.copy(), score=float(sc), last_seen=frame_idx)

    if not dets_norm:
        return track  # hold

    best_iou = -1.0
    best = None
    for box, sc in dets_norm:
        v = iou_xyxy(track.box_xyxy_norm, box)
        if v > best_iou:
            best_iou = v
            best = (box, sc)

    if best_iou >= iou_gate:
        box, sc = best
        track.box_xyxy_norm = ema * track.box_xyxy_norm + (1.0 - ema) * box
        track.score = float(sc)
        track.last_seen = frame_idx
        return track

    return track


class EventLogger:
    def __init__(self, enabled: bool, out_dir: str, prefix: str = "events"):
        self.enabled = enabled
        self.path = None
        self.f = None
        if enabled:
            ensure_dir(out_dir)
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            self.path = os.path.join(out_dir, f"{prefix}_{ts}.jsonl")
            self.f = open(self.path, "a", buffering=1)

    def write(self, obj: dict):
        if not self.enabled or self.f is None:
            return
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        try:
            if self.f:
                self.f.close()
        finally:
            self.f = None


class PersonStateMachine:
    NO_PERSON = "NO_PERSON"
    PERSON_PRESENT = "PERSON_PRESENT"
    PERSON_STAY = "PERSON_STAY"

    def __init__(self, stay_sec: float, lost_sec: float):
        self.stay_sec = float(stay_sec)
        self.lost_sec = float(lost_sec)
        self.state = self.NO_PERSON
        self.present_since = None
        self.last_seen_ts = None

    def update(self, now_ts: float, present: bool):
        """
        returns event string or None: ENTER/STAY/LOST
        """
        ev = None

        if present:
            self.last_seen_ts = now_ts
            if self.state == self.NO_PERSON:
                self.state = self.PERSON_PRESENT
                self.present_since = now_ts
                ev = "ENTER"
            elif self.state == self.PERSON_PRESENT:
                if self.present_since is not None and (now_ts - self.present_since) >= self.stay_sec:
                    self.state = self.PERSON_STAY
                    ev = "STAY"
            elif self.state == self.PERSON_STAY:
                pass
            return ev

        # not present
        if self.state != self.NO_PERSON:
            if self.last_seen_ts is None:
                self.last_seen_ts = now_ts
            if (now_ts - self.last_seen_ts) >= self.lost_sec:
                self.state = self.NO_PERSON
                self.present_since = None
                self.last_seen_ts = None
                ev = "LOST"
        return ev

    def duration(self, now_ts: float) -> float:
        if self.present_since is None:
            return 0.0
        return float(max(0.0, now_ts - self.present_since))


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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--engine", required=True)
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--video_size", default="640x480")
    ap.add_argument("--input_format", default="mjpeg")
    ap.add_argument("--fps", type=int, default=30)

    # YOLO engine tensor names (your engine)
    ap.add_argument("--input_name", default="images")
    ap.add_argument("--output_name", default="output0")
    ap.add_argument("--imgsz", type=int, default=640)

    ap.add_argument("--class_id", type=int, default=0)  # COCO person=0
    ap.add_argument("--score_thresh", type=float, default=0.35)
    ap.add_argument("--iou_thresh", type=float, default=0.50)
    ap.add_argument("--max_dets", type=int, default=20)

    ap.add_argument("--track", action="store_true")

    ap.add_argument("--out_dir", default="/workspace/outputs/live")
    ap.add_argument("--vis_name", default="vis.jpg")
    ap.add_argument("--save_every", type=int, default=2)

    ap.add_argument("--stats_every", type=float, default=5.0)
    ap.add_argument("--stats_window", type=int, default=300)
    ap.add_argument("--warmup_frames", type=int, default=60)

    # ffmpeg tuning
    ap.add_argument("--ffmpeg_lowlat", action="store_true")

    # events
    ap.add_argument("--event_log", action="store_true")
    ap.add_argument("--event_log_dir", default="/workspace/logs")
    ap.add_argument("--stay_sec", type=float, default=2.0)
    ap.add_argument("--lost_sec", type=float, default=0.7)

    ap.add_argument("--event_snapshot", action="store_true")
    ap.add_argument("--event_snapshot_dir", default="/workspace/outputs/live/events")
    ap.add_argument("--event_snapshot_quality", type=int, default=85)

    args = ap.parse_args()
    if args.save_every < 1:
        args.save_every = 1

    ensure_dir(args.out_dir)
    out_vis = os.path.join(args.out_dir, args.vis_name)

    # ffmpeg command (keep original resolution; letterbox happens in python)
    cmd = ["ffmpeg", "-loglevel", "error"]
    if args.ffmpeg_lowlat:
        cmd += ["-fflags", "nobuffer", "-flags", "low_delay", "-analyzeduration", "0", "-probesize", "32"]
    cmd += [
        "-f", "v4l2",
        "-input_format", args.input_format,
        "-video_size", args.video_size,
        "-i", args.device,
        "-vf", f"fps={args.fps}",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-q:v", "5",
        "-"
    ]

    reader = MJPEGStreamReader(cmd)
    reader.start()

    trt_runner = TRTInferYOLO(args.engine, input_name=args.input_name, output_name=args.output_name)

    # stats
    cap_q = deque(maxlen=args.stats_window)
    pre_q = deque(maxlen=args.stats_window)
    h2d_q = deque(maxlen=args.stats_window)
    infer_q = deque(maxlen=args.stats_window)
    post_q = deque(maxlen=args.stats_window)
    draw_q = deque(maxlen=args.stats_window)
    save_q = deque(maxlen=args.stats_window)
    e2e_q = deque(maxlen=args.stats_window)

    t_stats0 = time.perf_counter()
    fps_frames = 0
    fps_est = 0.0

    # events
    evlog = EventLogger(args.event_log, args.event_log_dir, prefix="events")
    sm = PersonStateMachine(stay_sec=args.stay_sec, lost_sec=args.lost_sec)
    track_state = None

    stop = False

    def _sig(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    frame = 0
    print("[live-yolo] start loop. Ctrl+C to stop.", flush=True)
    print(f"[live-yolo] writing: {out_vis}", flush=True)
    if args.event_log:
        print(f"[live-yolo] event log: {evlog.path}", flush=True)

    try:
        while not stop:
            t_e2e0 = now_ms()

            # capture
            t_cap0 = now_ms()
            im = reader.read_frame(timeout_s=2.0)
            t_cap1 = now_ms()

            # preprocess
            t_pre0 = now_ms()
            inp, scale, pad_x, pad_y = preprocess_yolo(im, imgsz=args.imgsz, dtype=trt_runner.in_host.dtype)
            t_pre1 = now_ms()

            # infer
            out0, t_h2d, t_infer = trt_runner.infer(inp)

            # post
            t_post0 = now_ms()
            dets_lb = yolo_v8_decode_person(
                out0,
                score_thresh=args.score_thresh,
                iou_thresh=args.iou_thresh,
                imgsz=args.imgsz,
                class_id=args.class_id,
                max_dets=args.max_dets,
            )

            W, H = im.size
            dets_norm = []
            for box_lb, sc in dets_lb:
                box_px = map_box_back_xyxy_px(box_lb, scale, pad_x, pad_y, W, H)
                box_norm = np.array([box_px[0] / W, box_px[1] / H, box_px[2] / W, box_px[3] / H], dtype=np.float32)
                dets_norm.append((box_norm, sc))

            # sort already handled, but keep safe
            dets_norm.sort(key=lambda x: x[1], reverse=True)

            if args.track:
                track_state = track_update(track_state, dets_norm, frame_idx=frame, iou_gate=0.3, ema=0.7, max_miss=10)
                draw_list = [(track_state.box_xyxy_norm, track_state.score)] if track_state is not None else []
            else:
                draw_list = dets_norm[:1]  # 대표 1개만

            present = (len(draw_list) > 0)
            now_ts = time.time()
            ev = sm.update(now_ts, present=present)

            t_post1 = now_ms()

            # draw
            t_draw0 = now_ms()
            vis = im.copy()
            dr = ImageDraw.Draw(vis)

            # state overlay
            dr.text((8, 6), f"state={sm.state}", fill=(255, 255, 0))
            if present and sm.present_since is not None:
                dr.text((8, 22), f"dur={sm.duration(now_ts):.1f}s", fill=(255, 255, 0))

            # bbox
            for b, sc in draw_list:
                x1 = int(b[0] * W)
                y1 = int(b[1] * H)
                x2 = int(b[2] * W)
                y2 = int(b[3] * H)
                dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                dr.text((x1, max(0, y1 - 16)), f"person {sc:.2f}", fill=(255, 0, 0))

            cap_ms = (t_cap1 - t_cap0)
            pre_ms = (t_pre1 - t_pre0)
            post_ms = (t_post1 - t_post0)
            draw_ms = (now_ms() - t_draw0)

            # save vis.jpg (periodic)
            save_ms = 0.0
            if (frame % args.save_every) == 0:
                t_save0 = now_ms()
                atomic_save_jpg(vis, out_vis, quality=85)
                save_ms = now_ms() - t_save0

            # event snapshot (only on event)
            if args.event_snapshot and ev is not None:
                ensure_dir(args.event_snapshot_dir)
                ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                snap_path = os.path.join(args.event_snapshot_dir, f"{ts}_{ev}.jpg")
                try:
                    atomic_save_jpg(vis, snap_path, quality=args.event_snapshot_quality)
                except Exception:
                    pass

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

            fps_frames += 1
            if (time.perf_counter() - t_stats0) >= args.stats_every and frame >= args.warmup_frames:
                dt = time.perf_counter() - t_stats0
                fps_est = fps_frames / dt
                print(f"[stats] fps~{fps_est:.1f}", flush=True)
                print("[stats] " + stats_line("cap", list(cap_q)), flush=True)
                print("[stats] " + stats_line("pre", list(pre_q)), flush=True)
                print("[stats] " + stats_line("h2d", list(h2d_q)), flush=True)
                print("[stats] " + stats_line("infer", list(infer_q)), flush=True)
                print("[stats] " + stats_line("post", list(post_q)), flush=True)
                print("[stats] " + stats_line("draw", list(draw_q)), flush=True)
                print("[stats] " + stats_line("save", list(save_q)), flush=True)
                print("[stats] " + stats_line("e2e", list(e2e_q)), flush=True)
                t_stats0 = time.perf_counter()
                fps_frames = 0

            # event log (on event only)
            if ev is not None and args.event_log:
                b = draw_list[0][0].tolist() if present else None
                sc = float(draw_list[0][1]) if present else None
                evlog.write({
                    "ts": utc_iso(),
                    "epoch": now_ts,
                    "frame_idx": frame,
                    "event": ev,
                    "state": sm.state,
                    "score": sc,
                    "bbox_xyxy_norm": b,
                    "duration_sec": sm.duration(now_ts) if present else 0.0,
                    "fps_est": fps_est if frame >= args.warmup_frames else None,
                    "lat_ms": {
                        "cap": cap_ms,
                        "pre": pre_ms,
                        "h2d": t_h2d,
                        "infer_d2h": t_infer,
                        "post": post_ms,
                        "draw": draw_ms,
                        "save": save_ms,
                        "e2e": e2e_ms,
                    }
                })

            if frame % 30 == 0:
                print(f"[{frame}] det={1 if present else 0} state={sm.state} cap={cap_ms:.1f} pre={pre_ms:.1f} infer={t_infer:.1f} post={post_ms:.1f}", flush=True)

            frame += 1

    finally:
        reader.close()
        evlog.close()
        print("\n[live-yolo] stopped.", flush=True)


if __name__ == "__main__":
    main()

