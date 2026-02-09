import argparse, json
import numpy as np

# -----------------------
# SSD priors (1917) for SSD-Mobilenet 300x300 style
# feature map sizes: [19,10,5,3,2,1]
# first layer: 3 anchors/location (ar: 1,2,0.5) with single ar=1 box
# later layers: 6 anchors/location (ar: 1,2,0.5,3,1/3) with TWO ar=1 boxes
# scales: min_scale=0.2, max_scale=0.95 (common)
# -----------------------
def generate_ssd_priors_1917(
    fm_sizes=(19, 10, 5, 3, 2, 1),
    min_scale=0.2,
    max_scale=0.95,
):
    m = len(fm_sizes)
    scales = [min_scale + (max_scale - min_scale) * k / (m - 1) for k in range(m)]
    scales.append(1.0)

    priors = []
    for k, f in enumerate(fm_sizes):
        s_k = scales[k]
        s_k1 = scales[k + 1]

        if k == 0:
            ars = [1.0, 2.0, 0.5]
            two_ar1 = False
        else:
            ars = [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]
            two_ar1 = True

        for i in range(f):
            cy = (i + 0.5) / f
            for j in range(f):
                cx = (j + 0.5) / f

                # ar=1, size=s_k
                priors.append([cx, cy, s_k, s_k])

                # second ar=1, size=sqrt(s_k*s_{k+1})
                if two_ar1:
                    s = float(np.sqrt(s_k * s_k1))
                    priors.append([cx, cy, s, s])

                # remaining aspect ratios (excluding the first ar=1 we already added)
                for ar in ars:
                    if ar == 1.0:
                        continue
                    w = s_k * float(np.sqrt(ar))
                    h = s_k / float(np.sqrt(ar))
                    priors.append([cx, cy, w, h])

    priors = np.asarray(priors, dtype=np.float32)  # (1917,4)
    if priors.shape[0] != 1917:
        raise RuntimeError(f"priors count mismatch: {priors.shape[0]} != 1917")
    return priors

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def decode_boxes(loc, priors, variances=(0.1, 0.2), order="tytxthtw"):
    """
    loc: (N,4)
    priors: (N,4) in [cx,cy,w,h] normalized
    order:
      - "tytxthtw" : [ty, tx, th, tw]
      - "txtytwth" : [tx, ty, tw, th]
    """
    if order == "tytxthtw":
        ty, tx, th, tw = loc[:, 0], loc[:, 1], loc[:, 2], loc[:, 3]
    elif order == "txtytwth":
        tx, ty, tw, th = loc[:, 0], loc[:, 1], loc[:, 2], loc[:, 3]
    else:
        raise ValueError(f"unknown order: {order}")

    cx = priors[:, 0] + tx * variances[0] * priors[:, 2]
    cy = priors[:, 1] + ty * variances[0] * priors[:, 3]
    w  = priors[:, 2] * np.exp(tw * variances[1])
    h  = priors[:, 3] * np.exp(th * variances[1])

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=1)

def iou_xyxy(a, b):
    xx1 = np.maximum(a[0], b[:, 0])
    yy1 = np.maximum(a[1], b[:, 1])
    xx2 = np.minimum(a[2], b[:, 2])
    yy2 = np.minimum(a[3], b[:, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a + area_b - inter + 1e-9
    return inter / union

def nms(boxes, scores, iou_thresh=0.5, topk=200):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < topk:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_xyxy(boxes[i], boxes[rest])
        rest = rest[ious <= iou_thresh]
        order = rest
    return np.array(keep, dtype=np.int32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", required=True, help="*.npy loc output (1,1917,4)")
    ap.add_argument("--scores", required=True, help="*.npy conf output (1,1917,91)")
    ap.add_argument("--out", required=True, help="output json")
    ap.add_argument("--score_thresh", type=float, default=0.30)
    ap.add_argument("--iou_thresh", type=float, default=0.50)
    ap.add_argument("--background", type=int, default=0)
    ap.add_argument("--use_softmax", action="store_true", help="softmax over classes (default: sigmoid)")
    ap.add_argument("--loc_order", type=str, default="tytxthtw",
                    choices=["tytxthtw", "txtytwth"],
                    help="loc encoding order")
    ap.add_argument("--class_filter", type=int, default=-1,
                    help="if set to >=0, only keep this class id (e.g. 1=person)")
    args = ap.parse_args()

    loc = np.load(args.boxes)
    conf = np.load(args.scores)

    loc = np.squeeze(loc, axis=0)   # (1917,4)
    conf = np.squeeze(conf, axis=0) # (1917,91)

    priors = generate_ssd_priors_1917()

    # class scores
    if args.use_softmax:
        probs = softmax(conf, axis=1)
    else:
        probs = sigmoid(conf)

    # decode boxes
    boxes = decode_boxes(loc, priors, order=args.loc_order)

    # clamp sanity
    unclamped = boxes.copy()
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, 1.0)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, 1.0)
    clamped_frac = float(np.mean(np.any(np.abs(unclamped - boxes) > 1e-6, axis=1)))
    print(f"[sanity] loc_order={args.loc_order} clamped fraction ~ {clamped_frac:.3f}")

    dets = []
    num_classes = probs.shape[1]

    for c in range(num_classes):
        if c == args.background:
            continue
        if args.class_filter >= 0 and c != args.class_filter:
            continue

        sc = probs[:, c]
        m = sc >= args.score_thresh
        if not np.any(m):
            continue

        b = boxes[m]
        s = sc[m]
        keep = nms(b, s, iou_thresh=args.iou_thresh)

        for idx in keep:
            x1, y1, x2, y2 = b[idx].tolist()
            dets.append({
                "class_id": int(c),
                "score": float(s[idx]),
                "bbox_xyxy_norm": [x1, y1, x2, y2],
            })

    dets.sort(key=lambda d: d["score"], reverse=True)
    with open(args.out, "w") as f:
        json.dump({"detections": dets}, f, indent=2)

    print(f"[done] detections={len(dets)} (score>={args.score_thresh}, iou<={args.iou_thresh})")
    print(f"[write] {args.out}")

if __name__ == "__main__":
    main()

