import argparse, json
from PIL import Image, ImageDraw, ImageFont

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--min_score", type=float, default=0.3)
    args = ap.parse_args()

    im = Image.open(args.image).convert("RGB")
    W,H = im.size
    d = json.load(open(args.dets))["detections"]

    d = [x for x in d if x["score"] >= args.min_score]
    d = sorted(d, key=lambda x: x["score"], reverse=True)[:args.topk]

    draw = ImageDraw.Draw(im)
    for x in d:
        x1,y1,x2,y2 = x["bbox_xyxy_norm"]
        x1 = int(x1 * W); x2 = int(x2 * W)
        y1 = int(y1 * H); y2 = int(y2 * H)
        draw.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=3)
        draw.text((x1, max(0,y1-14)), f'c{x["class_id"]}:{x["score"]:.2f}', fill=(255,0,0))

    im.save(args.out)
    print("[write]", args.out, "n=", len(d))

if __name__ == "__main__":
    main()
