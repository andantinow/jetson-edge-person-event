# SSD MobileNet v1 TensorRT Person Event System

Jetson Orin Nano에서 TensorRT FP16 엔진(SSD MobileNet v1)을 사용해 단일 사람(person)을 실시간으로 검출하고,
상태 변화(ENTER / STAY / LOST)를 이벤트(JSONL)로 기록하는 시스템이다.

이 파이프라인은 정확도 자체보다 엣지 환경에서의 안정적인 실시간 처리, 지연 시간 계측, 운영 형태에 초점을 둔다.

---

## Pipeline

/dev/video0 (MJPEG)
→ ffmpeg image2pipe
→ MJPEG frame parsing
→ PIL decode
→ 전처리 (300x300, NCHW)
→ TensorRT 추론 (SSD MobileNet v1)
→ SSD priors decode + NMS
→ 단일 person 선택 (tracking 옵션)
→ 상태 머신
→ 이벤트 로그(JSONL)
→ vis.jpg 저장
→ HTTP로 제공

---

## Model and Inference

- Model: SSD MobileNet v1 (COCO 계열, person class)
- Runtime: TensorRT
- Precision: FP16
- Input shape: (1, 3, 300, 300)
- Outputs:
  - Boxes: (1, 1917, 4)
  - Scores: (1, 1917, 91)

TensorRT 엔진 파일 (*.engine)은 저장소에 포함하지 않는다.

---

## Usage (inside container)

```bash
source /opt/venv/bin/activate

python3 /workspace/outputs/live_person_trt.py \
  --engine /workspace/models/ssd_mobilenet_v1_fp16.engine \
  --mode meanstd \
  --score_thresh 0.55 --iou_thresh 0.30 \
  --ar_filter --ar_min 0.40 --ar_max 2.6 \
  --ffmpeg_lowlat \
  --save_every 2 \
  --track \
  --event_log \
  --stay_sec 2.0 --lost_sec 0.7

