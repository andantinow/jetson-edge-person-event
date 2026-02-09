Jetson Edge Person Event System (YOLOv8 TensorRT)

Jetson Orin Nano에서 TensorRT를 사용해 단일 사람(person)을 실시간으로 검출하고,
상태 변화(ENTER / STAY / LOST)를 이벤트로 기록하는 엣지 비전 시스템이다.

모델 정확도 자체보다는 엣지 환경에서의 안정적인 실시간 처리,
지연 시간 계측, 그리고 실제 운영 가능한 구조에 초점을 둔다.


Overview

이 프로젝트는 고정 카메라 환경에서 다음을 수행한다.

- USB 카메라 (/dev/video0) 실시간 입력
- TensorRT FP16 엔진 기반 YOLOv8 person 검출
- 단일 person 기준 상태 머신
- 상태 변화 이벤트(JSONL) 기록
- 실시간 시각화 결과를 HTTP로 제공
- 파이프라인 전 구간 지연 시간 측정


System Architecture

입력부터 출력까지의 처리 흐름은 다음과 같다.

/dev/video0 (MJPEG)
→ ffmpeg image2pipe
→ MJPEG frame parsing
→ PIL image decode
→ 전처리 (640x640, NCHW)
→ TensorRT 추론 (YOLOv8)
→ YOLO decode + NMS
→ 단일 person 선택 (tracking 옵션)
→ 상태 머신
→ 이벤트 로그(JSONL)
→ vis.jpg 저장
→ HTTP 서버 제공


Model and Inference

- Model: YOLOv8n (COCO pretrained)
- Runtime: TensorRT
- Precision: FP16
- Input shape: (1, 3, 640, 640)
- Output shape: (1, 84, 8400)

TensorRT 엔진 파일 (*.engine)은 저장소에 포함하지 않는다.


State Machine

단일 person 기준 상태 머신을 사용한다.

States:
- NO_PERSON
- PERSON_PRESENT
- PERSON_STAY

Events:
- ENTER  
  사람이 처음 검출되었을 때 발생
- STAY  
  사람이 stay_sec 이상 연속 검출되었을 때 발생
- LOST  
  lost_sec 동안 검출되지 않았을 때 발생

이벤트는 상태 변화 시점에만 발생하며 JSONL 파일로 기록된다.


Event Logging

이벤트 로그는 JSONL 형식으로 append 기록된다.

기록 필드:
- timestamp (UTC ISO format)
- frame index
- event type (ENTER / STAY / LOST)
- current state
- detection score
- bounding box (normalized xyxy)
- duration (sec)
- fps estimate
- latency summary (cap, pre, h2d, infer, post, draw, save, e2e)

로그 파일 경로 예시:
/workspace/logs/events_YYYYMMDD_HHMMSS.jsonl


Visualization

- 실시간 결과는 vis.jpg 파일로 저장된다.
- atomic write 방식으로 항상 최신 프레임을 유지한다.
- 내장 HTTP 서버를 통해 외부에서 확인 가능하다.

접속 예시:
http://<JETSON_IP>:8000/vis.jpg


Performance Measurement

다음 구간에 대해 지연 시간을 측정한다.

- capture
- preprocess
- host to device
- inference
- postprocess
- draw
- save
- end-to-end

rolling window 기준 p50 / p90 / p99 / max 값을 주기적으로 출력한다.


Usage (inside container)

YOLO TensorRT 실행:

source /opt/venv/bin/activate
python3 /workspace/outputs/live_yolo_person_trt.py \
  --engine /workspace/models/yolo/yolov8n/yolov8n_fp16.engine \
  --score_thresh 0.35 --iou_thresh 0.50 \
  --ffmpeg_lowlat \
  --save_every 2 \
  --track \
  --event_log \
  --stay_sec 2.0 --lost_sec 0.7

