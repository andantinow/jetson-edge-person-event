# Jetson Edge Person Event System

Jetson Orin Nano에서 TensorRT를 사용해 사람(person)을 실시간 검출하고,
상태 변화(ENTER / STAY / LOST)를 이벤트로 기록하는 엣지 비전 시스템이다.

정확도 자체보다 엣지 환경에서의 안정적인 실시간 처리, 지연 시간 계측, 운영 가능한 구조에 초점을 둔다.

## Pipelines

- SSD MobileNet v1 (TensorRT FP16): docs/README_ssd.md
- YOLOv8n (TensorRT FP16): docs/README_yolo.md

