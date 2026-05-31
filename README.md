# HappyNewEar

HappyNewEar는 청각 장애인이 도로와 일상 공간에서 발생하는 위험 소리를 시각·촉각 정보로 인지할 수 있도록 제작한 웨어러블 임베디드 디바이스입니다. 4-mic-array로 소리를 수신하고, ODAS로 음원 방향과 RAW PCM 데이터를 분리한 뒤, TensorFlow Lite YAMNet 모델로 소리 종류를 분류합니다.

본 프로젝트는 제19회 임베디드SW경진대회 자유공모 부문 출품작이며 입선했습니다.

## 프로젝트 개요

시스템은 소리 수신, 소리 분석, 위험 정보 전달의 세 단계로 구성됩니다.

1. 4-mic-array가 주변 소리를 수신합니다.
2. ODAS가 음원 방향 데이터를 계산하고 beamforming을 통해 4개 채널의 RAW PCM 데이터를 생성합니다.
3. Python 프로그램이 ODAS와 socket으로 연결되어 위치 데이터와 RAW PCM 데이터를 실시간으로 수신합니다.
4. RAW PCM 데이터는 채널별로 분리되고 TensorFlow Lite YAMNet 모델을 통해 소리 종류로 분류됩니다.
5. 소리 방향은 Raspberry Pi Touch Display의 2D 좌표 그래프에 표시됩니다.
6. 위험군 소리가 감지되면 화면에 경고 이미지를 출력하고, 골전도 스피커로 위험 단계에 따른 알림음을 발생시킵니다.

## 사용 부품

| 구분 | 부품 |
| --- | --- |
| 메인 보드 | Raspberry Pi 4 |
| 입력 장치 | 4-mic-array |
| 표시 장치 | Raspberry Pi Touch Display (800x480) |
| 알림 장치 | 골전도 스피커 HUMBIRD |
| 전원 | 10000mAh 보조배터리 |
| 케이스 | 3D 프린팅 출력물 |

## 주요 기능

- ODAS 기반 음원 방향 추정
- 4채널 RAW PCM 데이터 수신 및 채널별 분리
- TensorFlow Lite YAMNet 기반 소리 분류
- 위험군 8종 소리 감지
- 소리 위치를 2D 좌표 그래프로 표시
- 위험도에 따른 골전도 스피커 알림
- 위험 상황별 이미지 표시
- 인터넷 없이 Raspberry Pi 단독 실행

## 위험군 소리

| 소리 | 위험 단계 | 출력 |
| --- | --- | --- |
| 고함 | 1 | 짧은 알림 |
| 알람 | 1 | 짧은 알림 |
| 응급차량 | 2 | 중간 알림 |
| 사이렌 | 2 | 중간 알림 |
| 경적 | 3 | 강한 알림 |
| 폭발 | 3 | 강한 알림 |
| 충돌 | 3 | 강한 알림 |
| 화재경보 | 3 | 강한 알림 |

## 음원 방향 추정

본 프로젝트에서는 4-mic-array와 ODAS(Open embeddeD Audition System)를 이용해 음원의 방향을 추정했습니다. 음원에서 발생한 소리는 각 마이크에 서로 다른 시간에 도달하며, 마이크 쌍 사이의 도달 시간차(TDOA, Time Difference of Arrival)는 두 마이크까지의 거리 차이로 변환할 수 있습니다.

두 마이크 `M_1`, `M_2`에 대해 음원 위치가 `S`일 때 다음 관계가 성립합니다.

```text
|d_1 - d_2| = c * Delta t
```

여기서 `d_1`, `d_2`는 음원과 각 마이크 사이의 거리, `c`는 음속, `Delta t`는 두 마이크 사이의 도달 시간차입니다. 이 식을 만족하는 음원 위치 후보들은 쌍곡선을 이루며, 여러 마이크 쌍에서 얻은 TDOA 정보를 함께 사용하면 음원의 위치 또는 도래 방향을 추정할 수 있습니다.

초기에는 이 쌍곡선 기반 위치 추정 방식을 직접 구현하려고 했지만, 실제 환경에서는 반사음, 주변 잡음, 샘플링 오차로 인해 여러 쌍곡선이 정확히 한 점에서 만나지 않았습니다. 실내외 소음이 섞이는 real world 환경에서는 직접 계산한 위치값이 안정적으로 수렴하지 않아, 다중 마이크 입력과 beamforming을 지원하는 ODAS 기반 구조로 전환했습니다.

ODAS는 4-mic-array 입력을 바탕으로 음원의 방향을 추정하고, Python 프로그램은 ODAS가 제공하는 `x`, `y` 좌표를 디바이스 기준 2D 평면에 표시합니다. 이를 통해 사용자는 소리 종류뿐 아니라 소리가 어느 방향에서 발생했는지도 함께 확인할 수 있습니다.

## 시스템 동작 Flow

1. `HappyNewEar.sh`가 Python 메인 프로그램을 실행합니다.
2. 메인 프로그램은 ODAS RAW PCM용 TCP server를 `127.0.0.1:9001`에 엽니다.
3. ODAS 위치 데이터용 TCP server를 `127.0.0.1:9000`에 엽니다.
4. RAW 수신, 위치 수신, 화면 출력을 각각 별도 thread로 실행합니다.
5. ODAS `odaslive` 프로세스를 실행합니다.
6. ODAS는 4-mic-array 입력을 처리하고 RAW PCM 데이터와 음원 위치 JSON을 Python server로 전송합니다.
7. RAW thread는 일정 시간 동안 수신한 PCM buffer를 하나의 window로 묶습니다.
8. 분류 thread는 PCM 데이터를 `numpy.float32` 배열로 변환하고 `[-1, 1]` 범위로 정규화합니다.
9. 정규화된 데이터는 4개 채널로 분리됩니다.
10. 각 채널 데이터는 YAMNet TFLite 모델에 입력되어 소리 종류로 변환됩니다.
11. 위치 thread는 ODAS JSON의 `x`, `y` 좌표를 Raspberry Pi Display 좌표로 변환합니다.
12. display thread는 4개 채널의 위치와 분류 결과를 화면에 표시합니다.
13. 분류 결과가 위험군 8종에 포함되면 해당 이미지를 화면 중앙에 표시합니다.
14. 위험 단계에 따라 골전도 스피커 알림음이 1~3회 재생됩니다.

## 코드 구조

```text
HappyNewEar/
├── HappyNewEar.sh
├── README.md
├── Report/
│   └── 2021ESWContest_자유공모_1155_HAPPYNEWEAR_개발완료보고서.pdf
├── csv/
│   ├── yamnet.tflite
│   └── yamnet_class_map.csv
├── img/
│   ├── 경적.png
│   ├── 고함.png
│   ├── 사이렌.png
│   ├── 알람.png
│   ├── 응급차량.png
│   ├── 충돌.png
│   ├── 폭발.png
│   └── 화재경보.png
└── src/
    ├── Classification.py
    └── main.py
```

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| `src/main.py` | ODAS socket 수신, 채널별 분류, pygame 화면 출력, 위험 알림 제어 |
| `src/Classification.py` | YAMNet TFLite 모델 로드, PCM 전처리, 소리 분류 |
| `HappyNewEar.sh` | Raspberry Pi에서 전체 프로그램 실행 |
| `csv/yamnet.tflite` | TensorFlow Lite 소리 분류 모델 |
| `csv/yamnet_class_map.csv` | 모델 출력 index와 class name 매핑 |
| `img/*.png` | 위험군 소리별 화면 경고 이미지 |

## 실행 방법

Raspberry Pi에서 4-mic-array, Raspberry Pi Touch Display, 골전도 스피커를 연결한 뒤 실행합니다.

```bash
chmod +x HappyNewEar.sh
./HappyNewEar.sh
```

직접 Python으로 실행할 수도 있습니다.

```bash
python3 src/main.py \
  --model-path csv/yamnet.tflite \
  --class-map-path csv/yamnet_class_map.csv \
  --image-dir img
```

ODAS 실행 파일과 설정 파일 위치가 다른 경우:

```bash
python3 src/main.py \
  --odas-bin odas/bin/odaslive \
  --odas-config odas/bin/odas.cfg
```

## 문제 해결 과정

- MAX9814 단일 마이크로는 거리와 방향을 안정적으로 계산하기 어려웠습니다. 4-mic-array와 ODAS를 사용해 음원 방향과 채널별 RAW PCM 데이터를 함께 얻도록 구조를 바꿨습니다.
- ODAS와 PyAudio가 동시에 같은 마이크를 사용할 수 없었습니다. PyAudio를 제거하고 ODAS socket에서 RAW PCM 데이터를 직접 수신하도록 변경했습니다.
- 32비트 Raspberry Pi OS에서 TensorFlow 전체 패키지가 동작하지 않는 문제가 있었습니다. TensorFlow Lite와 YAMNet TFLite 모델을 사용해 온디바이스 추론을 수행했습니다.
- 위치 수신, RAW PCM 수신, 화면 출력이 동시에 동작해야 했습니다. Python thread를 사용해 각 작업을 분리했습니다.
- 소켓 수신 버퍼가 밀리는 병목이 있었습니다. RAW PCM 분류 작업을 별도 thread로 분리해 수신 루프가 계속 동작하도록 구성했습니다.
