import cv2
import numpy as np
import time
import onnxruntime as ort
from simple_tracker import SimpleTracker

# ==================== 설정 ====================
VIDEO_PATH = 'test_video.mp4'
OUTPUT_PATH = 'output_simple_tracker.mp4'
YUNET_MODEL = 'yunet_face.onnx'
RECOGNITION_MODEL = 'w600k_r50.onnx'  # ArcFace 기반 얼굴 인식 모델

SCALE = 0.7  # YuNet 처리 해상도

# Tracker 파라미터
MAX_IOU_DISTANCE = 0.7  # IoU 매칭 임계값
MAX_COSINE_DISTANCE = 0.4  # 코사인 거리 임계값
N_INIT = 1  # 확정까지 필요한 연속 탐지 (즉시 확정)
MAX_AGE = 1  # 미탐지 후 즉시 삭제

# ==================== 모델 초기화 ====================
print("=== 모델 로딩 중 ===")

# 1. YuNet (얼굴 탐지)
detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 320),
    score_threshold=0.5,
    nms_threshold=0.3,
    top_k=5000
)
print("✓ YuNet 로드 완료")

# 2. w600k_r50 (Re-ID Feature)
session = ort.InferenceSession(RECOGNITION_MODEL, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"✓ w600k_r50 로드 완료")
print(f"  입력: {input_name}, Shape: {input_shape}")
print(f"  출력: {output_name}")

# 3. Simple Tracker
tracker = SimpleTracker(
    max_iou_distance=MAX_IOU_DISTANCE,
    max_cosine_distance=MAX_COSINE_DISTANCE,
    n_init=N_INIT,
    max_age=MAX_AGE
)
print("✓ Simple Tracker 초기화 완료\n")


# ==================== 유틸리티 함수 ====================
def preprocess_face(face_img):
    """112x112 정규화 (ArcFace 표준 전처리)"""
    face_resized = cv2.resize(face_img, (112, 112))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    # ArcFace 전처리: (pixel - 127.5) / 128.0
    face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
    # ONNX 입력 형식: [1, 3, 112, 112] (NCHW)
    face_transposed = np.transpose(face_normalized, (2, 0, 1))
    return np.expand_dims(face_transposed, axis=0)


def extract_embedding(face_img):
    """w600k_r50 임베딩 추출"""
    preprocessed = preprocess_face(face_img)
    outputs = session.run([output_name], {input_name: preprocessed})
    embedding = outputs[0][0]  # (512,) 또는 모델 출력 차원
    # L2 정규화 (ArcFace 표준)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


# 색상 팔레트
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255)
]


def get_color(track_id):
    """track_id에 따라 고유 색상 반환"""
    return COLORS[track_id % len(COLORS)]


# ==================== 동영상 처리 ====================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"동영상 파일을 열 수 없습니다: {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("=== 동영상 정보 ===")
print(f"해상도: {width}x{height}")
print(f"FPS: {fps:.1f}")
print(f"총 프레임: {total_frames}")
print(f"\nTracker 설정:")
print(f"  max_iou_distance: {MAX_IOU_DISTANCE}")
print(f"  max_cosine_distance: {MAX_COSINE_DISTANCE}")
print(f"  n_init: {N_INIT}")
print(f"  max_age: {MAX_AGE}")
print(f"\n출력: {OUTPUT_PATH}\n")

frame_count = 0
fps_start_time = time.time()
processing_times = []
track_stats = {}  # track_id별 통계

print("=== 처리 시작 ===")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    frame_start = time.time()

    # 1. YuNet 얼굴 탐지
    small_image = cv2.resize(image, None, fx=SCALE, fy=SCALE)
    detector.setInputSize((small_image.shape[1], small_image.shape[0]))
    _, faces = detector.detect(small_image)

    # 2. Detection 및 Embedding 준비
    detections = []
    embeddings = []

    if faces is not None:
        for face in faces:
            # 원본 크기로 좌표 복원
            x, y, w, h = (face[:4] / SCALE).astype(int)
            confidence = face[14]

            # 얼굴 crop
            face_crop = image[y:y + h, x:x + w]
            if face_crop.size == 0 or w < 30 or h < 30:
                continue

            # w600k_r50 임베딩 추출
            try:
                embedding = extract_embedding(face_crop)

                detections.append([x, y, w, h])
                embeddings.append(embedding)

            except Exception as e:
                continue

    # 3. Tracker 업데이트
    tracks = tracker.update(detections, embeddings)

    # 4. 추적 결과 시각화 (Detection과 매칭된 것만)
    confirmed_tracks = 0

    for track in tracks:
        if not track.is_confirmed():
            continue

        # 이번 프레임에서 Detection과 매칭되지 않은 경우 스킵 (예측만 있는 경우)
        if track.time_since_update > 0:
            continue

        confirmed_tracks += 1
        track_id = track.track_id

        # bbox 좌표
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # track_id별 통계 수집
        if track_id not in track_stats:
            track_stats[track_id] = {
                'first_frame': frame_count,
                'last_frame': frame_count,
                'count': 0
            }
        track_stats[track_id]['last_frame'] = frame_count
        track_stats[track_id]['count'] += 1

        # track_id별 고유 색상
        color = get_color(track_id)

        # 바운딩 박스
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 라벨
        label = f"ID:{track_id}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # 라벨 배경
        cv2.rectangle(image,
                      (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1),
                      color, -1)

        # 라벨 텍스트
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    frame_end = time.time()
    processing_time = (frame_end - frame_start) * 1000
    processing_times.append(processing_time)

    frame_count += 1
    progress = (frame_count / total_frames) * 100

    # 30프레임마다 진행상황
    if frame_count % 30 == 0:
        fps_end_time = time.time()
        current_fps = 30 / (fps_end_time - fps_start_time)
        fps_start_time = fps_end_time

        avg_time = sum(processing_times[-30:]) / 30
        print(f"진행: {progress:.1f}% | 탐지: {len(detections)}개 | "
              f"추적: {confirmed_tracks}개 (누적 ID: {len(track_stats)}) | "
              f"FPS: {current_fps:.1f} | 처리: {avg_time:.1f}ms")

    # 화면 정보
    cv2.putText(image, f'Detections: {len(detections)}',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f'Active Tracks: {confirmed_tracks}',
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f'Total IDs: {len(track_stats)}',
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 저장
    out.write(image)

cap.release()
out.release()
cv2.destroyAllWindows()

# ==================== 추적 통계 ====================
print("\n=== 추적 통계 ===")
print(f"총 Track ID 수: {len(track_stats)}")
print("\nTrack ID별 상세:")

# track_id별 정렬 (등장 프레임 순)
sorted_tracks = sorted(track_stats.items(), key=lambda x: x[1]['first_frame'])

for track_id, stats in sorted_tracks:
    duration = stats['last_frame'] - stats['first_frame']
    print(f"  ID {track_id:2d}: "
          f"Frame {stats['first_frame']:4d}~{stats['last_frame']:4d} "
          f"({duration:3d}프레임) | "
          f"탐지: {stats['count']:3d}회")

# 장기 추적 분석
long_tracks = [tid for tid, s in track_stats.items()
               if s['last_frame'] - s['first_frame'] > 100]
print(f"\n장기 추적 (100+ 프레임): {len(long_tracks)}개 ID")
if long_tracks:
    print(f"  ID: {long_tracks}")

# ==================== 처리 성능 ====================
print("\n=== 처리 성능 ===")
print(f"처리 프레임: {len(processing_times)}")
print(f"평균 처리 시간: {sum(processing_times) / len(processing_times):.2f}ms")
print(f"최소: {min(processing_times):.2f}ms")
print(f"최대: {max(processing_times):.2f}ms")
print(f"평균 FPS: {1000 / (sum(processing_times) / len(processing_times)):.1f}")

print(f"\n✅ 결과 저장: {OUTPUT_PATH}")
print("\n=== 확인 사항 ===")
print("1. 동일 인물이 같은 ID로 유지되는가? (Detection 있을 때만)")
print("2. ID 스위칭 발생하는가?")
print("3. Detection 없으면 박스가 즉시 사라지는가?")
print("4. 장기 추적 ID 중 가장 많이 등장하는 ID가 본인일 가능성 높음")