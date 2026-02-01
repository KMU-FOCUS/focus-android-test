import cv2
import numpy as np
import time
import onnxruntime as ort
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from simple_tracker import SimpleTracker

# ==================== 설정 ====================
VIDEO_PATH = 'test_video.mp4'
OUTPUT_PATH = 'output_recognition_tracking.mp4'
YUNET_MODEL = 'yunet_face.onnx'
RECOGNITION_MODEL = 'w600k_r50.onnx'  # ArcFace 기반 얼굴 인식 모델
FACEMESH_MODEL = 'face_landmarker.task'  # 얼굴 정렬용
# Master 이미지 설정 (여러 방법 지원)
MASTER_FACE_IMAGE = 'test_face.png'  # 단일 이미지 경로
MASTER_FACE_DIR = 'master_faces'  # 여러 이미지 폴더
MULTI_OWNER_DIR = 'test_faces'  # 여러 주인공 폴더 (test_face1.png, test_face2.png, ...)
MASTER_EXTRACT_FROM_VIDEO = False  # 동영상에서 자동 추출
MASTER_EXTRACT_FRAMES = [10, 50, 100, 150, 200]  # 추출할 프레임 번호 (5장)
USE_FACE_ALIGNMENT = True  # 얼굴 정렬 사용 여부
USE_MAX_SIMILARITY = True  # True: 각도별 Master 임베딩과 최대 유사도 사용, False: 평균 임베딩 사용
USE_DYNAMIC_MASTER = True  # 동적 Master 업데이트: OWNER 판별 후 새로운 각도 얼굴 자동 수집
DYNAMIC_MASTER_THRESHOLD = 0.6  # 새 각도로 판단할 유사도 임계값 (이 값보다 낮으면 새 각도)
SAVE_DYNAMIC_MASTER_IMAGES = True  # 동적으로 수집한 Master 이미지 저장 여부
DYNAMIC_MASTER_DIR = 'dynamic_master_faces'  # 동적 Master 이미지 저장 폴더

SCALE = 0.7  # YuNet 처리 해상도

# Tracker 파라미터 (각도 변화에 강건하도록 조정)
MAX_IOU_DISTANCE = 0.7  # IoU 매칭 임계값
MAX_COSINE_DISTANCE = 0.6  # 코사인 거리 임계값 (0.4 → 0.6: 각도 변화 허용)
N_INIT = 1  # 즉시 확정
MAX_AGE = 10  # 미탐지 후 삭제 프레임 (1 → 10: 각도 변화 시 일시적 탐지 실패 허용)
DECISION_FRAMES = 5  # 본인/타인 판별까지 필요한 프레임 수
SIMILARITY_THRESHOLD = 0.4  # 본인/타인 구분 임계값 (0.4~0.45 권장, 여러 Master 이미지 사용 시)
USE_EMBEDDING_HISTORY = True  # 임베딩 히스토리 활용 (여러 임베딩과 비교)

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

# 2. MediaPipe FaceLandmarker (얼굴 정렬용)
if USE_FACE_ALIGNMENT:
    base_options = python.BaseOptions(model_asset_path=FACEMESH_MODEL)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    print("✓ MediaPipe FaceLandmarker 로드 완료 (얼굴 정렬용)")

# 3. w600k_r50 (얼굴 인식)
session = ort.InferenceSession(RECOGNITION_MODEL, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"✓ w600k_r50 로드 완료")
print(f"  입력: {input_name}, Shape: {input_shape}")
print(f"  출력: {output_name}")


# ==================== 유틸리티 함수 ====================
def align_face(face_img):
    """
    얼굴 정렬: 눈 위치를 기준으로 회전/크롭
    MediaPipe 랜드마크 사용
    """
    if not USE_FACE_ALIGNMENT:
        return face_img
    
    try:
        # RGB 변환
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
        
        # 랜드마크 추출
        results = face_landmarker.detect(mp_image)
        
        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return face_img  # 정렬 실패 시 원본 반환
        
        landmarks = results.face_landmarks[0]
        h, w = face_img.shape[:2]
        
        # 왼쪽 눈 중심 (인덱스 33, 133)
        # 오른쪽 눈 중심 (인덱스 362, 263)
        left_eye = np.array([
            (landmarks[33].x * w + landmarks[133].x * w) / 2,
            (landmarks[33].y * h + landmarks[133].y * h) / 2
        ])
        right_eye = np.array([
            (landmarks[362].x * w + landmarks[263].x * w) / 2,
            (landmarks[362].y * h + landmarks[263].y * h) / 2
        ])
        
        # 눈 사이 각도 계산
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # 눈 사이 거리
        eye_dist = np.sqrt(dx**2 + dy**2)
        
        # 회전 중심 (얼굴 중심)
        center = ((left_eye + right_eye) / 2).astype(int)
        
        # 회전 행렬
        rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
        
        # 회전 적용
        aligned = cv2.warpAffine(face_img, rotation_matrix, (w, h), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_REPLICATE)
        
        return aligned
        
    except Exception as e:
        # 정렬 실패 시 원본 반환
        return face_img


def preprocess_face(face_img):
    """112x112 정규화 (얼굴 정렬 + ArcFace 표준 전처리)"""
    # 얼굴 정렬 적용
    if USE_FACE_ALIGNMENT:
        face_img = align_face(face_img)
    
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


def is_front_face(face_img):
    """
    정면 얼굴 판단 (랜드마크 기반)
    Returns: (is_front, angle) - 정면 여부와 얼굴 각도 (도)
    """
    if not USE_FACE_ALIGNMENT:
        return False, None
    
    try:
        # RGB 변환
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
        
        # 랜드마크 추출
        results = face_landmarker.detect(mp_image)
        
        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return False, None
        
        landmarks = results.face_landmarks[0]
        h, w = face_img.shape[:2]
        
        # 왼쪽 눈 중심 (인덱스 33, 133)
        left_eye = np.array([
            (landmarks[33].x * w + landmarks[133].x * w) / 2,
            (landmarks[33].y * h + landmarks[133].y * h) / 2
        ])
        # 오른쪽 눈 중심 (인덱스 362, 263)
        right_eye = np.array([
            (landmarks[362].x * w + landmarks[263].x * w) / 2,
            (landmarks[362].y * h + landmarks[263].y * h) / 2
        ])
        
        # 눈 사이 각도 계산
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        
        # 정면 판단 (각도가 5도 이하)
        is_front = angle <= 5.0
        
        return is_front, angle
        
    except Exception as e:
        return False, None


# ==================== Master Embedding 생성 ====================
print("\n=== Master Embedding 생성 ===")

def load_multi_owner_images():
    """여러 주인공 이미지 로드 (test_faces/test_face*.png)"""
    owner_images_list = []  # 각 주인공별 이미지 리스트
    
    if os.path.exists(MULTI_OWNER_DIR) and os.path.isdir(MULTI_OWNER_DIR):
        print(f"👥 여러 주인공 이미지 로드: {MULTI_OWNER_DIR}")
        import re
        
        # test_face*.png 파일 찾기
        image_files = []
        for filename in os.listdir(MULTI_OWNER_DIR):
            if re.match(r'test_face\d+\.png', filename, re.IGNORECASE):
                image_files.append(filename)
        
        # 숫자 순서로 정렬
        image_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        
        for filename in image_files:
            img_path = os.path.join(MULTI_OWNER_DIR, filename)
            img = cv2.imread(img_path)
            if img is not None:
                owner_images_list.append([img])  # 각 주인공은 1장씩 (동적 업데이트로 확장)
                print(f"  ✓ {filename}")
        
        if owner_images_list:
            return owner_images_list
    
    return None


def load_master_images():
    """Master 이미지들을 로드 (test_faces 폴더만 사용)"""
    # test_faces 폴더에서 여러 주인공 이미지 로드 (필수)
    multi_owner = load_multi_owner_images()
    if multi_owner:
        return multi_owner  # 리스트의 리스트 반환
    
    # test_faces 폴더가 없거나 이미지가 없으면 에러
    print(f"❌ {MULTI_OWNER_DIR} 폴더를 찾을 수 없거나 이미지가 없습니다.")
    print(f"   필수: {MULTI_OWNER_DIR}/test_face1.png, test_face2.png, ...")
    return []


def create_master_embeddings(master_images, use_max_similarity=True):
    """
    여러 Master 이미지의 임베딩 생성
    use_max_similarity=True: 각도별 임베딩을 각각 저장 (최대 유사도 사용)
    use_max_similarity=False: 평균 임베딩 사용 (기존 방식)
    """
    if not master_images:
        return None
    
    print(f"\n📊 {len(master_images)}장의 Master 이미지로 임베딩 생성 중...")
    embeddings = []
    
    for i, img in enumerate(master_images):
        try:
            embedding = extract_embedding(img)
            embeddings.append(embedding)
            print(f"  [{i+1}/{len(master_images)}] 임베딩 추출 완료")
        except Exception as e:
            print(f"  ⚠️  [{i+1}/{len(master_images)}] 임베딩 추출 실패: {e}")
    
    if not embeddings:
        return None
    
    if use_max_similarity:
        # 각도별 임베딩을 각각 저장 (최대 유사도 방식)
        print(f"\n✓ Master Embeddings 생성 완료 (각도별 최대 유사도 방식)")
        print(f"  이미지 수: {len(embeddings)}장")
        print(f"  얼굴 정렬: {'사용' if USE_FACE_ALIGNMENT else '미사용'}")
        print(f"  방식: 각 Master 임베딩과 비교 후 최대 유사도 사용")
        
        # 각 이미지 간 유사도 확인 (품질 검증)
        similarities_matrix = []
        for i, emb1 in enumerate(embeddings):
            row = []
            for j, emb2 in enumerate(embeddings):
                sim = np.dot(emb1, emb2)
                row.append(sim)
            similarities_matrix.append(row)
        
        # 대각선 제외한 유사도들
        cross_similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                cross_similarities.append(similarities_matrix[i][j])
        
        if cross_similarities:
            min_sim = np.min(cross_similarities)
            max_sim = np.max(cross_similarities)
            avg_sim = np.mean(cross_similarities)
            
            print(f"  이미지 간 유사도:")
            print(f"    평균: {avg_sim:.4f}")
            print(f"    최소: {min_sim:.4f}")
            print(f"    최대: {max_sim:.4f}")
            
            if min_sim < 0.6:
                print(f"  ⚠️  경고: 일부 Master 이미지가 서로 많이 다릅니다 (최소 유사도: {min_sim:.4f})")
                print(f"     → 각도 차이가 큰 경우 정상입니다. 최대 유사도 방식이 효과적입니다.")
        
        # 여러 임베딩을 리스트로 반환 (각도별 임베딩)
        return embeddings
    else:
        # 평균 임베딩 계산 (기존 방식)
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        similarities = [np.dot(emb, avg_embedding) for emb in embeddings]
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        avg_sim = np.mean(similarities)
        
        print(f"\n✓ Master Embedding 생성 완료 (평균 임베딩 방식)")
        print(f"  이미지 수: {len(embeddings)}장")
        print(f"  얼굴 정렬: {'사용' if USE_FACE_ALIGNMENT else '미사용'}")
        print(f"  Shape: {avg_embedding.shape}")
        print(f"  L2 norm: {np.linalg.norm(avg_embedding):.6f}")
        print(f"  이미지 간 유사도:")
        print(f"    평균: {avg_sim:.4f}")
        print(f"    최소: {min_sim:.4f}")
        print(f"    최대: {max_sim:.4f}")
        
        return avg_embedding


# Master Embedding 생성 (여러 주인공 지원)
master_images = load_master_images()
if master_images:
    # 여러 주인공 모드인지 확인 (리스트의 리스트)
    is_multi_owner = isinstance(master_images[0], list) if master_images else False
    
    if is_multi_owner:
        # 여러 주인공 모드
        print(f"\n👥 여러 주인공 모드 활성화")
        print(f"  주인공 수: {len(master_images)}명")
        
        master_embedding = []  # 각 주인공별 Master Embedding 리스트
        for i, owner_images in enumerate(master_images):
            if USE_DYNAMIC_MASTER and len(owner_images) > 0:
                # 각 주인공별로 첫 번째 이미지만 사용 (동적 업데이트)
                initial_master_img = owner_images[0]
                if initial_master_img is not None and initial_master_img.size > 0:
                    owner_embeddings = [extract_embedding(initial_master_img)]
                    master_embedding.append(owner_embeddings)
                    print(f"  ✓ 주인공 {i+1}: 초기 Master Embedding 생성 (1장)")
            else:
                # 각 주인공별로 모든 이미지 사용
                owner_embeddings = create_master_embeddings(owner_images, use_max_similarity=USE_MAX_SIMILARITY)
                if isinstance(owner_embeddings, list):
                    master_embedding.append(owner_embeddings)
                else:
                    master_embedding.append([owner_embeddings])
        
        # 동적 Master 이미지 저장 폴더 생성
        if SAVE_DYNAMIC_MASTER_IMAGES and USE_DYNAMIC_MASTER:
            os.makedirs(DYNAMIC_MASTER_DIR, exist_ok=True)
            for i, owner_images in enumerate(master_images):
                if len(owner_images) > 0 and owner_images[0] is not None:
                    cv2.imwrite(os.path.join(DYNAMIC_MASTER_DIR, f'owner{i+1}_000_initial.jpg'), owner_images[0])
            print(f"  📁 저장 폴더: {DYNAMIC_MASTER_DIR}")
    else:
        # 단일 주인공 모드 (기존)
        if USE_DYNAMIC_MASTER and len(master_images) > 0:
            print(f"\n🔄 동적 Master 업데이트 모드 활성화")
            print(f"  초기 Master 이미지: 1장만 사용")
            print(f"  새 각도 임계값: {DYNAMIC_MASTER_THRESHOLD}")
            initial_master_img = master_images[0]
            if initial_master_img is not None and initial_master_img.size > 0:
                master_embedding = [extract_embedding(initial_master_img)]  # 리스트로 시작
                print(f"  ✓ 초기 Master Embedding 생성 완료 (1장)")
                
                # 동적 Master 이미지 저장 폴더 생성
                if SAVE_DYNAMIC_MASTER_IMAGES:
                    os.makedirs(DYNAMIC_MASTER_DIR, exist_ok=True)
                    # 초기 이미지도 저장
                    cv2.imwrite(os.path.join(DYNAMIC_MASTER_DIR, 'master_000_initial.jpg'), initial_master_img)
                    print(f"  📁 저장 폴더: {DYNAMIC_MASTER_DIR}")
            else:
                print(f"  ❌ 초기 Master 이미지가 유효하지 않습니다")
                master_embedding = None
        else:
            # 기존 방식: 모든 Master 이미지 사용
            master_embedding = create_master_embeddings(master_images, use_max_similarity=USE_MAX_SIMILARITY)
else:
    print(f"❌ Master 이미지를 찾을 수 없습니다")
    print(f"   필수: {MULTI_OWNER_DIR} 폴더에 test_face1.png, test_face2.png, ... 파일이 있어야 합니다.")
    print(f"   모든 Track은 'OTHER'로 분류됩니다.")
    master_embedding = None

# 3. SimpleTracker (master_embedding 전달)
tracker = SimpleTracker(
    max_iou_distance=MAX_IOU_DISTANCE,
    max_cosine_distance=MAX_COSINE_DISTANCE,
    n_init=N_INIT,
    max_age=MAX_AGE,
    decision_frames=DECISION_FRAMES,
    master_embedding=master_embedding,
    similarity_threshold=SIMILARITY_THRESHOLD,
    use_embedding_history=USE_EMBEDDING_HISTORY
)
print("✓ SimpleTracker 초기화 완료")
print(f"  판별 프레임 수: {DECISION_FRAMES}")
print(f"  유사도 임계값: {SIMILARITY_THRESHOLD}")
print(f"  코사인 거리 임계값: {MAX_COSINE_DISTANCE} (각도 변화 허용)")
print(f"  최대 미탐지 프레임: {MAX_AGE} (각도 변화 시 일시적 실패 허용)")
print(f"  임베딩 히스토리 활용: {'사용' if USE_EMBEDDING_HISTORY else '미사용'}")
if USE_DYNAMIC_MASTER:
    print(f"  🔄 동적 Master 업데이트: 활성화")
    if master_embedding is not None:
        initial_count = len(master_embedding) if isinstance(master_embedding, list) else 1
        print(f"  초기 Master 수: {initial_count}개")
        print(f"  새 각도 임계값: {DYNAMIC_MASTER_THRESHOLD}")
else:
    if master_embedding is not None:
        master_count = len(master_embedding) if isinstance(master_embedding, list) else 1
        print(f"  Master 수: {master_count}개")
        print(f"  💡 여러 Master 이미지 사용 시 임계값을 0.4~0.45로 낮추는 것을 권장합니다")
print()

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
print(f"출력: {OUTPUT_PATH}\n")

frame_count = 0
fps_start_time = time.time()
processing_times = []
track_stats = {}  # track_id별 통계
reentry_events = []  # 재진입 이벤트 기록

print("=== 처리 시작 ===")

# 이전 프레임의 track_id 집합
prev_track_ids = set()

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
            x, y, w, h = (face[:4] / SCALE).astype(int)
            confidence = face[14]

            face_crop = image[y:y + h, x:x + w]
            if face_crop.size == 0 or w < 30 or h < 30:
                continue

            try:
                embedding = extract_embedding(face_crop)
                detections.append([x, y, w, h])
                embeddings.append(embedding)
            except Exception as e:
                continue

    # 3. Tracker 업데이트
    tracks = tracker.update(detections, embeddings)

    # 3.3. OTHER Track의 정면 얼굴 재검사 (최초 1회만)
    if master_embedding is not None:
        for track in tracks:
            # OTHER로 판별되었고, 아직 재검사 안 한 Track만
            if (track.label == 'OTHER' and track.is_confirmed() and 
                not track.front_face_checked and track.time_since_update == 0):
                # 현재 프레임에서 해당 Track의 얼굴 crop
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                face_crop = image[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    # 정면 얼굴 판단
                    is_front, angle = is_front_face(face_crop)
                    
                    if is_front:
                        # 정면 얼굴 감지 → Master와 재검사 (이 프레임의 임베딩만 사용)
                        current_embedding = track.embedding  # 현재 프레임 임베딩
                        current_embedding = current_embedding / np.linalg.norm(current_embedding)
                        
                        # Master 임베딩과 비교 (여러 주인공 지원)
                        current_master = tracker.get_master_embedding()
                        max_sim = 0.0
                        matched_owner_index = None
                        
                        if isinstance(current_master, list) and len(current_master) > 0 and isinstance(current_master[0], list):
                            # 여러 주인공 모드
                            for i, owner_embeddings in enumerate(current_master):
                                owner_sims = [np.dot(emb, current_embedding) for emb in owner_embeddings]
                                owner_max_sim = max(owner_sims)
                                if owner_max_sim > max_sim:
                                    max_sim = owner_max_sim
                                    matched_owner_index = i
                        elif isinstance(current_master, list):
                            # 단일 주인공 각도별 모드
                            max_sim = max([np.dot(master_emb, current_embedding) for master_emb in current_master])
                        else:
                            # 단일 임베딩
                            max_sim = np.dot(current_master, current_embedding)
                        
                        # 재검사 결과: OWNER로 판별
                        if max_sim > SIMILARITY_THRESHOLD:
                            track.label = 'OWNER'
                            track.similarity = max_sim
                            track.front_face_checked = True
                            print(f"    ✅ [Track {track.track_id}] 정면 얼굴 재검사 → OWNER (유사도: {max_sim:.4f}, 각도: {angle:.1f}°)")
                            
                            # 동적 Master 업데이트도 수행
                            if USE_DYNAMIC_MASTER:
                                if max_sim < DYNAMIC_MASTER_THRESHOLD:
                                    tracker.update_master_embedding(current_embedding, owner_index=matched_owner_index)
                                    master_embedding = tracker.get_master_embedding()
                                    owner_str = f" (주인공 {matched_owner_index+1})" if matched_owner_index is not None else ""
                                    print(f"      🔄 새로운 각도 Master 추가{owner_str}")
                        else:
                            # 재검사했지만 여전히 OTHER
                            track.front_face_checked = True
                            print(f"    ⚠️  [Track {track.track_id}] 정면 얼굴 재검사 → 여전히 OTHER (유사도: {max_sim:.4f}, 각도: {angle:.1f}°)")

    # 3.5. 동적 Master 업데이트 (OWNER 판별 후 새로운 각도 수집)
    if USE_DYNAMIC_MASTER and master_embedding is not None:
        for track in tracks:
            # OWNER로 판별된 Track만 처리 (이번 프레임에서 새로 판별된 경우)
            if track.label == 'OWNER' and track.is_confirmed() and track.hits == DECISION_FRAMES:
                # 현재 Track의 평균 임베딩
                track_avg_embedding = np.mean(track.embeddings, axis=0)
                track_avg_embedding = track_avg_embedding / np.linalg.norm(track_avg_embedding)
                
                # 현재 Master 임베딩들과 비교 (여러 주인공 지원)
                current_master = tracker.get_master_embedding()
                max_sim = 0.0
                matched_owner_index = None
                
                if isinstance(current_master, list) and len(current_master) > 0 and isinstance(current_master[0], list):
                    # 여러 주인공 모드: 모든 주인공과 비교
                    for i, owner_embeddings in enumerate(current_master):
                        owner_sims = [np.dot(emb, track_avg_embedding) for emb in owner_embeddings]
                        owner_max_sim = max(owner_sims)
                        if owner_max_sim > max_sim:
                            max_sim = owner_max_sim
                            matched_owner_index = i
                elif isinstance(current_master, list):
                    # 단일 주인공 각도별 모드
                    max_sim = max([np.dot(master_emb, track_avg_embedding) for master_emb in current_master])
                else:
                    # 단일 임베딩
                    max_sim = np.dot(current_master, track_avg_embedding)
                
                # 새 각도로 판단 (유사도가 임계값보다 낮으면)
                if max_sim < DYNAMIC_MASTER_THRESHOLD:
                    # 새로운 각도 임베딩 추가 (매칭된 주인공에 추가)
                    tracker.update_master_embedding(track_avg_embedding, owner_index=matched_owner_index)
                    master_embedding = tracker.get_master_embedding()  # 업데이트
                    
                    owner_str = f" (주인공 {matched_owner_index+1})" if matched_owner_index is not None else ""
                    print(f"    🔄 [Track {track.track_id}] 새로운 각도 Master 추가 (유사도: {max_sim:.4f}){owner_str}")
                    
                    # 얼굴 이미지 저장 (선택적)
                    if SAVE_DYNAMIC_MASTER_IMAGES:
                        # 현재 프레임에서 해당 Track의 얼굴 crop
                        x1, y1, x2, y2 = map(int, track.to_ltrb())
                        face_crop = image[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            if matched_owner_index is not None:
                                # 여러 주인공 모드
                                owner_embeddings = current_master[matched_owner_index]
                                save_path = os.path.join(DYNAMIC_MASTER_DIR, 
                                                        f'owner{matched_owner_index+1}_{len(owner_embeddings):03d}_frame{frame_count:05d}_id{track.track_id}.jpg')
                            else:
                                # 단일 주인공 모드
                                current_master_count = len(current_master) if isinstance(current_master, list) else 1
                                save_path = os.path.join(DYNAMIC_MASTER_DIR, 
                                                        f'master_{current_master_count:03d}_frame{frame_count:05d}_id{track.track_id}.jpg')
                            cv2.imwrite(save_path, face_crop)
                            print(f"      💾 저장: {save_path}")

    # 현재 프레임의 track_id 집합
    current_track_ids = set()

    # 4. 추적 결과 시각화
    confirmed_tracks = 0
    owner_count = 0
    other_count = 0
    pending_count = 0

    for track in tracks:
        if not track.is_confirmed():
            continue

        # Detection과 매칭된 것만 표시
        if track.time_since_update > 0:
            continue

        confirmed_tracks += 1
        track_id = track.track_id
        current_track_ids.add(track_id)

        # 재진입 감지 (이전 프레임에 없었는데 현재 프레임에 있음)
        is_reentry = track_id not in prev_track_ids and track.hits > 1

        # bbox 좌표
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # track_id별 통계 수집
        if track_id not in track_stats:
            track_stats[track_id] = {
                'first_frame': frame_count,
                'last_frame': frame_count,
                'count': 0,
                'label': track.label,
                'similarity': track.similarity,
                'reentry_count': 0
            }

        track_stats[track_id]['last_frame'] = frame_count
        track_stats[track_id]['count'] += 1
        track_stats[track_id]['label'] = track.label
        track_stats[track_id]['similarity'] = track.similarity

        # 재진입 기록
        if is_reentry:
            track_stats[track_id]['reentry_count'] += 1
            reentry_events.append({
                'frame': frame_count,
                'track_id': track_id,
                'label': track.label
            })

        # label별 카운트
        if track.label == 'OWNER':
            owner_count += 1
        elif track.label == 'OTHER':
            other_count += 1
        else:  # PENDING
            pending_count += 1

        # label별 색상
        if track.label == 'OWNER':
            color = (0, 255, 0)  # 초록
            label_text = f"ID:{track_id} OWNER"
            if is_reentry:
                label_text += " (RE)"  # 재진입 표시
        elif track.label == 'OTHER':
            color = (0, 0, 255)  # 빨강
            label_text = f"ID:{track_id} OTHER"
        else:  # PENDING
            color = (255, 255, 0)  # 노랑
            label_text = f"ID:{track_id} {track.hits}/{DECISION_FRAMES}"

        # 바운딩 박스
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        # 라벨 배경
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image,
                      (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0] + 10, y1),
                      color, -1)

        # 라벨 텍스트
        cv2.putText(image, label_text, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 이전 프레임 track_id 업데이트
    prev_track_ids = current_track_ids.copy()

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
              f"추적: {confirmed_tracks}개 (본인:{owner_count}, 타인:{other_count}, 판별중:{pending_count}) | "
              f"FPS: {current_fps:.1f} | 처리: {avg_time:.1f}ms")

    # 화면 정보
    cv2.putText(image, f'Owner: {owner_count} | Other: {other_count} | Pending: {pending_count}',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 저장
    out.write(image)

cap.release()
out.release()
cv2.destroyAllWindows()

# ==================== 재진입 이벤트 ====================
print("\n=== 재진입 이벤트 ===")
print(f"총 재진입 횟수: {len(reentry_events)}")

if reentry_events:
    print("\n재진입 상세:")
    for event in reentry_events[:10]:  # 최대 10개만 표시
        print(f"  Frame {event['frame']:4d}: ID {event['track_id']:2d} ({event['label']})")

    if len(reentry_events) > 10:
        print(f"  ... 외 {len(reentry_events) - 10}건")

# ==================== 유사도 분석 ====================
print("\n=== 유사도 분석 ===")

# 유사도 수집
owner_sims = [s['similarity'] for tid, s in track_stats.items() if s['label'] == 'OWNER' and s['similarity'] > 0]
other_sims = [s['similarity'] for tid, s in track_stats.items() if s['label'] == 'OTHER' and s['similarity'] > 0]

if owner_sims:
    print(f"\n본인 Track 유사도:")
    print(f"  평균: {np.mean(owner_sims):.4f}")
    print(f"  최소: {np.min(owner_sims):.4f}")
    print(f"  최대: {np.max(owner_sims):.4f}")
    print(f"  중앙값: {np.median(owner_sims):.4f}")

if other_sims:
    print(f"\n타인 Track 유사도:")
    print(f"  평균: {np.mean(other_sims):.4f}")
    print(f"  최소: {np.min(other_sims):.4f}")
    print(f"  최대: {np.max(other_sims):.4f}")
    print(f"  중앙값: {np.median(other_sims):.4f}")

# 임계값 분석
if owner_sims and other_sims:
    print(f"\n현재 임계값: {SIMILARITY_THRESHOLD}")

    # False Negative: 본인인데 임계값 이하
    false_negatives = [s for s in owner_sims if s <= SIMILARITY_THRESHOLD]
    # False Positive: 타인인데 임계값 초과
    false_positives = [s for s in other_sims if s > SIMILARITY_THRESHOLD]

    if false_negatives:
        print(f"  ⚠️  본인이지만 OTHER로 분류될 위험: {len(false_negatives)}개")
    if false_positives:
        print(f"  ⚠️  타인이지만 OWNER로 분류될 위험: {len(false_positives)}개")

    # 최적 임계값 제안
    min_owner = np.min(owner_sims) if owner_sims else 0
    max_other = np.max(other_sims) if other_sims else 1

    if min_owner > max_other:
        optimal = (min_owner + max_other) / 2
        print(f"  💡 제안 임계값: {optimal:.4f} (본인/타인 완전 분리 가능)")
    else:
        print(f"  ⚠️  본인/타인 유사도 범위 겹침 발생")
        print(f"     본인 최소: {min_owner:.4f}")
        print(f"     타인 최대: {max_other:.4f}")
        print(f"  💡 해결책: Master 이미지를 동영상에서 추출 (5장 평균)")

# ==================== 주인공 ID 분석 ====================
EXPECTED_OWNER_IDS = [1, 4, 5, 10, 19, 35, 36, 39]  # 실제 주인공 ID 목록
print("\n=== 주인공 ID 분석 ===")
print(f"예상 주인공 ID: {EXPECTED_OWNER_IDS}")

found_owner_ids = []
missed_owner_ids = []
false_positive_ids = []

for track_id in EXPECTED_OWNER_IDS:
    if track_id in track_stats:
        stats = track_stats[track_id]
        label = stats.get('label', 'PENDING')
        similarity = stats.get('similarity', 0.0)
        
        if label == 'OWNER':
            found_owner_ids.append((track_id, similarity))
            print(f"  ✓ ID {track_id:2d}: OWNER (유사도: {similarity:.4f})")
        else:
            missed_owner_ids.append((track_id, similarity))
            print(f"  ✗ ID {track_id:2d}: {label} (유사도: {similarity:.4f}) - 주인공인데 오분류!")
    else:
        missed_owner_ids.append((track_id, None))
        print(f"  ? ID {track_id:2d}: 통계 없음 (짧게 등장했거나 판별 전 사라짐)")

# OWNER로 판별되었지만 주인공이 아닌 ID
for track_id, stats in track_stats.items():
    if track_id not in EXPECTED_OWNER_IDS and stats.get('label') == 'OWNER':
        false_positive_ids.append((track_id, stats.get('similarity', 0.0)))

if found_owner_ids:
    print(f"\n✓ 정확히 판별된 주인공: {len(found_owner_ids)}/{len(EXPECTED_OWNER_IDS)}")
    avg_sim = np.mean([s for _, s in found_owner_ids])
    print(f"  평균 유사도: {avg_sim:.4f}")

if missed_owner_ids:
    print(f"\n⚠️  오분류된 주인공: {len(missed_owner_ids)}개")
    for track_id, sim in missed_owner_ids:
        if sim is not None:
            print(f"  ID {track_id:2d}: 유사도 {sim:.4f} (임계값 {SIMILARITY_THRESHOLD:.2f} 미만)")
            if sim < 0.3:
                print(f"    → Master 이미지와 매우 다름. 동영상에서 해당 ID의 얼굴을 Master로 추가 권장")
        else:
            print(f"  ID {track_id:2d}: 통계 없음")

if false_positive_ids:
    print(f"\n⚠️  타인인데 OWNER로 판별: {len(false_positive_ids)}개")
    for track_id, sim in false_positive_ids:
        print(f"  ID {track_id:2d}: 유사도 {sim:.4f}")

# ==================== 추적 통계 ====================
print("\n=== 추적 통계 ===")
print(f"총 Track ID 수: {len(track_stats)}")

# label별 분류
owner_tracks = []
other_tracks = []
pending_tracks = []

for track_id, stats in track_stats.items():
    label = stats.get('label', 'PENDING')
    if label == 'OWNER':
        owner_tracks.append((track_id, stats))
    elif label == 'OTHER':
        other_tracks.append((track_id, stats))
    else:
        pending_tracks.append((track_id, stats))

# 본인 Track 출력
print(f"\n본인 Track: {len(owner_tracks)}개")
if owner_tracks:
    # 출현 횟수로 정렬
    owner_tracks.sort(key=lambda x: x[1]['count'], reverse=True)
    for track_id, stats in owner_tracks:
        duration = stats['last_frame'] - stats['first_frame']
        reentry = stats.get('reentry_count', 0)
        print(f"  ID {track_id:2d}: "
              f"Frame {stats['first_frame']:4d}~{stats['last_frame']:4d} "
              f"({duration:3d}프레임) | "
              f"탐지: {stats['count']:3d}회 | "
              f"유사도: {stats['similarity']:.3f} | "
              f"재진입: {reentry}회")

# 타인 Track 출력
print(f"\n타인 Track: {len(other_tracks)}개")
if other_tracks:
    # 출현 횟수로 정렬
    other_tracks.sort(key=lambda x: x[1]['count'], reverse=True)
    display_count = min(5, len(other_tracks))
    for track_id, stats in other_tracks[:display_count]:
        duration = stats['last_frame'] - stats['first_frame']
        print(f"  ID {track_id:2d}: "
              f"Frame {stats['first_frame']:4d}~{stats['last_frame']:4d} "
              f"({duration:3d}프레임) | "
              f"탐지: {stats['count']:3d}회 | "
              f"유사도: {stats['similarity']:.3f}")

    if len(other_tracks) > display_count:
        print(f"  ... 외 {len(other_tracks) - display_count}개")

# 판별 중인 Track
if pending_tracks:
    print(f"\n판별 중 Track: {len(pending_tracks)}개 (5프레임 미만)")

# 장기 추적 분석
long_tracks = [(tid, s) for tid, s in track_stats.items()
               if s['last_frame'] - s['first_frame'] > 100]
print(f"\n장기 추적 (100+ 프레임): {len(long_tracks)}개")
if long_tracks:
    for track_id, stats in long_tracks:
        label_str = "본인" if stats['label'] == 'OWNER' else "타인"
        print(f"  ID {track_id}: {label_str} (유사도: {stats['similarity']:.3f})")

# ==================== 처리 성능 ====================
print("\n=== 처리 성능 ===")
print(f"처리 프레임: {len(processing_times)}")
print(f"평균 처리 시간: {sum(processing_times) / len(processing_times):.2f}ms")
print(f"최소: {min(processing_times):.2f}ms")
print(f"최대: {max(processing_times):.2f}ms")
print(f"평균 FPS: {1000 / (sum(processing_times) / len(processing_times)):.1f}")

print(f"\n✅ 결과 저장: {OUTPUT_PATH}")

# ==================== 동적 Master 업데이트 통계 ====================
if USE_DYNAMIC_MASTER and master_embedding is not None:
    final_master = tracker.get_master_embedding()
    
    # 여러 주인공 모드인지 확인
    is_multi_owner = isinstance(final_master, list) and len(final_master) > 0 and isinstance(final_master[0], list)
    
    if is_multi_owner:
        # 여러 주인공 모드
        print(f"\n=== 동적 Master 업데이트 통계 (여러 주인공) ===")
        print(f"주인공 수: {len(final_master)}명")
        
        total_added = 0
        for i, owner_embeddings in enumerate(final_master):
            initial_count = 1  # 초기 1장
            final_count = len(owner_embeddings)
            added_count = final_count - initial_count
            total_added += added_count
            
            print(f"  주인공 {i+1}: 초기 {initial_count}개 → 최종 {final_count}개 (추가: {added_count}개)")
        
        print(f"총 추가된 각도: {total_added}개")
        
        if SAVE_DYNAMIC_MASTER_IMAGES and os.path.exists(DYNAMIC_MASTER_DIR):
            saved_files = [f for f in os.listdir(DYNAMIC_MASTER_DIR) if f.endswith(('.jpg', '.png'))]
            print(f"저장된 이미지: {len(saved_files)}개")
            print(f"  폴더: {DYNAMIC_MASTER_DIR}")
        
        if total_added > 0:
            print(f"\n💡 다양한 각도의 Master가 자동으로 수집되었습니다.")
            print(f"   다음 실행 시 더 정확한 인식이 가능합니다.")
    elif isinstance(final_master, list):
        # 단일 주인공 각도별 모드
        final_count = len(final_master)
        initial_count = 1  # 초기 1장
        added_count = final_count - initial_count
        
        print(f"\n=== 동적 Master 업데이트 통계 ===")
        print(f"초기 Master 수: {initial_count}개")
        print(f"최종 Master 수: {final_count}개")
        print(f"추가된 각도: {added_count}개")
        
        if SAVE_DYNAMIC_MASTER_IMAGES and os.path.exists(DYNAMIC_MASTER_DIR):
            saved_files = [f for f in os.listdir(DYNAMIC_MASTER_DIR) if f.endswith(('.jpg', '.png'))]
            print(f"저장된 이미지: {len(saved_files)}개")
            print(f"  폴더: {DYNAMIC_MASTER_DIR}")
        
        if added_count > 0:
            print(f"\n💡 다양한 각도의 Master가 자동으로 수집되었습니다.")
            print(f"   다음 실행 시 더 정확한 인식이 가능합니다.")

# ==================== 검증 결과 ====================
print("\n=== 검증 결과 ===")
print("1. 초록 박스 (OWNER) = 본인으로 판별")
print("2. 빨간 박스 (OTHER) = 타인으로 판별")
print("3. 노란 박스 (N/5) = 판별 중 (N프레임 수집됨)")
print("4. 재진입 시 새 ID 할당 후 다시 본인으로 판별되는지 확인")
print(f"5. 재진입 이벤트: {len(reentry_events)}건")