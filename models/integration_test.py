import cv2
import numpy as np
import time
import onnxruntime as ort
import os
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from simple_tracker import SimpleTracker
from datetime import datetime

# ==================== 설정 ====================
VIDEO_PATH = 'test_video.mp4'
OUTPUT_VIDEO_PATH = 'output_integration_mosaic.mp4'
OUTPUT_METADATA_PATH = 'output_metadata.json'
YUNET_MODEL = 'yunet_face.onnx'
RECOGNITION_MODEL = 'w600k_r50.onnx'
FACEMESH_MODEL = 'face_landmarker.task'
MULTI_OWNER_DIR = 'test_faces'

# Tracker 파라미터
SCALE = 0.7
MAX_IOU_DISTANCE = 0.7
MAX_COSINE_DISTANCE = 0.6
N_INIT = 1
MAX_AGE = 10
DECISION_FRAMES = 5
SIMILARITY_THRESHOLD = 0.4
USE_EMBEDDING_HISTORY = True
USE_FACE_ALIGNMENT = True
USE_DYNAMIC_MASTER = True
DYNAMIC_MASTER_THRESHOLD = 0.6  # 새 각도로 판단할 유사도 임계값
SAVE_DYNAMIC_MASTER_IMAGES = True  # 동적으로 수집한 Master 이미지 저장 여부
DYNAMIC_MASTER_DIR = 'dynamic_master_faces'  # 동적 Master 이미지 저장 폴더

# 모자이크 설정
MOSAIC_BLOCK_SIZE = 16  # 16x16 모자이크 블록

# 디버그용 시각화 (박스/텍스트) 표시 여부
# 서버 전송용 결과물에는 사각형/텍스트가 보이면 안 되므로 기본값은 False
DEBUG_DRAW_OVERLAYS = False

# Action Units 랜드마크 인덱스 (MediaPipe 468 포인트)
AU_LANDMARKS = {
    'AU1_eyebrow_inner': [70, 63, 105, 66, 107],  # 눈썹 안쪽
    'AU2_eyebrow_outer': [336, 296, 334, 293, 300],  # 눈썹 바깥
    'AU4_eyebrow_down': [66, 107, 336, 296],  # 눈썹 내림
    'AU6_cheek': [234, 454],  # 볼 올림
    'AU12_mouth_corner': [61, 291],  # 입꼬리
    'AU25_lips': [13, 14],  # 입술
    'AU26_jaw': [152],  # 턱
    'AU45_eye': [33, 133, 362, 263]  # 눈 (깜빡임)
}

# AU 계산에 필요한 주요 랜드마크 인덱스만 추출 (중복 제거)
REQUIRED_LANDMARK_INDICES = set()
for indices in AU_LANDMARKS.values():
    REQUIRED_LANDMARK_INDICES.update(indices)
REQUIRED_LANDMARK_INDICES = sorted(list(REQUIRED_LANDMARK_INDICES))  # 정렬된 리스트

# 얼굴 윤곽 랜드마크 인덱스 (MediaPipe 468 포인트 기준)
# 얼굴 윤곽선: 얼굴 가장자리 포인트들 (Face Oval)
# MediaPipe Face Mesh의 얼굴 윤곽선 인덱스
FACE_CONTOUR_INDICES = [
    # 얼굴 윤곽선 (Face Oval) - 시계방향
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]
# 중복 제거 및 정렬
FACE_CONTOUR_INDICES = sorted(list(set(FACE_CONTOUR_INDICES)))

# ==================== 모델 초기화 ====================
print("=== 모델 로딩 중 ===")

# 1. YuNet
detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 320),
    score_threshold=0.5,
    nms_threshold=0.3,
    top_k=5000
)
print("✓ YuNet 로드 완료")

# 2. MediaPipe FaceLandmarker (랜드마크 추출용 - 여러 얼굴 지원)
base_options = python.BaseOptions(model_asset_path=FACEMESH_MODEL)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=10,  # 최대 10명 동시 처리
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)
print("✓ MediaPipe FaceLandmarker 로드 완료 (랜드마크 추출용)")

# 3. w600k_r50
session = ort.InferenceSession(RECOGNITION_MODEL, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("✓ w600k_r50 로드 완료")

# ==================== 유틸리티 함수 ====================
def detect_landmarks_align_and_check_front(face_img, face_landmarker_single):
    """
    단일 얼굴용 FaceLandmarker를 한 번만 호출해서:
      - 랜드마크 추출
      - 눈 위치 기반 정렬
      - 얼굴 각도/정면 여부
    를 동시에 계산한다.
    
    Returns:
        aligned (np.ndarray): 정렬된 얼굴 이미지 (실패 시 원본)
        is_front (bool): 정면 여부
        angle (float or None): 좌우 기울기 각도(도)
    """
    if not face_landmarker_single:
        # Landmarker 자체가 없으면 정렬/정면판별 모두 스킵
        return face_img, False, None

    try:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
        results = face_landmarker_single.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return face_img, False, None

        landmarks = results.face_landmarks[0]
        h, w = face_img.shape[:2]

        # 양쪽 눈 중심 계산
        left_eye = np.array([
            (landmarks[33].x * w + landmarks[133].x * w) / 2,
            (landmarks[33].y * h + landmarks[133].y * h) / 2
        ])
        right_eye = np.array([
            (landmarks[362].x * w + landmarks[263].x * w) / 2,
            (landmarks[362].y * h + landmarks[263].y * h) / 2
        ])

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # 정면 여부 (각도 기준)
        is_front = abs(angle) <= 5.0

        # 정렬
        center = ((left_eye + right_eye) / 2).astype(int)
        rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
        aligned = cv2.warpAffine(
            face_img,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned, is_front, angle
    except Exception:
        # 실패 시 원본 리턴하고, 정면 아님으로 처리
        return face_img, False, None

def preprocess_face(face_img, face_landmarker_single=None):
    """112x112 정규화"""
    if USE_FACE_ALIGNMENT and face_landmarker_single:
        # 한 번의 MediaPipe 호출로 정렬 + 정면 판별까지 수행하되,
        # 정면 여부/각도는 여기서는 사용하지 않고 단순 정렬 결과만 사용한다.
        face_img, _, _ = detect_landmarks_align_and_check_front(face_img, face_landmarker_single)
    
    face_resized = cv2.resize(face_img, (112, 112))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
    face_transposed = np.transpose(face_normalized, (2, 0, 1))
    return np.expand_dims(face_transposed, axis=0)

def extract_embedding(face_img, face_landmarker_single=None):
    """w600k_r50 임베딩 추출"""
    preprocessed = preprocess_face(face_img, face_landmarker_single)
    outputs = session.run([output_name], {input_name: preprocessed})
    embedding = outputs[0][0]
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def apply_mosaic(image, x, y, w, h, block_size=MOSAIC_BLOCK_SIZE):
    """bbox 전체에 모자이크 적용 (기본 함수)"""
    height, width = image.shape[:2]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = min(w, width - x)
    h = min(h, height - y)
    
    if w <= 0 or h <= 0:
        return image
    
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return image
    
    actual_h, actual_w = roi.shape[:2]
    small_w = max(1, actual_w // block_size)
    small_h = max(1, actual_h // block_size)
    
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (actual_w, actual_h), interpolation=cv2.INTER_NEAREST)
    image[y:y+actual_h, x:x+actual_w] = mosaic
    return image

def apply_mosaic_with_mask(image, landmarks, block_size=MOSAIC_BLOCK_SIZE):
    """
    랜드마크로 얼굴 윤곽 마스크를 만들어 얼굴 영역만 모자이크 적용
    landmarks: 랜드마크 포인트 리스트 (index, x, y 포함)
    """
    if not landmarks or len(landmarks) == 0:
        return image
    
    height, width = image.shape[:2]
    
    # 얼굴 윤곽 포인트 추출
    contour_points = []
    for lm in landmarks:
        if lm['index'] in FACE_CONTOUR_INDICES:
            x, y = int(lm['x']), int(lm['y'])
            if 0 <= x < width and 0 <= y < height:
                contour_points.append([x, y])
    
    if len(contour_points) < 3:
        # 윤곽 포인트가 부족하면 bbox 전체 모자이크
        return image
    
    # Convex Hull을 사용하여 얼굴 영역 확실히 채우기
    contour_array = np.array(contour_points, dtype=np.int32)
    hull = cv2.convexHull(contour_array)
    
    # 모자이크 영역 계산 (hull을 포함하는 bbox)
    x_min = max(0, min([p[0][0] for p in hull]))
    y_min = max(0, min([p[0][1] for p in hull]))
    x_max = min(width, max([p[0][0] for p in hull]))
    y_max = min(height, max([p[0][1] for p in hull]))
    
    w = x_max - x_min
    h = y_max - y_min
    
    if w <= 0 or h <= 0:
        return image
    
    # ROI 영역의 상대 좌표로 변환
    roi_hull = hull.copy()
    roi_hull[:, 0, 0] -= x_min
    roi_hull[:, 0, 1] -= y_min
    
    # ROI 영역의 마스크 생성 (convex hull로 채우기)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_hull], 255)
    
    # 마스크가 제대로 생성되었는지 확인 (디버깅용)
    if np.sum(roi_mask) == 0:
        # 마스크 생성 실패 시 bbox 전체 모자이크
        return apply_mosaic(image, x_min, y_min, w, h, block_size)
    
    # 페더링 적용 (경계 부드럽게)
    roi_mask = cv2.GaussianBlur(roi_mask, (5, 5), 0)
    
    # 해당 영역만 모자이크 처리
    roi = image[y_min:y_max, x_min:x_max].copy()
    if roi.size == 0:
        return image
    
    actual_h, actual_w = roi.shape[:2]
    small_w = max(1, actual_w // block_size)
    small_h = max(1, actual_h // block_size)
    
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (actual_w, actual_h), interpolation=cv2.INTER_NEAREST)
    
    # 마스크 영역만 모자이크 적용 (알파 블렌딩)
    mask_normalized = roi_mask.astype(np.float32) / 255.0
    mask_normalized = np.expand_dims(mask_normalized, axis=2)  # (h, w, 1)
    
    image[y_min:y_max, x_min:x_max] = (
        mosaic * mask_normalized + 
        image[y_min:y_max, x_min:x_max] * (1 - mask_normalized)
    ).astype(np.uint8)
    
    return image

def extract_landmarks(image, bbox, padding_ratio=0.2, include_contour=True):
    """
    bbox 영역에서 랜드마크 추출
    - AU 계산에 필요한 주요 포인트
    - 얼굴 윤곽 포인트 (모자이크 마스크 생성용)
    """
    x, y, w, h = bbox
    width, height = image.shape[1], image.shape[0]
    
    # padding 추가
    padding = int(max(w, h) * padding_ratio)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    
    face_crop = image[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None
    
    # RGB 변환
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
    
    # 랜드마크 추출
    results = face_landmarker.detect(mp_image)
    
    if not results.face_landmarks or len(results.face_landmarks) == 0:
        return None
    
    # 첫 번째 얼굴의 랜드마크 사용
    landmarks = results.face_landmarks[0]
    
    # 필요한 인덱스 수집 (AU + 얼굴 윤곽)
    required_indices = set(REQUIRED_LANDMARK_INDICES)
    if include_contour:
        required_indices.update(FACE_CONTOUR_INDICES)
    required_indices = sorted(list(required_indices))
    
    # 랜드마크 포인트 추출
    landmark_points = []
    for idx in required_indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            # crop 좌표를 원본 이미지 좌표로 변환
            lm_x = lm.x * face_crop.shape[1] + x1
            lm_y = lm.y * face_crop.shape[0] + y1
            landmark_points.append({
                'index': int(idx),  # 인덱스도 함께 저장 (참조용)
                'x': round(float(lm_x), 3),
                'y': round(float(lm_y), 3),
                'z': round(float(lm.z), 3) if hasattr(lm, 'z') else 0.0
            })
    
    return landmark_points if len(landmark_points) > 0 else None

def calculate_action_units(landmarks):
    """
    랜드마크로부터 Action Units 계산 (간단한 거리/각도 기반)
    
    주의: 모바일에서는 사용하지 않음. 서버에서 랜드마크로 AU를 계산할 때 참고용으로 남겨둠.
    """
    if not landmarks or len(landmarks) == 0:
        return {f'AU{i}': 0.0 for i in [1, 2, 4, 6, 12, 25, 26, 45]}
    
    # 인덱스로 랜드마크 찾기 (dict 리스트에서)
    def get_point(idx):
        for lm in landmarks:
            if lm.get('index') == idx:
                return np.array([lm['x'], lm['y']])
        return None
    
    def distance(p1_idx, p2_idx):
        p1 = get_point(p1_idx)
        p2 = get_point(p2_idx)
        if p1 is None or p2 is None:
            return 0.0
        return np.linalg.norm(p1 - p2)
    
    def angle(p1_idx, p2_idx, p3_idx):
        p1 = get_point(p1_idx)
        p2 = get_point(p2_idx)
        p3 = get_point(p3_idx)
        if p1 is None or p2 is None or p3 is None:
            return 0.0
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.arccos(np.clip(cos_angle, -1, 1))
    
    # 기준값 (정면 얼굴 기준, 실제로는 더 정교한 계산 필요)
    # 여기서는 간단한 상대적 변화량으로 계산
    au_values = {}
    
    # AU1: 눈썹 안쪽 올림 (눈썹-눈 거리)
    if all(get_point(i) is not None for i in [70, 33]):
        au_values['AU1'] = min(1.0, distance(70, 33) / 50.0)  # 정규화
    else:
        au_values['AU1'] = 0.0
    
    # AU2: 눈썹 바깥 올림
    if all(get_point(i) is not None for i in [336, 362]):
        au_values['AU2'] = min(1.0, distance(336, 362) / 50.0)
    else:
        au_values['AU2'] = 0.0
    
    # AU4: 눈썹 내림
    au_values['AU4'] = max(0.0, 1.0 - au_values['AU1'] - au_values['AU2'])
    
    # AU6: 볼 올림 (볼 포인트 높이)
    if all(get_point(i) is not None for i in [234, 454]):
        cheek_y = (get_point(234)[1] + get_point(454)[1]) / 2
        eye_y = (get_point(33)[1] + get_point(362)[1]) / 2 if all(get_point(i) is not None for i in [33, 362]) else cheek_y
        au_values['AU6'] = min(1.0, max(0.0, (eye_y - cheek_y) / 100.0))
    else:
        au_values['AU6'] = 0.0
    
    # AU12: 입꼬리 당김 (입꼬리 거리)
    if all(get_point(i) is not None for i in [61, 291]):
        au_values['AU12'] = min(1.0, distance(61, 291) / 80.0)
    else:
        au_values['AU12'] = 0.0
    
    # AU25: 입술 벌림
    if all(get_point(i) is not None for i in [13, 14]):
        au_values['AU25'] = min(1.0, distance(13, 14) / 30.0)
    else:
        au_values['AU25'] = 0.0
    
    # AU26: 턱 내림
    if get_point(152) is not None and get_point(13) is not None:
        au_values['AU26'] = min(1.0, max(0.0, (get_point(152)[1] - get_point(13)[1]) / 50.0))
    else:
        au_values['AU26'] = 0.0
    
    # AU45: 눈 깜빡임 (눈 높이)
    if all(get_point(i) is not None for i in [33, 133, 362, 263]):
        eye_height = (distance(33, 133) + distance(362, 263)) / 2
        au_values['AU45'] = min(1.0, max(0.0, 1.0 - eye_height / 20.0))
    else:
        au_values['AU45'] = 0.0
    
    return au_values

def create_alpha_mask(image_shape, bbox):
    """얼굴 영역 알파 마스크 생성 (1채널)"""
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    x, y, w, h = bbox
    
    # 이미지 경계 체크
    height, width = image_shape[0], image_shape[1]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = min(w, width - x)
    h = min(h, height - y)
    
    if w <= 0 or h <= 0:
        return mask
    
    # 페더링 적용 (5px, bbox 크기에 맞춰 조정)
    feather = min(5, w // 2, h // 2)
    if feather <= 0:
        # 페더링 불가능하면 전체 영역 채우기
        mask[y:y+h, x:x+w] = 255
        return mask
    
    # 중앙 영역 (255)
    y1_center = y + feather
    y2_center = y + h - feather
    x1_center = x + feather
    x2_center = x + w - feather
    
    if y2_center > y1_center and x2_center > x1_center:
        mask[y1_center:y2_center, x1_center:x2_center] = 255
    
    # 페더링 영역 (그라데이션)
    for i in range(feather):
        alpha = int(255 * (i + 1) / feather)
        
        # 상단/하단 경계
        if y + i < height and x1_center < width and x2_center > 0:
            x_start = max(0, x1_center)
            x_end = min(width, x2_center)
            if x_end > x_start:
                mask[y + i, x_start:x_end] = alpha
        
        if y + h - i - 1 >= 0 and y + h - i - 1 < height and x1_center < width and x2_center > 0:
            x_start = max(0, x1_center)
            x_end = min(width, x2_center)
            if x_end > x_start:
                mask[y + h - i - 1, x_start:x_end] = alpha
        
        # 좌측/우측 경계
        if x + i < width and y1_center < height and y2_center > 0:
            y_start = max(0, y1_center)
            y_end = min(height, y2_center)
            if y_end > y_start:
                mask[y_start:y_end, x + i] = alpha
        
        if x + w - i - 1 >= 0 and x + w - i - 1 < width and y1_center < height and y2_center > 0:
            y_start = max(0, y1_center)
            y_end = min(height, y2_center)
            if y_end > y_start:
                mask[y_start:y_end, x + w - i - 1] = alpha
    
    return mask

def is_front_face(face_img, face_landmarker_single):
    """
    정면 얼굴 판단 (랜드마크 기반)
    Returns: (is_front, angle) - 정면 여부와 얼굴 각도 (도)
    """
    if not USE_FACE_ALIGNMENT or face_landmarker_single is None:
        return False, None

    # 정렬 + 정면판별을 한 번에 수행하고, 여기서는 정면 여부/각도만 사용
    _, is_front, angle = detect_landmarks_align_and_check_front(face_img, face_landmarker_single)
    return is_front, angle

# ==================== Master Embedding 생성 ====================
print("\n=== Master Embedding 생성 ===")

# 단일 얼굴용 FaceLandmarker (얼굴 정렬용) - 전역 변수로 선언
face_landmarker_single = None
if USE_FACE_ALIGNMENT:
    base_options_single = python.BaseOptions(model_asset_path=FACEMESH_MODEL)
    options_single = vision.FaceLandmarkerOptions(
        base_options=base_options_single,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5
    )
    face_landmarker_single = vision.FaceLandmarker.create_from_options(options_single)
    print("✓ 단일 얼굴용 FaceLandmarker 로드 완료 (얼굴 정렬용)")

def load_multi_owner_images():
    owner_images_list = []
    if os.path.exists(MULTI_OWNER_DIR) and os.path.isdir(MULTI_OWNER_DIR):
        print(f"👥 여러 주인공 이미지 로드: {MULTI_OWNER_DIR}")
        import re
        image_files = []
        for filename in os.listdir(MULTI_OWNER_DIR):
            if re.match(r'test_face\d+\.png', filename, re.IGNORECASE):
                image_files.append(filename)
        image_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        
        for filename in image_files:
            img_path = os.path.join(MULTI_OWNER_DIR, filename)
            img = cv2.imread(img_path)
            if img is not None:
                owner_images_list.append([img])
                print(f"  ✓ {filename}")
        if owner_images_list:
            return owner_images_list
    return None

master_images = load_multi_owner_images()
if not master_images:
    print(f"❌ {MULTI_OWNER_DIR} 폴더를 찾을 수 없거나 이미지가 없습니다.")
    print(f"   모든 Track은 'OTHER'로 분류됩니다.")
    master_embedding = None
else:
    print(f"\n👥 여러 주인공 모드 활성화")
    print(f"  주인공 수: {len(master_images)}명")
    
    # 동적 Master 이미지 저장 폴더 생성
    if SAVE_DYNAMIC_MASTER_IMAGES and USE_DYNAMIC_MASTER:
        os.makedirs(DYNAMIC_MASTER_DIR, exist_ok=True)
        print(f"  📁 저장 폴더: {DYNAMIC_MASTER_DIR}")
    
    master_embedding = []
    for i, owner_images in enumerate(master_images):
        if len(owner_images) > 0:
            initial_master_img = owner_images[0]
            if initial_master_img is not None and initial_master_img.size > 0:
                owner_embeddings = [extract_embedding(initial_master_img, face_landmarker_single)]
                master_embedding.append(owner_embeddings)
                print(f"  ✓ 주인공 {i+1}: 초기 Master Embedding 생성 (1장)")
                
                # 초기 이미지 저장
                if SAVE_DYNAMIC_MASTER_IMAGES:
                    cv2.imwrite(os.path.join(DYNAMIC_MASTER_DIR, f'owner{i+1}_000_initial.jpg'), initial_master_img)

# Tracker 초기화
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
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

print("\n=== 동영상 정보 ===")
print(f"해상도: {width}x{height}")
print(f"FPS: {fps:.1f}")
print(f"총 프레임: {total_frames}")
print(f"출력 비디오: {OUTPUT_VIDEO_PATH}")
print(f"출력 메타데이터: {OUTPUT_METADATA_PATH}\n")

frame_count = 0
fps_start_time = time.time()
processing_times = []
metadata_frames = []  # 프레임별 메타데이터
prev_track_ids = set()
track_stats = {}  # track_id별 통계
reentry_events = []  # 재진입 이벤트 기록

print("=== 처리 시작 ===")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    frame_start = time.time()
    timestamp = frame_count / fps  # 초 단위 타임스탬프
    
    # 원본 복사 (모자이크 적용용)
    mosaic_image = image.copy()
    
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
                # 단일 얼굴용 FaceLandmarker로 임베딩 추출
                if USE_FACE_ALIGNMENT and face_landmarker_single is not None:
                    embedding = extract_embedding(face_crop, face_landmarker_single)
                else:
                    embedding = extract_embedding(face_crop)
                
                detections.append([x, y, w, h])
                embeddings.append(embedding)
            except Exception as e:
                continue
    
    # 3. Tracker 업데이트
    tracks = tracker.update(detections, embeddings)
    
    # 3.1. OTHER Track의 정면 얼굴 재검사 (최초 1회만)
    if master_embedding is not None and face_landmarker_single is not None:
        for track in tracks:
            # OTHER로 판별되었고, 아직 재검사 안 한 Track만
            if (track.label == 'OTHER' and track.is_confirmed() and 
                not track.front_face_checked and track.time_since_update == 0):
                # 현재 프레임에서 해당 Track의 얼굴 crop
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                face_crop = image[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    # 정면 얼굴 판단
                    is_front, angle = is_front_face(face_crop, face_landmarker_single)
                    
                    if is_front:
                        # 정면 얼굴 감지 → Master와 재검사
                        current_embedding = track.embedding
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
    
    # 3.2. 동적 Master 업데이트 (OWNER 판별 후 새로운 각도 수집)
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
                    # 여러 주인공 모드
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
                    # 새로운 각도 임베딩 추가
                    tracker.update_master_embedding(track_avg_embedding, owner_index=matched_owner_index)
                    master_embedding = tracker.get_master_embedding()
                    
                    owner_str = f" (주인공 {matched_owner_index+1})" if matched_owner_index is not None else ""
                    print(f"    🔄 [Track {track.track_id}] 새로운 각도 Master 추가 (유사도: {max_sim:.4f}){owner_str}")
                    
                    # 얼굴 이미지 저장 (선택적)
                    if SAVE_DYNAMIC_MASTER_IMAGES:
                        x1, y1, x2, y2 = map(int, track.to_ltrb())
                        face_crop = image[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            os.makedirs(DYNAMIC_MASTER_DIR, exist_ok=True)
                            if matched_owner_index is not None:
                                owner_embeddings = current_master[matched_owner_index]
                                save_path = os.path.join(DYNAMIC_MASTER_DIR, 
                                                        f'owner{matched_owner_index+1}_{len(owner_embeddings):03d}_frame{frame_count:05d}_id{track.track_id}.jpg')
                            else:
                                current_master_count = len(current_master) if isinstance(current_master, list) else 1
                                save_path = os.path.join(DYNAMIC_MASTER_DIR, 
                                                        f'master_{current_master_count:03d}_frame{frame_count:05d}_id{track.track_id}.jpg')
                            cv2.imwrite(save_path, face_crop)
    
    # 4. 프레임별 메타데이터 초기화
    frame_metadata = {
        'frame_number': frame_count,
        'timestamp': float(timestamp),
        'faces': []
    }
    
    # 5. OTHER/PENDING 얼굴 처리: 모자이크 + 랜드마크 추출
    current_track_ids = set()
    
    for track in tracks:
        # Detection과 매칭된 것만 처리 (예측만 있는 경우 제외)
        if track.time_since_update > 0:
            continue
        
        # PENDING 상태도 포함 (안전을 위해 판별 전에도 모자이크)
        # is_confirmed() 체크 제거하여 PENDING 상태도 처리
        
        track_id = track.track_id
        current_track_ids.add(track_id)
        
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]
        
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
        
        # 재진입 감지
        is_reentry = track_id not in prev_track_ids and track.hits > 1
        if is_reentry:
            track_stats[track_id]['reentry_count'] += 1
            reentry_events.append({
                'frame': frame_count,
                'track_id': track_id,
                'label': track.label
            })
        
        # OTHER 또는 PENDING 얼굴 처리
        if track.label == 'OTHER' or track.label == 'PENDING':
            # OTHER는 랜드마크 추출 먼저 시도 (랜드마크가 없으면 식별 어려운 얼굴이므로 스킵)
            if track.label == 'OTHER':
                landmarks = extract_landmarks(image, bbox)
                
                # 랜드마크가 없으면 모자이크 처리 및 메타데이터 추가 스킵
                if landmarks is None or len(landmarks) == 0:
                    # 디버그 시각화만 (회색 박스로 표시 - 처리 스킵됨)
                    if DEBUG_DRAW_OVERLAYS:
                        cv2.rectangle(mosaic_image, (x1, y1), (x2, y2), (128, 128, 128), 2)
                        cv2.putText(mosaic_image, f"ID:{track_id} SKIP", (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                    continue  # 다음 얼굴로
                
                # 랜드마크가 있으면 모자이크 처리 및 메타데이터 생성
                # 랜드마크 기반 얼굴 윤곽 마스크로 정확한 얼굴 영역만 모자이크
                mosaic_image = apply_mosaic_with_mask(mosaic_image, landmarks)
                
                # Action Units는 서버에서 랜드마크로 계산하므로 모바일에서는 계산하지 않음
                
                # 알파 마스크는 서버에서 랜드마크로 재구성 가능하므로 JSON에는 포함하지 않음
                # alpha_mask = create_alpha_mask(image.shape, bbox)
                
                # 메타데이터에 추가 (서버에 필요한 필드만 전송)
                face_metadata = {
                    'tracking_id': int(track_id),
                    'bbox': bbox,
                    'landmarks': landmarks  # 서버에서 AU 계산 및 아바타 합성에 사용
                }
                frame_metadata['faces'].append(face_metadata)
                
                # 디버그 시각화 (빨간 박스)
                if DEBUG_DRAW_OVERLAYS:
                    cv2.rectangle(mosaic_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(mosaic_image, f"ID:{track_id} OTHER", (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            else:  # PENDING
                # PENDING은 안전을 위해 모자이크 처리 (랜드마크 추출 안함)
                mosaic_image = apply_mosaic(mosaic_image, bbox[0], bbox[1], bbox[2], bbox[3])
                
                # 디버그 시각화 (노란 박스)
                if DEBUG_DRAW_OVERLAYS:
                    cv2.rectangle(mosaic_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(mosaic_image, f"ID:{track_id} {track.hits}/{DECISION_FRAMES}", (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        elif track.label == 'OWNER':
            # OWNER는 초록 박스만 (모자이크 없음, 디버그 시각화 전용)
            if DEBUG_DRAW_OVERLAYS:
                cv2.rectangle(mosaic_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(mosaic_image, f"ID:{track_id} OWNER", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    prev_track_ids = current_track_ids.copy()
    
    # 프레임 메타데이터 저장
    metadata_frames.append(frame_metadata)
    
    # 진행 상황 출력
    frame_end = time.time()
    processing_time = (frame_end - frame_start) * 1000
    processing_times.append(processing_time)
    
    frame_count += 1
    progress = (frame_count / total_frames) * 100
    
    if frame_count % 30 == 0:
        fps_end_time = time.time()
        current_fps = 30 / (fps_end_time - fps_start_time)
        fps_start_time = fps_end_time
        
        avg_time = sum(processing_times[-30:]) / 30
        other_count = sum(1 for t in tracks if t.time_since_update == 0 and t.label == 'OTHER')
        pending_count = sum(1 for t in tracks if t.time_since_update == 0 and t.label == 'PENDING')
        owner_count = sum(1 for t in tracks if t.time_since_update == 0 and t.label == 'OWNER')
        
        print(f"진행: {progress:.1f}% | 탐지: {len(detections)}개 | "
              f"추적: {len([t for t in tracks if t.time_since_update == 0])}개 "
              f"(본인:{owner_count}, 타인:{other_count}, 판별중:{pending_count}) | "
              f"FPS: {current_fps:.1f} | 처리: {avg_time:.1f}ms")
    
    # 화면 정보
    if DEBUG_DRAW_OVERLAYS:
        other_count = sum(1 for t in tracks if t.time_since_update == 0 and t.label == 'OTHER')
        pending_count = sum(1 for t in tracks if t.time_since_update == 0 and t.label == 'PENDING')
        owner_count = sum(1 for t in tracks if t.time_since_update == 0 and t.label == 'OWNER')
        cv2.putText(mosaic_image, f'Owner: {owner_count} | Other: {other_count} | Pending: {pending_count}',
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 저장
    out.write(mosaic_image)

cap.release()
out.release()

# ==================== 메타데이터 저장 ====================
print("\n=== 메타데이터 저장 ===")

metadata = {
    'video_info': {
        'width': int(width),
        'height': int(height),
        'fps': float(fps)
    },
    'frames': metadata_frames
}

with open(OUTPUT_METADATA_PATH, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, separators=(',', ':'))

print(f"✓ 메타데이터 저장 완료: {OUTPUT_METADATA_PATH}")
print(f"  총 프레임: {len(metadata_frames)}")
print(f"  OTHER 얼굴 총 수: {sum(len(f['faces']) for f in metadata_frames)}")

# ==================== 통계 ====================
print("\n=== 처리 통계 ===")
print(f"처리 프레임: {len(processing_times)}")
print(f"평균 처리 시간: {sum(processing_times) / len(processing_times):.2f}ms")
print(f"최소: {min(processing_times):.2f}ms")
print(f"최대: {max(processing_times):.2f}ms")
print(f"평균 FPS: {1000 / (sum(processing_times) / len(processing_times)):.1f}")

# OTHER 얼굴 통계
total_other_faces = sum(len(f['faces']) for f in metadata_frames)
frames_with_other = sum(1 for f in metadata_frames if len(f['faces']) > 0)
frames_with_landmarks = sum(1 for f in metadata_frames for face in f['faces'] if face['landmarks'] is not None)

print(f"\nOTHER 얼굴 통계:")
print(f"  총 OTHER 얼굴 수: {total_other_faces}")
print(f"  OTHER 얼굴이 있는 프레임: {frames_with_other}")
print(f"  랜드마크 추출 성공: {frames_with_landmarks}")

print(f"\n✅ 통합 테스트 완료")
print(f"  비디오: {OUTPUT_VIDEO_PATH}")
print(f"  메타데이터: {OUTPUT_METADATA_PATH}")

# ==================== 재진입 이벤트 ====================
print("\n=== 재진입 이벤트 ===")
print(f"총 재진입 횟수: {len(reentry_events)}")
if reentry_events:
    print("\n재진입 상세:")
    for event in reentry_events[:10]:
        print(f"  Frame {event['frame']:4d}: ID {event['track_id']:2d} ({event['label']})")
    if len(reentry_events) > 10:
        print(f"  ... 외 {len(reentry_events) - 10}건")

# ==================== 유사도 분석 ====================
print("\n=== 유사도 분석 ===")
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

# ==================== 추적 통계 ====================
print("\n=== 추적 통계 ===")
print(f"총 Track ID 수: {len(track_stats)}")

owner_tracks = [(tid, s) for tid, s in track_stats.items() if s['label'] == 'OWNER']
other_tracks = [(tid, s) for tid, s in track_stats.items() if s['label'] == 'OTHER']

print(f"\n본인 Track: {len(owner_tracks)}개")
if owner_tracks:
    owner_tracks.sort(key=lambda x: x[1]['count'], reverse=True)
    for track_id, stats in owner_tracks[:5]:
        duration = stats['last_frame'] - stats['first_frame']
        reentry = stats.get('reentry_count', 0)
        print(f"  ID {track_id:2d}: Frame {stats['first_frame']:4d}~{stats['last_frame']:4d} "
              f"({duration:3d}프레임) | 탐지: {stats['count']:3d}회 | "
              f"유사도: {stats['similarity']:.3f} | 재진입: {reentry}회")

print(f"\n타인 Track: {len(other_tracks)}개")
if other_tracks:
    other_tracks.sort(key=lambda x: x[1]['count'], reverse=True)
    for track_id, stats in other_tracks[:5]:
        duration = stats['last_frame'] - stats['first_frame']
        print(f"  ID {track_id:2d}: Frame {stats['first_frame']:4d}~{stats['last_frame']:4d} "
              f"({duration:3d}프레임) | 탐지: {stats['count']:3d}회 | "
              f"유사도: {stats['similarity']:.3f}")

# ==================== 동적 Master 업데이트 통계 ====================
if USE_DYNAMIC_MASTER and master_embedding is not None:
    final_master = tracker.get_master_embedding()
    is_multi_owner = isinstance(final_master, list) and len(final_master) > 0 and isinstance(final_master[0], list)
    
    if is_multi_owner:
        print(f"\n=== 동적 Master 업데이트 통계 (여러 주인공) ===")
        print(f"주인공 수: {len(final_master)}명")
        total_added = 0
        for i, owner_embeddings in enumerate(final_master):
            initial_count = 1
            final_count = len(owner_embeddings)
            added_count = final_count - initial_count
            total_added += added_count
            print(f"  주인공 {i+1}: 초기 {initial_count}개 → 최종 {final_count}개 (추가: {added_count}개)")
        print(f"총 추가된 각도: {total_added}개")
    elif isinstance(final_master, list):
        final_count = len(final_master)
        initial_count = 1
        added_count = final_count - initial_count
        print(f"\n=== 동적 Master 업데이트 통계 ===")
        print(f"초기 Master 수: {initial_count}개")
        print(f"최종 Master 수: {final_count}개")
        print(f"추가된 각도: {added_count}개")

