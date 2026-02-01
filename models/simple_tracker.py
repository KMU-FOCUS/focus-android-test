import numpy as np
from scipy.optimize import linear_sum_assignment


# ==================== Kalman Filter ====================
class KalmanFilter:
    """
    간단한 Kalman Filter (상태: [x, y, w, h, vx, vy, vw, vh])
    x, y: bbox 중심, w, h: bbox 크기, v*: 속도
    """

    def __init__(self):
        # 상태 차원: 8 (위치 4 + 속도 4)
        # 측정 차원: 4 (bbox만 측정)
        self.dt = 1  # 시간 간격

        # 상태 전이 행렬 (F)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = self.dt

        # 측정 행렬 (H) - bbox만 측정
        self.H = np.eye(4, 8)

        # 프로세스 노이즈 (Q)
        self.Q = np.eye(8) * 0.01

        # 측정 노이즈 (R)
        self.R = np.eye(4) * 1.0

        # 상태 및 공분산
        self.x = None  # 상태 벡터
        self.P = None  # 공분산 행렬

    def initiate(self, measurement):
        """초기화 (첫 측정값)"""
        # [x, y, w, h, 0, 0, 0, 0]
        self.x = np.zeros(8)
        self.x[:4] = measurement
        self.P = np.eye(8) * 10.0
        return self.x.copy()

    def predict(self):
        """예측 단계"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].copy()  # bbox만 반환

    def update(self, measurement):
        """업데이트 단계"""
        # 혁신 (innovation)
        y = measurement - self.H @ self.x

        # 혁신 공분산
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 상태 업데이트
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        return self.x[:4].copy()


# ==================== Track ====================
class Track:
    """단일 추적 객체"""
    _id_counter = 0

    def __init__(self, detection, embedding, n_init=3, max_age=30, decision_frames=5):
        """
        detection: [x, y, w, h]
        embedding: Re-ID feature (192-dim)
        n_init: 확정까지 필요한 연속 탐지 횟수
        max_age: 미탐지 후 삭제까지 프레임 수
        decision_frames: 판별까지 필요한 프레임 수
        """
        self.track_id = Track._id_counter
        Track._id_counter += 1

        self.kf = KalmanFilter()
        self.kf.initiate(detection)

        self.embedding = embedding
        self.embeddings = [embedding]  # 임베딩 히스토리

        self.hits = 1  # 연속 탐지 횟수
        self.age = 1  # 총 프레임 수
        self.time_since_update = 0  # 마지막 업데이트 이후 프레임

        self.n_init = n_init
        self.max_age = max_age
        self.decision_frames = decision_frames

        self.state = 'tentative'  # tentative or confirmed
        self.label = 'PENDING'  # PENDING, OWNER, OTHER
        self.similarity = 0.0  # Master와의 유사도
        self.front_face_checked = False  # 정면 얼굴 재검사 수행 여부 (한 번만)

    def predict(self):
        """Kalman 예측"""
        self.age += 1
        self.time_since_update += 1
        bbox = self.kf.predict()
        return bbox

    def update(self, detection, embedding, master_embedding=None, similarity_threshold=0.5):
        """측정값으로 업데이트"""
        self.kf.update(detection)
        self.embedding = embedding
        self.embeddings.append(embedding)

        # 최근 100개 임베딩만 유지
        if len(self.embeddings) > 100:
            self.embeddings = self.embeddings[-100:]

        self.hits += 1
        self.time_since_update = 0

        # 확정 조건
        if self.state == 'tentative' and self.hits >= self.n_init:
            self.state = 'confirmed'

        # 판별 조건 (decision_frames 프레임 수집 완료)
        if self.hits == self.decision_frames and self.label == 'PENDING' and master_embedding is not None:
            self.decide_label(master_embedding, threshold=similarity_threshold)

    def decide_label(self, master_embedding, threshold=0.5):
        """
        Master 임베딩으로 본인/타인 판별
        master_embedding 구조:
        - 단일 임베딩: [emb] (배열)
        - 단일 주인공 각도별: [[emb1, emb2, ...]] (리스트의 리스트, 길이 1)
        - 여러 주인공 각도별: [[emb1, emb2, ...], [emb1, emb2, ...], ...] (리스트의 리스트)
        """
        avg_embedding = np.mean(self.embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        # 여러 주인공 모드인지 확인 (리스트의 리스트)
        if isinstance(master_embedding, list) and len(master_embedding) > 0 and isinstance(master_embedding[0], list):
            # 여러 주인공 모드: 모든 주인공과 비교 후 최대 유사도 사용
            all_similarities = []
            for owner_embeddings in master_embedding:
                # 각 주인공의 모든 각도 임베딩과 비교
                owner_similarities = [np.dot(emb, avg_embedding) for emb in owner_embeddings]
                all_similarities.append(max(owner_similarities))  # 각 주인공의 최대 유사도
            
            self.similarity = max(all_similarities)  # 모든 주인공 중 최대 유사도
        elif isinstance(master_embedding, list):
            # 단일 주인공 각도별 임베딩
            similarities = [np.dot(emb, avg_embedding) for emb in master_embedding]
            self.similarity = max(similarities)  # 최대 유사도
        else:
            # 단일 Master 임베딩 (평균 임베딩)
            self.similarity = np.dot(master_embedding, avg_embedding)

        if self.similarity > threshold:
            self.label = 'OWNER'
        else:
            self.label = 'OTHER'

        # 디버깅: 판별 결과 로그
        print(f"    [Track {self.track_id}] 판별 완료: {self.label} (유사도: {self.similarity:.4f})")

    def mark_missed(self):
        """미탐지 표시"""
        if self.state == 'tentative':
            self.state = 'deleted'

    def is_confirmed(self):
        return self.state == 'confirmed'

    def is_deleted(self):
        return self.state == 'deleted' or self.time_since_update > self.max_age

    def get_bbox(self):
        """현재 bbox 반환 [x, y, w, h]"""
        return self.kf.x[:4].copy()

    def to_ltrb(self):
        """bbox를 [left, top, right, bottom]으로 변환"""
        bbox = self.get_bbox()
        x, y, w, h = bbox
        return [x, y, x + w, y + h]


# ==================== Tracker ====================
class SimpleTracker:
    """경량 DeepSORT 트래커"""

    def __init__(self, max_iou_distance=0.7, max_cosine_distance=0.4,
                 n_init=3, max_age=30, decision_frames=5, master_embedding=None,
                 similarity_threshold=0.5, use_embedding_history=True):
        """
        max_iou_distance: IoU 매칭 임계값 (높을수록 엄격)
        max_cosine_distance: 코사인 거리 임계값 (낮을수록 엄격)
        n_init: 확정까지 필요한 연속 탐지
        max_age: 미탐지 후 삭제 프레임
        decision_frames: 판별까지 필요한 프레임 수
        master_embedding: 본인 얼굴 임베딩 (192-dim)
        similarity_threshold: 본인/타인 구분 임계값
        use_embedding_history: 임베딩 히스토리 활용 여부 (각도 변화에 강건)
        """
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.n_init = n_init
        self.max_age = max_age
        self.decision_frames = decision_frames
        self.master_embedding = master_embedding
        self.similarity_threshold = similarity_threshold
        self.use_embedding_history = use_embedding_history

        self.tracks = []
    
    def update_master_embedding(self, new_embedding, owner_index=None):
        """
        Master 임베딩 동적 업데이트 (새로운 각도 추가)
        owner_index: 여러 주인공 모드에서 특정 주인공 지정 (None이면 첫 번째 주인공)
        """
        # 여러 주인공 모드인지 확인
        if isinstance(self.master_embedding, list) and len(self.master_embedding) > 0 and isinstance(self.master_embedding[0], list):
            # 여러 주인공 모드: 특정 주인공의 임베딩에 추가
            if owner_index is None:
                owner_index = 0  # 기본값: 첫 번째 주인공
            if 0 <= owner_index < len(self.master_embedding):
                self.master_embedding[owner_index].append(new_embedding)
        elif isinstance(self.master_embedding, list):
            # 단일 주인공 각도별 모드
            self.master_embedding.append(new_embedding)
        else:
            # 단일 임베딩이었으면 리스트로 변환
            self.master_embedding = [self.master_embedding, new_embedding]
    
    def get_master_embedding(self):
        """현재 Master 임베딩 반환"""
        return self.master_embedding

    def update(self, detections, embeddings):
        """
        detections: List of [x, y, w, h]
        embeddings: List of embeddings (192-dim)
        """
        # 1. 예측
        for track in self.tracks:
            track.predict()

        # 2. 매칭
        matches, unmatched_detections, unmatched_tracks = self._match(
            detections, embeddings
        )

        # 3. 매칭된 track 업데이트
        for det_idx, track_idx in matches:
            self.tracks[track_idx].update(
                detections[det_idx],
                embeddings[det_idx],
                master_embedding=self.master_embedding,
                similarity_threshold=self.similarity_threshold
            )

        # 4. 미매칭 track 처리
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 5. 새로운 track 생성
        for det_idx in unmatched_detections:
            self.tracks.append(Track(
                detections[det_idx],
                embeddings[det_idx],
                n_init=self.n_init,
                max_age=self.max_age,
                decision_frames=self.decision_frames
            ))

        # 6. 삭제된 track 제거
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        return self.tracks

    def _match(self, detections, embeddings):
        """Detection과 Track 매칭"""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # 거리 행렬 계산 (IoU + 코사인)
        cost_matrix = self._calculate_cost_matrix(detections, embeddings, 
                                                  use_embedding_history=self.use_embedding_history)

        # Hungarian Algorithm
        det_indices, track_indices = linear_sum_assignment(cost_matrix)

        # 임계값 필터링
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        for det_idx, track_idx in zip(det_indices, track_indices):
            if cost_matrix[det_idx, track_idx] < 1.0:  # 유효한 매칭
                matches.append((det_idx, track_idx))
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(track_idx)

        return matches, unmatched_detections, unmatched_tracks

    def _calculate_cost_matrix(self, detections, embeddings, use_embedding_history=True):
        """
        거리 행렬 계산 (IoU + 코사인 거리)
        use_embedding_history: True면 Track의 임베딩 히스토리와 비교 (각도 변화에 강건)
        """
        cost_matrix = np.zeros((len(detections), len(self.tracks)))

        for i, (det, emb) in enumerate(zip(detections, embeddings)):
            for j, track in enumerate(self.tracks):
                # IoU 거리
                iou = self._iou(det, track.get_bbox())
                iou_distance = 1 - iou

                # 코사인 거리 (임베딩 히스토리 활용)
                if use_embedding_history and len(track.embeddings) > 1:
                    # 여러 임베딩과 비교 후 최대 유사도 사용 (각도 변화에 강건)
                    cosine_sims = [np.dot(emb, track_emb) for track_emb in track.embeddings[-5:]]  # 최근 5개만
                    cosine_sim = max(cosine_sims)  # 최대 유사도
                else:
                    # 최신 임베딩만 사용
                    cosine_sim = np.dot(emb, track.embedding)
                
                cosine_distance = 1 - cosine_sim

                # IoU가 높으면 임베딩 거리 임계값 완화 (위치가 확실하면 임베딩 차이 허용)
                if iou > 0.5:
                    adjusted_cosine_threshold = self.max_cosine_distance * 1.2  # 20% 완화
                else:
                    adjusted_cosine_threshold = self.max_cosine_distance

                # 결합 (IoU 가중치 증가: 위치 정보를 더 신뢰)
                if iou_distance > self.max_iou_distance or cosine_distance > adjusted_cosine_threshold:
                    cost_matrix[i, j] = 1e5  # 무한대 (매칭 불가)
                else:
                    # IoU 가중치 증가 (0.6) + 코사인 가중치 감소 (0.4)
                    cost_matrix[i, j] = 0.6 * iou_distance + 0.4 * cosine_distance

        return cost_matrix

    def _iou(self, bbox1, bbox2):
        """IoU 계산"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # 교집합
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # 합집합
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area