# 3DMM JSON 출력 형식 (계수 모드)

서버 전송용 **3DMM(3D Morphable Model)** JSON — **id/exp/pose 계수만** 전송. 아바타 합성 등에서 동일 3DMM 정의로 메시 복원용.

---

## 1. 전체 구조

```json
{
  "video_info": { "width": 1280, "height": 720, "fps": 30.0, "format": "3dmm" },
  "frames": [
    { "frame_number": 0, "timestamp": 0.0, "faces": [] },
    { "frame_number": 1, "timestamp": 0.033..., "faces": [...] }
  ]
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `video_info` | object | 영상 메타 + 포맷 식별 |
| `frames` | array | 프레임별 얼굴·3DMM 계수 |

---

## 2. video_info

| 필드 | 타입 | 설명 |
|------|------|------|
| `width` | int | 영상 가로 해상도 |
| `height` | int | 영상 세로 해상도 |
| `fps` | float | 초당 프레임 수 |
| `format` | string | `"3dmm"` — 3DMM 계수 전송 포맷 |

---

## 3. frames[] 항목

| 필드 | 타입 | 설명 |
|------|------|------|
| `frame_number` | int | 0-based 프레임 인덱스 |
| `timestamp` | float | 재생 시간(초). `frame_number / fps` |
| `faces` | array | 해당 프레임에서 검출된 얼굴별 3DMM 계수 |

---

## 4. faces[] 항목

| 필드 | 타입 | 설명 |
|------|------|------|
| `tracking_id` | int | 프레임 내 얼굴 ID. 추후 트래킹 적용 시 동일 인물은 동일 ID 유지 |
| `bbox` | int[4] | `[x, y, width, height]` 픽셀 단위 바운딩 박스 |
| `3dmm` | object | **id_coeffs**, **exp_coeffs**, **pose** 세 필드 (계수만) |

---

## 5. faces[].3dmm (계수)

모델이 `[1, K]` float 배열을 내보낼 때, **앞에서부터** id → exp → pose 순으로 분할해 전송.

| 필드 | 타입 | 설명 |
|------|------|------|
| `id_coeffs` | float[] | Identity(형태) 계수 — 앞쪽 `idDim`개 (기본 80) |
| `exp_coeffs` | float[] | Expression(표정) 계수 — 다음 `expDim`개 (기본 64) |
| `pose` | float[] | Pose(자세: 회전·이동 등) — 나머지 (K - idDim - expDim) |

- **분할 차원**: `FacialLandmarkDetector.idDim`(기본 80), `expDim`(기본 64). 모델 정의에 맞게 앱에서 조정 가능.
- 계수가 없으면 빈 배열 `[]`로 전송.

---

## 6. 예시

```json
{
  "video_info": { "width": 1280, "height": 720, "fps": 30.0, "format": "3dmm" },
  "frames": [
    { "frame_number": 0, "timestamp": 0.0, "faces": [] },
    {
      "frame_number": 4,
      "timestamp": 0.13333333333333333,
      "faces": [
        {
          "tracking_id": 0,
          "bbox": [659, 177, 49, 64],
          "3dmm": {
            "id_coeffs": [ 0.12, -0.05, 0.0, ... ],
            "exp_coeffs": [ 0.01, 0.02, ... ],
            "pose": [ 0.0, 0.0, 0.1, 0.0, 0.0, 0.0 ]
          }
        }
      ]
    }
  ]
}
```

---

## 7. 저장 위치

- **우선**: 다운로드 폴더 (`Environment.DIRECTORY_DOWNLOADS`).
- **실패 시**: 비디오 저장 경로와 같은 디렉터리 → 앱 외부 Documents → `filesDir`.
- **반환**: `ProcessingResult.exportJsonPath`에 실제 저장된 JSON 파일 절대 경로.

---

## 8. 서버 측 사용

- `video_info.format`: `"3dmm"`이면 3DMM 계수 전송용.
- `video_info.width/height/fps`: 해상도·fps로 타임라인·스케일 해석.
- `frames[].timestamp`: 아바타 합성 시 프레임-시간 매핑.
- `faces[].tracking_id`: 동일 인물 추적용 (추후 트래킹 구현 시 동일 ID 유지).
- `faces[].bbox`: 얼굴 영역 참고.
- `faces[].3dmm.id_coeffs`, `exp_coeffs`, `pose`: 동일 3DMM 정의로 메시 복원.
