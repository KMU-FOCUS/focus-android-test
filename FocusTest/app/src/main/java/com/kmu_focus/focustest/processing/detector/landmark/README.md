# landmark 패키지

얼굴 랜드마크 관련 코드를 용도별로 나눈 패키지.

## yunet

**YuNet 5-point 랜드마크** (얼굴 검출기와 함께 출력)

- `FaceLandmarks5`: 오른쪽 눈, 왼쪽 눈, 코, 오른쪽 입꼬리, 왼쪽 입꼬리 (2D)
- 용도: 모자이크 타원 근사, 얼굴 정렬, 시각화

## model3d

**3DMM 랜드마크** (TFLite 별도 모델 출력)

- `Landmark`, `Landmark3D`, `FaceLandmarks`: 정규화/이미지 좌표 랜드마크
- `FacialLandmarkDetector`: facial_landmark.tflite 추론
- 용도: 서버 전송용 JSON(3DMM), 아바타 합성
