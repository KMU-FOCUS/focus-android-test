import cv2
import numpy as np
import time

VIDEO_PATH = 'test_video.mp4'
MODEL_PATH = 'yunet_face.onnx'

try:
    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (320, 320),
        score_threshold=0.5,  # 0.6 → 0.5로 낮춤 (가림 처리)
        nms_threshold=0.3,
        top_k=5000
    )
    print("✓ YuNet 모델 로드 성공\n")
except Exception as e:
    print(f"✗ 모델 로드 실패: {e}")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"동영상 파일을 열 수 없습니다: {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# VideoWriter 설정 - 여기 추가!
OUTPUT_PATH = 'output_yunet_face_detection.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("=== 동영상 정보 ===")
print(f"파일: {VIDEO_PATH}")
print(f"해상도: {width}x{height}")
print(f"FPS: {fps:.1f}")
print(f"총 프레임: {total_frames}")
print(f"\n모델: YuNet (최적화)")
print(f"신뢰도 임계값: 0.5")
print(f"출력 파일: {OUTPUT_PATH}")  # 출력 파일 경로 표시
print("ESC: 종료 | SPACE: 일시정지\n")

frame_count = 0
fps_start_time = time.time()
frame_times = []
paused = False
delay = 1  # 최대 속도
current_threshold = 1.0

# 다운스케일 설정
scale = 0.7  # 70% 크기 (속도 향상)

while cap.isOpened():
    if not paused:
        success, image = cap.read()
        if not success:
            print("\n동영상 재생 완료")
            break
        
        frame_start = time.time()
        
        # 해상도 축소
        small_image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # 얼굴 탐지
        detector.setInputSize((small_image.shape[1], small_image.shape[0]))
        _, faces = detector.detect(small_image)
        
        frame_end = time.time()
        processing_time = (frame_end - frame_start) * 1000
        frame_times.append(processing_time)
        
        detected_faces = 0
        if faces is not None:
            detected_faces = len(faces)
            
            for face in faces:
                # 원본 크기로 좌표 복원
                x, y, w, h = (face[:4] / scale).astype(int)
                confidence = face[14]
                
                # 신뢰도에 따라 색상
                if confidence < 0.6:
                    color = (0, 165, 255)  # 주황
                elif confidence < 0.8:
                    color = (0, 255, 255)  # 노랑
                else:
                    color = (0, 255, 0)    # 초록
                
                # 바운딩 박스
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                # 신뢰도 표시
                cv2.putText(image, f'{confidence:.2f}', (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        
        if frame_count % 30 == 0:
            fps_end_time = time.time()
            current_fps = 30 / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
            
            avg_processing = sum(frame_times[-30:]) / 30
            print(f"진행: {progress:.1f}% | 탐지: {detected_faces}명 | FPS: {current_fps:.1f} | 처리: {avg_processing:.1f}ms")
        
        # 화면 정보
        cv2.putText(image, f'Faces: {detected_faces}', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f'YuNet (Scale: {scale})', (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 프레임을 출력 파일에 저장 - 여기 추가!
        out.write(image)
    
    cv2.imshow('Face Detection - YuNet Optimized', image)
    
    key = cv2.waitKey(delay) & 0xFF
    if key == 27:
        break
    elif key == 32:
        paused = not paused

cap.release()
out.release()  # VideoWriter 종료
cv2.destroyAllWindows()

if frame_times:
    print("\n=== 테스트 결과 ===")
    print(f"처리한 프레임: {len(frame_times)}")
    print(f"평균 처리 시간: {sum(frame_times)/len(frame_times):.2f}ms")
    print(f"최소: {min(frame_times):.2f}ms")
    print(f"최대: {max(frame_times):.2f}ms")
    print(f"평균 FPS: {1000/(sum(frame_times)/len(frame_times)):.1f}")
    print(f"\n✅ 결과 동영상 저장 완료: {OUTPUT_PATH}")
