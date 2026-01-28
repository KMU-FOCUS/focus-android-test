package com.kmu_focus.focustest.processing

import android.content.Context
import android.graphics.BitmapFactory
import android.net.Uri
import org.opencv.android.OpenCVLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * 비디오 처리 파이프라인
 * face_detection_yunet.py의 로직을 그대로 이식
 * VideoFrameSource -> FrameProcessor -> VideoWriter
 */
class VideoProcessor(
    private val context: Context
) {
    
    companion object {
        /** 사용할 검출기 타입 */
        @JvmStatic
        var detectorType: DetectorType = DetectorType.YOLO_TFLITE
    }
    
    private var faceDetector: FaceDetector? = null
    private var frameProcessor: FrameProcessor? = null
    
    /**
     * 비디오 처리 실행
     * @param videoUri 입력 비디오 URI
     * @param outputPath 출력 파일 경로
     * @param progressCallback 진행률 콜백 (0.0 ~ 1.0)
     */
    suspend fun processVideo(
        videoUri: Uri,
        outputPath: String,
        progressCallback: (Float) -> Unit
    ): ProcessingResult = withContext(Dispatchers.Default) {
        
        // OpenCV 초기화 확인
        if (!OpenCVLoader.initDebug()) {
            return@withContext ProcessingResult(
                success = false,
                error = "OpenCV 초기화 실패"
            )
        }
        
        // 초기화 - 검출기 타입에 따라 생성
        if (faceDetector == null) {
            faceDetector = when (detectorType) {
                DetectorType.YOLO_TFLITE -> YuNetFaceDetector(context)      // YOLO TFLite + NNAPI
                DetectorType.YUNET_ONNX -> YuNetOnnxDetector(context)       // YuNet ONNX Runtime + NNAPI
                DetectorType.YUNET_OPENCV -> YuNetOpenCVDetector(context)   // YuNet OpenCV (CPU)
            }
            android.util.Log.i("VideoProcessor", "검출기 초기화: ${faceDetector!!.getDetectorType()}")
        }
        
        if (frameProcessor == null) {
            frameProcessor = FrameProcessor(faceDetector!!)
        }
        
        val videoSource = FileVideoSource(context, videoUri)
        val videoInfo = videoSource.getVideoInfo()
        
        // 영상 정보 로그
        val durationSec = if (videoInfo.fps > 0) videoInfo.totalFrames / videoInfo.fps else 0f
        android.util.Log.i("VideoProcessor", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        android.util.Log.i("VideoProcessor", "영상 정보:")
        android.util.Log.i("VideoProcessor", "  - 해상도: ${videoInfo.width}x${videoInfo.height}")
        android.util.Log.i("VideoProcessor", "  - FPS: ${videoInfo.fps}")
        android.util.Log.i("VideoProcessor", "  - 총 프레임: ${videoInfo.totalFrames}")
        android.util.Log.i("VideoProcessor", "  - 재생 시간: ${String.format("%.1f", durationSec)}초")
        android.util.Log.i("VideoProcessor", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        // VideoWriter 초기화 (OpenCV 사용으로 변경)
        val videoWriter = OpenCVVideoWriter(
            outputPath,
            videoInfo.width,
            videoInfo.height,
            videoInfo.fps
        )
        
        var processedFrames = 0
        var totalFaces = 0
        val totalFrames = if (videoInfo.totalFrames > 0) videoInfo.totalFrames else Int.MAX_VALUE
        
        // 성능 측정용
        val processingTimes = mutableListOf<Long>()
        val processingStartTime = System.currentTimeMillis()
        
        try {
            var firstFrameSkipped = false
            
            while (true) {
                val frameStart = System.currentTimeMillis()
                
                val frame = videoSource.readFrame()
                if (frame == null) break
                
                // 프레임 처리 (Python: 얼굴 검출 → 바운딩 박스 그리기)
                val processedFrame = frameProcessor!!.processFrame(frame)
                
                // 첫 프레임은 스킵 (인코더 초기화 문제 방지)
                if (!firstFrameSkipped) {
                    firstFrameSkipped = true
                    frame.recycle()
                    if (processedFrame != frame) {
                        processedFrame.recycle()
                    }
                    continue
                }
                
                // 비디오에 쓰기
                videoWriter.writeFrame(processedFrame)
                
                processedFrames++
                
                // 성능 측정 및 로그 출력 (Python과 동일)
                val frameTime = System.currentTimeMillis() - frameStart
                processingTimes.add(frameTime)
                
                // 30프레임마다 평균 처리 시간 로그 출력 (Python: if frame_count % 30 == 0)
                // 로그 출력 최소화 (성능 최적화)
                if (processedFrames % 30 == 0 && processingTimes.size >= 30) {
                    val avgTime = processingTimes.takeLast(30).average()
                    val currentFps = 1000.0 / avgTime
                    android.util.Log.d("VideoProcessor", 
                        "진행: ${(processedFrames.toFloat() / totalFrames * 100).toInt()}% | " +
                        "FPS: ${currentFps.toInt()} | " +
                        "처리: ${avgTime.toInt()}ms")
                }
                
                // 진행률 업데이트 (0.0 ~ 1.0 범위로 제한)
                val progress = if (totalFrames > 0 && totalFrames != Int.MAX_VALUE) {
                    (processedFrames.toFloat() / totalFrames).coerceIn(0f, 1f)
                } else {
                    // totalFrames를 알 수 없는 경우, 최소한 0.0 이상으로 설정
                    0f.coerceAtLeast(0f)
                }
                progressCallback(progress)
                
                // 메모리 정리
                frame.recycle()
                processedFrame.recycle()
            }
            
            videoWriter.release()
            
            // 처리 완료 로그
            val totalProcessingTime = (System.currentTimeMillis() - processingStartTime) / 1000.0
            val videoDuration = if (videoInfo.fps > 0) processedFrames / videoInfo.fps else 0f
            val speedRatio = if (totalProcessingTime > 0) videoDuration / totalProcessingTime else 0.0
            
            android.util.Log.i("VideoProcessor", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            android.util.Log.i("VideoProcessor", "처리 완료!")
            android.util.Log.i("VideoProcessor", "  - 처리된 프레임: $processedFrames")
            android.util.Log.i("VideoProcessor", "  - 영상 길이: ${String.format("%.1f", videoDuration)}초")
            android.util.Log.i("VideoProcessor", "  - 처리 시간: ${String.format("%.1f", totalProcessingTime)}초")
            android.util.Log.i("VideoProcessor", "  - 속도 비율: ${String.format("%.2f", speedRatio)}x (1.0 = 실시간)")
            android.util.Log.i("VideoProcessor", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            ProcessingResult(
                success = true,
                outputPath = outputPath,
                totalFrames = processedFrames,
                totalFaces = totalFaces
            )
        } catch (e: Exception) {
            videoWriter.release()
            ProcessingResult(
                success = false,
                error = e.message ?: "알 수 없는 오류",
                totalFrames = processedFrames,
                totalFaces = totalFaces
            )
        } finally {
            videoSource.release()
        }
    }
    
    fun release() {
        faceDetector?.release()
        faceDetector = null
        frameProcessor = null
    }
}

data class ProcessingResult(
    val success: Boolean,
    val outputPath: String = "",
    val error: String = "",
    val totalFrames: Int = 0,
    val totalFaces: Int = 0
)
