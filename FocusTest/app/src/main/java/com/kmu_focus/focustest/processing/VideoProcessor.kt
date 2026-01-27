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
    
    private var faceDetector: YuNetFaceDetector? = null
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
        
        // 초기화 (Python: detector = cv2.FaceDetectorYN.create(...))
        if (faceDetector == null) {
            faceDetector = YuNetFaceDetector(context)
        }
        
        if (frameProcessor == null) {
            frameProcessor = FrameProcessor(faceDetector!!)
        }
        
        val videoSource = FileVideoSource(context, videoUri)
        val videoInfo = videoSource.getVideoInfo()
        
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
        
        try {
            while (true) {
                val frame = videoSource.readFrame()
                if (frame == null) break
                
                // 프레임 처리 (Python: 얼굴 검출 → 바운딩 박스 그리기)
                val processedFrame = frameProcessor!!.processFrame(frame)
                val detectedFaces = faceDetector!!.detectFaces(frame)
                totalFaces += detectedFaces.size
                
                // 비디오에 쓰기
                videoWriter.writeFrame(processedFrame)
                
                processedFrames++
                
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
