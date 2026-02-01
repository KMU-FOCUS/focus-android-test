package com.kmu_focus.focustest.processing.detector

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.FaceDetectorYN
import java.io.File
import java.io.FileOutputStream

/**
 * YuNet 얼굴 검출기 (OpenCV CPU)
 * 
 * 특징:
 * - OpenCV의 FaceDetectorYN 사용 (자체 SIMD/NEON 최적화)
 * - 동적 입력 크기 (scale로 다운스케일)
 * - CPU 기반이지만 매우 빠름 (~15-20ms)
 */
class YuNetOpenCVDetector(context: Context) : FaceDetector {
    
    companion object {
        private const val TAG = "YuNetOpenCVDetector"
        private const val MODEL_NAME = "yunet_face.onnx"
        
        /** 검출용 입력 크기 (작을수록 빠름, 클수록 정확) */
        @JvmStatic
        var inputSize: Int = 320  // 160, 320, 640 중 선택
        
        /** 벤치마크 로그 활성화 */
        @JvmStatic
        var enableBenchmark: Boolean = true
    }
    
    private var detector: FaceDetectorYN? = null
    
    // 벤치마크
    private var frameCounter: Int = 0
    private var totalPreprocessMs: Long = 0
    private var totalInferenceMs: Long = 0
    private var totalPostprocessMs: Long = 0
    
    // Mat 재사용
    private var originalMat: Mat? = null
    private var bgrMat: Mat? = null
    private var smallMat: Mat? = null
    private var facesMat: Mat? = null
    
    init {
        initializeDetector(context)
    }
    
    private fun initializeDetector(context: Context) {
        try {
            // assets에서 모델 파일 복사
            val modelFile = File(context.filesDir, MODEL_NAME)
            if (!modelFile.exists()) {
                context.assets.open(MODEL_NAME).use { input ->
                    FileOutputStream(modelFile).use { output ->
                        input.copyTo(output)
                    }
                }
                Log.i(TAG, "모델 파일 복사 완료: ${modelFile.absolutePath}")
            }
            
            // YuNet 초기화 (320x320 기본, 동적으로 변경됨)
            detector = FaceDetectorYN.create(
                modelFile.absolutePath,
                "",
                Size(320.0, 320.0),
                0.5f,  // score_threshold
                0.3f,  // nms_threshold
                5000   // top_k
            )
            
            if (detector == null) {
                throw IllegalStateException("YuNet 검출기 초기화 실패")
            }
            
            Log.i(TAG, "✓ YuNet OpenCV 초기화 완료 (inputSize: $inputSize)")
        } catch (e: Exception) {
            throw IllegalStateException("YuNet 초기화 중 오류: ${e.message}", e)
        }
    }
    
    override fun detectFaces(frame: Bitmap): List<DetectedFace> {
        val det = detector ?: return emptyList()
        
        try {
            val t0 = SystemClock.elapsedRealtimeNanos()
            
            // 1. Bitmap → Mat (RGBA → BGR)
            val orgMat = originalMat ?: Mat().also { originalMat = it }
            Utils.bitmapToMat(frame, orgMat)
            
            val bgr = bgrMat ?: Mat().also { bgrMat = it }
            if (orgMat.type() == CvType.CV_8UC4 && orgMat.channels() == 4) {
                Imgproc.cvtColor(orgMat, bgr, Imgproc.COLOR_RGBA2BGR)
            } else {
                Imgproc.cvtColor(orgMat, bgr, Imgproc.COLOR_RGB2BGR)
            }
            
            // 2. 비율 유지 리사이즈 (핵심 최적화)
            // 가로를 inputSize로 고정, 세로는 비율에 맞춤
            // 예: 1920x1080 + inputSize=320 → 320x180 (57,600 픽셀)
            val origWidth = bgr.cols()
            val origHeight = bgr.rows()
            val scale = inputSize.toFloat() / origWidth
            val smallWidth = inputSize
            val smallHeight = (origHeight * scale).toInt()
            val small = smallMat ?: Mat().also { smallMat = it }
            Imgproc.resize(bgr, small, Size(smallWidth.toDouble(), smallHeight.toDouble()))
            
            // 3. 입력 크기 동적 설정
            det.setInputSize(Size(small.cols().toDouble(), small.rows().toDouble()))
            
            val t1 = SystemClock.elapsedRealtimeNanos()
            
            // 4. 검출
            val faces = facesMat ?: Mat().also { facesMat = it }
            det.detect(small, faces)
            
            val t2 = SystemClock.elapsedRealtimeNanos()
            
            // 5. 결과 파싱
            val result = mutableListOf<DetectedFace>()
            
            if (faces.rows() > 0 && faces.cols() >= 15) {
                for (i in 0 until faces.rows()) {
                    val row = faces.row(i)
                    val data = FloatArray(15)
                    row.get(0, 0, data)
                    
                    // 원본 크기로 좌표 복원
                    val x = (data[0] / scale).toInt()
                    val y = (data[1] / scale).toInt()
                    val width = (data[2] / scale).toInt()
                    val height = (data[3] / scale).toInt()
                    val confidence = data[14]
                    
                    val clippedX = x.coerceIn(0, frame.width - 1)
                    val clippedY = y.coerceIn(0, frame.height - 1)
                    val clippedWidth = width.coerceIn(1, frame.width - clippedX)
                    val clippedHeight = height.coerceIn(1, frame.height - clippedY)
                    
                    result.add(DetectedFace(clippedX, clippedY, clippedWidth, clippedHeight, confidence))
                }
            }
            
            val t3 = SystemClock.elapsedRealtimeNanos()
            
            // 벤치마크
            frameCounter++
            if (enableBenchmark) {
                val preprocessMs = (t1 - t0) / 1_000_000
                val inferenceMs = (t2 - t1) / 1_000_000
                val postprocessMs = (t3 - t2) / 1_000_000
                
                totalPreprocessMs += preprocessMs
                totalInferenceMs += inferenceMs
                totalPostprocessMs += postprocessMs
                
                if (frameCounter % 30 == 0) {
                    val avgPre = totalPreprocessMs / frameCounter
                    val avgInf = totalInferenceMs / frameCounter
                    val avgPost = totalPostprocessMs / frameCounter
                    val totalAvg = avgPre + avgInf + avgPost
                    val fps = if (totalAvg > 0) 1000.0 / totalAvg else 0.0
                    
                    Log.d(TAG, "[OpenCV CPU] 전처리:${avgPre}ms 추론:${avgInf}ms 후처리:${avgPost}ms = ${totalAvg}ms (${String.format("%.1f", fps)} FPS)")
                    Log.d(TAG, "  검출된 얼굴: ${result.size}")
                }
            }
            
            return result
        } catch (e: Exception) {
            Log.e(TAG, "얼굴 검출 중 오류: ${e.message}", e)
            return emptyList()
        }
    }
    
    override fun release() {
        detector = null
        originalMat?.release()
        bgrMat?.release()
        smallMat?.release()
        facesMat?.release()
    }
    
    override fun getDetectorType(): String = "YuNet OpenCV (CPU)"
}
