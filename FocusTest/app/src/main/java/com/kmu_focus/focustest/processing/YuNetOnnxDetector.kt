package com.kmu_focus.focustest.processing

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.EnumSet

/**
 * YuNet 얼굴 검출기 (ONNX Runtime + NNAPI)
 * 
 * Galaxy S25 NPU 가속:
 * - ONNX Runtime의 NNAPI Execution Provider 사용
 * - FP16 연산 허용으로 NPU 최적화
 */
class YuNetOnnxDetector(context: Context) : FaceDetector {
    
    companion object {
        private const val TAG = "YuNetOnnxDetector"
        private const val MODEL_NAME = "yunet_face.onnx"
        private const val INPUT_SIZE = 640  // 모델 고정 입력 크기
        private const val SCORE_THRESHOLD = 0.5f
        private const val NMS_THRESHOLD = 0.3f
        
        /** NNAPI 사용 여부 */
        @JvmStatic
        var useNNAPI: Boolean = true
        
        /** 벤치마크 로그 활성화 */
        @JvmStatic
        var enableBenchmark: Boolean = true
    }
    
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var acceleratorType: String = "CPU"
    
    // 입력 버퍼 재사용
    private var inputBuffer: FloatBuffer? = null
    private val inputShape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
    
    // 벤치마크
    private var frameCounter: Int = 0
    private var totalPreprocessMs: Long = 0
    private var totalInferenceMs: Long = 0
    private var totalPostprocessMs: Long = 0
    
    init {
        initializeOnnxRuntime(context)
    }
    
    private fun initializeOnnxRuntime(context: Context) {
        try {
            // 모델 파일 복사
            val modelFile = File(context.filesDir, MODEL_NAME)
            if (!modelFile.exists()) {
                context.assets.open(MODEL_NAME).use { input ->
                    FileOutputStream(modelFile).use { output ->
                        input.copyTo(output)
                    }
                }
                Log.i(TAG, "모델 파일 복사 완료: ${modelFile.absolutePath}")
            }
            
            // ONNX Runtime 환경 생성
            ortEnv = OrtEnvironment.getEnvironment()
            
            // 세션 옵션 설정
            val sessionOptions = OrtSession.SessionOptions().apply {
                // 기본 최적화
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                
                // NNAPI Execution Provider 추가 (NPU 가속)
                if (useNNAPI) {
                    try {
                        // FP16만 사용 (CPU fallback 허용해서 호환성 확보)
                        addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                        acceleratorType = "NNAPI (NPU)"
                        Log.i(TAG, "NNAPI Execution Provider 활성화 (FP16)")
                    } catch (e: Exception) {
                        Log.w(TAG, "NNAPI 초기화 실패, CPU 사용: ${e.message}")
                        acceleratorType = "CPU"
                    }
                }
            }
            
            // 세션 생성
            ortSession = ortEnv!!.createSession(modelFile.absolutePath, sessionOptions)
            
            // 입력 버퍼 초기화
            inputBuffer = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)
            
            Log.i(TAG, "✓ YuNet ONNX Runtime 초기화 완료")
            Log.i(TAG, "  - 가속기: $acceleratorType")
            Log.i(TAG, "  - 입력 크기: ${INPUT_SIZE}x${INPUT_SIZE}")
            
        } catch (e: Exception) {
            throw IllegalStateException("YuNet ONNX Runtime 초기화 중 오류: ${e.message}", e)
        }
    }
    
    override fun detectFaces(frame: Bitmap): List<DetectedFace> {
        val session = ortSession ?: return emptyList()
        val env = ortEnv ?: return emptyList()
        
        try {
            val t0 = SystemClock.elapsedRealtimeNanos()
            
            // 1. 전처리: Bitmap → 정규화된 float 텐서 (CHW)
            val buffer = inputBuffer!!
            buffer.rewind()
            prepareInputBuffer(frame, buffer)
            
            val t1 = SystemClock.elapsedRealtimeNanos()
            
            // 2. 추론
            buffer.rewind()
            val inputTensor = OnnxTensor.createTensor(env, buffer, inputShape)
            val inputs = mapOf("input" to inputTensor)
            
            val outputs = session.run(inputs)
            
            val t2 = SystemClock.elapsedRealtimeNanos()
            
            // 3. 후처리
            val result = parseOutputs(outputs, frame.width, frame.height)
            
            val t3 = SystemClock.elapsedRealtimeNanos()
            
            // 리소스 정리
            inputTensor.close()
            outputs.close()
            
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
                    
                    Log.d(TAG, "[$acceleratorType] 전처리:${avgPre}ms 추론:${avgInf}ms 후처리:${avgPost}ms = ${totalAvg}ms (${String.format("%.1f", fps)} FPS)")
                    Log.d(TAG, "  검출된 얼굴: ${result.size}")
                }
            }
            
            return result
        } catch (e: Exception) {
            Log.e(TAG, "얼굴 검출 중 오류: ${e.message}", e)
            return emptyList()
        }
    }
    
    /**
     * Bitmap을 ONNX 입력 텐서로 변환
     * - 리사이즈 (INPUT_SIZE x INPUT_SIZE)
     * - 정규화 (0-255 → 0-1)
     * - HWC → CHW 변환
     */
    private fun prepareInputBuffer(bitmap: Bitmap, buffer: FloatBuffer) {
        // 리사이즈
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        
        if (resized != bitmap) {
            resized.recycle()
        }
        
        // CHW 순서로 변환 (R, G, B 채널 순서)
        val size = INPUT_SIZE * INPUT_SIZE
        for (i in 0 until size) {
            val pixel = pixels[i]
            // R 채널
            buffer.put(((pixel shr 16) and 0xFF) / 255.0f)
        }
        for (i in 0 until size) {
            val pixel = pixels[i]
            // G 채널
            buffer.put(((pixel shr 8) and 0xFF) / 255.0f)
        }
        for (i in 0 until size) {
            val pixel = pixels[i]
            // B 채널
            buffer.put((pixel and 0xFF) / 255.0f)
        }
    }
    
    /**
     * ONNX 출력 파싱 (YuNet 멀티스케일 출력)
     * 
     * 출력 구조:
     * - cls_8/16/32: [1, N, 1] - face classification score
     * - obj_8/16/32: [1, N, 1] - objectness score  
     * - bbox_8/16/32: [1, N, 4] - bounding box (dx, dy, dw, dh)
     * - kps_8/16/32: [1, N, 10] - keypoints
     */
    private fun parseOutputs(outputs: OrtSession.Result, origWidth: Int, origHeight: Int): List<DetectedFace> {
        val candidates = mutableListOf<DetectedFace>()
        
        try {
            // 디버그 출력 (첫 프레임만)
            if (frameCounter == 1) {
                outputs.forEach { (name, value) ->
                    if (value is OnnxTensor) {
                        Log.d(TAG, "출력: $name, shape: ${value.info.shape.contentToString()}")
                    }
                }
            }
            
            val scaleX = origWidth.toFloat() / INPUT_SIZE
            val scaleY = origHeight.toFloat() / INPUT_SIZE
            
            // 각 스케일(8, 16, 32) 처리
            val strides = intArrayOf(8, 16, 32)
            
            for (stride in strides) {
                val clsTensor = outputs.get("cls_$stride")?.orElse(null) as? OnnxTensor ?: continue
                val objTensor = outputs.get("obj_$stride")?.orElse(null) as? OnnxTensor ?: continue
                val bboxTensor = outputs.get("bbox_$stride")?.orElse(null) as? OnnxTensor ?: continue
                
                val clsData = clsTensor.floatBuffer
                val objData = objTensor.floatBuffer
                val bboxData = bboxTensor.floatBuffer
                
                val gridSize = INPUT_SIZE / stride  // 80, 40, 20
                val numAnchors = gridSize * gridSize
                
                for (i in 0 until numAnchors) {
                    // score = cls * obj (sigmoid 이미 적용된 상태로 가정)
                    val cls = clsData.get(i)
                    val obj = objData.get(i)
                    val score = cls * obj
                    
                    if (score < SCORE_THRESHOLD) continue
                    
                    // 그리드 좌표
                    val gridX = i % gridSize
                    val gridY = i / gridSize
                    
                    // bbox 디코딩 (anchor-free: 그리드 중심 기준)
                    val dx = bboxData.get(i * 4 + 0)
                    val dy = bboxData.get(i * 4 + 1)
                    val dw = bboxData.get(i * 4 + 2)
                    val dh = bboxData.get(i * 4 + 3)
                    
                    // 실제 좌표 계산
                    val cx = (gridX + 0.5f + dx) * stride
                    val cy = (gridY + 0.5f + dy) * stride
                    val w = kotlin.math.exp(dw) * stride
                    val h = kotlin.math.exp(dh) * stride
                    
                    // x, y, w, h (왼쪽 상단 기준)
                    val x = ((cx - w / 2) * scaleX).toInt().coerceIn(0, origWidth - 1)
                    val y = ((cy - h / 2) * scaleY).toInt().coerceIn(0, origHeight - 1)
                    val finalW = (w * scaleX).toInt().coerceIn(1, origWidth - x)
                    val finalH = (h * scaleY).toInt().coerceIn(1, origHeight - y)
                    
                    candidates.add(DetectedFace(x, y, finalW, finalH, score))
                }
            }
            
            // NMS 적용
            return applyNMS(candidates, NMS_THRESHOLD)
            
        } catch (e: Exception) {
            Log.e(TAG, "출력 파싱 오류: ${e.message}", e)
            return emptyList()
        }
    }
    
    /**
     * Non-Maximum Suppression
     */
    private fun applyNMS(candidates: List<DetectedFace>, threshold: Float): List<DetectedFace> {
        if (candidates.isEmpty()) return emptyList()
        
        val sorted = candidates.sortedByDescending { it.confidence }
        val selected = mutableListOf<DetectedFace>()
        val suppressed = BooleanArray(sorted.size)
        
        for (i in sorted.indices) {
            if (suppressed[i]) continue
            
            val face = sorted[i]
            selected.add(face)
            
            for (j in i + 1 until sorted.size) {
                if (suppressed[j]) continue
                
                if (calculateIoU(face, sorted[j]) > threshold) {
                    suppressed[j] = true
                }
            }
        }
        
        return selected
    }
    
    private fun calculateIoU(a: DetectedFace, b: DetectedFace): Float {
        val x1 = maxOf(a.x, b.x)
        val y1 = maxOf(a.y, b.y)
        val x2 = minOf(a.x + a.width, b.x + b.width)
        val y2 = minOf(a.y + a.height, b.y + b.height)
        
        val intersection = maxOf(0, x2 - x1) * maxOf(0, y2 - y1)
        val areaA = a.width * a.height
        val areaB = b.width * b.height
        val union = areaA + areaB - intersection
        
        return if (union > 0) intersection.toFloat() / union else 0f
    }
    
    override fun release() {
        ortSession?.close()
        ortSession = null
        ortEnv?.close()
        ortEnv = null
    }
    
    override fun getDetectorType(): String = "YuNet ONNX ($acceleratorType)"
}
