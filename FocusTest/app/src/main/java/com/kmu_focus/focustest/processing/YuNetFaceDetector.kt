package com.kmu_focus.focustest.processing

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.java.TfLite
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 얼굴 검출기 (YOLOv12n-face TFLite + NPU/GPU 가속)
 *
 * Galaxy S25 최적화:
 * - INT8 양자화 모델 사용 (yolov12n-face_full_integer_quant.tflite)
 * - NNAPI Delegate + 모델 캐싱
 * - LUT 기반 고속 양자화
 * - YOLOv12n: nano 버전으로 s 대비 3-4배 빠름
 */
class YuNetFaceDetector(context: Context) : FaceDetector {

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null

    // 모델 입력 정보
    private var inputWidth: Int = 0
    private var inputHeight: Int = 0
    private var inputDataType: DataType = DataType.FLOAT32

    // 모델 출력 정보
    private var outputDataType: DataType = DataType.FLOAT32
    private var numAnchors: Int = 0

    // 양자화 파라미터
    private var inputScale: Float = 1f
    private var inputZeroPoint: Int = 0
    private var outputScale: Float = 1f
    private var outputZeroPoint: Int = 0

    // INT8 양자화 LUT (Look-Up Table) - 0~255 → -128~127 매핑
    private var quantizeLUT: ByteArray? = null

    // 임계값
    private val confThreshold = 0.5f
    private val nmsThreshold = 0.3f

    // 전처리용 버퍼 재사용
    private var rgbaMat: Mat? = null
    private var resizedMat: Mat? = null
    private var rgbMat: Mat? = null
    private var inputBuffer: ByteBuffer? = null
    private var inputByteArray: ByteArray? = null

    // 출력 버퍼 재사용
    private var outputByteArray: Array<Array<ByteArray>>? = null
    private var outputFloatArray: Array<Array<FloatArray>>? = null

    // 캐시 디렉토리
    private var cacheDir: File? = null

    // 디버그/벤치마크
    private var frameCounter: Int = 0
    private var acceleratorType: String = "Unknown"
    
    // 벤치마크 누적
    private var totalPreprocessMs: Long = 0
    private var totalInferenceMs: Long = 0
    private var totalPostprocessMs: Long = 0

    companion object {
        private const val TAG = "YuNetFaceDetector"
        
        private const val MODEL_NAME_QUANTIZED = "yolov12n-face_full_integer_quant.tflite"
        // Float32 모델 없음 - INT8 양자화 모델만 사용
        private const val MODEL_NAME_FLOAT = "yolov12n-face_full_integer_quant.tflite"

        enum class AcceleratorMode {
            AUTO,
            NPU_NNAPI,  // NNAPI (범용, S25에서 부분 NPU 가속)
            GPU,
            CPU
        }

        @JvmStatic
        var acceleratorMode: AcceleratorMode = AcceleratorMode.AUTO

        @JvmStatic
        var useQuantizedModel: Boolean = true
        
        /** 벤치마크 로그 활성화 */
        @JvmStatic
        var enableBenchmark: Boolean = true
    }

    init {
        cacheDir = context.cacheDir
        initializeGooglePlayTfLite(context)
        initializeDetector(context)
    }

    private fun initializeGooglePlayTfLite(context: Context) {
        try {
            val options = TfLiteInitializationOptions.builder()
                .setEnableGpuDelegateSupport(true)
                .build()
            val initTask = TfLite.initialize(context, options)
            Tasks.await(initTask)
            Log.i(TAG, "Google Play Services TFLite 초기화 완료")
        } catch (e: Exception) {
            Log.w(TAG, "Google Play Services TFLite 초기화 실패: ${e.message}")
        }
    }

    private fun loadModelFile(context: Context, filename: String): ByteBuffer {
        context.assets.open(filename).use { input ->
            val bytes = input.readBytes()
            return ByteBuffer.allocateDirect(bytes.size).apply {
                order(ByteOrder.nativeOrder())
                put(bytes)
                rewind()
            }
        }
    }

    private fun initializeDetector(context: Context) {
        try {
            val modelName = if (useQuantizedModel) {
                val exists = try {
                    context.assets.open(MODEL_NAME_QUANTIZED).close()
                    true
                } catch (e: Exception) { false }
                if (exists) MODEL_NAME_QUANTIZED else MODEL_NAME_FLOAT
            } else {
                MODEL_NAME_FLOAT
            }

            val modelExists = try {
                context.assets.open(modelName).close()
                true
            } catch (e: Exception) {
                Log.e(TAG, "모델 파일을 찾을 수 없음: $modelName")
                false
            }

            if (!modelExists) {
                throw IllegalStateException("모델이 assets에 없습니다: $modelName")
            }

            val modelBuffer = loadModelFile(context, modelName)
            val isQuantized = modelName.contains("quant")

            Log.i(TAG, "모델 로드: $modelName (양자화: $isQuantized)")

            val options = createInterpreterOptions(context, isQuantized)
            val tflite = Interpreter(modelBuffer, options)
            interpreter = tflite

            // 입력 텐서 정보
            val inputTensor = tflite.getInputTensor(0)
            val inShape = inputTensor.shape()
            inputDataType = inputTensor.dataType()

            if (inShape.size != 4) {
                throw IllegalStateException("예상치 못한 입력 shape: ${inShape.contentToString()}")
            }
            inputHeight = inShape[1]
            inputWidth = inShape[2]

            // 양자화 파라미터
            if (inputDataType == DataType.UINT8 || inputDataType == DataType.INT8) {
                val quantParams = inputTensor.quantizationParams()
                inputScale = quantParams.scale
                inputZeroPoint = quantParams.zeroPoint
                Log.i(TAG, "입력 양자화: scale=$inputScale, zeroPoint=$inputZeroPoint")
                
                // LUT 생성 (한 번만)
                buildQuantizeLUT()
            }

            // 출력 텐서 정보
            val outputTensor = tflite.getOutputTensor(0)
            val outShape = outputTensor.shape()
            outputDataType = outputTensor.dataType()
            numAnchors = outShape[2]

            if (outputDataType == DataType.UINT8 || outputDataType == DataType.INT8) {
                val quantParams = outputTensor.quantizationParams()
                outputScale = quantParams.scale
                outputZeroPoint = quantParams.zeroPoint
                Log.i(TAG, "출력 양자화: scale=$outputScale, zeroPoint=$outputZeroPoint")
            }

            // 버퍼 사전 할당
            preallocateBuffers()

            Log.i(TAG, "✓ YOLOv12n-face 초기화 완료")
            Log.i(TAG, "  - 입력: ${inputWidth}x$inputHeight, dtype=$inputDataType")
            Log.i(TAG, "  - 출력: 앵커=$numAnchors, dtype=$outputDataType")
            Log.i(TAG, "  - 가속기: $acceleratorType")

        } catch (e: Exception) {
            throw IllegalStateException("YOLOv12n-face 초기화 중 오류: ${e.message}", e)
        }
    }

    /**
     * INT8 양자화 LUT 생성 (0~255 픽셀 → INT8 양자화 값)
     * 런타임에 매 픽셀 연산 대신 테이블 조회로 고속화
     */
    private fun buildQuantizeLUT() {
        quantizeLUT = ByteArray(256) { pixel ->
            val realValue = pixel / 255.0f
            val qval = ((realValue / inputScale) + inputZeroPoint).toInt().coerceIn(-128, 127)
            qval.toByte()
        }
        Log.i(TAG, "양자화 LUT 생성 완료 (256 entries)")
    }

    /**
     * 버퍼 사전 할당
     */
    private fun preallocateBuffers() {
        val numElements = inputWidth * inputHeight * 3
        
        when (inputDataType) {
            DataType.INT8, DataType.UINT8 -> {
                inputBuffer = ByteBuffer.allocateDirect(numElements).order(ByteOrder.nativeOrder())
                inputByteArray = ByteArray(numElements)
            }
            else -> {
                inputBuffer = ByteBuffer.allocateDirect(numElements * 4).order(ByteOrder.nativeOrder())
            }
        }
        
        when (outputDataType) {
            DataType.INT8, DataType.UINT8 -> {
                outputByteArray = Array(1) { Array(5) { ByteArray(numAnchors) } }
            }
            else -> {
                outputFloatArray = Array(1) { Array(5) { FloatArray(numAnchors) } }
            }
        }
    }

    private fun createInterpreterOptions(context: Context, isQuantized: Boolean): Interpreter.Options {
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }

        when (acceleratorMode) {
            AcceleratorMode.AUTO -> {
                // NNAPI > GPU > CPU 순서로 시도
                if (isQuantized) {
                    if (tryAddNnApiDelegate(context, options)) {
                        acceleratorType = "NNAPI (NPU/DSP)"
                        return options
                    }
                }
                if (tryAddGpuDelegate(options, isQuantized)) {
                    acceleratorType = "GPU"
                    return options
                }
                acceleratorType = "CPU"
            }

            AcceleratorMode.NPU_NNAPI -> {
                if (tryAddNnApiDelegate(context, options)) {
                    acceleratorType = "NNAPI (NPU/DSP)"
                } else {
                    acceleratorType = "CPU (NNAPI 실패)"
                }
            }

            AcceleratorMode.GPU -> {
                if (tryAddGpuDelegate(options, isQuantized)) {
                    acceleratorType = "GPU"
                } else {
                    acceleratorType = "CPU (GPU 실패)"
                }
            }

            AcceleratorMode.CPU -> {
                acceleratorType = "CPU"
            }
        }

        return options
    }

    /**
     * NNAPI Delegate + 모델 캐싱
     * 
     * S25 NPU 최적화 설정:
     * - EXECUTION_PREFERENCE_SUSTAINED_SPEED: 지속적 고속 추론
     * - allowFp16: FP16 연산 허용 (일부 연산 NPU 지원 확대)
     * - useNnapiCpu: false (CPU 폴백 시 TFLite CPU 사용, NNAPI CPU보다 빠름)
     */
    private fun tryAddNnApiDelegate(context: Context, options: Interpreter.Options): Boolean {
        return try {
            val nnApiOptions = NnApiDelegate.Options().apply {
                // 성능 우선 모드
                setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
                
                // FP16 비허용 - INT8 모델에 맞게
                setAllowFp16(false)
                
                // NNAPI CPU 폴백 비활성화 
                // (폴백 시 TFLite의 XNNPACK이 NNAPI CPU보다 빠름)
                setUseNnapiCpu(false)
                
                // 모델 컴파일 캐싱 (초기화 속도 대폭 향상)
                cacheDir?.let { dir ->
                    val cacheFile = File(dir, "nnapi_cache")
                    if (!cacheFile.exists()) cacheFile.mkdirs()
                    setCacheDir(cacheFile.absolutePath)
                    setModelToken("yolov12n_face_int8")
                    Log.i(TAG, "NNAPI 캐시 설정: ${cacheFile.absolutePath}")
                }
            }
            
            nnApiDelegate = NnApiDelegate(nnApiOptions)
            options.addDelegate(nnApiDelegate)
            Log.i(TAG, "NNAPI Delegate 활성화 (FP16 허용)")
            true
        } catch (e: Exception) {
            Log.w(TAG, "NNAPI Delegate 실패: ${e.message}")
            nnApiDelegate = null
            false
        }
    }

    private fun tryAddGpuDelegate(options: Interpreter.Options, isQuantized: Boolean): Boolean {
        return try {
            val compatList = CompatibilityList()
            Log.i(TAG, "GPU 호환성 체크: isDelegateSupported=${compatList.isDelegateSupportedOnThisDevice}")
            
            if (!compatList.isDelegateSupportedOnThisDevice) {
                Log.w(TAG, "GPU Delegate 미지원 디바이스 - CompatibilityList 체크 실패")
                return false
            }

            val gpuOptions = GpuDelegate.Options().apply {
                if (isQuantized) {
                    Log.i(TAG, "GPU: 양자화 모델 허용 설정")
                    setQuantizedModelsAllowed(true)
                }
                setPrecisionLossAllowed(true)
                setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
            }

            Log.i(TAG, "GPU Delegate 생성 시도...")
            gpuDelegate = GpuDelegate(gpuOptions)
            options.addDelegate(gpuDelegate)
            Log.i(TAG, "GPU Delegate 활성화 성공")
            true
        } catch (e: Exception) {
            Log.e(TAG, "GPU Delegate 실패: ${e.javaClass.simpleName}: ${e.message}")
            e.printStackTrace()
            gpuDelegate = null
            false
        }
    }

    /**
     * 프레임에서 얼굴 검출
     */
    override fun detectFaces(frame: Bitmap): List<DetectedFace> {
        val tflite = interpreter ?: return emptyList()

        val origWidth = frame.width
        val origHeight = frame.height

        try {
            // === 전처리 ===
            val t0 = SystemClock.elapsedRealtimeNanos()
            
            // 1. Bitmap → Mat (RGBA)
            val rgba = rgbaMat ?: Mat().also { rgbaMat = it }
            Utils.bitmapToMat(frame, rgba)

            // 2. Resize (RGBA 상태로)
            val resized = resizedMat ?: Mat().also { resizedMat = it }
            Imgproc.resize(rgba, resized, Size(inputWidth.toDouble(), inputHeight.toDouble()))

            // 3. RGBA → RGB (BGR 거치지 않음)
            val rgb = rgbMat ?: Mat().also { rgbMat = it }
            Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_RGBA2RGB)

            // 4. 입력 버퍼 준비
            val inputBuf = prepareInputBufferFast(rgb)
            
            val t1 = SystemClock.elapsedRealtimeNanos()

            // === 추론 ===
            val predictions = runInferenceFast(tflite, inputBuf)
            
            val t2 = SystemClock.elapsedRealtimeNanos()

            // === 후처리 ===
            val boxes = mutableListOf<IntArray>()
            val scores = mutableListOf<Float>()

            for (i in 0 until numAnchors) {
                val conf = predictions[4][i]
                if (conf < confThreshold) continue

                val xCenter = predictions[0][i]
                val yCenter = predictions[1][i]
                val wNorm = predictions[2][i]
                val hNorm = predictions[3][i]

                val xCenterAbs = xCenter * origWidth
                val yCenterAbs = yCenter * origHeight
                val wAbs = wNorm * origWidth
                val hAbs = hNorm * origHeight

                val x1 = (xCenterAbs - wAbs / 2f).toInt()
                val y1 = (yCenterAbs - hAbs / 2f).toInt()
                val boxW = wAbs.toInt()
                val boxH = hAbs.toInt()

                val clippedX = x1.coerceIn(0, origWidth)
                val clippedY = y1.coerceIn(0, origHeight)
                val clippedW = boxW.coerceIn(1, origWidth - clippedX)
                val clippedH = boxH.coerceIn(1, origHeight - clippedY)

                boxes.add(intArrayOf(clippedX, clippedY, clippedW, clippedH))
                scores.add(conf)
            }

            val keep = nms(boxes, scores, nmsThreshold)

            val faces = mutableListOf<DetectedFace>()
            for (idx in keep) {
                val b = boxes[idx]
                faces.add(DetectedFace(b[0], b[1], b[2], b[3], scores[idx]))
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
                    
                    Log.d(TAG, "[$acceleratorType] 전처리:${avgPre}ms 추론:${avgInf}ms 후처리:${avgPost}ms = ${totalAvg}ms (${String.format("%.1f", fps)} FPS)")
                    Log.d(TAG, "  raw=${boxes.size}, NMS=${faces.size}")
                }
            }

            return faces
        } catch (e: Exception) {
            Log.e(TAG, "검출 중 오류: ${e.message}")
            return emptyList()
        }
    }

    /**
     * 고속 입력 버퍼 준비 (LUT 사용)
     */
    private fun prepareInputBufferFast(rgb: Mat): ByteBuffer {
        val buf = inputBuffer!!
        buf.clear()

        when (inputDataType) {
            DataType.INT8 -> {
                val lut = quantizeLUT!!
                val byteArr = inputByteArray!!
                rgb.get(0, 0, byteArr)
                
                // LUT 기반 양자화 (연산 없이 테이블 조회만)
                for (i in byteArr.indices) {
                    val pixel = byteArr[i].toInt() and 0xFF
                    byteArr[i] = lut[pixel]
                }
                buf.put(byteArr)
            }
            
            DataType.UINT8 -> {
                val byteArr = inputByteArray!!
                rgb.get(0, 0, byteArr)
                buf.put(byteArr)
            }
            
            else -> {
                val floatMat = Mat()
                rgb.convertTo(floatMat, CvType.CV_32F, 1.0 / 255.0)
                val numElements = inputWidth * inputHeight * 3
                val floatArr = FloatArray(numElements)
                floatMat.get(0, 0, floatArr)
                buf.asFloatBuffer().put(floatArr)
                floatMat.release()
            }
        }
        
        buf.rewind()
        return buf
    }

    /**
     * 고속 추론 (버퍼 재사용)
     */
    private fun runInferenceFast(tflite: Interpreter, inputBuf: ByteBuffer): Array<FloatArray> {
        val result = Array(5) { FloatArray(numAnchors) }

        when (outputDataType) {
            DataType.INT8 -> {
                val output = outputByteArray!!
                tflite.run(inputBuf, output)
                
                // 역양자화
                val scale = outputScale
                val zp = outputZeroPoint
                for (c in 0 until 5) {
                    val outC = output[0][c]
                    val resC = result[c]
                    for (i in 0 until numAnchors) {
                        resC[i] = (outC[i].toInt() - zp) * scale
                    }
                }
            }
            
            DataType.UINT8 -> {
                val output = outputByteArray!!
                tflite.run(inputBuf, output)
                
                val scale = outputScale
                val zp = outputZeroPoint
                for (c in 0 until 5) {
                    val outC = output[0][c]
                    val resC = result[c]
                    for (i in 0 until numAnchors) {
                        resC[i] = ((outC[i].toInt() and 0xFF) - zp) * scale
                    }
                }
            }
            
            else -> {
                val output = outputFloatArray!!
                tflite.run(inputBuf, output)
                for (c in 0 until 5) {
                    System.arraycopy(output[0][c], 0, result[c], 0, numAnchors)
                }
            }
        }

        return result
    }

    private fun nms(boxes: List<IntArray>, scores: List<Float>, iouThreshold: Float): List<Int> {
        if (boxes.isEmpty()) return emptyList()

        val indices = boxes.indices.sortedByDescending { scores[it] }
        val keep = mutableListOf<Int>()
        val suppressed = BooleanArray(boxes.size)

        for (i in indices) {
            if (suppressed[i]) continue
            keep.add(i)
            val boxA = boxes[i]
            for (j in indices) {
                if (j == i || suppressed[j]) continue
                val boxB = boxes[j]
                
                val x1 = maxOf(boxA[0], boxB[0])
                val y1 = maxOf(boxA[1], boxB[1])
                val x2 = minOf(boxA[0] + boxA[2], boxB[0] + boxB[2])
                val y2 = minOf(boxA[1] + boxA[3], boxB[1] + boxB[3])

                val interW = (x2 - x1).coerceAtLeast(0)
                val interH = (y2 - y1).coerceAtLeast(0)
                val interArea = interW * interH

                if (interArea > 0) {
                    val areaA = boxA[2] * boxA[3]
                    val areaB = boxB[2] * boxB[3]
                    val iou = interArea.toFloat() / (areaA + areaB - interArea)
                    if (iou > iouThreshold) {
                        suppressed[j] = true
                    }
                }
            }
        }

        return keep
    }

    fun getAcceleratorType(): String = acceleratorType

    override fun release() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
        nnApiDelegate?.close()
        nnApiDelegate = null
    }
    
    override fun getDetectorType(): String = "YOLO TFLite ($acceleratorType)"
}
