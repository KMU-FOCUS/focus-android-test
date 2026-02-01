package com.kmu_focus.focustest.processing.detector.landmark.model3d

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * 3DMM 얼굴 랜드마크 검출기 (TFLite)
 *
 * - 입력: 128x128 얼굴 이미지 (crop)
 * - 출력: 랜드마크 좌표 (2D 또는 3D, 정규화 0–1)
 * - NPU 가속 지원 (NNAPI)
 */
class FacialLandmarkDetector(context: Context) {
    
    companion object {
        private const val TAG = "FacialLandmarkDetector"
        private const val MODEL_NAME = "facial_landmark.tflite"
        private const val INPUT_SIZE = 128
        /** 3DMM id 계수 차원 (모델 출력 [1,K]에서 앞쪽 K_id개) */
        @JvmStatic
        var idDim: Int = 80
        /** 3DMM exp 계수 차원 (다음 K_exp개) */
        @JvmStatic
        var expDim: Int = 64
        /** pose 차원 = K - idDim - expDim */
        @JvmStatic
        var enableBenchmark: Boolean = true
    }
    
    private var interpreter: Interpreter? = null
    private var nnApiDelegate: NnApiDelegate? = null
    private var gpuDelegate: GpuDelegate? = null

    private var inputBuffer: ByteBuffer? = null
    
    private var numLandmarks: Int = 0
    private var numChannels: Int = 2
    private var outputShape: IntArray = intArrayOf()
    private var isQuantized: Boolean = false
    /** true = 3DMM 계수(id/exp/pose 등) 출력 — 원본 float 배열 전송 */
    private var isCoefficientMode: Boolean = false
    private var coefficientSize: Int = 0
    
    private var frameCounter: Int = 0
    private var totalInferenceMs: Long = 0
    
    init {
        initialize(context)
    }
    
    private fun initialize(context: Context) {
        try {
            val modelBuffer = loadModelFile(context, MODEL_NAME)

            val options = Interpreter.Options().apply {
                setNumThreads(4)
                try {
                    nnApiDelegate = NnApiDelegate(
                        NnApiDelegate.Options()
                            .setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
                            .setAllowFp16(true)
                    )
                    addDelegate(nnApiDelegate)
                    Log.i(TAG, "NNAPI Delegate 활성화")
                } catch (e: Exception) {
                    Log.w(TAG, "NNAPI 실패, GPU 시도: ${e.message}")
                    try {
                        if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                            gpuDelegate = GpuDelegate()
                            addDelegate(gpuDelegate)
                            Log.i(TAG, "GPU Delegate 활성화")
                        }
                    } catch (e2: Exception) {
                        Log.w(TAG, "GPU 실패, CPU 사용: ${e2.message}")
                    }
                }
            }

            interpreter = Interpreter(modelBuffer, options)
            
            val inputTensor = interpreter!!.getInputTensor(0)
            Log.i(TAG, "입력 shape: ${inputTensor.shape().contentToString()}")
            
            val outputCount = try {
                interpreter!!.javaClass.getMethod("getOutputTensorCount").invoke(interpreter!!) as Int
            } catch (_: Exception) { 1 }
            for (i in 0 until outputCount) {
                val t = interpreter!!.getOutputTensor(i)
                Log.i(TAG, "출력[$i] shape: ${t.shape().contentToString()}, dtype: ${t.dataType()}")
            }
            
            val outputTensor = interpreter!!.getOutputTensor(0)
            outputShape = outputTensor.shape()
            
            if (outputShape.size == 2 && outputShape[0] == 1) {
                coefficientSize = outputShape[1]
                isCoefficientMode = true
                numLandmarks = 0
                numChannels = 0
                Log.i(TAG, "✓ 3DMM 계수 모드: 출력 [1, $coefficientSize] — 원본 float 배열 전송")
            } else {
                if (outputShape.size == 3) {
                    numLandmarks = outputShape[1]
                    numChannels = outputShape[2].coerceIn(2, 3)
                } else {
                    numChannels = 2
                    numLandmarks = outputShape[1] / 2
                    if (outputShape[1] % 3 == 0) {
                        numChannels = 3
                        numLandmarks = outputShape[1] / 3
                    }
                }
                Log.i(TAG, "✓ 랜드마크(정점) 모드: ${numLandmarks}개, ${numChannels}D")
                if (numChannels == 2) {
                    Log.w(TAG, "모델 출력 2채널(x,y만) — 전송 시 z=0")
                }
            }
            
            isQuantized = inputTensor.dataType() == org.tensorflow.lite.DataType.UINT8
            
            val bytesPerChannel = if (isQuantized) 1 else 4
            inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * bytesPerChannel)
                .order(ByteOrder.nativeOrder())
            
        } catch (e: Exception) {
            throw IllegalStateException("랜드마크 검출기 초기화 오류: ${e.message}", e)
        }
    }
    
    private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun detectLandmarks(frame: Bitmap, faceRect: Rect): FaceLandmarks? {
        val interp = interpreter ?: return null
        
        try {
            val t0 = SystemClock.elapsedRealtimeNanos()
            
            val safeFaceRect = Rect(
                faceRect.left.coerceIn(0, frame.width - 1),
                faceRect.top.coerceIn(0, frame.height - 1),
                faceRect.right.coerceIn(1, frame.width),
                faceRect.bottom.coerceIn(1, frame.height)
            )
            
            if (safeFaceRect.width() <= 0 || safeFaceRect.height() <= 0) {
                return null
            }
            
            val faceCrop = Bitmap.createBitmap(
                frame,
                safeFaceRect.left,
                safeFaceRect.top,
                safeFaceRect.width(),
                safeFaceRect.height()
            )
            
            val resized = Bitmap.createScaledBitmap(faceCrop, INPUT_SIZE, INPUT_SIZE, true)
            if (faceCrop != resized) {
                faceCrop.recycle()
            }
            
            prepareInputBuffer(resized)
            resized.recycle()
            
            if (isCoefficientMode) {
                val outTensor = interp.getOutputTensor(0)
                val size = coefficientSize
                val coeffs = if (outTensor.dataType() == org.tensorflow.lite.DataType.UINT8) {
                    val outBuf = Array(1) { ByteArray(size) }
                    interp.run(inputBuffer, outBuf)
                    val scale = outTensor.quantizationParams().scale
                    val zeroPoint = outTensor.quantizationParams().zeroPoint
                    FloatArray(size) { i -> ((outBuf[0][i].toInt() and 0xFF) - zeroPoint) * scale }
                } else {
                    val outBuf = Array(1) { FloatArray(size) }
                    interp.run(inputBuffer, outBuf)
                    outBuf[0].copyOf()
                }
                val poseDim = size - idDim - expDim
                val idCoeffs = if (idDim > 0 && size >= idDim) coeffs.copyOfRange(0, idDim) else floatArrayOf()
                val expCoeffs = if (expDim > 0 && size >= idDim + expDim) coeffs.copyOfRange(idDim, idDim + expDim) else floatArrayOf()
                val pose = if (poseDim > 0 && size >= idDim + expDim + poseDim) coeffs.copyOfRange(idDim + expDim, size) else floatArrayOf()
                if (frameCounter == 0 && (idCoeffs.isEmpty() || expCoeffs.isEmpty() || pose.isEmpty())) {
                    Log.w(TAG, "3DMM 분할: id=$idDim exp=$expDim pose=$poseDim (총 $size) — idDim/expDim 조정 필요할 수 있음")
                }
                val t1 = SystemClock.elapsedRealtimeNanos()
                frameCounter++
                if (enableBenchmark && frameCounter % 30 == 0) {
                    Log.d(TAG, "[3DMM 계수] 평균 추론: ${(t1 - t0) / 1_000_000}ms (id=${idCoeffs.size} exp=${expCoeffs.size} pose=${pose.size})")
                }
                if (frameCounter == 1) {
                    Log.d(TAG, "3DMM id 처음 3개: ${idCoeffs.take(3).map { String.format("%.4f", it) }}, exp 처음 3개: ${expCoeffs.take(3).map { String.format("%.4f", it) }}, pose 처음 3개: ${pose.take(3).map { String.format("%.4f", it) }}")
                }
                return FaceLandmarks(emptyList(), safeFaceRect, null, null, raw3DMM = Face3DMMCoeffs(idCoeffs, expCoeffs, pose))
            }
            
            val outputSize = outputShape[1] * if (outputShape.size == 3) outputShape[2] else 1
            val landmarks: List<Landmark>
            val landmarks3D: List<Landmark3D>?
            
            if (isQuantized) {
                val outputBuffer = Array(1) { ByteArray(outputSize) }
                interp.run(inputBuffer, outputBuffer)
                val parsed = parseLandmarksQuantized(outputBuffer[0])
                landmarks = parsed.first
                landmarks3D = parsed.second
            } else {
                val outputBuffer = Array(1) { FloatArray(outputSize) }
                interp.run(inputBuffer, outputBuffer)
                val parsed = parseLandmarks(outputBuffer[0])
                landmarks = parsed.first
                landmarks3D = parsed.second
            }
            
            val t1 = SystemClock.elapsedRealtimeNanos()
            
            frameCounter++
            if (enableBenchmark) {
                val inferenceMs = (t1 - t0) / 1_000_000
                totalInferenceMs += inferenceMs
                if (frameCounter % 30 == 0) {
                    val avgMs = totalInferenceMs / frameCounter
                    Log.d(TAG, "[Landmark] 평균 추론: ${avgMs}ms (${numLandmarks}개)")
                }
            }
            
            val absolute3D = landmarks3D?.let { toAbsoluteLandmarks3D(safeFaceRect, it) }
            return FaceLandmarks(landmarks, safeFaceRect, absolute3D, rawLandmarks3D = landmarks3D, raw3DMM = null)
            
        } catch (e: Exception) {
            Log.e(TAG, "랜드마크 검출 오류: ${e.message}", e)
            return null
        }
    }
    
    private fun prepareInputBuffer(bitmap: Bitmap) {
        val buffer = inputBuffer ?: return
        buffer.rewind()
        
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        
        if (isQuantized) {
            for (pixel in pixels) {
                buffer.put(((pixel shr 16) and 0xFF).toByte())
                buffer.put(((pixel shr 8) and 0xFF).toByte())
                buffer.put((pixel and 0xFF).toByte())
            }
        } else {
            for (pixel in pixels) {
                buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
                buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
                buffer.putFloat((pixel and 0xFF) / 255.0f)
            }
        }
    }
    
    private fun parseLandmarks(output: FloatArray): Pair<List<Landmark>, List<Landmark3D>?> {
        val landmarks = mutableListOf<Landmark>()
        val list3D = if (numChannels == 3) mutableListOf<Landmark3D>() else null
        val stride = numChannels
        
        for (i in 0 until numLandmarks) {
            val xNorm = output[i * stride].coerceIn(0f, 1f)
            val yNorm = output[i * stride + 1].coerceIn(0f, 1f)
            landmarks.add(Landmark(xNorm, yNorm))
            if (numChannels == 3) {
                val xRaw = output[i * stride]
                val yRaw = output[i * stride + 1]
                val zRaw = output[i * stride + 2]
                list3D!!.add(Landmark3D(i, xRaw, yRaw, zRaw))
                if (frameCounter == 0 && i < 3) {
                    Log.d(TAG, "raw 3DMM [${i}]: x=$xRaw y=$yRaw z=$zRaw")
                }
            }
        }
        return landmarks to list3D
    }
    
    private fun parseLandmarksQuantized(output: ByteArray): Pair<List<Landmark>, List<Landmark3D>?> {
        val landmarks = mutableListOf<Landmark>()
        val list3D = if (numChannels == 3) mutableListOf<Landmark3D>() else null
        val stride = numChannels
        
        val outTensor = interpreter?.getOutputTensor(0)
        val scale = outTensor?.quantizationParams()?.scale ?: 0.0563f
        val zeroPoint = outTensor?.quantizationParams()?.zeroPoint ?: 132
        
        if (frameCounter == 0) {
            val dequantized = output.take(20).mapIndexed { _, b -> ((b.toInt() and 0xFF) - zeroPoint) * scale }
            Log.d(TAG, "dequant 처음 20개: ${dequantized.map { String.format("%.3f", it) }}")
        }
        
        for (i in 0 until numLandmarks) {
            val rawX = output[i * stride].toInt() and 0xFF
            val rawY = output[i * stride + 1].toInt() and 0xFF
            val dequantX = (rawX - zeroPoint) * scale
            val dequantY = (rawY - zeroPoint) * scale
            val x = dequantX.toFloat().coerceIn(0f, 1f)
            val y = dequantY.toFloat().coerceIn(0f, 1f)
            landmarks.add(Landmark(x, y))
            if (numChannels == 3) {
                val rawZ = output[i * stride + 2].toInt() and 0xFF
                val zRaw = (rawZ - zeroPoint) * scale
                list3D!!.add(Landmark3D(i, dequantX.toFloat(), dequantY.toFloat(), zRaw))
                if (frameCounter == 0 && i < 3) {
                    Log.d(TAG, "raw 3DMM quant [${i}]: x=${dequantX} y=${dequantY} z=$zRaw")
                }
            }
        }
        return landmarks to list3D
    }
    
    fun toAbsoluteLandmarks3D(faceRect: Rect, normalized3D: List<Landmark3D>): List<Landmark3D> {
        return normalized3D.map { lm ->
            Landmark3D(
                index = lm.index,
                x = faceRect.left + lm.x * faceRect.width(),
                y = faceRect.top + lm.y * faceRect.height(),
                z = lm.z
            )
        }
    }
    
    fun release() {
        interpreter?.close()
        interpreter = null
        nnApiDelegate?.close()
        nnApiDelegate = null
        gpuDelegate?.close()
        gpuDelegate = null
    }
    
    fun getLandmarkCount(): Int = numLandmarks
}
