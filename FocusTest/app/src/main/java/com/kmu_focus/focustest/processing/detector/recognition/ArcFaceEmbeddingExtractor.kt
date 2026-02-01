package com.kmu_focus.focustest.processing.detector.recognition

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * ArcFace 기반 얼굴 임베딩 추출기 (w600k_mbf.onnx)
 *
 * recognition_tracking_test.py와 동일:
 * - 입력: 112x112 RGB, (pixel - 127.5) / 128.0, NCHW [1, 3, 112, 112]
 * - 출력: L2 정규화된 임베딩 벡터 (512차원 또는 모델 출력 차원)
 */
class ArcFaceEmbeddingExtractor(context: Context) {

    companion object {
        private const val TAG = "ArcFaceEmbedding"
        private const val MODEL_NAME = "w600k_mbf.onnx"
        private const val INPUT_SIZE = 112
        private const val NORM_MEAN = 127.5f
        private const val NORM_STD = 128.0f
    }

    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val outputName: String
    val embeddingDim: Int

    init {
        val modelBytes = context.assets.open(MODEL_NAME).use { it.readBytes() }
        val sessionOptions = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = ortEnvironment.createSession(modelBytes, sessionOptions)

        val inNames = session.inputNames
        val outNames = session.outputNames
        inputName = (inNames?.iterator()?.next() ?: inNames?.firstOrNull()) ?: ""
        outputName = (outNames?.iterator()?.next() ?: outNames?.firstOrNull()) ?: ""

        var dim = 512
        try {
            val outInfo = session.outputInfo[outputName]?.info
            if (outInfo is ai.onnxruntime.TensorInfo) {
                val shape = outInfo.shape
                if (shape != null && shape.isNotEmpty()) {
                    var d = 1
                    for (i in shape.indices) d *= shape[i].toInt()
                    dim = d
                }
            }
        } catch (_: Exception) {}
        embeddingDim = dim
        Log.i(TAG, "w600k_mbf 로드 완료: 입력=$inputName, 출력=$outputName, embeddingDim=$embeddingDim")
    }

    /**
     * 얼굴 crop Bitmap → ArcFace 표준 전처리 → [1, 3, 112, 112] float NCHW
     * (face alignment 없이 resize + RGB + (p-127.5)/128)
     */
    private fun preprocess(faceBitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(faceBitmap, INPUT_SIZE, INPUT_SIZE, true)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        if (resized != faceBitmap) resized.recycle()

        // NCHW: [1, 3, 112, 112] → 인덱스 c*112*112 + y*112 + x
        val input = FloatArray(1 * 3 * INPUT_SIZE * INPUT_SIZE)
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val pixel = pixels[y * INPUT_SIZE + x]
                val r = (Color.red(pixel) - NORM_MEAN) / NORM_STD
                val g = (Color.green(pixel) - NORM_MEAN) / NORM_STD
                val b = (Color.blue(pixel) - NORM_MEAN) / NORM_STD
                input[0 * (INPUT_SIZE * INPUT_SIZE) + y * INPUT_SIZE + x] = r
                input[1 * (INPUT_SIZE * INPUT_SIZE) + y * INPUT_SIZE + x] = g
                input[2 * (INPUT_SIZE * INPUT_SIZE) + y * INPUT_SIZE + x] = b
            }
        }
        return input
    }

    /**
     * L2 정규화 (ArcFace 표준)
     */
    private fun l2Normalize(vec: FloatArray): FloatArray {
        var norm = 0.0
        for (v in vec) norm += v * v
        norm = sqrt(norm)
        if (norm < 1e-8) return vec
        return FloatArray(vec.size) { vec[it] / norm.toFloat() }
    }

    /**
     * 얼굴 이미지(crop)에서 임베딩 추출. L2 정규화된 벡터 반환.
     */
    fun extractEmbedding(faceBitmap: Bitmap): FloatArray? {
        if (faceBitmap.width < 16 || faceBitmap.height < 16) return null
        return try {
            val inputArr = preprocess(faceBitmap)
            val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            val inputTensor = OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(inputArr), shape)
            try {
                val result = session.run(mapOf(inputName to inputTensor))
                try {
                    val output = result.get(0) as OnnxTensor
                    val outputBuffer = output.floatBuffer
                    val dim = outputBuffer.remaining()
                    val embedding = FloatArray(dim)
                    outputBuffer.get(embedding)
                    l2Normalize(embedding)
                } finally {
                    result.close()
                }
            } finally {
                inputTensor.close()
            }
        } catch (e: Exception) {
            Log.e(TAG, "임베딩 추출 실패: ${e.message}")
            null
        }
    }

    fun release() {
        try {
            session.close()
        } catch (_: Exception) {}
    }
}
