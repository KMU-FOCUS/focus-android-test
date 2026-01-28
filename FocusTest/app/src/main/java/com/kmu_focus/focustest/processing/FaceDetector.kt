package com.kmu_focus.focustest.processing

import android.graphics.Bitmap

/**
 * 얼굴 검출기 인터페이스
 */
interface FaceDetector {
    fun detectFaces(frame: Bitmap): List<DetectedFace>
    fun release()
    fun getDetectorType(): String
}

/**
 * 검출된 얼굴 정보
 */
data class DetectedFace(
    val x: Int,
    val y: Int,
    val width: Int,
    val height: Int,
    val confidence: Float
) {
    fun toRect(): android.graphics.Rect {
        return android.graphics.Rect(x, y, x + width, y + height)
    }
}

/**
 * 검출기 타입
 */
enum class DetectorType {
    YOLO_TFLITE,   // YOLOv12n + TFLite + NNAPI (NPU)
    YUNET_ONNX,    // YuNet + ONNX Runtime + NNAPI (NPU) - 느림
    YUNET_OPENCV   // YuNet + OpenCV (CPU) - 빠름
}
