package com.kmu_focus.focustest.processing.detector

import android.graphics.Bitmap
import com.kmu_focus.focustest.processing.detector.landmark.yunet.FaceLandmarks5

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
    val confidence: Float,
    val landmarks: FaceLandmarks5? = null
) {
    fun toRect(): android.graphics.Rect {
        return android.graphics.Rect(x, y, x + width, y + height)
    }
}
