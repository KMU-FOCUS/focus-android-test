package com.kmu_focus.focustest.processing.detector.recognition

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import com.kmu_focus.focustest.processing.detector.landmark.yunet.FaceLandmarks5
import kotlin.math.abs

/**
 * 랜드마크 기반 얼굴 정렬 (ArcFace 입력 품질 향상)
 *
 * YuNet 5점(눈·코·입)으로 눈을 수평에 맞춰 회전.
 * 적당히 측면일 때도 정렬된 crop으로 인식률 개선.
 */
object FaceAlignment {

    /** 회전 각도가 이 값(도) 이하면 정렬 스킵 (이미 거의 정면) */
    private const val MIN_ANGLE_DEG = 1.5f

    /**
     * 얼굴 crop을 눈 기준으로 수평 정렬.
     * @param crop 얼굴 영역 crop (프레임 좌표 기준 rect로 잘린 비트맵)
     * @param landmarks YuNet 5점 (프레임 좌표)
     * @param faceRect 프레임 내 얼굴 bbox (crop의 원본 영역)
     * @return 정렬된 비트맵 (회전 필요 없으면 crop 그대로 반환, 아니면 새 비트맵 — 호출측에서 recycle)
     */
    fun alignFaceForRecognition(
        crop: Bitmap,
        landmarks: FaceLandmarks5,
        faceRect: Rect
    ): Bitmap {
        val angleRad = landmarks.getFaceAngle()
        val angleDeg = java.lang.Math.toDegrees(angleRad.toDouble()).toFloat()
        if (abs(angleDeg) < MIN_ANGLE_DEG) return crop

        val centerX = crop.width / 2f
        val centerY = crop.height / 2f
        val matrix = Matrix().apply {
            postRotate(-angleDeg, centerX, centerY)
        }
        val out = Bitmap.createBitmap(crop.width, crop.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.BLACK)
        canvas.drawBitmap(crop, matrix, Paint(Paint.FILTER_BITMAP_FLAG))
        return out
    }
}
