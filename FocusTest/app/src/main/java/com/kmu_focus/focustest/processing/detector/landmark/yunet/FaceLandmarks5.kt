package com.kmu_focus.focustest.processing.detector.landmark.yunet

import android.graphics.PointF

/**
 * 5-point 얼굴 랜드마크 (YuNet 출력)
 * 오른쪽 눈, 왼쪽 눈, 코 끝, 오른쪽 입꼬리, 왼쪽 입꼬리
 */
data class FaceLandmarks5(
    val rightEye: PointF,
    val leftEye: PointF,
    val nose: PointF,
    val rightMouth: PointF,
    val leftMouth: PointF
) {
    /** 두 눈의 중심점 */
    fun getEyeCenter(): PointF {
        return PointF(
            (rightEye.x + leftEye.x) / 2f,
            (rightEye.y + leftEye.y) / 2f
        )
    }

    /** 두 눈 사이 거리 */
    fun getEyeDistance(): Float {
        val dx = leftEye.x - rightEye.x
        val dy = leftEye.y - rightEye.y
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    /** 입 중심점 */
    fun getMouthCenter(): PointF {
        return PointF(
            (rightMouth.x + leftMouth.x) / 2f,
            (rightMouth.y + leftMouth.y) / 2f
        )
    }

    /** 얼굴 기울기 각도 (라디안) */
    fun getFaceAngle(): Float {
        val dx = leftEye.x - rightEye.x
        val dy = leftEye.y - rightEye.y
        return kotlin.math.atan2(dy, dx)
    }

    /**
     * 정면 응시 여부 (YuNet 5점 기준)
     * - 코가 두 눈 중심에서 벗어난 정도로 yaw 근사
     * - 좌우 대칭이면 정면
     */
    fun isFrontal(symmetryThreshold: Float = 0.2f): Boolean {
        val eyeCenterX = (leftEye.x + rightEye.x) / 2f
        val eyeDist = getEyeDistance()
        if (eyeDist < 1e-6f) return false
        val noseOffset = kotlin.math.abs(nose.x - eyeCenterX) / eyeDist
        return noseOffset < symmetryThreshold
    }
}
