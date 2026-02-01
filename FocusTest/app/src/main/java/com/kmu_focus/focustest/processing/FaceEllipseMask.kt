package com.kmu_focus.focustest.processing

import android.graphics.*
import com.kmu_focus.focustest.processing.detector.landmark.yunet.FaceLandmarks5
import kotlin.math.cos
import kotlin.math.sin

/**
 * 5-point 랜드마크 기반 타원 근사 얼굴 마스크
 *
 * 5-point 랜드마크(눈2, 코1, 입꼬리2)를 사용하여 얼굴 영역을 타원으로 근사
 * - 타원 중심: 눈 중심과 입 중심의 중간점 (약간 위로 보정)
 * - 장축: 얼굴 세로 길이 (눈-입 거리 기반 추정)
 * - 단축: 얼굴 가로 길이 (눈 간 거리 기반 추정)
 * - 회전: 눈 기울기 각도
 */
object FaceEllipseMask {

    /**
     * 얼굴 타원 마스크 생성
     *
     * @param width 이미지 너비
     * @param height 이미지 높이
     * @param landmarks 5-point 랜드마크
     * @param paddingRatio 타원 크기 확장 비율 (1.0 = 원본, 1.2 = 20% 확대)
     * @return 얼굴 영역이 흰색(255)인 마스크 비트맵
     */
    fun createMask(
        width: Int,
        height: Int,
        landmarks: FaceLandmarks5,
        paddingRatio: Float = 1.05f
    ): Bitmap {
        val mask = Bitmap.createBitmap(width, height, Bitmap.Config.ALPHA_8)
        val canvas = Canvas(mask)

        // 배경을 검정(0)으로
        canvas.drawColor(Color.TRANSPARENT)

        // 타원 파라미터 계산
        val ellipse = calculateEllipseParams(landmarks, paddingRatio)

        // 타원 그리기 (흰색 = 얼굴 영역)
        val paint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        canvas.save()
        canvas.rotate(
            Math.toDegrees(ellipse.angle.toDouble()).toFloat(),
            ellipse.centerX,
            ellipse.centerY
        )
        canvas.drawOval(
            ellipse.centerX - ellipse.radiusX,
            ellipse.centerY - ellipse.radiusY,
            ellipse.centerX + ellipse.radiusX,
            ellipse.centerY + ellipse.radiusY,
            paint
        )
        canvas.restore()

        return mask
    }

    /**
     * 얼굴 영역에 블러 모자이크 적용
     *
     * @param source 원본 이미지
     * @param landmarks 5-point 랜드마크
     * @param blurRadius 블러 반경 (픽셀 기반 모자이크의 경우 블록 크기)
     * @param paddingRatio 타원 크기 확장 비율
     * @return 얼굴 영역이 모자이크된 이미지
     */
    fun applyMosaic(
        source: Bitmap,
        landmarks: FaceLandmarks5,
        blurRadius: Int = 20,
        paddingRatio: Float = 1.05f
    ): Bitmap {
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val ellipse = calculateEllipseParams(landmarks, paddingRatio)

        // 타원 바운딩 박스 계산
        val bounds = calculateEllipseBounds(ellipse)
        val left = bounds.left.toInt().coerceIn(0, source.width - 1)
        val top = bounds.top.toInt().coerceIn(0, source.height - 1)
        val right = bounds.right.toInt().coerceIn(1, source.width)
        val bottom = bounds.bottom.toInt().coerceIn(1, source.height)

        if (right <= left || bottom <= top) return result

        // 픽셀화 모자이크 적용 (타원 내부만)
        val blockSize = blurRadius.coerceAtLeast(4)

        for (by in top until bottom step blockSize) {
            for (bx in left until right step blockSize) {
                // 블록 중심이 타원 내부인지 확인
                val cx = bx + blockSize / 2f
                val cy = by + blockSize / 2f

                if (!isInsideEllipse(cx, cy, ellipse)) continue

                // 블록 평균 색상 계산
                var r = 0
                var g = 0
                var b = 0
                var count = 0

                val blockRight = (bx + blockSize).coerceAtMost(right)
                val blockBottom = (by + blockSize).coerceAtMost(bottom)

                for (py in by until blockBottom) {
                    for (px in bx until blockRight) {
                        if (isInsideEllipse(px.toFloat(), py.toFloat(), ellipse)) {
                            val pixel = source.getPixel(px, py)
                            r += Color.red(pixel)
                            g += Color.green(pixel)
                            b += Color.blue(pixel)
                            count++
                        }
                    }
                }

                if (count > 0) {
                    val avgColor = Color.rgb(r / count, g / count, b / count)

                    // 블록 내 타원 영역에 평균 색상 적용
                    for (py in by until blockBottom) {
                        for (px in bx until blockRight) {
                            if (isInsideEllipse(px.toFloat(), py.toFloat(), ellipse)) {
                                result.setPixel(px, py, avgColor)
                            }
                        }
                    }
                }
            }
        }

        return result
    }

    /**
     * 타원 테두리 그리기 (시각화용)
     */
    fun drawEllipseOutline(
        canvas: Canvas,
        landmarks: FaceLandmarks5,
        paint: Paint,
        paddingRatio: Float = 1.05f
    ) {
        val ellipse = calculateEllipseParams(landmarks, paddingRatio)

        canvas.save()
        canvas.rotate(
            Math.toDegrees(ellipse.angle.toDouble()).toFloat(),
            ellipse.centerX,
            ellipse.centerY
        )
        canvas.drawOval(
            ellipse.centerX - ellipse.radiusX,
            ellipse.centerY - ellipse.radiusY,
            ellipse.centerX + ellipse.radiusX,
            ellipse.centerY + ellipse.radiusY,
            paint
        )
        canvas.restore()
    }

    /**
     * 타원 파라미터 계산
     */
    private fun calculateEllipseParams(
        landmarks: FaceLandmarks5,
        paddingRatio: Float
    ): EllipseParams {
        val eyeCenter = landmarks.getEyeCenter()
        val mouthCenter = landmarks.getMouthCenter()
        val eyeDistance = landmarks.getEyeDistance()
        val angle = landmarks.getFaceAngle()

        // 눈-입 거리
        val eyeMouthDist = kotlin.math.sqrt(
            (mouthCenter.x - eyeCenter.x) * (mouthCenter.x - eyeCenter.x) +
            (mouthCenter.y - eyeCenter.y) * (mouthCenter.y - eyeCenter.y)
        )

        // 타원 중심: 눈 중심과 입 중심 사이 (눈 쪽으로 약간 치우침)
        // 이마 영역을 포함하기 위해 눈 위쪽으로 확장
        val centerX = eyeCenter.x + (mouthCenter.x - eyeCenter.x) * 0.3f
        val centerY = eyeCenter.y + (mouthCenter.y - eyeCenter.y) * 0.3f

        // 타원 크기 (얼굴에 타이트하게 fit)
        val radiusX = eyeDistance * 1.0f * paddingRatio   // 가로 반경
        val radiusY = eyeMouthDist * 1.35f * paddingRatio // 세로 반경 (이마+턱 포함)

        return EllipseParams(centerX, centerY, radiusX, radiusY, angle)
    }

    /**
     * 타원의 바운딩 박스 계산 (회전 고려)
     */
    private fun calculateEllipseBounds(ellipse: EllipseParams): RectF {
        val cos = cos(ellipse.angle)
        val sin = sin(ellipse.angle)

        // 회전된 타원의 바운딩 박스
        val ux = ellipse.radiusX * cos
        val uy = ellipse.radiusX * sin
        val vx = ellipse.radiusY * -sin
        val vy = ellipse.radiusY * cos

        val halfWidth = kotlin.math.sqrt(ux * ux + vx * vx)
        val halfHeight = kotlin.math.sqrt(uy * uy + vy * vy)

        return RectF(
            ellipse.centerX - halfWidth,
            ellipse.centerY - halfHeight,
            ellipse.centerX + halfWidth,
            ellipse.centerY + halfHeight
        )
    }

    /**
     * 점이 타원 내부에 있는지 확인
     */
    private fun isInsideEllipse(x: Float, y: Float, ellipse: EllipseParams): Boolean {
        // 타원 중심 기준 좌표
        val dx = x - ellipse.centerX
        val dy = y - ellipse.centerY

        // 회전 역변환
        val cos = cos(-ellipse.angle)
        val sin = sin(-ellipse.angle)
        val rx = dx * cos - dy * sin
        val ry = dx * sin + dy * cos

        // 타원 방정식: (x/a)^2 + (y/b)^2 <= 1
        val normalizedX = rx / ellipse.radiusX
        val normalizedY = ry / ellipse.radiusY

        return normalizedX * normalizedX + normalizedY * normalizedY <= 1.0f
    }

    /**
     * 타원 파라미터 데이터 클래스
     */
    data class EllipseParams(
        val centerX: Float,
        val centerY: Float,
        val radiusX: Float,  // 가로 반경 (단축)
        val radiusY: Float,  // 세로 반경 (장축)
        val angle: Float     // 회전 각도 (라디안)
    )
}
