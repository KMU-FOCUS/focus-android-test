package com.kmu_focus.focustest.processing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import com.kmu_focus.focustest.processing.detector.FaceDetector
import com.kmu_focus.focustest.processing.detector.landmark.model3d.FacialLandmarkDetector

/**
 * 프레임 처리 결과 (비트맵 + 서버 전송용 3DMM export)
 */
data class ProcessedFrameResult(
    val bitmap: Bitmap,
    val frameExport: FrameExport? = null
)

/**
 * 프레임 처리기
 * 얼굴 검출 + 랜드마크 검출 + 시각화 + 타원 모자이크
 */
class FrameProcessor(
    private val faceDetector: FaceDetector,
    private val landmarkDetector: FacialLandmarkDetector? = null
) {

    companion object {
        /** 3DMM 랜드마크 시각화 (서버 전송용만 쓸 경우 false) */
        @JvmStatic
        var drawLandmarks: Boolean = false

        /** 5-point 랜드마크 시각화 활성화 */
        @JvmStatic
        var draw5PointLandmarks: Boolean = true

        /** 타원 근사 얼굴 영역 표시 */
        @JvmStatic
        var drawFaceEllipse: Boolean = true

        /** 얼굴 모자이크 적용 */
        @JvmStatic
        var applyFaceMosaic: Boolean = false

        /** 모자이크 블록 크기 */
        @JvmStatic
        var mosaicBlockSize: Int = 15
    }
    
    /**
     * 프레임 처리
     * @param frame 원본 프레임
     * @param frameIndex export 시 사용 (null이면 export 미수집)
     * @param timestamp export 시 사용 (초 단위)
     * @return ProcessedFrameResult(비트맵, 3DMM export 데이터)
     */
    fun processFrame(
        frame: Bitmap,
        frameIndex: Int? = null,
        timestamp: Double? = null
    ): ProcessedFrameResult {
        val detectedFaces = faceDetector.detectFaces(frame)
            .filter { it.confidence >= 0.5f }

        // 서버 전송용 3DMM 수집 (frameIndex/timestamp 있을 때)
        val frameExport: FrameExport? = if (frameIndex != null && timestamp != null) {
            val facesExport = detectedFaces.mapIndexed { idx, face ->
                val faceRect = Rect(face.x, face.y, face.x + face.width, face.y + face.height)
                val raw3dmm = landmarkDetector?.detectLandmarks(frame, faceRect)?.raw3DMM
                FaceExport(
                    trackingId = idx,
                    bbox = intArrayOf(face.x, face.y, face.width, face.height),
                    idCoeffs = raw3dmm?.idCoeffs,
                    expCoeffs = raw3dmm?.expCoeffs,
                    pose = raw3dmm?.pose
                )
            }
            FrameExport(frameNumber = frameIndex, timestamp = timestamp, faces = facesExport)
        } else null

        if (detectedFaces.isEmpty()) {
            return ProcessedFrameResult(frame, frameExport)
        }

        // 모자이크 적용 (5-point 랜드마크 기반 타원 모자이크)
        var result = if (applyFaceMosaic) {
            var mosaicFrame = frame.copy(Bitmap.Config.ARGB_8888, true)
            for (face in detectedFaces) {
                face.landmarks?.let { landmarks ->
                    mosaicFrame = FaceEllipseMask.applyMosaic(
                        mosaicFrame,
                        landmarks,
                        mosaicBlockSize
                    )
                }
            }
            mosaicFrame
        } else {
            frame.copy(Bitmap.Config.ARGB_8888, true)
        }

        // 캔버스 및 페인트 준비
        val canvas = Canvas(result)
        val paint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }

        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 24f
            isAntiAlias = true
        }

        for (face in detectedFaces) {
            val x = face.x
            val y = face.y
            val width = face.width
            val height = face.height
            val confidence = face.confidence

            // 신뢰도에 따라 색상
            val color = when {
                confidence < 0.6f -> Color.rgb(255, 165, 0)  // 주황
                confidence < 0.8f -> Color.rgb(255, 255, 0)  // 노랑
                else -> Color.rgb(0, 255, 0)  // 초록
            }

            paint.color = color

            // 바운딩 박스 그리기
            canvas.drawRect(Rect(x, y, x + width, y + height), paint)

            // 신뢰도 텍스트
            textPaint.color = color
            val confidenceText = String.format("%.2f", confidence)
            val textY = (y - 10).toFloat().coerceAtLeast(textPaint.textSize)
            canvas.drawText(confidenceText, x.toFloat(), textY, textPaint)

            // 5-point 랜드마크 시각화
            if (draw5PointLandmarks && face.landmarks != null) {
                val lm = face.landmarks
                val landmarkPaint = Paint().apply {
                    this.color = Color.MAGENTA
                    style = Paint.Style.FILL
                    isAntiAlias = true
                }

                // 5개 포인트 그리기
                canvas.drawCircle(lm.rightEye.x, lm.rightEye.y, 4f, landmarkPaint)
                canvas.drawCircle(lm.leftEye.x, lm.leftEye.y, 4f, landmarkPaint)
                landmarkPaint.color = Color.YELLOW
                canvas.drawCircle(lm.nose.x, lm.nose.y, 4f, landmarkPaint)
                landmarkPaint.color = Color.CYAN
                canvas.drawCircle(lm.rightMouth.x, lm.rightMouth.y, 4f, landmarkPaint)
                canvas.drawCircle(lm.leftMouth.x, lm.leftMouth.y, 4f, landmarkPaint)
            }

            // 타원 근사 얼굴 영역 시각화
            if (drawFaceEllipse && face.landmarks != null) {
                val ellipsePaint = Paint().apply {
                    this.color = Color.argb(180, 0, 255, 255)  // 반투명 시안
                    style = Paint.Style.STROKE
                    strokeWidth = 2f
                    isAntiAlias = true
                }
                FaceEllipseMask.drawEllipseOutline(canvas, face.landmarks, ellipsePaint)
            }

            // 3DMM 기반 랜드마크 검출 (FacialLandmarkDetector 사용 시)
            if (drawLandmarks && landmarkDetector != null) {
                val faceRect = Rect(x, y, x + width, y + height)
                val landmarks = landmarkDetector.detectLandmarks(result, faceRect)

                if (landmarks != null) {
                    val landmarkPaint = Paint().apply {
                        this.color = Color.CYAN
                        style = Paint.Style.FILL
                        isAntiAlias = true
                    }

                    for (lm in landmarks.getAbsoluteLandmarks()) {
                        canvas.drawCircle(lm.x, lm.y, 2f, landmarkPaint)
                    }
                }
            }
        }

        return ProcessedFrameResult(result, frameExport)
    }
    
    fun release() {
        landmarkDetector?.release()
    }
}
