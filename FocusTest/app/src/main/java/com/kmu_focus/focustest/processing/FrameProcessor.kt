package com.kmu_focus.focustest.processing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import com.kmu_focus.focustest.processing.detector.FaceDetector
import com.kmu_focus.focustest.processing.detector.landmark.model3d.FacialLandmarkDetector
import com.kmu_focus.focustest.processing.detector.tracking.FaceTracker

/**
 * 프레임 처리 결과 (비트맵 + 서버 전송용 3DMM export)
 */
data class ProcessedFrameResult(
    val bitmap: Bitmap,
    val frameExport: FrameExport? = null
)

/**
 * 프레임 처리기
 * 얼굴 검출 + 랜드마크 검출 + 추적 + 시각화 + 타원 모자이크
 */
class FrameProcessor(
    private val faceDetector: FaceDetector,
    private val landmarkDetector: FacialLandmarkDetector? = null,
    private val faceTracker: FaceTracker? = null
) {

    companion object {
        /** 얼굴 모자이크 적용 */
        @JvmStatic
        var applyFaceMosaic: Boolean = false

        @JvmStatic
        var mosaicBlockSize: Int = 15

        /** ID별 박스 색상 (구분용) */
        private val TRACK_COLORS = intArrayOf(
            Color.rgb(255, 0, 0),
            Color.rgb(0, 255, 0),
            Color.rgb(0, 0, 255),
            Color.rgb(255, 255, 0),
            Color.rgb(255, 0, 255),
            Color.rgb(0, 255, 255),
            Color.rgb(255, 165, 0),
            Color.rgb(128, 0, 255),
            Color.rgb(0, 255, 128),
            Color.rgb(255, 128, 0),
            Color.rgb(128, 255, 0),
            Color.rgb(0, 128, 255),
            Color.rgb(255, 0, 128),
            Color.rgb(128, 128, 255),
            Color.rgb(255, 128, 255)
        )
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

        val raw3dmmList = if (detectedFaces.isNotEmpty() && (frameIndex != null || landmarkDetector != null)) {
            detectedFaces.map { face ->
                val faceRect = Rect(face.x, face.y, face.x + face.width, face.y + face.height)
                landmarkDetector?.detectLandmarks(frame, faceRect)?.raw3DMM
            }
        } else emptyList()

        val trackingIds: List<Int> = if (frameIndex != null && detectedFaces.isNotEmpty()) {
            val detections = detectedFaces.map { intArrayOf(it.x, it.y, it.width, it.height) }
            faceTracker?.update(detections, raw3dmmList.map { it?.idCoeffs })
                ?: detectedFaces.indices.toList()
        } else {
            detectedFaces.indices.toList()
        }

        val frameExport: FrameExport? = if (frameIndex != null && timestamp != null && detectedFaces.isNotEmpty()) {
            val facesExport = detectedFaces.mapIndexed { idx, face ->
                val raw3dmm = raw3dmmList.getOrNull(idx)
                FaceExport(
                    trackingId = trackingIds.getOrElse(idx) { idx },
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

        val canvas = Canvas(result)
        val paint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 4f
            isAntiAlias = true
        }
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 24f
            isAntiAlias = true
        }

        for ((idx, face) in detectedFaces.withIndex()) {
            val x = face.x
            val y = face.y
            val width = face.width
            val height = face.height
            val trackId = trackingIds.getOrElse(idx) { idx }
            val color = TRACK_COLORS[trackId % TRACK_COLORS.size]

            paint.color = color
            canvas.drawRect(Rect(x, y, x + width, y + height), paint)

            textPaint.color = color
            val labelText = "ID:$trackId"
            val textY = (y - 8).toFloat().coerceAtLeast(textPaint.textSize)
            canvas.drawText(labelText, x.toFloat(), textY, textPaint)
        }

        return ProcessedFrameResult(result, frameExport)
    }
    
    fun release() {
        landmarkDetector?.release()
    }
}
