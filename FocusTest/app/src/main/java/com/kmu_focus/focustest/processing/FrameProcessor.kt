package com.kmu_focus.focustest.processing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import com.kmu_focus.focustest.processing.detector.FaceDetector
import com.kmu_focus.focustest.processing.detector.landmark.model3d.FacialLandmarkDetector
import com.kmu_focus.focustest.processing.detector.recognition.ArcFaceEmbeddingExtractor
import com.kmu_focus.focustest.processing.detector.recognition.FaceAlignment
import com.kmu_focus.focustest.processing.detector.mosaic.FaceMosaicApplier
import com.kmu_focus.focustest.processing.detector.recognition.TrackLabelState
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
 * 얼굴 검출 + 랜드마크 검출 + 추적 + (선택) Owner/Other 판별 + 시각화 + 타원 모자이크
 */
class FrameProcessor(
    private val faceDetector: FaceDetector,
    private val landmarkDetector: FacialLandmarkDetector? = null,
    private val faceTracker: FaceTracker? = null,
    private val embeddingExtractor: ArcFaceEmbeddingExtractor? = null,
    private val trackLabelState: TrackLabelState? = null
) {

    companion object {
        /** 얼굴 모자이크 적용 (Owner/Other 사용 시 PENDING·OTHER만, 미사용 시 전체) */
        @JvmStatic
        var applyFaceMosaic: Boolean = true

        /** 화질 저하 방식: 1/n 해상도로 축소 후 복원 (16 = 극단적 저화질, 픽셀 루프 없음) */
        @JvmStatic
        var mosaicScaleDownFactor: Int = 16

        /** Owner/Other: 앞 N프레임 스킵(얼굴 잘릴 가능성) */
        const val SKIP_FRAMES = 5
        /** Owner/Other: 스킵 후 수집할 프레임 수 (이만큼 모이면 판별) */
        const val COLLECT_FRAMES = 3

        /** tracking_id별 박스 색상 (Owner/Other 미사용 시) */
        private val TRACK_COLORS = intArrayOf(
            Color.rgb(255, 0, 0), Color.rgb(0, 255, 0), Color.rgb(0, 0, 255),
            Color.rgb(255, 255, 0), Color.rgb(255, 0, 255), Color.rgb(0, 255, 255)
        )
        /** 뒷모습/가려진 얼굴(랜드마크 미판별) 박스 색상 */
        private val GRAY_BOX_COLOR = Color.GRAY
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
                val trackId = trackingIds.getOrElse(idx) { idx }
                FaceExport(
                    trackingId = trackId,
                    bbox = intArrayOf(face.x, face.y, face.width, face.height),
                    idCoeffs = raw3dmm?.idCoeffs,
                    expCoeffs = raw3dmm?.expCoeffs,
                    pose = raw3dmm?.pose
                )
            }
            FrameExport(frameNumber = frameIndex, timestamp = timestamp, faces = facesExport)
        } else null

        // 랜드마크 유효 여부: 3DMM 성공 시에만 인식 가능(뒷모습/가림은 스킵 → 속도 개선)
        fun hasValidLandmarks(idx: Int): Boolean =
            if (landmarkDetector != null) raw3dmmList.getOrNull(idx) != null
            else detectedFaces.getOrNull(idx)?.landmarks != null

        // Owner/Other 판별: 랜드마크 완전 판별된 얼굴만 ArcFace 검사
        trackLabelState?.beginFrame(trackingIds.toSet())
        val recheckedThisFrame = mutableSetOf<Int>()
        if (embeddingExtractor != null && trackLabelState != null && detectedFaces.isNotEmpty()) {
            for (idx in detectedFaces.indices) {
                if (!hasValidLandmarks(idx)) continue
                val face = detectedFaces[idx]
                val trackId = trackingIds.getOrElse(idx) { idx }
                trackLabelState!!.recordFrameSeen(trackId)
                val isRecognitionFriendly = face.landmarks?.isFrontal(0.4f) ?: false
                if (!trackLabelState!!.needsEmbeddingThisFrame(trackId, isRecognitionFriendly)) continue
                val rect = Rect(face.x, face.y, face.x + face.width, face.y + face.height)
                if (rect.width() < 16 || rect.height() < 16) continue
                var crop = Bitmap.createBitmap(
                    frame,
                    rect.left.coerceIn(0, frame.width - 1),
                    rect.top.coerceIn(0, frame.height - 1),
                    rect.width().coerceIn(1, frame.width - rect.left),
                    rect.height().coerceIn(1, frame.height - rect.top)
                )
                face.landmarks?.let { lm ->
                    val aligned = FaceAlignment.alignFaceForRecognition(crop, lm, rect)
                    if (aligned != crop) {
                        crop.recycle()
                        crop = aligned
                    }
                }
                embeddingExtractor.extractEmbedding(crop)?.let { emb ->
                    val label = trackLabelState.getLabel(trackId)
                    when (label) {
                        null -> trackLabelState.addEmbedding(trackId, emb)
                        false -> {
                            trackLabelState.recheckFrontal(trackId, emb)
                            recheckedThisFrame.add(trackId)
                        }
                        true -> { /* OWNER: 추가 검사 없음 */ }
                    }
                }
                crop.recycle()
            }
        }

        if (detectedFaces.isEmpty()) {
            return ProcessedFrameResult(frame, frameExport)
        }

        // 모자이크: 랜드마크 유효(식별 가능)한 얼굴만 처리. PENDING·OTHER만 적용, 뒷모습/가림은 스킵
        var result = if (applyFaceMosaic) {
            if (trackLabelState != null) {
                val faceLandmarksPerTrack = detectedFaces.mapIndexed { idx, face ->
                    Triple(idx, trackingIds.getOrElse(idx) { idx }, face.landmarks)
                }.filter { (idx, _, landmarks) -> hasValidLandmarks(idx) && landmarks != null }
                    .map { (_, trackId, landmarks) -> trackId to landmarks!! }
                FaceMosaicApplier.applyMosaicToPendingAndOther(
                    frame,
                    faceLandmarksPerTrack,
                    { trackId -> trackLabelState!!.getLabel(trackId) },
                    mosaicScaleDownFactor,
                    1.05f
                )
            } else {
                val landmarksList = detectedFaces.mapIndexed { idx, face -> idx to face.landmarks }
                    .filter { (idx, _) -> hasValidLandmarks(idx) }
                    .mapNotNull { (_, lm) -> lm }
                FaceMosaicApplier.applyMosaicToFaces(frame, landmarksList, mosaicScaleDownFactor, 1.05f)
            }
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

        val useOwnerOther = trackLabelState != null

        for ((idx, face) in detectedFaces.withIndex()) {
            val x = face.x
            val y = face.y
            val width = face.width
            val height = face.height
            val trackId = trackingIds.getOrElse(idx) { idx }
            val validLandmarks = hasValidLandmarks(idx)

            val (color, labelText) = when {
                !validLandmarks -> GRAY_BOX_COLOR to "ID:$trackId -"
                useOwnerOther -> {
                    if (trackId in recheckedThisFrame) {
                        Color.rgb(255, 255, 0) to "ID:$trackId 1/1"
                    } else {
                        val label = trackLabelState!!.getLabel(trackId)
                        when (label) {
                            true -> Color.rgb(0, 255, 0) to "ID:$trackId OWNER"
                            false -> Color.rgb(255, 0, 0) to "ID:$trackId OTHER"
                            null -> {
                                val framesSeen = trackLabelState!!.getFramesSeen(trackId)
                                val collectFrames = trackLabelState!!.getCollectFrames()
                                val lbl = if (framesSeen <= SKIP_FRAMES) {
                                    "ID:$trackId 대기"
                                } else {
                                    val hits = trackLabelState!!.getEmbeddingCount(trackId).coerceAtMost(collectFrames)
                                    "ID:$trackId $hits/$collectFrames"
                                }
                                Color.rgb(255, 255, 0) to lbl
                            }
                        }
                    }
                }
                else -> TRACK_COLORS[trackId % TRACK_COLORS.size] to "ID:$trackId"
            }

            paint.color = color
            canvas.drawRect(Rect(x, y, x + width, y + height), paint)

            textPaint.color = color
            val textY = (y - 8).toFloat().coerceAtLeast(textPaint.textSize)
            canvas.drawText(labelText, x.toFloat(), textY, textPaint)
        }

        return ProcessedFrameResult(result, frameExport)
    }
    
    fun release() {
        landmarkDetector?.release()
    }
}
