package com.kmu_focus.focustest.processing.detector.tracking

/**
 * 방법 5: IoU + 3DMM id_coeffs 코사인 유사도
 * ID 스위칭 감소, 추가 모델 없음 (기존 FacialLandmarkDetector id_coeffs 활용)
 */
class IoU3DMMTracker(
    private val maxIouDistance: Float = 0.7f,
    private val maxCosineDistance: Float = 0.4f,  // 1 - cosineSim 최소
    private val iouWeight: Float = 0.6f,
    private val cosineWeight: Float = 0.4f,
    private val nInit: Int = 1,
    private val maxAge: Int = 30
) : FaceTracker {

    private val tracks = mutableListOf<Track>()

    override fun update(
        detections: List<IntArray>,
        idCoeffs: List<FloatArray?>?
    ): List<Int> {
        tracks.forEach { it.predict() }

        if (detections.isEmpty()) {
            tracks.removeAll { it.isDeleted() }
            return emptyList()
        }

        if (tracks.isEmpty()) {
            return detections.mapIndexed { i, det ->
                val coeffs = idCoeffs?.getOrNull(i)
                tracks.add(Track(TrackingUtils.intBboxToFloat(det), coeffs, nInit, maxAge))
                tracks.last().trackId
            }
        }

        val costMatrix = Array(detections.size) { FloatArray(tracks.size) }
        for (i in detections.indices) {
            val detCoeffs = idCoeffs?.getOrNull(i)
            for (j in tracks.indices) {
                val iou = TrackingUtils.iou(detections[i], tracks[j].getBbox())
                val iouDist = 1f - iou

                val cosineDist = if (detCoeffs != null) {
                    val trackCoeffs = tracks[j].getLatestIdCoeffs()
                    if (trackCoeffs != null && trackCoeffs.size == detCoeffs.size) {
                        val sim = TrackingUtils.cosineSimilarity(
                            TrackingUtils.normalizeL2(detCoeffs),
                            TrackingUtils.normalizeL2(trackCoeffs)
                        )
                        1f - sim
                    } else 0.5f  // coeffs 없으면 중립
                } else 0.5f

                val adjustedCosine = if (iou > 0.5f) maxCosineDistance * 1.2f else maxCosineDistance

                costMatrix[i][j] = when {
                    iouDist > maxIouDistance -> 1e5f
                    cosineDist > adjustedCosine -> 1e5f
                    else -> iouWeight * iouDist + cosineWeight * cosineDist
                }
            }
        }

        val (matches, unmatchedDets, unmatchedTracks) = TrackingUtils.greedyAssign(
            detections.size, tracks.size, costMatrix, 1e4f
        )

        val result = IntArray(detections.size) { -1 }
        for ((detIdx, trackIdx) in matches) {
            tracks[trackIdx].update(
                TrackingUtils.intBboxToFloat(detections[detIdx]),
                idCoeffs?.getOrNull(detIdx)
            )
            result[detIdx] = tracks[trackIdx].trackId
        }
        for (trackIdx in unmatchedTracks) {
            tracks[trackIdx].markMissed()
        }
        for (detIdx in unmatchedDets) {
            val t = Track(
                TrackingUtils.intBboxToFloat(detections[detIdx]),
                idCoeffs?.getOrNull(detIdx),
                nInit,
                maxAge
            )
            tracks.add(t)
            result[detIdx] = t.trackId
        }

        tracks.removeAll { it.isDeleted() }

        return result.toList()
    }

    override fun reset() {
        tracks.clear()
        Track.resetIdCounter()
    }
}
