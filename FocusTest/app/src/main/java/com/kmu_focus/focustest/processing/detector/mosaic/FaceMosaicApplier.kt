package com.kmu_focus.focustest.processing.detector.mosaic

import android.graphics.Bitmap
import com.kmu_focus.focustest.processing.FaceEllipseMask
import com.kmu_focus.focustest.processing.detector.landmark.yunet.FaceLandmarks5

/**
 * YuNet 5점 랜드마크 기반 얼굴 테두리(타원)로 모자이크 적용.
 * 판별중(PENDING)·Other 얼굴에만 모자이크를 칠할 때 사용.
 */
object FaceMosaicApplier {

    /**
     * 랜드마크 리스트에 대해 바운딩 박스 영역만 화질 저하 (다운스케일→업스케일).
     * 픽셀 루프 없이 Bitmap.createScaledBitmap만 사용 → 연산량 최소.
     *
     * @param scaleDownFactor 1/n 해상도로 축소 (16 = 극단적 저화질)
     */
    fun applyMosaicToFaces(
        source: Bitmap,
        landmarksList: List<FaceLandmarks5>,
        scaleDownFactor: Int = 16,
        paddingRatio: Float = 1.05f
    ): Bitmap {
        if (landmarksList.isEmpty()) return source.copy(Bitmap.Config.ARGB_8888, true)
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        for (landmarks in landmarksList) {
            FaceEllipseMask.applyLowResInPlaceBbox(result, landmarks, scaleDownFactor, paddingRatio)
        }
        return result
    }

    /**
     * trackId별 라벨에 따라 PENDING(null)·OTHER(false) 얼굴에만 화질 저하 적용.
     * OWNER(true) 또는 랜드마크 없는 얼굴은 제외.
     */
    fun applyMosaicToPendingAndOther(
        source: Bitmap,
        faceLandmarksPerTrack: List<Pair<Int, FaceLandmarks5?>>,
        getLabel: (Int) -> Boolean?,
        scaleDownFactor: Int = 16,
        paddingRatio: Float = 1.05f
    ): Bitmap {
        val toMosaic = faceLandmarksPerTrack
            .filter { (trackId, landmarks) ->
                landmarks != null && (getLabel(trackId) == null || getLabel(trackId) == false)
            }
            .mapNotNull { (_, landmarks) -> landmarks }
        return applyMosaicToFaces(source, toMosaic, scaleDownFactor, paddingRatio)
    }
}
