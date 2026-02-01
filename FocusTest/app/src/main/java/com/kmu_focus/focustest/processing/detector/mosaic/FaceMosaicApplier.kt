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
     * 랜드마크 리스트에 대해 바운딩 박스 영역만 화질 저하.
     *
     * @param mosaicBufferScale 1f = 720p에서 직접 처리. 0.5f = 반 해상도 버퍼에서 처리 후 합성 (픽셀 연산 약 1/4).
     */
    fun applyMosaicToFaces(
        source: Bitmap,
        landmarksList: List<FaceLandmarks5>,
        scaleDownFactor: Int = 16,
        paddingRatio: Float = 1.05f,
        mosaicBufferScale: Float = 0.5f
    ): Bitmap {
        if (landmarksList.isEmpty()) return source.copy(Bitmap.Config.ARGB_8888, true)
        if (mosaicBufferScale >= 1f) {
            val result = source.copy(Bitmap.Config.ARGB_8888, true)
            for (landmarks in landmarksList) {
                FaceEllipseMask.applyLowResInPlaceBbox(result, landmarks, scaleDownFactor, paddingRatio)
            }
            return result
        }
        // 저해상도 버퍼에서 모자이크 후 720p에 합성
        val sw = (source.width * mosaicBufferScale).toInt().coerceAtLeast(2)
        val sh = (source.height * mosaicBufferScale).toInt().coerceAtLeast(2)
        val small = Bitmap.createScaledBitmap(source, sw, sh, true)
        val scale = mosaicBufferScale
        for (landmarks in landmarksList) {
            FaceEllipseMask.applyLowResInPlaceBbox(small, landmarks, scaleDownFactor, paddingRatio, coordScale = scale)
        }
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        for (landmarks in landmarksList) {
            val bounds = FaceEllipseMask.getEllipseBounds(landmarks, paddingRatio)
            val left = bounds.left.toInt().coerceIn(0, source.width - 1)
            val top = bounds.top.toInt().coerceIn(0, source.height - 1)
            val right = bounds.right.toInt().coerceIn(1, source.width)
            val bottom = bounds.bottom.toInt().coerceIn(1, source.height)
            if (right <= left || bottom <= top) continue
            val w = right - left
            val h = bottom - top
            val sl = (left * scale).toInt().coerceIn(0, sw - 1)
            val st = (top * scale).toInt().coerceIn(0, sh - 1)
            val sr = (right * scale).toInt().coerceIn(1, sw)
            val sb = (bottom * scale).toInt().coerceIn(1, sh)
            val sw2 = sr - sl
            val sh2 = sb - st
            if (sw2 < 2 || sh2 < 2) continue
            val region = Bitmap.createBitmap(small, sl, st, sw2, sh2)
            val up = Bitmap.createScaledBitmap(region, w, h, false)
            region.recycle()
            val pixels = IntArray(w * h)
            up.getPixels(pixels, 0, w, 0, 0, w, h)
            up.recycle()
            result.setPixels(pixels, 0, w, left, top, w, h)
        }
        small.recycle()
        return result
    }

    /**
     * trackId별 라벨에 따라 PENDING(null)·OTHER(false) 얼굴에만 화질 저하 적용.
     */
    fun applyMosaicToPendingAndOther(
        source: Bitmap,
        faceLandmarksPerTrack: List<Pair<Int, FaceLandmarks5?>>,
        getLabel: (Int) -> Boolean?,
        scaleDownFactor: Int = 16,
        paddingRatio: Float = 1.05f,
        mosaicBufferScale: Float = 0.5f
    ): Bitmap {
        val toMosaic = faceLandmarksPerTrack
            .filter { (trackId, landmarks) ->
                landmarks != null && (getLabel(trackId) == null || getLabel(trackId) == false)
            }
            .mapNotNull { (_, landmarks) -> landmarks }
        return applyMosaicToFaces(source, toMosaic, scaleDownFactor, paddingRatio, mosaicBufferScale)
    }
}
