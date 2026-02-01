package com.kmu_focus.focustest.processing.detector.tracking

/**
 * 얼굴 추적기 인터페이스
 * @param detections bbox [x, y, w, h] 리스트
 * @param idCoeffs 3DMM id_coeffs (방법 5에서 사용, null 가능)
 * @return 각 detection에 할당된 tracking_id 리스트 (순서 유지)
 */
interface FaceTracker {
    fun update(
        detections: List<IntArray>,
        idCoeffs: List<FloatArray?>?
    ): List<Int>

    fun reset()
}
