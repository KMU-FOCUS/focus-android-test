package com.kmu_focus.focustest.processing.detector.landmark.model3d

/**
 * 3DMM 계수 (id/exp/pose) — 모델 출력 [1,K]를 id_coeffs, exp_coeffs, pose로 분할
 */
data class Face3DMMCoeffs(
    val idCoeffs: FloatArray,
    val expCoeffs: FloatArray,
    val pose: FloatArray
) {
    override fun equals(other: Any?) = (other is Face3DMMCoeffs) &&
        idCoeffs.contentEquals(other.idCoeffs) &&
        expCoeffs.contentEquals(other.expCoeffs) &&
        pose.contentEquals(other.pose)
    override fun hashCode() = idCoeffs.contentHashCode() + 31 * expCoeffs.contentHashCode() + 31 * 31 * pose.contentHashCode()
}

/**
 * 얼굴 랜드마크 좌표 (2D, 정규화 0–1)
 */
data class Landmark(
    val x: Float,
    val y: Float
)

/**
 * 3D 랜드마크 (서버 전송용, index 포함)
 */
data class Landmark3D(
    val index: Int,
    val x: Float,
    val y: Float,
    val z: Float
)

/**
 * 검출된 얼굴의 3DMM 랜드마크 결과
 */
data class FaceLandmarks(
    val landmarks: List<Landmark>,
    val faceRect: android.graphics.Rect,  // 원본 얼굴 영역 (좌표 복원용)
    /** 3D 랜드마크 (모델이 3D 출력 시, 이미지 좌표 — 표시용) */
    val landmarks3D: List<Landmark3D>? = null,
    /** 원본 3DMM 정점 (정규화 x,y + 모델 z) — 서버 전송용 */
    val rawLandmarks3D: List<Landmark3D>? = null,
    /** 원본 3DMM 계수(id/exp/pose) — 출력이 [1,K]일 때 id_coeffs, exp_coeffs, pose 분할 */
    val raw3DMM: Face3DMMCoeffs? = null
) {
    /**
     * 원본 이미지 좌표로 변환된 랜드마크 반환
     */
    fun getAbsoluteLandmarks(): List<Landmark> {
        return landmarks.map { lm ->
            Landmark(
                x = faceRect.left + lm.x * faceRect.width(),
                y = faceRect.top + lm.y * faceRect.height()
            )
        }
    }

    /**
     * 서버 전송용: 원본 3DMM 정점 (모델이 정점 출력 시)
     * rawLandmarks3D가 있으면 사용, 없으면 2D+z=0
     */
    fun getLandmarks3DForExport(): List<Landmark3D> {
        if (!rawLandmarks3D.isNullOrEmpty()) return rawLandmarks3D
        if (!landmarks3D.isNullOrEmpty()) return landmarks3D
        return landmarks.mapIndexed { i, lm ->
            Landmark3D(
                index = i,
                x = faceRect.left + lm.x * faceRect.width(),
                y = faceRect.top + lm.y * faceRect.height(),
                z = 0f
            )
        }
    }

    /**
     * 특정 인덱스의 랜드마크 반환 (원본 좌표)
     */
    fun getLandmark(index: Int): Landmark? {
        if (index < 0 || index >= landmarks.size) return null
        val lm = landmarks[index]
        return Landmark(
            x = faceRect.left + lm.x * faceRect.width(),
            y = faceRect.top + lm.y * faceRect.height()
        )
    }
    
    /**
     * 랜드마크 개수
     */
    val count: Int get() = landmarks.size
}
