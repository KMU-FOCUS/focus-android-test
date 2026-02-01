package com.kmu_focus.focustest.processing.detector.tracking

/**
 * 단일 얼굴 추적 객체 (IoU + 3DMM id_coeffs)
 */
class Track(
    private var bbox: FloatArray,
    idCoeffs: FloatArray? = null,
    private val nInit: Int = 1,
    private val maxAge: Int = 30
) {

    var trackId: Int = 0
        private set

    var idCoeffsHistory: MutableList<FloatArray> = mutableListOf()
        private set

    var hits: Int = 1
        private set
    var timeSinceUpdate: Int = 0
        private set

    var state: TrackState = TrackState.TENTATIVE
        private set

    init {
        trackId = nextId++
        idCoeffs?.let { idCoeffsHistory.add(it) }
    }

    fun predict(): FloatArray {
        timeSinceUpdate++
        return bbox
    }

    fun update(detection: FloatArray, idCoeffs: FloatArray? = null) {
        bbox = detection
        idCoeffs?.let {
            idCoeffsHistory.add(it)
            if (idCoeffsHistory.size > 20) idCoeffsHistory.removeAt(0)
        }
        hits++
        timeSinceUpdate = 0
        if (state == TrackState.TENTATIVE && hits >= nInit) {
            state = TrackState.CONFIRMED
        }
    }

    fun markMissed() {
        if (state == TrackState.TENTATIVE) {
            state = TrackState.DELETED
        }
    }

    fun isDeleted(): Boolean =
        state == TrackState.DELETED || timeSinceUpdate > maxAge

    fun getBbox(): FloatArray = bbox

    fun getLatestIdCoeffs(): FloatArray? = idCoeffsHistory.lastOrNull()

    companion object {
        private var nextId = 0
        fun resetIdCounter() { nextId = 0 }
    }
}

enum class TrackState { TENTATIVE, CONFIRMED, DELETED }
