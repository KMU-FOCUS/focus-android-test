package com.kmu_focus.focustest.processing.detector.recognition

import android.util.Log

/**
 * Track별 임베딩 누적 및 Owner/Other 라벨 관리
 *
 * - 새 ID: 앞 5프레임 스킵(잘릴 가능성) → 그다음 3프레임만 수집 후 판별
 * - OTHER: 정면 얼굴일 때 1회만 재검사 (front_face_checked)
 * - OTHER 박스가 사라졌다가 같은 ID로 다시 나타나면 5프레임 스킵 후 3프레임 수집으로 재판별
 */
class TrackLabelState(
    private val classifier: OwnerOtherClassifier,
    /** 스킵할 프레임 수 (얼굴이 잘릴 가능성 높은 초반) */
    private val skipFrames: Int = 5,
    /** 스킵 후 수집할 프레임 수 (이만큼 모이면 판별) */
    private val collectFrames: Int = 3
) {

    companion object {
        private const val TAG = "TrackLabelState"
    }

    private data class Entry(
        val embeddings: MutableList<FloatArray>,
        var isOwner: Boolean? = null,
        var similarity: Float = 0f,
        var frontFaceChecked: Boolean = false,
        var framesSeen: Int = 0,
        /** 직전 프레임에 미노출이었으면 true. 다시 나타나면 PENDING으로 리셋 */
        var wasAbsentLastFrame: Boolean = false
    )

    private val state = mutableMapOf<Int, Entry>()

    /**
     * 이 프레임에 노출된 track ID 집합으로 미노출 track 표시.
     * 다음 프레임에서 해당 track이 다시 나타나면 recordFrameSeen 시 PENDING으로 리셋됨.
     */
    fun beginFrame(seenTrackIds: Set<Int>) {
        for ((id, entry) in state) if (id !in seenTrackIds) entry.wasAbsentLastFrame = true
    }

    /**
     * 이 프레임에서 해당 track을 봤음을 기록 (스킵 카운트용).
     * 직전 프레임에 없었던 track이면 PENDING으로 리셋 후 5프레임 스킵·3프레임 수집으로 재판별.
     */
    fun recordFrameSeen(trackId: Int) {
        val entry = state.getOrPut(trackId) { Entry(mutableListOf()) }
        if (entry.wasAbsentLastFrame) {
            entry.embeddings.clear()
            entry.isOwner = null
            entry.similarity = 0f
            entry.frontFaceChecked = false
            entry.framesSeen = 0
            entry.wasAbsentLastFrame = false
            Log.d(TAG, "[Track $trackId] 재등장 → PENDING (5프레임 스킵 후 3프레임 수집)")
        }
        entry.framesSeen++
    }

    /**
     * 이 프레임에서 ArcFace 임베딩을 뽑을 필요가 있는지.
     * - PENDING: 스킵 구간 지난 뒤, 수집 프레임 수 < collectFrames 이고, 정면일 때만 (측면 프레임은 수집 안 함)
     * - OTHER: 정면 재검사 아직 안 했고, 이번에 정면일 때만 (1회)
     */
    fun needsEmbeddingThisFrame(trackId: Int, isFrontal: Boolean): Boolean {
        val entry = state[trackId] ?: return true
        return when (entry.isOwner) {
            null -> entry.framesSeen > skipFrames && entry.embeddings.size < collectFrames && isFrontal
            false -> !entry.frontFaceChecked && isFrontal
            true -> false
        }
    }

    /**
     * PENDING track에 임베딩 추가. collectFrames개 도달 시 한 번만 판별.
     */
    fun addEmbedding(trackId: Int, embedding: FloatArray) {
        val entry = state.getOrPut(trackId) { Entry(mutableListOf()) }
        if (entry.isOwner != null) return
        entry.embeddings.add(embedding)
        if (entry.embeddings.size >= collectFrames) {
            val (isOwner, sim) = classifier.decideLabel(entry.embeddings)
            entry.isOwner = isOwner
            entry.similarity = sim
            Log.d(TAG, "[Track $trackId] 판별 완료: ${if (isOwner) "OWNER" else "OTHER"} (유사도: $sim)")
        }
    }

    /**
     * OTHER track 정면 재검사 (1회만). 현재 프레임 임베딩으로 Master와 비교.
     * @return OWNER로 바뀌었으면 true
     */
    fun recheckFrontal(trackId: Int, embedding: FloatArray): Boolean {
        val entry = state[trackId] ?: return false
        if (entry.isOwner != false || entry.frontFaceChecked) return false
        entry.frontFaceChecked = true
        val (isOwner, sim) = classifier.decideLabel(listOf(embedding))
        entry.similarity = sim
        if (isOwner) {
            entry.isOwner = true
            Log.d(TAG, "[Track $trackId] 정면 재검사 → OWNER (유사도: $sim)")
            return true
        }
        Log.d(TAG, "[Track $trackId] 정면 재검사 → 여전히 OTHER (유사도: $sim)")
        return false
    }

    fun getLabel(trackId: Int): Boolean? = state[trackId]?.isOwner

    fun getSimilarity(trackId: Int): Float = state[trackId]?.similarity ?: 0f

    fun isPending(trackId: Int): Boolean = state[trackId]?.isOwner == null

    fun getEmbeddingCount(trackId: Int): Int = state[trackId]?.embeddings?.size ?: 0

    fun getFramesSeen(trackId: Int): Int = state[trackId]?.framesSeen ?: 0

    fun getCollectFrames(): Int = collectFrames

    fun getFrontFaceChecked(trackId: Int): Boolean = state[trackId]?.frontFaceChecked ?: false

    fun removeTrack(trackId: Int) {
        state.remove(trackId)
    }

    fun clear() {
        state.clear()
    }
}
