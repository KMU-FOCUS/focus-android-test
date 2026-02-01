package com.kmu_focus.focustest.processing.detector.recognition

import kotlin.math.sqrt

/**
 * Owner/Other 판별 (recognition_tracking_test.py + simple_tracker.py decide_label)
 *
 * - Track의 임베딩 리스트 평균 → L2 정규화
 * - Master 임베딩과 코사인 유사도(dot product) 비교
 * - 여러 주인공: 각 주인공별 최대 유사도 → 전체 최대 유사도
 * - 유사도 > threshold → OWNER, 아니면 OTHER
 */
class OwnerOtherClassifier(
    /** 주인공별 임베딩 리스트 (각도별). List<List<FloatArray>> */
    private val masterEmbedding: List<List<FloatArray>>,
    /** 본인/타인 구분 임계값 (0.4~0.45 권장) */
    private val similarityThreshold: Float = 0.4f
) {

    /**
     * 임베딩 벡터 L2 정규화
     */
    private fun l2Normalize(v: FloatArray): FloatArray {
        var norm = 0.0
        for (x in v) norm += x * x
        norm = sqrt(norm)
        if (norm < 1e-8) return v
        return FloatArray(v.size) { v[it] / norm.toFloat() }
    }

    /**
     * 두 벡터 코사인 유사도 (L2 정규화된 벡터에 대해 dot = cosine similarity)
     */
    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return dot
    }

    /**
     * Track 임베딩 리스트로 Owner/Other 판별.
     * DECISION_FRAMES개 수집 후 평균 임베딩으로 판별.
     *
     * @param embeddings Track에 쌓인 임베딩 리스트
     * @return Pair(isOwner, similarity) — similarity는 0~1 (코사인 유사도)
     */
    fun decideLabel(embeddings: List<FloatArray>): Pair<Boolean, Float> {
        if (embeddings.isEmpty() || masterEmbedding.isEmpty()) {
            return false to 0f
        }

        // 평균 임베딩 (recognition_tracking_test / simple_tracker decide_label)
        val dim = embeddings.first().size
        val avg = FloatArray(dim)
        for (emb in embeddings) {
            for (i in emb.indices) avg[i] += emb[i]
        }
        for (i in avg.indices) avg[i] /= embeddings.size
        val avgNorm = l2Normalize(avg)

        // 여러 주인공: 각 주인공의 모든 임베딩과 비교 후 최대 유사도 → 전체 최대
        var maxSimilarity = 0f
        for (ownerEmbeddings in masterEmbedding) {
            var ownerMax = 0f
            for (emb in ownerEmbeddings) {
                val sim = cosineSimilarity(l2Normalize(emb), avgNorm)
                if (sim > ownerMax) ownerMax = sim
            }
            if (ownerMax > maxSimilarity) maxSimilarity = ownerMax
        }

        val isOwner = maxSimilarity > similarityThreshold
        return isOwner to maxSimilarity
    }
}
