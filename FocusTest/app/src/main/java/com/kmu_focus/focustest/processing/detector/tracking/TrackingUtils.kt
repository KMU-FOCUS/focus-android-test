package com.kmu_focus.focustest.processing.detector.tracking

import kotlin.math.max
import kotlin.math.min

object TrackingUtils {

    fun intBboxToFloat(bbox: IntArray): FloatArray =
        floatArrayOf(bbox[0].toFloat(), bbox[1].toFloat(), bbox[2].toFloat(), bbox[3].toFloat())

    /** IoU 계산 - bbox [x, y, w, h] */
    fun iou(bbox1: IntArray, bbox2: FloatArray): Float {
        val x1 = bbox1[0].toFloat()
        val y1 = bbox1[1].toFloat()
        val w1 = bbox1[2].toFloat()
        val h1 = bbox1[3].toFloat()
        val x2 = bbox2[0]
        val y2 = bbox2[1]
        val w2 = bbox2[2]
        val h2 = bbox2[3]

        val xi1 = max(x1, x2)
        val yi1 = max(y1, y2)
        val xi2 = min(x1 + w1, x2 + w2)
        val yi2 = min(y1 + h1, y2 + h2)

        val interArea = max(0f, xi2 - xi1) * max(0f, yi2 - yi1)
        val unionArea = w1 * h1 + w2 * h2 - interArea

        return if (unionArea > 0) interArea / unionArea else 0f
    }

    fun iou(bbox1: FloatArray, bbox2: FloatArray): Float {
        val xi1 = max(bbox1[0], bbox2[0])
        val yi1 = max(bbox1[1], bbox2[1])
        val xi2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        val yi2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

        val interArea = max(0f, xi2 - xi1) * max(0f, yi2 - yi1)
        val unionArea = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - interArea

        return if (unionArea > 0) interArea / unionArea else 0f
    }

    /** Cosine similarity (L2 정규화 가정) */
    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        var dot = 0f
        var na = 0f
        var nb = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        val denom = kotlin.math.sqrt(na) * kotlin.math.sqrt(nb)
        return if (denom > 1e-8f) dot / denom else 0f
    }

    /** L2 정규화 */
    fun normalizeL2(v: FloatArray): FloatArray {
        var norm = 0f
        for (x in v) norm += x * x
        norm = kotlin.math.sqrt(norm)
        return if (norm > 1e-8f) FloatArray(v.size) { v[it] / norm } else v
    }

    /** Greedy assignment: cost가 낮은 순으로 매칭 */
    fun greedyAssign(
        numDets: Int,
        numTracks: Int,
        costMatrix: Array<FloatArray>,
        maxCost: Float
    ): Triple<List<Pair<Int, Int>>, List<Int>, List<Int>> {
        val pairs = mutableListOf<Triple<Float, Int, Int>>()
        for (i in 0 until numDets) {
            for (j in 0 until numTracks) {
                if (costMatrix[i][j] < maxCost) {
                    pairs.add(Triple(costMatrix[i][j], i, j))
                }
            }
        }
        pairs.sortBy { it.first }

        val detAssigned = BooleanArray(numDets)
        val trackAssigned = BooleanArray(numTracks)
        val matches = mutableListOf<Pair<Int, Int>>()

        for ((_, detIdx, trackIdx) in pairs) {
            if (!detAssigned[detIdx] && !trackAssigned[trackIdx]) {
                detAssigned[detIdx] = true
                trackAssigned[trackIdx] = true
                matches.add(detIdx to trackIdx)
            }
        }

        val unmatchedDets = (0 until numDets).filter { !detAssigned[it] }
        val unmatchedTracks = (0 until numTracks).filter { !trackAssigned[it] }

        return Triple(matches, unmatchedDets, unmatchedTracks)
    }
}
