package com.kmu_focus.focustest.processing.detector.recognition

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.util.Log
import com.kmu_focus.focustest.processing.detector.FaceDetector
import com.kmu_focus.focustest.processing.detector.DetectedFace
import java.util.regex.Pattern

/**
 * Owner 얼굴 이미지에서 Master 임베딩 로드 (recognition_tracking_test.py load_multi_owner_images + create_master_embeddings)
 *
 * - assets/owner/ 내 test_face1.png, test_face2.png, ... (숫자 순) 또는 owner1.jpg, owner2.jpg
 * - 각 이미지 = Owner 1명, 첫 번째 검출된 얼굴 사용
 * - 반환: List<List<FloatArray>> — 주인공별 임베딩 리스트 (각도별 확장 시 내부 리스트에 여러 개)
 */
class OwnerEmbeddingStore(
    private val context: Context,
    private val faceDetector: FaceDetector,
    private val embeddingExtractor: ArcFaceEmbeddingExtractor
) {

    companion object {
        private const val TAG = "OwnerEmbeddingStore"
        private const val OWNER_ASSET_DIR = "owner"
        private val TEST_FACE_PATTERN = Pattern.compile("test_face(\\d+)\\.(png|jpg|jpeg|webp)", Pattern.CASE_INSENSITIVE)
        private val OWNER_PATTERN = Pattern.compile("owner(\\d+)\\.(png|jpg|jpeg|webp)", Pattern.CASE_INSENSITIVE)
        private const val MIN_FACE_CONFIDENCE = 0.5f
    }

    /**
     * assets/owner/ 내 이미지 파일명을 숫자 순으로 정렬해 반환 (test_face1, test_face2, ... 또는 owner1, owner2, ...)
     */
    private fun listOwnerImageFiles(): List<String> {
        val files = context.assets.list(OWNER_ASSET_DIR) ?: return emptyList()
        val withNumber = mutableListOf<Pair<Int, String>>()
        for (name in files) {
            var matcher = TEST_FACE_PATTERN.matcher(name)
            if (matcher.matches()) {
                withNumber.add(matcher.group(1)!!.toInt() to name)
                continue
            }
            matcher = OWNER_PATTERN.matcher(name)
            if (matcher.matches()) {
                withNumber.add(matcher.group(1)!!.toInt() to name)
            }
        }
        withNumber.sortBy { it.first }
        return withNumber.map { it.second }
    }

    /**
     * Master 임베딩 생성: 각 Owner 이미지에서 첫 얼굴 crop → 임베딩 추출.
     * 반환: List<List<FloatArray>> — 주인공별로 1개 이상의 임베딩 (초기에는 1개씩)
     */
    fun loadOwnerEmbeddings(): List<List<FloatArray>> {
        val fileNames = listOwnerImageFiles()
        if (fileNames.isEmpty()) {
            Log.w(TAG, "owner 폴더에 test_face*.png 또는 owner*.jpg 없음")
            return emptyList()
        }

        val result = mutableListOf<List<FloatArray>>()
        for (fileName in fileNames) {
            val path = "$OWNER_ASSET_DIR/$fileName"
            val bitmap = try {
                context.assets.open(path).use { BitmapFactory.decodeStream(it) }
            } catch (e: Exception) {
                Log.e(TAG, "이미지 로드 실패: $path — ${e.message}")
                continue
            } ?: continue

            val faces = faceDetector.detectFaces(bitmap).filter { it.confidence >= MIN_FACE_CONFIDENCE }
            if (faces.isEmpty()) {
                Log.w(TAG, "얼굴 미검출: $path")
                bitmap.recycle()
                continue
            }

            val face = faces.first()
            val rect = Rect(
                face.x.coerceIn(0, bitmap.width - 1),
                face.y.coerceIn(0, bitmap.height - 1),
                (face.x + face.width).coerceIn(1, bitmap.width),
                (face.y + face.height).coerceIn(1, bitmap.height)
            )
            if (rect.width() <= 0 || rect.height() <= 0) {
                bitmap.recycle()
                continue
            }

            val crop = Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.width(), rect.height())
            bitmap.recycle()

            val embedding = embeddingExtractor.extractEmbedding(crop)
            crop.recycle()
            if (embedding == null) {
                Log.w(TAG, "임베딩 추출 실패: $path")
                continue
            }
            result.add(listOf(embedding))
            Log.i(TAG, "Owner 임베딩 로드: $fileName (1장)")
        }

        Log.i(TAG, "Master Embedding: ${result.size}명")
        return result
    }
}
