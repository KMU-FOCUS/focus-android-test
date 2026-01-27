package com.kmu_focus.focustest.processing

import android.graphics.Bitmap
import android.graphics.Rect
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfInt
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs

/**
 * 얼굴 매칭 로직
 * 주인공 얼굴과 검출된 얼굴을 비교하여 동일인 여부 판단
 */
class FaceMatcher {
    
    private val referenceFaces = mutableListOf<Mat>()
    
    /**
     * 주인공 얼굴 이미지 추가
     */
    fun addReferenceFace(bitmap: Bitmap) {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        
        // RGBA -> BGR 변환 (명시적 타입 지정)
        val bgrMat = Mat(mat.rows(), mat.cols(), CvType.CV_8UC3)
        if (mat.type() == CvType.CV_8UC4 && mat.channels() == 4) {
            Imgproc.cvtColor(mat, bgrMat, Imgproc.COLOR_RGBA2BGR)
        } else {
            val converted = Mat(mat.rows(), mat.cols(), CvType.CV_8UC3)
            mat.convertTo(converted, CvType.CV_8UC3, 1.0, 0.0)
            Imgproc.cvtColor(converted, bgrMat, Imgproc.COLOR_RGB2BGR)
            converted.release()
        }
        
        // 정규화된 크기로 리사이즈 (매칭 정확도 향상)
        val normalizedMat = Mat(128, 128, CvType.CV_8UC3)
        Imgproc.resize(bgrMat, normalizedMat, Size(128.0, 128.0))
        
        referenceFaces.add(normalizedMat)
        
        mat.release()
        bgrMat.release()
    }
    
    /**
     * 검출된 얼굴이 주인공 얼굴인지 확인
     * @param frame 전체 프레임
     * @param detectedFace 검출된 얼굴 영역
     * @return true면 주인공 얼굴, false면 모자이크 대상
     */
    fun isMainCharacter(frame: Bitmap, detectedFace: DetectedFace): Boolean {
        if (referenceFaces.isEmpty()) {
            // 주인공 얼굴이 없으면 모든 얼굴을 모자이크
            return false
        }
        
        // 얼굴 영역 추출
        val faceRect = detectedFace.toRect()
        val faceBitmap = Bitmap.createBitmap(
            frame,
            faceRect.left.coerceAtLeast(0),
            faceRect.top.coerceAtLeast(0),
            faceRect.width().coerceAtMost(frame.width - faceRect.left),
            faceRect.height().coerceAtMost(frame.height - faceRect.top)
        )
        
        val faceMat = Mat()
        Utils.bitmapToMat(faceBitmap, faceMat)
        
        // RGBA -> BGR 변환 (명시적 타입 지정)
        val bgrMat = Mat(faceMat.rows(), faceMat.cols(), CvType.CV_8UC3)
        if (faceMat.type() == CvType.CV_8UC4 && faceMat.channels() == 4) {
            Imgproc.cvtColor(faceMat, bgrMat, Imgproc.COLOR_RGBA2BGR)
        } else {
            val converted = Mat(faceMat.rows(), faceMat.cols(), CvType.CV_8UC3)
            faceMat.convertTo(converted, CvType.CV_8UC3, 1.0, 0.0)
            Imgproc.cvtColor(converted, bgrMat, Imgproc.COLOR_RGB2BGR)
            converted.release()
        }
        
        val normalizedMat = Mat(128, 128, CvType.CV_8UC3)
        Imgproc.resize(bgrMat, normalizedMat, Size(128.0, 128.0))
        
        // 각 주인공 얼굴과 비교
        var maxSimilarity = 0.0
        for (refFace in referenceFaces) {
            val similarity = calculateSimilarity(refFace, normalizedMat)
            maxSimilarity = maxOf(maxSimilarity, similarity)
        }
        
        faceMat.release()
        bgrMat.release()
        normalizedMat.release()
        faceBitmap.recycle()
        
        // 유사도 임계값 (조정 가능)
        return maxSimilarity > 0.6
    }
    
    /**
     * 두 얼굴 이미지의 유사도 계산 (간단한 히스토그램 기반)
     * 실제 프로덕션에서는 더 정교한 얼굴 인식 모델 사용 권장
     */
    private fun calculateSimilarity(face1: Mat, face2: Mat): Double {
        // 입력 Mat이 CV_8UC3인지 확인
        val mat1 = if (face1.type() == CvType.CV_8UC3 && face1.channels() == 3) {
            face1
        } else {
            val converted = Mat(face1.rows(), face1.cols(), CvType.CV_8UC3)
            face1.convertTo(converted, CvType.CV_8UC3, 1.0, 0.0)
            converted
        }
        
        val mat2 = if (face2.type() == CvType.CV_8UC3 && face2.channels() == 3) {
            face2
        } else {
            val converted = Mat(face2.rows(), face2.cols(), CvType.CV_8UC3)
            face2.convertTo(converted, CvType.CV_8UC3, 1.0, 0.0)
            converted
        }
        
        // 히스토그램 비교 (CV_32F 타입으로 생성)
        val hist1 = Mat()
        val hist2 = Mat()
        
        val histSize = 256
        
        // 채널 인덱스를 MatOfInt로 변환
        val channels = MatOfInt(0, 1, 2)
        val histSizeArray = MatOfInt(histSize, histSize, histSize)
        // MatOfFloat는 가변 인자로 생성 (각 채널의 범위: 0~256)
        val histRange = MatOfFloat(0f, 256f, 0f, 256f, 0f, 256f)
        
        Imgproc.calcHist(
            listOf(mat1),
            channels,
            Mat(),
            hist1,
            histSizeArray,
            histRange
        )
        
        Imgproc.calcHist(
            listOf(mat2),
            channels,
            Mat(),
            hist2,
            histSizeArray,
            histRange
        )
        
        // 리소스 정리
        if (mat1 != face1) mat1.release()
        if (mat2 != face2) mat2.release()
        
        // 히스토그램 상관관계
        val correlation = Imgproc.compareHist(hist1, hist2, Imgproc.HISTCMP_CORREL)
        
        hist1.release()
        hist2.release()
        
        return correlation
    }
    
    fun release() {
        referenceFaces.forEach { it.release() }
        referenceFaces.clear()
    }
}
