package com.kmu_focus.focustest.processing

import android.content.Context
import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.FaceDetectorYN
import java.io.File
import java.io.FileOutputStream

/**
 * YuNet 얼굴 검출기
 * Python face_detection_yunet.py의 로직을 그대로 이식
 */
class YuNetFaceDetector(context: Context) {
    
    private var detector: FaceDetectorYN? = null
    private val scale = 0.7f  // Python: scale = 0.7
    
    init {
        initializeDetector(context)
    }
    
    private fun initializeDetector(context: Context) {
        try {
            // assets에서 모델 파일 복사
            val modelFile = File(context.filesDir, "yunet_face.onnx")
            if (!modelFile.exists()) {
                context.assets.open("yunet_face.onnx").use { input ->
                    FileOutputStream(modelFile).use { output ->
                        input.copyTo(output)
                    }
                }
            }
            
            // YuNet 초기화 (Python face_detection_yunet.py와 동일한 파라미터)
            // Python: cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.5, nms_threshold=0.3, top_k=5000)
            detector = FaceDetectorYN.create(
                modelFile.absolutePath,  // MODEL_PATH
                "",                      // config path (empty for ONNX)
                Size(320.0, 320.0),      // input size (320, 320)
                0.5f,                    // score_threshold=0.5
                0.3f,                    // nms_threshold=0.3
                5000                      // top_k=5000
            )
            
            if (detector == null) {
                throw IllegalStateException("YuNet 검출기 초기화 실패")
            }
        } catch (e: Exception) {
            throw IllegalStateException("YuNet 초기화 중 오류: ${e.message}", e)
        }
    }
    
    /**
     * 프레임에서 얼굴 검출
     * @param frame 원본 프레임 (Bitmap)
     * @return 검출된 얼굴 리스트 (x, y, width, height, confidence 포함)
     */
    fun detectFaces(frame: Bitmap): List<DetectedFace> {
        val detector = this.detector ?: return emptyList()
        
        try {
            // Bitmap을 Mat으로 변환 (Utils.bitmapToMat은 RGBA Mat 생성)
            val originalMat = Mat()
            Utils.bitmapToMat(frame, originalMat)
            
            // RGBA -> BGR 변환 (Python: image는 BGR)
            val bgrMat = Mat(originalMat.rows(), originalMat.cols(), CvType.CV_8UC3)
            if (originalMat.type() == CvType.CV_8UC4 && originalMat.channels() == 4) {
                Imgproc.cvtColor(originalMat, bgrMat, Imgproc.COLOR_RGBA2BGR)
            } else if (originalMat.type() == CvType.CV_8UC3 && originalMat.channels() == 3) {
                Imgproc.cvtColor(originalMat, bgrMat, Imgproc.COLOR_RGB2BGR)
            } else {
                originalMat.convertTo(bgrMat, CvType.CV_8UC3)
            }
            originalMat.release()
            
            // 다운스케일 (Python: small_image = cv2.resize(image, None, fx=scale, fy=scale))
            val smallWidth = (bgrMat.cols() * scale).toInt()
            val smallHeight = (bgrMat.rows() * scale).toInt()
            val smallMat = Mat(smallHeight, smallWidth, CvType.CV_8UC3)
            Imgproc.resize(bgrMat, smallMat, Size(smallWidth.toDouble(), smallHeight.toDouble()))
            
            // 검출기 입력 크기 설정 (Python: detector.setInputSize((small_image.shape[1], small_image.shape[0])))
            detector.setInputSize(Size(smallMat.cols().toDouble(), smallMat.rows().toDouble()))
            
            // 얼굴 검출 (Python: _, faces = detector.detect(small_image))
            val facesMat = Mat()
            detector.detect(smallMat, facesMat)
            
            val faces = mutableListOf<DetectedFace>()
            
            if (facesMat.rows() > 0 && facesMat.cols() >= 15) {
                // facesMat은 [N x 15] 형태
                // Python: for face in faces: x, y, w, h = (face[:4] / scale).astype(int), confidence = face[14]
                for (i in 0 until facesMat.rows()) {
                    val row = facesMat.row(i)
                    
                    // Float 타입으로 데이터 읽기 (CV_32F)
                    val data = FloatArray(15)
                    row.get(0, 0, data)
                    
                    // Python: x, y, w, h = (face[:4] / scale).astype(int)
                    // 원본 크기로 좌표 복원
                    val x = (data[0] / scale).toInt()
                    val y = (data[1] / scale).toInt()
                    val width = (data[2] / scale).toInt()
                    val height = (data[3] / scale).toInt()
                    // Python: confidence = face[14]
                    val confidence = data[14]
                    
                    // 원본 프레임 크기(Bitmap 크기) 내로 클리핑
                    val clippedX = x.coerceIn(0, frame.width - 1)
                    val clippedY = y.coerceIn(0, frame.height - 1)
                    val clippedWidth = width.coerceIn(1, frame.width - clippedX)
                    val clippedHeight = height.coerceIn(1, frame.height - clippedY)
                    
                    faces.add(
                        DetectedFace(
                            x = clippedX,
                            y = clippedY,
                            width = clippedWidth,
                            height = clippedHeight,
                            confidence = confidence
                        )
                    )
                }
            }
            
            // 리소스 정리
            bgrMat.release()
            smallMat.release()
            facesMat.release()
            
            return faces
        } catch (e: Exception) {
            android.util.Log.e("YuNetFaceDetector", "얼굴 검출 중 오류: ${e.message}", e)
            return emptyList()
        }
    }
    
    fun release() {
        detector = null
    }
}

data class DetectedFace(
    val x: Int,
    val y: Int,
    val width: Int,
    val height: Int,
    val confidence: Float
) {
    fun toRect(): android.graphics.Rect {
        return android.graphics.Rect(x, y, x + width, y + height)
    }
}
