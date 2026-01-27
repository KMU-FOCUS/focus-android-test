package com.kmu_focus.focustest.processing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * 프레임 처리기
 * face_detection_yunet.py의 로직을 그대로 이식
 * 얼굴 검출 후 바운딩 박스 그리기 (신뢰도에 따라 색상 변경)
 */
class FrameProcessor(
    private val faceDetector: YuNetFaceDetector
) {
    
    /**
     * 프레임 처리
     * Python: 얼굴 검출 → 바운딩 박스 그리기 → 신뢰도 표시
     * @param frame 원본 프레임
     * @return 처리된 프레임 (바운딩 박스 그려진 프레임)
     */
    fun processFrame(frame: Bitmap): Bitmap {
        // 원본 복사
        val result = frame.copy(Bitmap.Config.ARGB_8888, true)
        
        // 얼굴 검출 (Python: _, faces = detector.detect(small_image))
        val detectedFaces = faceDetector.detectFaces(frame)
        
        // 각 얼굴에 대해 바운딩 박스 그리기
        val canvas = Canvas(result)
        val paint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }
        
        val textPaint = Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 24f
            isAntiAlias = true
        }
        
        for (face in detectedFaces) {
            // Python: x, y, w, h = (face[:4] / scale).astype(int)
            val x = face.x
            val y = face.y
            val width = face.width
            val height = face.height
            val confidence = face.confidence
            
            // Python: 신뢰도에 따라 색상
            // if confidence < 0.6: color = (0, 165, 255)  # 주황 (BGR)
            // elif confidence < 0.8: color = (0, 255, 255)  # 노랑 (BGR)
            // else: color = (0, 255, 0)  # 초록 (BGR)
            val color = when {
                confidence < 0.6f -> android.graphics.Color.rgb(255, 165, 0)  // 주황 (RGB)
                confidence < 0.8f -> android.graphics.Color.rgb(255, 255, 0)  // 노랑 (RGB)
                else -> android.graphics.Color.rgb(0, 255, 0)  // 초록 (RGB)
            }
            
            paint.color = color
            
            // Python: cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            canvas.drawRect(
                Rect(x, y, x + width, y + height),
                paint
            )
            
            // Python: cv2.putText(image, f'{confidence:.2f}', (x, y - 5), ...)
            textPaint.color = color
            val confidenceText = String.format("%.2f", confidence)
            val textY = (y - 10).toFloat().coerceAtLeast(textPaint.textSize)
            canvas.drawText(confidenceText, x.toFloat(), textY, textPaint)
        }
        
        return result
    }
}
