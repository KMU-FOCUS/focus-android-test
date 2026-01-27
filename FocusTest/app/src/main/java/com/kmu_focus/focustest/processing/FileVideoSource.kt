package com.kmu_focus.focustest.processing

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.videoio.VideoCapture
import java.io.File

/**
 * 파일 기반 비디오 프레임 소스
 * OpenCV VideoCapture 사용 (빠른 순차 읽기)
 */
class FileVideoSource(
    private val context: Context,
    private val videoUri: Uri
) : VideoFrameSource {

    private var videoCapture: VideoCapture? = null
    private var currentFrameIndex = 0
    private var videoInfo: VideoInfo? = null

    init {
        initialize()
    }

    private fun initialize() {
        val filePath = getFilePathFromUri(videoUri)
        videoCapture = VideoCapture(filePath)
        
        if (!videoCapture!!.isOpened) {
            throw IllegalStateException("비디오 파일을 열 수 없습니다: $filePath")
        }

        // MediaMetadataRetriever로 메타데이터만 가져오기 (색상 정보 확인용)
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, videoUri)
            
            val width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toInt() ?: 0
            val height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toInt() ?: 0
            
            // OpenCV로 FPS와 프레임 수 가져오기
            val fps = videoCapture!!.get(org.opencv.videoio.Videoio.CAP_PROP_FPS).toFloat()
            var frameCount = videoCapture!!.get(org.opencv.videoio.Videoio.CAP_PROP_FRAME_COUNT).toInt()
            
            // frameCount가 0 이하이면 duration으로 계산 시도
            if (frameCount <= 0) {
                val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
                if (duration > 0 && fps > 0) {
                    frameCount = ((duration / 1000.0) * fps).toInt()
                }
            }
            
            // 여전히 0 이하면 최소값 1로 설정
            if (frameCount <= 0) {
                frameCount = 1
            }
            
            videoInfo = VideoInfo(
                width = width,
                height = height,
                fps = if (fps > 0) fps else 30f,
                totalFrames = frameCount
            )
        } finally {
            retriever.release()
        }
    }

    private fun getFilePathFromUri(uri: Uri): String {
        return when (uri.scheme) {
            "file" -> uri.path ?: throw IllegalArgumentException("Invalid file URI")
            "content" -> {
                // content:// URI의 경우 임시 파일로 복사
                val cursor = context.contentResolver.query(uri, null, null, null, null)
                cursor?.use {
                    if (it.moveToFirst()) {
                        val index = it.getColumnIndex(android.provider.MediaStore.Video.Media.DATA)
                        if (index >= 0) {
                            return it.getString(index)
                        }
                    }
                }
                // 대체 방법: 임시 파일로 복사
                val inputStream = context.contentResolver.openInputStream(uri)
                val tempFile = File(context.cacheDir, "temp_video_${System.currentTimeMillis()}.mp4")
                inputStream?.use { input ->
                    tempFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                tempFile.absolutePath
            }
            else -> throw IllegalArgumentException("Unsupported URI scheme: ${uri.scheme}")
        }
    }

    override fun getVideoInfo(): VideoInfo {
        return videoInfo ?: throw IllegalStateException("비디오가 초기화되지 않았습니다")
    }

    override fun readFrame(): Bitmap? {
        val capture = videoCapture ?: return null
        
        val frame = Mat()
        val success = capture.read(frame)
        
        if (!success || frame.empty()) {
            frame.release()
            return null
        }

        // VideoCapture로 읽은 Mat은 BGR 타입 (CV_8UC3)
        // BGR -> RGB 변환 (OpenCV Mat은 BGR, Android Bitmap은 RGB)
        val rgbMat = Mat()
        org.opencv.imgproc.Imgproc.cvtColor(frame, rgbMat, org.opencv.imgproc.Imgproc.COLOR_BGR2RGB)
        
        // Mat을 Bitmap으로 변환
        val bitmap = Bitmap.createBitmap(rgbMat.cols(), rgbMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgbMat, bitmap)
        
        // 리소스 정리
        frame.release()
        rgbMat.release()

        currentFrameIndex++
        return bitmap
    }

    override fun release() {
        videoCapture?.release()
        videoCapture = null
    }

    override fun getCurrentFrameIndex(): Int {
        return currentFrameIndex
    }
}
