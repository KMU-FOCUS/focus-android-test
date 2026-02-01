package com.kmu_focus.focustest.processing

import android.content.Context
import android.net.Uri
import android.os.Environment
import com.kmu_focus.focustest.processing.detector.FaceDetector
import com.kmu_focus.focustest.processing.detector.YuNetOpenCVDetector
import com.kmu_focus.focustest.processing.detector.landmark.model3d.FacialLandmarkDetector
import com.kmu_focus.focustest.processing.detector.recognition.ArcFaceEmbeddingExtractor
import com.kmu_focus.focustest.processing.detector.recognition.OwnerEmbeddingStore
import com.kmu_focus.focustest.processing.detector.recognition.OwnerOtherClassifier
import com.kmu_focus.focustest.processing.detector.recognition.TrackLabelState
import com.kmu_focus.focustest.processing.detector.tracking.TrackingMethod
import com.kmu_focus.focustest.processing.detector.tracking.createFaceTracker
import com.kmu_focus.focustest.processing.video.FileVideoSource
import com.kmu_focus.focustest.processing.video.OpenCVVideoWriter
import org.opencv.android.OpenCVLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * 비디오 처리 파이프라인
 * face_detection_yunet.py의 로직을 그대로 이식
 * VideoFrameSource -> FrameProcessor -> VideoWriter
 */
class VideoProcessor(
    private val context: Context
) {
    
    
    private var faceDetector: FaceDetector? = null
    private var landmarkDetector: FacialLandmarkDetector? = null
    private var frameProcessor: FrameProcessor? = null
    private var embeddingExtractor: ArcFaceEmbeddingExtractor? = null
    
    /**
     * 비디오 처리 실행
     * @param videoUri 입력 비디오 URI
     * @param outputPath 출력 파일 경로
     * @param trackingMethod 얼굴 추적 방식 (고정: IoU+3DMM)
     * @param progressCallback 진행률 콜백 (0.0 ~ 1.0)
     */
    suspend fun processVideo(
        videoUri: Uri,
        outputPath: String,
        trackingMethod: TrackingMethod = TrackingMethod.IoU_3DMM,
        progressCallback: (Float) -> Unit
    ): ProcessingResult = withContext(Dispatchers.Default) {
        
        // OpenCV 초기화 확인
        if (!OpenCVLoader.initDebug()) {
            return@withContext ProcessingResult(
                success = false,
                error = "OpenCV 초기화 실패"
            )
        }
        
        // 초기화 - YuNet OpenCV 고정 (inputSize=480)
        if (faceDetector == null) {
            YuNetOpenCVDetector.inputSize = 480
            faceDetector = YuNetOpenCVDetector(context)
            android.util.Log.i("VideoProcessor", "검출기 초기화: ${faceDetector!!.getDetectorType()}")
        }
        
        // 랜드마크 검출기 초기화
        if (landmarkDetector == null) {
            try {
                landmarkDetector = FacialLandmarkDetector(context)
                android.util.Log.i("VideoProcessor", "랜드마크 검출기 초기화: ${landmarkDetector!!.getLandmarkCount()}개 포인트")
            } catch (e: Exception) {
                android.util.Log.e("VideoProcessor", "랜드마크 검출기 초기화 실패: ${e.message}")
            }
        }
        
        if (frameProcessor == null) {
            val faceTracker = createFaceTracker(trackingMethod)
            var recognitionExtractor: ArcFaceEmbeddingExtractor? = null
            var trackLabelState: TrackLabelState? = null
            try {
                val arcFace = ArcFaceEmbeddingExtractor(context)
                val ownerStore = OwnerEmbeddingStore(context, faceDetector!!, arcFace)
                val masterEmbedding = ownerStore.loadOwnerEmbeddings()
                if (masterEmbedding.isNotEmpty()) {
                    recognitionExtractor = arcFace
                    val classifier = OwnerOtherClassifier(masterEmbedding, 0.4f)
                    trackLabelState = TrackLabelState(
                        classifier,
                        skipFrames = FrameProcessor.SKIP_FRAMES,
                        collectFrames = FrameProcessor.COLLECT_FRAMES
                    )
                    android.util.Log.i("VideoProcessor", "Owner/Other 판별: Master ${masterEmbedding.size}명, 임계값 0.4")
                } else {
                    arcFace.release()
                }
            } catch (e: Exception) {
                android.util.Log.w("VideoProcessor", "Owner/Other 판별 초기화 실패 (무시): ${e.message}")
            }
            embeddingExtractor = recognitionExtractor
            frameProcessor = FrameProcessor(
                faceDetector!!,
                landmarkDetector,
                faceTracker,
                recognitionExtractor,
                trackLabelState
            )
            android.util.Log.i("VideoProcessor", "추적: IoU+3DMM")
        }
        
        val videoSource = FileVideoSource(context, videoUri)
        val videoInfo = videoSource.getVideoInfo()
        
        // 영상 정보 로그
        val durationSec = if (videoInfo.fps > 0) videoInfo.totalFrames / videoInfo.fps else 0f
        android.util.Log.i("VideoProcessor", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        android.util.Log.i("VideoProcessor", "영상 정보:")
        android.util.Log.i("VideoProcessor", "  - 해상도: ${videoInfo.width}x${videoInfo.height}")
        android.util.Log.i("VideoProcessor", "  - FPS: ${videoInfo.fps}")
        android.util.Log.i("VideoProcessor", "  - 총 프레임: ${videoInfo.totalFrames}")
        android.util.Log.i("VideoProcessor", "  - 재생 시간: ${String.format("%.1f", durationSec)}초")
        android.util.Log.i("VideoProcessor", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        // VideoWriter 초기화 (Surface 기반)
        val videoWriter = OpenCVVideoWriter(
            outputPath,
            videoInfo.width,
            videoInfo.height,
            videoInfo.fps
        )
        
        var processedFrames = 0
        var totalFaces = 0
        val totalFrames = if (videoInfo.totalFrames > 0) videoInfo.totalFrames else Int.MAX_VALUE
        val exportFrames = mutableListOf<FrameExport>()
        // JSON 저장: 다운로드 폴더 우선 (휴대폰에서 확인 가능), 실패 시 비디오 옆 → 앱 외부
        val videoFile = java.io.File(outputPath)
        val jsonFilename = videoFile.nameWithoutExtension + "_3dmm.json"
        val jsonDirDownload = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        val jsonDirFallback = videoFile.parentFile?.takeIf { it.exists() }
            ?: context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            ?: context.filesDir
        val jsonFileDownload = java.io.File(jsonDirDownload, jsonFilename).also { it.parentFile?.mkdirs() }
        val jsonFileFallback = java.io.File(jsonDirFallback, jsonFilename).also { it.parentFile?.mkdirs() }
        
        // 성능 측정용
        val processingTimes = mutableListOf<Long>()
        val processingStartTime = System.currentTimeMillis()
        
        try {
            while (true) {
                val frameStart = System.currentTimeMillis()
                
                val frame = videoSource.readFrame()
                if (frame == null) break
                
                val fps = videoInfo.fps.toDouble().coerceAtLeast(1.0)
                val timestamp = processedFrames / fps
                val result = frameProcessor!!.processFrame(frame, processedFrames, timestamp)
                
                result.frameExport?.let { fe ->
                    exportFrames.add(fe)
                    totalFaces += fe.faces.size
                }
                videoWriter.writeFrame(result.bitmap)
                
                processedFrames++
                
                val frameTime = System.currentTimeMillis() - frameStart
                processingTimes.add(frameTime)
                
                if (processedFrames % 30 == 0 && processingTimes.size >= 30) {
                    val avgTime = processingTimes.takeLast(30).average()
                    val currentFps = 1000.0 / avgTime
                    android.util.Log.d("VideoProcessor", 
                        "진행: ${(processedFrames.toFloat() / totalFrames * 100).toInt()}% | " +
                        "FPS: ${currentFps.toInt()} | " +
                        "처리: ${avgTime.toInt()}ms")
                }
                
                val progress = if (totalFrames > 0 && totalFrames != Int.MAX_VALUE) {
                    (processedFrames.toFloat() / totalFrames).coerceIn(0f, 1f)
                } else {
                    0f.coerceAtLeast(0f)
                }
                progressCallback(progress)
                
                frame.recycle()
                result.bitmap.recycle()
            }
            
            videoWriter.release()
            
            // 3DMM JSON 스트리밍 저장: 다운로드 폴더 시도 → 실패 시 폴백
            fun writeJsonTo(file: java.io.File) {
                file.bufferedWriter(Charsets.UTF_8).use { jsonWriter ->
                    VideoExportStreaming.writeHeader(jsonWriter, VideoInfo(videoInfo.width, videoInfo.height, videoInfo.fps))
                    exportFrames.forEachIndexed { i, fe ->
                        VideoExportStreaming.writeFrame(jsonWriter, fe, i == 0)
                    }
                    VideoExportStreaming.writeFooter(jsonWriter)
                }
            }
            val jsonFile = try {
                writeJsonTo(jsonFileDownload)
                jsonFileDownload
            } catch (e: Exception) {
                android.util.Log.w("VideoProcessor", "다운로드 폴더 저장 실패, 대체 경로 사용: ${e.message}")
                writeJsonTo(jsonFileFallback)
                jsonFileFallback
            }
            android.util.Log.i("VideoProcessor", "3DMM JSON 저장: ${jsonFile.absolutePath}")
            
            // 처리 완료 로그
            val totalProcessingTime = (System.currentTimeMillis() - processingStartTime) / 1000.0
            val videoDuration = if (videoInfo.fps > 0) processedFrames / videoInfo.fps else 0f
            val speedRatio = if (totalProcessingTime > 0) videoDuration / totalProcessingTime else 0.0
            
            android.util.Log.i("VideoProcessor", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            android.util.Log.i("VideoProcessor", "처리 완료!")
            android.util.Log.i("VideoProcessor", "  - 처리된 프레임: $processedFrames")
            android.util.Log.i("VideoProcessor", "  - 영상 길이: ${String.format("%.1f", videoDuration)}초")
            android.util.Log.i("VideoProcessor", "  - 처리 시간: ${String.format("%.1f", totalProcessingTime)}초")
            android.util.Log.i("VideoProcessor", "  - 속도 비율: ${String.format("%.2f", speedRatio)}x (1.0 = 실시간)")
            android.util.Log.i("VideoProcessor", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            ProcessingResult(
                success = true,
                outputPath = outputPath,
                exportJsonPath = jsonFile.absolutePath,
                totalFrames = processedFrames,
                totalFaces = totalFaces
            )
        } catch (e: Exception) {
            videoWriter.release()
            ProcessingResult(
                success = false,
                error = e.message ?: "알 수 없는 오류",
                totalFrames = processedFrames,
                totalFaces = totalFaces
            )
        } finally {
            videoSource.release()
        }
    }
    
    fun release() {
        faceDetector?.release()
        faceDetector = null
        landmarkDetector?.release()
        landmarkDetector = null
        frameProcessor?.release()
        frameProcessor = null
        embeddingExtractor?.release()
        embeddingExtractor = null
    }
}

data class ProcessingResult(
    val success: Boolean,
    val outputPath: String = "",
    val exportJsonPath: String = "",
    val error: String = "",
    val totalFrames: Int = 0,
    val totalFaces: Int = 0
)
