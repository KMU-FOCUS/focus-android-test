package com.kmu_focus.focustest.processing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.util.Log
import android.view.Surface
import java.io.File

/**
 * MediaCodec 기반 비디오 작성기
 * Surface를 사용하여 Bitmap을 직접 그리는 방식
 */
class OpenCVVideoWriter(
    outputPath: String,
    width: Int,
    height: Int,
    fps: Float
) {
    
    companion object {
        private const val TAG = "OpenCVVideoWriter"
        private const val MIME_TYPE = MediaFormat.MIMETYPE_VIDEO_AVC
        private const val TIMEOUT_US = 10000L
    }
    
    private var encoder: MediaCodec? = null
    private var muxer: MediaMuxer? = null
    private var inputSurface: Surface? = null
    private var trackIndex = -1
    private var muxerStarted = false
    private var frameIndex = 0L
    private var presentationTimeUs = 0L
    private val fps: Int
    
    private val bufferInfo = MediaCodec.BufferInfo()
    
    init {
        val file = File(outputPath)
        file.parentFile?.mkdirs()
        
        this.fps = fps.toInt().coerceAtLeast(1)
        initialize(outputPath, width, height)
    }
    
    private fun initialize(outputPath: String, width: Int, height: Int) {
        Log.d(TAG, "Initializing encoder: ${width}x${height} @ ${fps}fps")
        
        // 비디오 포맷 생성
        val format = MediaFormat.createVideoFormat(MIME_TYPE, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, width * height * 4) // 비트레이트
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        
        // 인코더 생성 및 설정
        encoder = MediaCodec.createEncoderByType(MIME_TYPE).apply {
            configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            inputSurface = createInputSurface()
            start()
        }
        
        // Muxer 생성
        muxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        
        Log.d(TAG, "Encoder initialized successfully")
    }
    
    fun writeFrame(bitmap: Bitmap) {
        val surface = inputSurface ?: return
        val encoder = encoder ?: return
        
        try {
            // Bitmap을 Surface에 그리기
            val canvas = surface.lockHardwareCanvas()
            try {
                // 캔버스 클리어
                canvas.drawColor(android.graphics.Color.BLACK)
                
                // Bitmap을 Surface 크기에 맞게 그리기
                val srcRect = Rect(0, 0, bitmap.width, bitmap.height)
                val dstRect = Rect(0, 0, encoder.inputFormat.getInteger(MediaFormat.KEY_WIDTH), 
                                         encoder.inputFormat.getInteger(MediaFormat.KEY_HEIGHT))
                canvas.drawBitmap(bitmap, srcRect, dstRect, null)
            } finally {
                surface.unlockCanvasAndPost(canvas)
            }
            
            // 타임스탬프 계산
            presentationTimeUs = frameIndex * 1_000_000L / fps
            
            // 인코더 출력 처리
            drainEncoder(false)
            
            frameIndex++
        } catch (e: Exception) {
            Log.e(TAG, "Error writing frame", e)
        }
    }
    
    /**
     * 인코더 출력 버퍼 처리
     */
    private fun drainEncoder(endOfStream: Boolean) {
        val encoder = encoder ?: return
        val muxer = muxer ?: return
        
        while (true) {
            val outputBufferIndex = encoder.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)
            
            when {
                outputBufferIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream) {
                        break  // 아직 출력 없음
                    }
                    // EOS 대기 중이면 계속 시도
                }
                
                outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    if (muxerStarted) {
                        Log.w(TAG, "Format changed after muxer started")
                        continue
                    }
                    
                    val newFormat = encoder.outputFormat
                    Log.d(TAG, "Encoder output format changed: $newFormat")
                    
                    trackIndex = muxer.addTrack(newFormat)
                    muxer.start()
                    muxerStarted = true
                    Log.d(TAG, "Muxer started, track index: $trackIndex")
                }
                
                outputBufferIndex >= 0 -> {
                    val encodedData = encoder.getOutputBuffer(outputBufferIndex) ?: continue
                    
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                        // 코덱 설정 데이터 - 스킵
                        bufferInfo.size = 0
                    }
                    
                    if (bufferInfo.size != 0) {
                        if (!muxerStarted) {
                            Log.w(TAG, "Muxer not started, skipping frame")
                        } else {
                            encodedData.position(bufferInfo.offset)
                            encodedData.limit(bufferInfo.offset + bufferInfo.size)
                            
                            // 실제 프레젠테이션 타임 사용
                            bufferInfo.presentationTimeUs = presentationTimeUs
                            
                            muxer.writeSampleData(trackIndex, encodedData, bufferInfo)
                        }
                    }
                    
                    encoder.releaseOutputBuffer(outputBufferIndex, false)
                    
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        Log.d(TAG, "End of stream reached")
                        return
                    }
                }
            }
        }
    }
    
    fun release() {
        Log.d(TAG, "Finishing encoder, total frames: $frameIndex")
        
        try {
            // EOS 신호 전송
            encoder?.signalEndOfInputStream()
            
            // 남은 출력 처리
            drainEncoder(true)
        } catch (e: Exception) {
            Log.e(TAG, "Error signaling end of stream", e)
        } finally {
            // 리소스 해제
            try {
                encoder?.stop()
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping encoder", e)
            }
            
            try {
                encoder?.release()
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing encoder", e)
            }
            encoder = null
            
            try {
                if (muxerStarted) {
                    muxer?.stop()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping muxer", e)
            }
            
            try {
                muxer?.release()
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing muxer", e)
            }
            muxer = null
            
            inputSurface?.release()
            inputSurface = null
            
            Log.d(TAG, "Encoder finished and released")
        }
    }
}
