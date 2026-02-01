package com.kmu_focus.focustest.processing.video

import android.graphics.Bitmap
import android.graphics.Rect
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaFormat
import android.media.MediaMuxer
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
    private var isFirstFrame = true  // 첫 프레임 플래그
    
    private val bufferInfo = MediaCodec.BufferInfo()
    
    init {
        val file = File(outputPath)
        file.parentFile?.mkdirs()
        
        this.fps = fps.toInt().coerceAtLeast(1)
        initialize(outputPath, width, height)
    }
    
    private fun initialize(outputPath: String, width: Int, height: Int) {
        // 비디오 포맷 생성
        val format = MediaFormat.createVideoFormat(MIME_TYPE, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, width * height * 4) // 비트레이트
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1) // 1초마다 키프레임
        }
        
        // 하드웨어 인코더 우선 선택
        encoder = try {
            // 하드웨어 코덱 목록에서 H.264 인코더 찾기
            val codecList = MediaCodecList(MediaCodecList.REGULAR_CODECS)
            var hardwareEncoder: MediaCodec? = null
            
            for (codecInfo in codecList.codecInfos) {
                if (!codecInfo.isEncoder) continue
                if (codecInfo.isHardwareAccelerated && codecInfo.supportedTypes.contains(MIME_TYPE)) {
                    try {
                        hardwareEncoder = MediaCodec.createByCodecName(codecInfo.name)
                        android.util.Log.d(TAG, "하드웨어 인코더 사용: ${codecInfo.name}")
                        break
                    } catch (e: Exception) {
                        // 이 코덱 사용 불가, 다음 시도
                        continue
                    }
                }
            }
            
            // 하드웨어 코덱을 찾지 못하면 기본 방식 사용
            hardwareEncoder ?: MediaCodec.createEncoderByType(MIME_TYPE)
        } catch (e: Exception) {
            // 하드웨어 코덱 선택 실패 시 기본 방식
            android.util.Log.w(TAG, "하드웨어 인코더 선택 실패, 기본 인코더 사용: ${e.message}")
            MediaCodec.createEncoderByType(MIME_TYPE)
        }
        
        encoder?.apply {
            configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            inputSurface = createInputSurface()
            start()
        }
        
        // Muxer 생성
        muxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        
        // 인코더 워밍업 (SPS/PPS 헤더 준비까지만)
        warmupEncoder(width, height)
    }
    
    /**
     * 인코더 워밍업 - SPS/PPS 헤더가 준비될 때까지만 대기 (데이터는 muxer에 쓰지 않음)
     */
    private fun warmupEncoder(width: Int, height: Int) {
        val surface = inputSurface ?: return
        val encoder = encoder ?: return
        
        // 검은색 더미 프레임으로 인코더 활성화
        repeat(3) {
            try {
                val canvas = surface.lockHardwareCanvas()
                canvas.drawColor(android.graphics.Color.BLACK)
                surface.unlockCanvasAndPost(canvas)
                // 워밍업 중에는 muxer에 쓰지 않고 인코더 준비만
                drainEncoderForWarmup()
            } catch (e: Exception) {
                // 무시
            }
        }
        
        // 워밍업 완료 후 리셋
        frameIndex = 0
        isFirstFrame = true
        
        android.util.Log.d(TAG, "인코더 워밍업 완료 (muxerStarted: $muxerStarted)")
    }
    
    /**
     * 워밍업용 drainEncoder - muxer 시작만 하고 실제 데이터는 버림
     */
    private fun drainEncoderForWarmup() {
        val encoder = encoder ?: return
        val muxer = muxer ?: return
        
        while (true) {
            val outputBufferIndex = encoder.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)
            
            when {
                outputBufferIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> break
                
                outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    if (!muxerStarted) {
                        val newFormat = encoder.outputFormat
                        trackIndex = muxer.addTrack(newFormat)
                        muxer.start()
                        muxerStarted = true
                        android.util.Log.d(TAG, "Muxer 시작됨 (워밍업 중)")
                    }
                }
                
                outputBufferIndex >= 0 -> {
                    // 워밍업 데이터는 버림 (muxer에 쓰지 않음)
                    encoder.releaseOutputBuffer(outputBufferIndex, false)
                }
            }
        }
    }
    
    fun writeFrame(bitmap: Bitmap) {
        val surface = inputSurface ?: return
        val encoder = encoder ?: return
        
        try {
            // 첫 프레임은 타임스탬프를 0으로 설정
            presentationTimeUs = if (isFirstFrame) {
                0L
            } else {
                frameIndex * 1_000_000L / fps
            }
            
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
            
            // 인코더 출력 처리
            drainEncoder(false)
            
            // 첫 프레임 처리 후 플래그 해제
            if (isFirstFrame) {
                isFirstFrame = false
            }
            
            frameIndex++
        } catch (e: Exception) {
            // 에러 로그 최소화
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
                        continue
                    }
                    
                    val newFormat = encoder.outputFormat
                    trackIndex = muxer.addTrack(newFormat)
                    muxer.start()
                    muxerStarted = true
                }
                
                outputBufferIndex >= 0 -> {
                    val encodedData = encoder.getOutputBuffer(outputBufferIndex) ?: continue
                    
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                        // 코덱 설정 데이터 - 스킵
                        bufferInfo.size = 0
                    }
                    
                    if (bufferInfo.size != 0) {
                        // muxer가 시작되지 않았으면 대기
                        if (!muxerStarted) {
                            encoder.releaseOutputBuffer(outputBufferIndex, false)
                            continue
                        }
                        
                        encodedData.position(bufferInfo.offset)
                        encodedData.limit(bufferInfo.offset + bufferInfo.size)
                        
                        // 첫 프레임은 키프레임이어야 함 - 키프레임이 나올 때까지 대기
                        val isKeyFrame = (bufferInfo.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME) != 0
                        
                        if (isFirstFrame && !isKeyFrame) {
                            // 첫 프레임인데 키프레임이 아니면 스킵하고 다음 버퍼 대기
                            encoder.releaseOutputBuffer(outputBufferIndex, false)
                            continue
                        }
                        
                        // 프레젠테이션 타임 설정 및 쓰기
                        bufferInfo.presentationTimeUs = presentationTimeUs
                        muxer.writeSampleData(trackIndex, encodedData, bufferInfo)
                    }
                    
                    encoder.releaseOutputBuffer(outputBufferIndex, false)
                    
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        return
                    }
                }
            }
        }
    }
    
    fun release() {
        try {
            // EOS 신호 전송
            encoder?.signalEndOfInputStream()
            
            // 남은 출력 처리
            drainEncoder(true)
        } catch (e: Exception) {
            // 릴리즈 중 에러는 무시
        } finally {
            // 리소스 해제
            try {
                encoder?.stop()
            } catch (e: Exception) {
                // 무시
            }
            
            try {
                encoder?.release()
            } catch (e: Exception) {
                // 무시
            }
            encoder = null
            
            try {
                if (muxerStarted) {
                    muxer?.stop()
                }
            } catch (e: Exception) {
                // 무시
            }
            
            try {
                muxer?.release()
            } catch (e: Exception) {
                // 무시
            }
            muxer = null
            
            inputSurface?.release()
            inputSurface = null
        }
    }
}
