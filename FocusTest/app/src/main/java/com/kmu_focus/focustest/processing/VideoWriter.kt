package com.kmu_focus.focustest.processing

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.File
import java.nio.ByteBuffer

/**
 * 비디오 작성기
 * 처리된 프레임들을 MP4 파일로 저장
 */
class VideoWriter(
    private val outputPath: String,
    private val width: Int,
    private val height: Int,
    private val fps: Int
) {
    
    private var mediaMuxer: MediaMuxer? = null
    private var encoder: MediaCodec? = null
    private var muxerStarted = false
    private var trackIndex = -1
    
    init {
        initialize()
    }
    
    private fun initialize() {
        val outputFile = File(outputPath)
        outputFile.parentFile?.mkdirs()
        
        mediaMuxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        
        // H.264 인코더 설정
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height)
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
        format.setInteger(MediaFormat.KEY_BIT_RATE, width * height * 3)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, fps)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        
        encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        encoder!!.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        encoder!!.start()
    }
    
    /**
     * 프레임 추가
     */
    fun writeFrame(bitmap: Bitmap) {
        val codec = encoder ?: return
        
        // Bitmap을 YUV420P로 변환
        val yuvData = bitmapToYuv420(bitmap, width, height)
        
        // 인코더에 입력
        val inputBufferIndex = codec.dequeueInputBuffer(10000)
        if (inputBufferIndex >= 0) {
            val inputBuffer = codec.getInputBuffer(inputBufferIndex)
            inputBuffer?.clear()
            inputBuffer?.put(yuvData)
            codec.queueInputBuffer(
                inputBufferIndex,
                0,
                yuvData.size,
                System.nanoTime() / 1000,
                0
            )
        }
        
        // 인코더 출력 처리
        val bufferInfo = MediaCodec.BufferInfo()
        var outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, 10000)
        
        while (outputBufferIndex >= 0) {
            if (outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                val newFormat = codec.outputFormat
                trackIndex = mediaMuxer!!.addTrack(newFormat)
                mediaMuxer!!.start()
                muxerStarted = true
            } else if (outputBufferIndex >= 0) {
                val outputBuffer = codec.getOutputBuffer(outputBufferIndex)
                if (outputBuffer != null && bufferInfo.size > 0 && muxerStarted) {
                    outputBuffer.position(bufferInfo.offset)
                    outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                    mediaMuxer!!.writeSampleData(trackIndex, outputBuffer, bufferInfo)
                }
                codec.releaseOutputBuffer(outputBufferIndex, false)
            }
            outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, 0)
        }
    }
    
    /**
     * Bitmap을 YUV420P로 변환
     */
    private fun bitmapToYuv420(bitmap: Bitmap, width: Int, height: Int): ByteArray {
        val yuvSize = width * height * 3 / 2
        val yuv = ByteArray(yuvSize)
        val pixels = IntArray(width * height)
        
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        
        var yIndex = 0
        var uvIndex = width * height
        
        for (j in 0 until height) {
            for (i in 0 until width) {
                val pixel = pixels[j * width + i]
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                
                // Y (luminance)
                val y = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                yuv[yIndex++] = y.toByte().coerceIn(-128, 127)
                
                // U, V (chrominance) - 2x2 서브샘플링
                if (j % 2 == 0 && i % 2 == 0) {
                    val u = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                    val v = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                    yuv[uvIndex++] = u.toByte().coerceIn(-128, 127)
                    yuv[uvIndex++] = v.toByte().coerceIn(-128, 127)
                }
            }
        }
        
        return yuv
    }
    
    /**
     * 리소스 해제
     */
    fun release() {
        try {
            encoder?.stop()
            encoder?.release()
            encoder = null
            
            if (muxerStarted) {
                mediaMuxer?.stop()
            }
            mediaMuxer?.release()
            mediaMuxer = null
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}
