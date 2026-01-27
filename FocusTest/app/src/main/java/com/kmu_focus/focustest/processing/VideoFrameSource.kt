package com.kmu_focus.focustest.processing

import android.graphics.Bitmap

/**
 * 비디오 프레임 소스 인터페이스
 * 파일/카메라 등 다양한 소스로 확장 가능
 */
interface VideoFrameSource {
    /**
     * 비디오 정보 가져오기
     */
    fun getVideoInfo(): VideoInfo

    /**
     * 다음 프레임 읽기
     * @return 프레임 Bitmap 또는 null (끝)
     */
    fun readFrame(): Bitmap?

    /**
     * 리소스 해제
     */
    fun release()

    /**
     * 현재 프레임 인덱스 (0부터 시작)
     */
    fun getCurrentFrameIndex(): Int
}

data class VideoInfo(
    val width: Int,
    val height: Int,
    val fps: Float,
    val totalFrames: Int
)
