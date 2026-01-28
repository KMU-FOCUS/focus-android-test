package com.kmu_focus.focustest

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.kmu_focus.focustest.processing.ProcessingResult
import com.kmu_focus.focustest.processing.VideoProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream

class MainViewModel : ViewModel() {
    
    private val _videoUri = MutableStateFlow<Uri?>(null)
    val videoUri: StateFlow<Uri?> = _videoUri.asStateFlow()
    
    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()
    
    private val _progress = MutableStateFlow(0f)
    val progress: StateFlow<Float> = _progress.asStateFlow()
    
    private val _result = MutableStateFlow<ProcessingResult?>(null)
    val result: StateFlow<ProcessingResult?> = _result.asStateFlow()
    
    private var videoProcessor: VideoProcessor? = null
    
    fun setVideoUri(uri: Uri) {
        _videoUri.value = uri
        _result.value = null
    }
    
    fun processVideo(context: Context) {
        val videoUri = _videoUri.value ?: return
        
        // 가속기 모드 변경 시 재초기화 필요하므로 항상 새로 생성
        videoProcessor?.release()
        videoProcessor = VideoProcessor(context)
        
        // 임시 파일로 먼저 처리 (앱 내부 저장소)
        val tempFile = File(
            context.cacheDir,
            "temp_output_${System.currentTimeMillis()}.mp4"
        )
        
        _isProcessing.value = true
        _progress.value = 0f
        _result.value = null
        
        viewModelScope.launch {
            try {
                val result = videoProcessor!!.processVideo(
                    videoUri = videoUri,
                    outputPath = tempFile.absolutePath,
                    progressCallback = { progress ->
                        _progress.value = progress
                    }
                )
                
                // 처리 성공 시 Download 폴더로 복사
                if (result.success) {
                    val finalPath = copyToDownloads(context, tempFile)
                    _result.value = result.copy(outputPath = finalPath)
                } else {
                    _result.value = result
                }
            } catch (e: Exception) {
                _result.value = ProcessingResult(
                    success = false,
                    error = e.message ?: "알 수 없는 오류"
                )
            } finally {
                _isProcessing.value = false
            }
        }
    }
    
    /**
     * 임시 파일을 Movies 폴더로 복사 (Android 10 이상에서는 Download 폴더에 비디오 저장 불가)
     */
    private suspend fun copyToDownloads(context: Context, tempFile: File): String = withContext(Dispatchers.IO) {
        val fileName = "focus_output_${System.currentTimeMillis()}.mp4"
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // Android 10 이상: MediaStore API 사용 (Movies 폴더 사용)
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_MOVIES)
            }
            
            val uri = context.contentResolver.insert(
                MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                contentValues
            ) ?: throw IllegalStateException("Movies 폴더에 파일을 생성할 수 없습니다")
            
            context.contentResolver.openOutputStream(uri)?.use { output ->
                FileInputStream(tempFile).use { input ->
                    input.copyTo(output)
                }
            }
            
            // 임시 파일 삭제
            tempFile.delete()
            
            uri.toString()
        } else {
            // Android 9 이하: 직접 파일 시스템 사용 (Download 폴더 사용 가능)
            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            if (!downloadsDir.exists()) {
                downloadsDir.mkdirs()
            }
            
            val finalFile = File(downloadsDir, fileName)
            FileInputStream(tempFile).use { input ->
                finalFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            
            // 임시 파일 삭제
            tempFile.delete()
            
            finalFile.absolutePath
        }
    }
    
    override fun onCleared() {
        super.onCleared()
        videoProcessor?.release()
    }
}
