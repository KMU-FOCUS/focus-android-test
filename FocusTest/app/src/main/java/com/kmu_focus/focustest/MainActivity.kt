package com.kmu_focus.focustest

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.lifecycle.viewmodel.compose.viewModel
import com.kmu_focus.focustest.processing.DetectorType
import com.kmu_focus.focustest.processing.VideoProcessor
import com.kmu_focus.focustest.processing.YuNetFaceDetector
import com.kmu_focus.focustest.processing.YuNetOpenCVDetector
import com.kmu_focus.focustest.ui.theme.FocusTestTheme
import kotlinx.coroutines.launch
import org.opencv.android.OpenCVLoader

class MainActivity : ComponentActivity() {
    
    companion object {
        init {
            // OpenCV 네이티브 라이브러리 로드 시도
            try {
                System.loadLibrary("opencv_java4")
            } catch (e: UnsatisfiedLinkError) {
                // 라이브러리가 이미 로드되었거나 다른 이름일 수 있음
            }
        }
    }
    
    override fun onResume() {
        super.onResume()
        // OpenCV 초기화 시도
        if (!OpenCVLoader.initDebug()) {
            android.util.Log.e("MainActivity", "OpenCV 초기화 실패")
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        enableEdgeToEdge()
        setContent {
            FocusTestTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen()
                }
            }
        }
    }
}

@Composable
fun MainScreen(viewModel: MainViewModel = viewModel()) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    // StateFlow를 State로 변환
    val videoUri by viewModel.videoUri.collectAsState()
    val isProcessing by viewModel.isProcessing.collectAsState()
    val progress by viewModel.progress.collectAsState()
    val result by viewModel.result.collectAsState()
    
    val videoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { viewModel.setVideoUri(it) }
    }
    
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (!allGranted) {
            Toast.makeText(context, "권한이 필요합니다", Toast.LENGTH_SHORT).show()
        }
    }
    
    LaunchedEffect(Unit) {
        val permissions = mutableListOf<String>().apply {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                add(Manifest.permission.READ_MEDIA_VIDEO)
            } else {
                add(Manifest.permission.READ_EXTERNAL_STORAGE)
                add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            }
        }
        
        val needsPermission = permissions.any {
            ContextCompat.checkSelfPermission(context, it) != PackageManager.PERMISSION_GRANTED
        }
        
        if (needsPermission) {
            permissionLauncher.launch(permissions.toTypedArray())
        }
    }
    
    // 검출기 타입 상태
    var selectedDetector by remember {
        mutableStateOf(VideoProcessor.detectorType)
    }
    
    // 가속기 모드 상태 (YOLO TFLite용)
    var selectedMode by remember { 
        mutableStateOf(YuNetFaceDetector.acceleratorMode) 
    }
    
    // YuNet OpenCV 입력 크기 상태
    var selectedInputSize by remember {
        mutableStateOf(YuNetOpenCVDetector.inputSize)
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "얼굴 검출 처리",
            style = MaterialTheme.typography.headlineMedium
        )
        
        // 검출기 선택 UI
        Card(
            modifier = Modifier.fillMaxWidth(),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "검출기 선택",
                    style = MaterialTheme.typography.titleSmall
                )
                
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    AcceleratorButton(
                        text = "YOLO\nNNAPI",
                        selected = selectedDetector == DetectorType.YOLO_TFLITE,
                        onClick = {
                            selectedDetector = DetectorType.YOLO_TFLITE
                            VideoProcessor.detectorType = selectedDetector
                        },
                        modifier = Modifier.weight(1f)
                    )
                    AcceleratorButton(
                        text = "YuNet\nOpenCV",
                        selected = selectedDetector == DetectorType.YUNET_OPENCV,
                        onClick = {
                            selectedDetector = DetectorType.YUNET_OPENCV
                            VideoProcessor.detectorType = selectedDetector
                        },
                        modifier = Modifier.weight(1f)
                    )
                    AcceleratorButton(
                        text = "YuNet\nONNX",
                        selected = selectedDetector == DetectorType.YUNET_ONNX,
                        onClick = {
                            selectedDetector = DetectorType.YUNET_ONNX
                            VideoProcessor.detectorType = selectedDetector
                        },
                        modifier = Modifier.weight(1f)
                    )
                }
                
                // YuNet OpenCV 선택 시 입력 크기 옵션 표시
                if (selectedDetector == DetectorType.YUNET_OPENCV) {
                    Text(
                        text = "입력 크기 (작을수록 빠름)",
                        style = MaterialTheme.typography.labelMedium,
                        modifier = Modifier.padding(top = 8.dp)
                    )
                    
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        AcceleratorButton(
                            text = "160",
                            selected = selectedInputSize == 160,
                            onClick = { 
                                selectedInputSize = 160
                                YuNetOpenCVDetector.inputSize = 160 
                            },
                            modifier = Modifier.weight(1f)
                        )
                        AcceleratorButton(
                            text = "320",
                            selected = selectedInputSize == 320,
                            onClick = { 
                                selectedInputSize = 320
                                YuNetOpenCVDetector.inputSize = 320 
                            },
                            modifier = Modifier.weight(1f)
                        )
                        AcceleratorButton(
                            text = "480",
                            selected = selectedInputSize == 480,
                            onClick = { 
                                selectedInputSize = 480
                                YuNetOpenCVDetector.inputSize = 480 
                            },
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
                
                // YOLO TFLite 선택 시 가속기 옵션 표시
                if (selectedDetector == DetectorType.YOLO_TFLITE) {
                    Text(
                        text = "가속기 (YOLO용)",
                        style = MaterialTheme.typography.labelMedium,
                        modifier = Modifier.padding(top = 8.dp)
                    )
                    
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        AcceleratorButton(
                            text = "NNAPI",
                            selected = selectedMode == YuNetFaceDetector.Companion.AcceleratorMode.NPU_NNAPI,
                            onClick = {
                                selectedMode = YuNetFaceDetector.Companion.AcceleratorMode.NPU_NNAPI
                                YuNetFaceDetector.acceleratorMode = selectedMode
                            },
                            modifier = Modifier.weight(1f)
                        )
                        AcceleratorButton(
                            text = "GPU",
                            selected = selectedMode == YuNetFaceDetector.Companion.AcceleratorMode.GPU,
                            onClick = {
                                selectedMode = YuNetFaceDetector.Companion.AcceleratorMode.GPU
                                YuNetFaceDetector.acceleratorMode = selectedMode
                            },
                            modifier = Modifier.weight(1f)
                        )
                        AcceleratorButton(
                            text = "CPU",
                            selected = selectedMode == YuNetFaceDetector.Companion.AcceleratorMode.CPU,
                            onClick = {
                                selectedMode = YuNetFaceDetector.Companion.AcceleratorMode.CPU
                                YuNetFaceDetector.acceleratorMode = selectedMode
                            },
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
                
                Text(
                    text = "※ 변경 후 '처리 시작'하면 재초기화됨",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
        
        // 동영상 선택
        Button(
            onClick = { videoPickerLauncher.launch("video/*") },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = if (videoUri != null) "동영상 선택됨" else "동영상 선택"
            )
        }
        
        // 처리 시작
        Button(
            onClick = {
                if (videoUri == null) {
                    Toast.makeText(context, "동영상을 선택해주세요", Toast.LENGTH_SHORT).show()
                    return@Button
                }
                
                scope.launch {
                    viewModel.processVideo(context)
                }
            },
            modifier = Modifier.fillMaxWidth(),
            enabled = !isProcessing
        ) {
            Text(if (isProcessing) "처리 중..." else "처리 시작")
        }
        
        // 진행률 표시
        if (isProcessing) {
            LinearProgressIndicator(
                progress = progress,
                modifier = Modifier.fillMaxWidth()
            )
            Text(
                text = "${(progress * 100).toInt()}%",
                style = MaterialTheme.typography.bodyMedium
            )
        }
        
        // 결과 표시
        result?.let { result ->
            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = if (result.success) "처리 완료" else "처리 실패",
                        style = MaterialTheme.typography.titleLarge
                    )
                    
                    if (result.success) {
                        Text("프레임 수: ${result.totalFrames}")
                        Text("검출된 얼굴 수: ${result.totalFaces}")
                        Text("출력 경로: ${result.outputPath}")
                        
                        Button(
                            onClick = {
                                // 결과 동영상 재생
                                val uri = try {
                                    if (result.outputPath.startsWith("content://")) {
                                        // MediaStore URI
                                        Uri.parse(result.outputPath)
                                    } else {
                                        // 파일 경로
                                        val file = java.io.File(result.outputPath)
                                        if (file.exists()) {
                                            androidx.core.content.FileProvider.getUriForFile(
                                                context,
                                                "${context.packageName}.fileprovider",
                                                file
                                            )
                                        } else {
                                            null
                                        }
                                    }
                                } catch (e: Exception) {
                                    null
                                }
                                
                                uri?.let {
                                    val intent = Intent(Intent.ACTION_VIEW).apply {
                                        setDataAndType(it, "video/*")
                                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                                    }
                                    try {
                                        context.startActivity(intent)
                                    } catch (e: Exception) {
                                        Toast.makeText(context, "동영상 재생 실패: ${e.message}", Toast.LENGTH_SHORT).show()
                                    }
                                } ?: run {
                                    Toast.makeText(context, "동영상 파일을 찾을 수 없습니다", Toast.LENGTH_SHORT).show()
                                }
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("결과 동영상 재생")
                        }
                    } else {
                        Text("오류: ${result.error}", color = MaterialTheme.colorScheme.error)
                    }
                }
            }
        }
    }
}

@Composable
fun AcceleratorButton(
    text: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    if (selected) {
        Button(
            onClick = onClick,
            modifier = modifier,
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.primary
            )
        ) {
            Text(text)
        }
    } else {
        OutlinedButton(
            onClick = onClick,
            modifier = modifier
        ) {
            Text(text)
        }
    }
}
