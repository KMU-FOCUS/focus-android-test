package com.kmu_focus.focustest.processing

import com.kmu_focus.focustest.processing.detector.landmark.model3d.Landmark3D
import org.json.JSONArray
import org.json.JSONObject

/**
 * 서버 전송용 3DMM JSON 스키마 (계수 모드만)
 *
 * frames[].faces[].3dmm.id_coeffs, exp_coeffs, pose
 */
data class VideoInfo(
    val width: Int,
    val height: Int,
    val fps: Float
)

data class LandmarkExport(
    val index: Int,
    val x: Float,
    val y: Float,
    val z: Float
)

data class FaceExport(
    val trackingId: Int,
    val bbox: IntArray,  // [x, y, w, h]
    /** 3DMM 계수(id/exp/pose) — id_coeffs, exp_coeffs, pose 전송 */
    val idCoeffs: FloatArray? = null,
    val expCoeffs: FloatArray? = null,
    val pose: FloatArray? = null
) {
    override fun equals(other: Any?) = (other is FaceExport) &&
        trackingId == other.trackingId &&
        bbox.contentEquals(other.bbox) &&
        (idCoeffs == null && other.idCoeffs == null || idCoeffs != null && other.idCoeffs != null && idCoeffs.contentEquals(other.idCoeffs)) &&
        (expCoeffs == null && other.expCoeffs == null || expCoeffs != null && other.expCoeffs != null && expCoeffs.contentEquals(other.expCoeffs)) &&
        (pose == null && other.pose == null || pose != null && other.pose != null && pose.contentEquals(other.pose))
    override fun hashCode() = 31 * trackingId + bbox.contentHashCode() +
        (idCoeffs?.contentHashCode() ?: 0) + 31 * (expCoeffs?.contentHashCode() ?: 0) + 31 * 31 * (pose?.contentHashCode() ?: 0)
}

data class FrameExport(
    val frameNumber: Int,
    val timestamp: Double,
    val faces: List<FaceExport>
) {
    fun toJsonObject(): JSONObject = JSONObject().apply {
        put("frame_number", frameNumber)
        put("timestamp", timestamp)
        put("faces", JSONArray().apply {
            faces.forEach { face ->
                put(JSONObject().apply {
                    put("tracking_id", face.trackingId)
                    put("bbox", JSONArray().apply { face.bbox.forEach { put(it) } })
                    put("3dmm", JSONObject().apply {
                        put("id_coeffs", JSONArray().apply { (face.idCoeffs ?: floatArrayOf()).forEach { put(it.toDouble()) } })
                        put("exp_coeffs", JSONArray().apply { (face.expCoeffs ?: floatArrayOf()).forEach { put(it.toDouble()) } })
                        put("pose", JSONArray().apply { (face.pose ?: floatArrayOf()).forEach { put(it.toDouble()) } })
                    })
                })
            }
        })
    }
}

data class VideoExportData(
    val videoInfo: VideoInfo,
    val frames: List<FrameExport>
) {
    fun toJsonString(): String = toJsonObject().toString(2)

    fun toJsonObject(): JSONObject {
        val jo = JSONObject()
        jo.put("video_info", JSONObject().apply {
            put("width", videoInfo.width)
            put("height", videoInfo.height)
            put("fps", videoInfo.fps.toDouble())
            put("format", "3dmm")
        })
        jo.put("frames", JSONArray().apply {
            frames.forEach { frame -> put(frame.toJsonObject()) }
        })
        return jo
    }
}

/**
 * 스트리밍 JSON 저장 (OOM 방지: 전체 문자열을 메모리에 올리지 않음)
 */
object VideoExportStreaming {

    fun writeHeader(writer: java.io.Writer, videoInfo: VideoInfo) {
        writer.write("{\"video_info\":{\"width\":${videoInfo.width},\"height\":${videoInfo.height},\"fps\":${videoInfo.fps},\"format\":\"3dmm\"}},\"frames\":[")
    }

    fun writeFrame(writer: java.io.Writer, frame: FrameExport, isFirst: Boolean) {
        if (!isFirst) writer.write(",")
        writer.write(frame.toJsonObject().toString())
    }

    fun writeFooter(writer: java.io.Writer) {
        writer.write("]}")
    }
}

fun Landmark3D.toExport(): LandmarkExport = LandmarkExport(index, x, y, z)
