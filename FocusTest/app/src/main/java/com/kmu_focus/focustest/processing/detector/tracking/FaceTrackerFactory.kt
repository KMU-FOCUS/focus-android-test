package com.kmu_focus.focustest.processing.detector.tracking

fun createFaceTracker(method: TrackingMethod): FaceTracker = IoU3DMMTracker()
