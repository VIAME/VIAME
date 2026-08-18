# ============ Native Pipelines ============ #

add_pipeline_script_test(
        test_measurement_calibrate_cameras_default
        test_measurement.py
        TestMeasurementCalibrateCamerasDefault
)

add_pipeline_script_test(
        test_measurement_calibrate_cameras_fast
        test_measurement.py
        TestMeasurementCalibrateCamerasFast
)

add_pipeline_script_test(
        test_measurement_compute_rectified_disparity
        test_measurement.py
        TestMeasurementComputeRectifiedDisparity
)

add_pipeline_script_test(
        test_measurement_detect_calibration_target
        test_measurement.py
        TestMeasurementDetectCalibrationTarget
)

add_pipeline_script_test(
        test_measurement_from_annotations_default
        test_measurement.py
        TestMeasurementFromAnnotationsDefault
)

add_pipeline_script_test(
        test_measurement_from_annotations_template
        test_measurement.py
        TestMeasurementFromAnnotationsTemplate
)

add_pipeline_script_test(
        test_measurement_fully_auto_fish_default
        test_measurement.py
        TestMeasurementFullyAutoFishDefault
)

add_pipeline_script_test(
        test_measurement_fully_auto_gmm_motion
        test_measurement.py
        TestMeasurementFullyAutoGmmMotion
)

# ============ ADD-ON DINO Pipelines ============ #

add_pipeline_script_test(
        test_measurement_from_annotations_ncc_dino
        test_measurement.py
        TestMeasurementFromAnnotationsNccDINO
        REQ_ADDON VIAME_DOWNLOAD_MODELS-DINO
)