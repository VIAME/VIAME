import pytest

from .validators import check_csv


CALIBRATION_PARAMS = {
    "measurer:calibration_file": "calibration_matrices.json",
    "calibration_reader:file": "calibration_matrices.json",
}


def run_measurement_viame_pipeline(runner, env_dir, pipe, params: dict = None):
    if params is None:
        params = {}
    params["input:video_reader:type"] = 'image_list'
    params["input:video_filename"] = 'input1_images.txt'
    params["input1:video_filename"] = 'input1_images.txt'
    params["input2:video_filename"] = 'input2_images.txt'
    params["detection_reader:file_name"] = 'detections1.csv'
    params["detection_reader1:file_name"] = 'detections1.csv'
    params["detection_reader2:file_name"] = 'detections2.csv'
    params["track_reader:file_name"] = 'detections1.csv'
    params["track_reader1:file_name"] = 'detections1.csv'
    params["track_reader2:file_name"] = 'detections2.csv'
    params["detector_writer1:file_name"] = 'output/detector_output1.csv'
    params["detector_writer2:file_name"] = 'output/detector_output2.csv'
    params["track_writer1:file_name"] = 'output/track_output1.csv'
    params["track_writer2:file_name"] = 'output/track_output2.csv'

    res = runner.run(pipe, env_dir, overrides=params)
    assert res.returncode == 0


class TestMeasurementCalibrateCamerasDefault:
    def test_measurement_calibrate_cameras_default(self, runner, env_stereo_checkerboards, env_dir):
        run_measurement_viame_pipeline(runner, env_dir,
                                   "pipelines/measurement_calibrate_cameras_default.pipe"
                                   )
        check_csv(env_dir, expected_detections=9*6*8, is_stereo=True)
        assert (env_dir / "calibration_matrices.json").is_file()


class TestMeasurementCalibrateCamerasFast:
    def test_measurement_calibrate_cameras_fast(self, runner, env_stereo_checkerboards, env_dir):
        run_measurement_viame_pipeline(runner, env_dir,
                                       "pipelines/measurement_calibrate_cameras_fast.pipe"
                                       )
        check_csv(env_dir, expected_detections=9*6*8, is_stereo=True)
        assert (env_dir / "calibration_matrices.json").is_file()


class TestMeasurementComputeRectifiedDisparity:
    def test_measurement_compute_rectified_disparity(self, runner, env_stereo_fish, env_dir):
        (env_dir / "output" / "depthMap").mkdir()
        run_measurement_viame_pipeline(runner, env_dir,
                                       "pipelines/measurement_compute_rectified_disparity.pipe",
                                        params=CALIBRATION_PARAMS | {
                                            'depth_map:computer:ocv_stereo_disparity:calibration_file': './',
                                            'output:file_name_template': 'output/depthMap/depth_map%06d.png'
                                        }
                                       )
        depth_maps_dir = env_dir / "output" / "depthMap"
        assert depth_maps_dir.is_dir()
        assert len(list(depth_maps_dir.glob("*.png"))) == 2


class TestMeasurementDetectCalibrationTarget:
    def test_measurement_detect_calibration_target(self, runner, env_stereo_checkerboards, env_dir):
        run_measurement_viame_pipeline(runner, env_dir,
                                       "pipelines/measurement_detect_calibration_target.pipe",
                                       )
        check_csv(env_dir, expected_detections=9*6*8, is_stereo=True)


class TestMeasurementFromAnnotationsDefault:
    def test_measurement_from_annotations_default(self, runner, env_stereo_fish_with_polygons, env_dir):
        run_measurement_viame_pipeline(runner, env_dir,
                                       "pipelines/measurement_from_annotations_default.pipe",
                                       params=CALIBRATION_PARAMS
                                       )
        check_csv(env_dir, expected_detections=2, comparison_detection='min', is_stereo=True)


class TestMeasurementFromAnnotationsTemplate:
    def test_measurement_from_annotations_template(self, runner, env_stereo_fish_with_polygons, env_dir):
        run_measurement_viame_pipeline(runner, env_dir,
                                       "pipelines/measurement_from_annotations_template.pipe",
                                       params=CALIBRATION_PARAMS
                                       )
        check_csv(env_dir, expected_detections=2, comparison_detection='min', is_stereo=True)


class TestMeasurementFullyAutoFishDefault:
    def test_measurement_fully_auto_fish_default(self, runner, env_stereo_fish, env_dir):
        run_measurement_viame_pipeline(runner, env_dir,
                                       "pipelines/measurement_fully_auto_fish_default.pipe",
                                       params=CALIBRATION_PARAMS
                                       )
        check_csv(env_dir, expected_detections=2, comparison_detection='min', is_stereo=True)


class TestMeasurementFullyAutoGmmMotion:
    def test_measurement_fully_auto_gmm_motion(self, runner, env_stereo_fish, env_dir):
        run_measurement_viame_pipeline(runner, env_dir,
                                       "pipelines/measurement_fully_auto_gmm_motion.pipe",
                                       params=CALIBRATION_PARAMS
                                       )
        check_csv(env_dir, expected_detections=0, comparison_detection='min', is_stereo=True)
        # TODO: Check on sequence >= 10 to expect detections


class TestMeasurementFromAnnotationsNccDINO:
    def test_measurement_from_annotations_ncc_dino(self, runner, env_stereo_fish_with_polygons, env_dir):
        run_measurement_viame_pipeline(runner, env_dir,
                                       "pipelines/measurement_from_annotations_ncc_dino.pipe",
                                       params=CALIBRATION_PARAMS
                                       )
        check_csv(env_dir, expected_detections=2, comparison_detection='min', is_stereo=True)