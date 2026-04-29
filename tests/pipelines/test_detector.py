import pytest

from .validators import check_csv

def run_detector_viame_pipeline(runner, env_dir, pipe, params: dict = None):
    if params is None:
        params = {}
    params["input:video_filename"] = 'image-manifest.txt'
    params["input:video_reader:type"] = 'image_list'
    params["input:video_reader:image_list:image_reader:type"] = 'vxl'
    params["detector_writer:file_name"] = 'output/detector_output.csv'
    params["track_writer:file_name"] = 'output/track_output.csv'

    res = runner.run(pipe, env_dir, overrides=params)
    assert res.returncode == 0

class TestDetectorSimpleHough:
    def test_detector_simple_hough_no_circles(self, runner, env_single_empty, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
            "pipelines/detector_simple_hough.pipe",
        )
        check_csv(env_dir, expected_detections=0)

    def test_detector_simple_hough_3_circles(self, runner, env_circles_3, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
            "pipelines/detector_simple_hough.pipe",
        )
        check_csv(env_dir, expected_detections=3)


class TestDetectorCalibrationTarget:
    def test_detector_calibration_target_9_6(self, runner, env_checkerboard_9_6, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
            "pipelines/detector_calibration_target.pipe",
            {
                'detector1:detector:ocv_detect_calibration_targets:target_width': 9,
                'detector1:detector:ocv_detect_calibration_targets:target_height': 6,
            },
        )
        check_csv(env_dir, expected_detections=54)


    def test_detector_calibration_target_square(self, runner, env_checkerboard_4_4, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
            "pipelines/detector_calibration_target.pipe",
            {
                'detector1:detector:ocv_detect_calibration_targets:target_width': 4,
                'detector1:detector:ocv_detect_calibration_targets:target_height': 4,
            },
        )
        check_csv(env_dir, expected_detections=16)


class TestDetectorGenericProposals:
    def test_detector_generic_proposals(self, runner, env_fish, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
            "pipelines/detector_generic_proposals.pipe",
        )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorHuggingfaceZeroshot:
    def test_detector_huggingface_zeroshot(self, runner, env_fish, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
            "pipelines/detector_huggingface_zeroshot.pipe",
        )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorArcticSealEOYolo:
    def test_detector_arctic_seal_eo_yolo(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
            "pipelines/detector_arctic_seal_eo_yolo.pipe",
        )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorArcticSealIRYolo:
    def test_detector_arctic_seal_ir_yolo(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
            "pipelines/detector_arctic_seal_ir_yolo.pipe",
        )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorCommunityFish:
    def test_detector_community_fish(self, runner, env_fish, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_community_fish.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorDefaultFish:
    def test_detector_default_fish(self, runner, env_fish, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_default_fish.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')