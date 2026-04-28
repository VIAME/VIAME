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


class TestDetectorPenguinAerialMultiEO:
    def test_detector_penguin_aerial_multi_eo(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_penguin_aerial_multi_eo.pipe",
                                    )
        check_csv(env_dir)


class TestDetectorPenguinAerialMultiIR:
    def test_detector_penguin_aerial_multi_ir(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_penguin_aerial_multi_ir.pipe",
                                    )
        check_csv(env_dir)


class TestDetectorPenguinAerialSingleEO:
    def test_detector_penguin_aerial_single_eo(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_penguin_aerial_single_eo.pipe",
                                    )
        check_csv(env_dir)


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


class TestDetectorEMTuna:
    def test_detector_em_tuna(self, runner, env_fish, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_em_tuna.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorFishFusion:
    def test_detector_fish_fusion(self, runner, env_fish_sequence, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_fish_fusion.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorFishWithMotion:
    def test_detector_fish_with_motion_1080(self, runner, env_fish_sequence, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_fish_with_motion_1080.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorFishWithoutMotion:
    def test_detector_fish_with_motion_640(self, runner, env_fish_sequence, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_fish_with_motion_640.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')

    def test_detector_fish_with_motion_800(self, runner, env_fish_sequence, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_fish_with_motion_800.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')

    def test_detector_fish_with_motion_1920(self, runner, env_fish_sequence, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_fish_with_motion_1920.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorGenericProposals:
    def test_detector_generic_proposals(self, runner, env_fish, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_generic_proposals.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorGrouperMoonV1:
    def test_detector_grouper_moon_v1(self, runner, env_fish, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_grouper_moon_v1.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorMotion:
    def test_detector_motion(self, runner, env_fish_sequence, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_motion.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorMoussDeep7VN:
    def test_detector_grouper_mouss_deep7(self, runner, env_fish, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_mouss_deep7_vn.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorSealionYoloAllClasses:
    def test_detector_sealion_yolo_all_classes(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_sealion_yolo_all_classes.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorSealionBackgroundClassifier:
    def test_detector_sealion_background_classifier(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_sealion_background_classifier.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorSealionCFRNNAllClass:
    def test_detector_sealion_cfrnn_all_class(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_sealion_cfrnn_all_class.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorSealionCFRNNTwoClass:
    def test_detector_sealion_cfrnn_two_class(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_sealion_cfrnn_two_class.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorSealionFusionAllClass:
    def test_detector_sealion_fusion_all_class(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_sealion_fusion_all_class.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorSealionFusionTwoClass:
    def test_detector_sealion_fusion_two_class(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_sealion_fusion_two_class.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorSealionMaskRCNNTwoClass:
    def test_detector_sealion_mask_rcnn_two_class(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_sealion_mask_rcnn_two_class.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorSealionMaskRCNNFiveClass:
    def test_detector_sealion_mask_rcnn_five_class(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_sealion_mask_rcnn_five_class.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorSealionMaskReclassifierFiveClass:
    def test_detector_sealion_reclassifier_five_class(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_sealion_reclassifier_five_class.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorPengcamKW:
    def test_detector_pengcam_kw(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_pengcam_kw.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestDetectorPengcamSWFSC:
    def test_detector_swfsc(self, runner, env_seal, env_dir):
        run_detector_viame_pipeline(runner, env_dir,
                                    "pipelines/detector_swfsc.pipe",
                                    )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')