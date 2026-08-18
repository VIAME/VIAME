import pytest

from .validators import check_csv

def run_tracker_viame_pipeline(runner, env_dir, pipe, params: dict = None):
    if params is None:
        params = {}
    params["input:video_filename"] = 'image-manifest.txt'
    params["input:video_reader:type"] = 'image_list'
    params["input:video_reader:image_list:image_reader:type"] = 'vxl'
    params["detector_writer:file_name"] = 'output/detector_output.csv'
    params["track_writer:file_name"] = 'output/track_output.csv'

    res = runner.run(pipe, env_dir, overrides=params)
    assert res.returncode == 0

class TestTrackerCalibrationTarget:
    def test_tracker_calibration_target(self, runner, env_checkerboard_sequence, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                    "pipelines/tracker_calibration_target.pipe",
                                    )
        check_csv(env_dir, expected_detections=9*6*8)


class TestTrackerGenericProposals:
    def test_tracker_generic_proposals(self, runner, env_fish_sequence, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_generic_proposals.pipe",
                                   )
        check_csv(env_dir, expected_detections=2, comparison_detection='min')


class TestTrackerPenguinAerialMultiEO:
    def test_tracker_penguin_aerial_multi_eo(self, runner, env_seal, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_penguin_aerial_multi_eo.pipe",
                                   )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestTrackerPenguinAerialMultiIR:
    def test_tracker_penguin_aerial_multi_ir(self, runner, env_seal, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_penguin_aerial_multi_ir.pipe",
                                   )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestTrackerPenguinAerialSingleEO:
    def test_tracker_penguin_aerial_single_eo(self, runner, env_seal, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_penguin_aerial_single_eo.pipe",
                                   )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')

class TestTrackerCommunityFish:
    def test_tracker_community_fish(self, runner, env_fish_sequence, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_community_fish.pipe",
                                   )
        check_csv(env_dir, expected_detections=2, comparison_detection='min')


class TestTrackerDefaultFish:
    def test_tracker_default_fish(self, runner, env_fish_sequence, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_default_fish.pipe",
                                   )
        check_csv(env_dir, expected_detections=2, comparison_detection='min')


class TestTrackerFishSFD:
    def test_tracker_fish_sfd(self, runner, env_fish_sequence, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_fish.sfd.pipe",
                                   )
        check_csv(env_dir, expected_detections=2, comparison_detection='min')

class TestTrackerFishSFDCPU:
    def test_tracker_fish_sfd_cpu(self, runner, env_fish_sequence, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_fish.sfd.cpu.pipe",
                                   )
        check_csv(env_dir, expected_detections=2, comparison_detection='min')

class TestTrackerEMTuna:
    def test_tracker_em_tuna(self, runner, env_fish_sequence, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_em_tuna.pipe",
                                   )
        check_csv(env_dir, expected_detections=2, comparison_detection='min')

class TestTrackerGrouperMoonV1:
    def test_tracker_grouper_moon_v1(self, runner, env_fish_sequence, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_grouper_moon_v1.pipe",
                                   )
        check_csv(env_dir, expected_detections=2, comparison_detection='min')

class TestTrackerMotion:
    def test_tracker_motion(self, runner, env_fish_sequence, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_motion.pipe",
                                   )
        check_csv(env_dir, expected_detections=2, comparison_detection='min')

class TestTrackerMoussDeep7VN:
    def test_tracker_mouss_deep7(self, runner, env_fish_sequence, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_mouss_deep7.pipe",
                                   )
        check_csv(env_dir, expected_detections=2, comparison_detection='min')


class TestTrackerSealionSuppressorFusionAllClass:
    def test_tracker_sealion_suppressor_fusion_all_class(self, runner, env_seal, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_sea_lion_suppressor_fusion_all_class.pipe",
                                   )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestTrackerSealionSuppressorFusionTwoClass:
    def test_tracker_sealion_suppressor_fusion_two_class(self, runner, env_seal, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_sea_lion_suppressor_fusion_two_class.pipe",
                                   )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestTrackerSealionSuppressorReclassifierAllClass:
    def test_tracker_sealion_suppressor_reclassifier_all_class(self, runner, env_seal, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_sea_lion_suppressor_reclassifier_all_class.pipe",
                                   )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestTrackerSealionTrackerFusionAllClass:
    def test_tracker_sealion_tracker_fusion_all_class(self, runner, env_seal, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_sea_lion_tracker_fusion_all_class.pipe",
                                   )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestTrackerSealionTrackerFusionTwoClass:
    def test_tracker_sealion_tracker_fusion_two_class(self, runner, env_seal, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_sea_lion_tracker_fusion_two_class.pipe",
                                   )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')


class TestTrackerSealionTrackerReclassifierAllClass:
    def test_tracker_sealion_tracker_reclassifier_all_class(self, runner, env_seal, env_dir):
        run_tracker_viame_pipeline(runner, env_dir,
                                   "pipelines/tracker_sea_lion_tracker_reclassifier_all_class.pipe",
                                   )
        check_csv(env_dir, expected_detections=1, comparison_detection='min')