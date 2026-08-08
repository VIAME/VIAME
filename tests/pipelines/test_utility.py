import pytest

from .validators import check_csv


def run_utility_viame_pipeline(runner, env_dir, pipe, params: dict = None):
    if params is None:
        params = {}
    params["input:video_filename"] = 'image-manifest.txt'
    params["input:video_reader:type"] = 'image_list'
    params["input:video_reader:image_list:image_reader:type"] = 'vxl'
    params["detection_reader:file_name"] = 'groundtruth.csv'
    params["track_reader:file_name"] = 'groundtruth.csv'
    params["detector_writer:file_name"] = 'output/detector_output.csv'
    params["track_writer:file_name"] = 'output/track_output.csv'

    res = runner.run(pipe, env_dir, overrides=params)
    assert res.returncode == 0


class TestUtilityAddHeadTailKeypointsFromDets:
    def test_utility_add_head_tail_keypoints_from_dets(self, runner, env_fish_with_polygons, env_dir):
        run_utility_viame_pipeline(runner, env_dir,
            "pipelines/utility_add_head_tail_keypoints_from_dets.pipe",
        )
        check_csv(env_dir, all_types='head-tail')


class TestUtilityAddHeadTailKeypointsSAM2:
    def test_utility_add_head_tail_keypoints_sam2(self, runner, env_fish_with_detections, env_dir):
        run_utility_viame_pipeline(runner, env_dir,
                                   "pipelines/utility_add_head_tail_keypoints_sam2.pipe",
                                   )
        check_csv(env_dir, all_types='head-tail')


class TestUtilityAddSegmentationSAM2:
    def test_utility_add_segmentation_sam2(self, runner, env_fish_with_detections, env_dir):
        run_utility_viame_pipeline(runner, env_dir,
                                   "pipelines/utility_add_segmentations_sam2.pipe",
                                   )
        check_csv(env_dir, all_types='polygon')


class TestUtilityAddSegmentationSAM3:
    def test_utility_add_segmentation_sam3(self, runner, env_fish_with_detections, env_dir):
        run_utility_viame_pipeline(runner, env_dir,
                                   "pipelines/utility_add_segmentations_sam3.pipe",
                                   )
        check_csv(env_dir, all_types='polygon')


class TestUtilityAddSegmentationWatershed:
    def test_utility_add_segmentation_watershed(self, runner, env_fish_with_detections, env_dir):
        run_utility_viame_pipeline(runner, env_dir,
                                   "pipelines/utility_add_segmentations_watershed.pipe",
                                   )
        check_csv(env_dir, all_types='polygon')

    def test_utility_add_segmentation_watershed_2x(self, runner, env_fish_with_detections, env_dir):
        run_utility_viame_pipeline(runner, env_dir,
                                   "pipelines/utility_add_segmentations_watershed.pipe",
                                   {'detection_refiner:refiner:ocv_watershed:seed_scale_factor': 0.05}
                                   )
        check_csv(env_dir, all_types='polygon')


class TestUtilityEmptyFrameLbls:
    def test_utility_empty_frame_lbls(self, runner, env_fish_sequence, env_dir):
        run_utility_viame_pipeline(runner, env_dir,
                                   "pipelines/utility_empty_frame_lbls_10fr.pipe",
                                   )
        check_csv(env_dir, expected_detections=9)


class TestUtilityMaxPointsPerPoly:
    def test_utility_max_points_per_poly(self, runner, env_fish_with_polygons, env_dir):
        run_utility_viame_pipeline(runner, env_dir,
                                   "pipelines/utility_max_points_per_poly.pipe",
                                   )
        check_csv(env_dir, all_types='polygon')


class TestUtilityRegisterFrames:
    def test_utility_register_frames(self, runner, env_fish_sequence, env_dir):
        run_utility_viame_pipeline(runner, env_dir,
                                   "pipelines/utility_register_frames.pipe",
                                   )
        check_csv(env_dir)


class TestUtilityRemoveDetsInIgnoreRegions:
    def test_utility_remove_dets_in_ignore_regions(self, runner, env_fish_sequence, env_dir):
        run_utility_viame_pipeline(runner, env_dir,
                                   "pipelines/utility_remove_dets_in_ignore_regions.pipe",
                                   )
        check_csv(env_dir)