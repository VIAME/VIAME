import pytest

from .validators import check_csv


def run_utility_viame_pipeline(runner, env_dir, pipe, params):
    params["input:video_filename"] = 'image-manifest.txt'
    params["input:video_reader:type"] = 'image_list'
    params["input:video_reader:image_list:image_reader:type"] = 'vxl'
    params["detection_reader:file_name"] = 'groundtruth.csv'
    params["track_reader:file_name"] = 'groundtruth.csv'
    params["detector_writer:file_name"] = 'output/detector_output.csv'
    params["track_writer:file_name"] = 'output/track_output.csv'

    res = runner.run(pipe, env_dir, overrides=params)
    assert res.returncode == 0


class TestAddHeadTailKeypointsFromDets:
    def test_add_head_tail_keypoints_from_dets(self, runner, env_fish_with_polygons):
        run_utility_viame_pipeline(
            runner,
            env_fish_with_polygons,
            "pipelines/utility_add_head_tail_keypoints_from_dets.pipe",
            {},
        )
        check_csv(env_fish_with_polygons, all_types='head-tail')
