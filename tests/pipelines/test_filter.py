import pytest

from .validators import check_generated_frames


def run_filter_viame_pipeline(runner, env_dir, pipe, params: dict = None):
    if params is None:
        params = {}
    params["input:video_filename"] = 'image-manifest.txt'
    params["input:video_reader:type"] = 'image_list'
    params["input:video_reader:image_list:image_reader:type"] = 'vxl'
    params["detection_reader:file_name"] = 'groundtruth.csv'
    params["track_reader:file_name"] = 'groundtruth.csv'
    params["kwa_writer:output_directory"] = 'output/'
    params["image_writer:file_name_prefix"] = 'output/'

    res = runner.run(pipe, env_dir, overrides=params)
    assert res.returncode == 0


class TestFilterDebayer:
    def test_filter_debayer(self, runner, env_fish, env_dir):
        run_filter_viame_pipeline(runner, env_dir,
                                   "pipelines/filter_debayer.pipe",
                                   )
        check_generated_frames(env_dir)


class TestFilterDebayerAndEnhance:
    def test_filter_debayer_and_enhance(self, runner, env_fish, env_dir):
        run_filter_viame_pipeline(runner, env_dir, "pipelines/filter_debayer_and_enhance.pipe")
        check_generated_frames(env_dir)


class TestFilterDrawDets:
    def test_filter_draw_dets(self, runner, env_fish_with_detections, env_dir):
        run_filter_viame_pipeline(runner, env_dir, "pipelines/filter_draw_dets.pipe")
        check_generated_frames(env_dir)


class TestFilterEnhance:
    def test_filter_enhance(self, runner, env_fish, env_dir):
        run_filter_viame_pipeline(runner, env_dir, "pipelines/filter_enhance.pipe")
        check_generated_frames(env_dir)


class TestFilterExtractChips:
    def test_filter_extract_chips(self, runner, env_fish_with_detections, env_dir):
        run_filter_viame_pipeline(runner, env_dir, "pipelines/filter_extract_chips.pipe")
        # TODO: check_generated_frames could be different here
        check_generated_frames(env_dir)


class TestFilterNormalize16bit:
    def test_filter_normalize_16bit(self, runner, env_fish, env_dir):
        run_filter_viame_pipeline(runner, env_dir, "pipelines/filter_normalize_16bit.pipe")
        check_generated_frames(env_dir)


class TestFilterSplitAndDebayer:
    def test_filter_split_and_debayer(self, runner, env_fish, env_dir):
        run_filter_viame_pipeline(runner, env_dir, "pipelines/filter_split_and_debayer.pipe")
        check_generated_frames(env_dir)


class TestFilterSplitLeftSide:
    def test_filter_split_left_side(self, runner, env_fish, env_dir):
        run_filter_viame_pipeline(runner, env_dir, "pipelines/filter_split_left_side.pipe")
        check_generated_frames(env_dir)


class TestFilterSplitRightSide:
    def test_filter_split_right_side(self, runner, env_fish, env_dir):
        run_filter_viame_pipeline(runner, env_dir, "pipelines/filter_split_right_side.pipe")
        check_generated_frames(env_dir)


class TestFilterStereoDepthMap:
    def test_filter_stereo_depth_map(self, runner, env_fish, env_dir):
        run_filter_viame_pipeline(runner, env_dir, "pipelines/filter_stereo_depth_map.pipe")
        check_generated_frames(env_dir, match_names=False)


class TestFilterToKWA:
    def test_filter_to_kwa(self, runner, env_fish, env_dir):
        run_filter_viame_pipeline(runner, env_dir, "pipelines/filter_to_kwa.pipe", {
            'kwa_writer:base_filename': 'kwa'
        })
        data_file = env_dir / "output" / "kwa.data"
        index_file = env_dir / "output" / "kwa.index"
        meta_file = env_dir / "output" / "kwa.meta"

        assert data_file.is_file()
        assert index_file.is_file()
        assert meta_file.is_file()

        assert data_file.stat().st_size >= 20_000
        assert index_file.stat().st_size >= 25
        assert meta_file.stat().st_size >= 70


class TestFilterToVideo:
    def test_filter_to_video(self, runner, env_fish, env_dir):
        run_filter_viame_pipeline(runner, env_dir, "pipelines/filter_to_video.pipe", {
            'video_writer:video_filename': 'output/video.mp4'
        })
        video_file = env_dir / "output" / "video.mp4"
        assert video_file.is_file()
        assert video_file.stat().st_size >= 10000

class TestFilterTracksOnly:
    def test_filter_tracks_only(self, runner, env_fish_sequence_with_detections, env_dir):
        run_filter_viame_pipeline(runner, env_dir, "pipelines/filter_tracks_only.pipe")
        check_generated_frames(env_dir, match_names=False, delta=-2)

class TestFilterTracksOnlyAdjustCsv:
    def test_filter_tracks_only_adjust_csv(self, runner, env_fish_sequence_with_detections, env_dir):
        run_filter_viame_pipeline(runner, env_dir, "pipelines/filter_tracks_only_adjust_csv.pipe")
        check_generated_frames(env_dir, match_names=False, delta=-2)
