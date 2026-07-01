import pytest

from .validators import check_generated_video


def run_transcode_viame_pipeline(runner, env_dir, pipe, params: dict = None):
    if params is None:
        params = {}
    params["input:video_filename"] = 'image-manifest.txt'
    params["input:video_reader:type"] = 'image_list'
    params["input:video_reader:image_list:image_reader:type"] = 'vxl'
    params["detection_reader:file_name"] = 'groundtruth.csv'
    params["track_reader:file_name"] = 'groundtruth.csv'
    params["video_writer:video_filename"] = 'output/output.mp4'

    res = runner.run(pipe, env_dir, overrides=params)
    assert res.returncode == 0


class TestTranscodeCompress:
    def test_transcode_compress(self, runner, env_fish, env_dir):
        run_transcode_viame_pipeline(runner, env_dir,
                                  "pipelines/transcode_compress.pipe",
                                  )
        check_generated_video(env_dir, min_size=100)


class TestTranscodeDefault:
    def test_transcode_default(self, runner, env_fish, env_dir):
        run_transcode_viame_pipeline(runner, env_dir,
                                      "pipelines/transcode_default.pipe",
                                      )
        check_generated_video(env_dir, min_size=100)


class TestTranscodeDrawDets:
    def test_transcode_draw_dets(self, runner, env_fish, env_dir):
        run_transcode_viame_pipeline(runner, env_dir,
                                      "pipelines/transcode_draw_dets.pipe",
                                      )
        check_generated_video(env_dir, min_size=100)


class TestTranscodeEnhance:
    def test_transcode_enhance(self, runner, env_fish, env_dir):
        run_transcode_viame_pipeline(runner, env_dir,
                                      "pipelines/transcode_enhance.pipe",
                                      )
        check_generated_video(env_dir, min_size=100)


class TestTranscodeTracksOnly:
    def test_transcode_tracks_only(self, runner, env_fish, env_dir):
        run_transcode_viame_pipeline(runner, env_dir,
                                      "pipelines/transcode_tracks_only.pipe",
                                      )
        check_generated_video(env_dir, min_size=100)
