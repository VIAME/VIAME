# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Tests for the tools exposed as applets of the viame tool runner."""

import shutil
import subprocess
import sys
import time

from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from viame_env import find_viame_install, get_sourced_env, get_viame_source


# Applets the runner must always know about
CORE_APPLETS = ["csv", "get-configs", "resample-tracks", "score", "train", "runner"]

# Longest a lazily dispatched applet may take to print its own help. Loading
# every plugin costs upwards of ten seconds, so this fails if the applet
# dispatch stops being lazy.
LAZY_DISPATCH_SECONDS = 8.0


@pytest.fixture(scope="module")
def viame_env():
    install = find_viame_install()
    if install is None:
        pytest.skip("No VIAME install found")
    return get_sourced_env(install)


@pytest.fixture(scope="module")
def scoring_data():
    folder = get_viame_source() / "examples" / "scoring_and_evaluation"
    computed = folder / "detections.csv"
    truth = folder / "groundtruth.csv"
    if not computed.exists() or not truth.exists():
        pytest.skip("Missing scoring example data")
    return computed, truth


@pytest.fixture(scope="module")
def detections_csv():
    path = (
        get_viame_source()
        / "examples"
        / "annotation_and_visualization"
        / "example_detections.csv"
    )
    if not path.exists():
        pytest.skip(f"Missing test data: {path}")
    return path


def run_viame(env, *args, timeout=300):
    return subprocess.run(
        ["viame", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestAppletDispatch:
    def test_help_lists_applets(self, viame_env):
        result = run_viame(viame_env, "help")
        assert result.returncode == 0

        for applet in CORE_APPLETS:
            assert f"  {applet} " in result.stdout, f"{applet} missing from help"

    def test_unknown_applet_is_reported(self, viame_env):
        result = run_viame(viame_env, "no-such-applet")
        assert result.returncode != 0
        assert "not found" in result.stderr

    def test_dispatch_does_not_load_every_plugin(self, viame_env):
        start = time.monotonic()
        result = run_viame(viame_env, "csv", "--help")
        elapsed = time.monotonic() - start

        assert result.returncode == 0
        assert elapsed < LAZY_DISPATCH_SECONDS, (
            f"csv --help took {elapsed:.1f}s; applet dispatch is loading "
            f"plugins it does not need"
        )


class TestCsvApplet:
    def test_print_types(self, viame_env, detections_csv, tmp_path):
        work = tmp_path / "types.csv"
        shutil.copy(detections_csv, work)

        result = run_viame(viame_env, "csv", "-i", str(work), "--print-types")

        assert result.returncode == 0
        assert "fish" in result.stdout
        assert "scallop" in result.stdout

    def test_track_count(self, viame_env, detections_csv, tmp_path):
        work = tmp_path / "counts.csv"
        shutil.copy(detections_csv, work)

        result = run_viame(viame_env, "csv", "-i", str(work), "--track-count")

        assert result.returncode == 0
        assert "Track count: 992" in result.stdout

    def test_frame_id_shift_round_trips(self, viame_env, detections_csv, tmp_path):
        work = tmp_path / "shift.csv"
        shutil.copy(detections_csv, work)
        original = work.read_text()

        assert run_viame(viame_env, "csv", "-i", str(work), "--increase-fid").returncode == 0
        assert work.read_text() != original

        assert run_viame(viame_env, "csv", "-i", str(work), "--decrease-fid").returncode == 0
        assert work.read_text() == original

    def test_assign_uid_renumbers_from_one(self, viame_env, detections_csv, tmp_path):
        work = tmp_path / "uid.csv"
        shutil.copy(detections_csv, work)

        assert run_viame(viame_env, "csv", "-i", str(work), "--assign-uid").returncode == 0

        ids = [
            line.split(",")[0]
            for line in work.read_text().splitlines()
            if line and not line.startswith("#")
        ]
        assert ids[:3] == ["1", "2", "3"]

    def test_malformed_number_names_file_and_line(self, viame_env, tmp_path):
        work = tmp_path / "bad.csv"
        work.write_text("# comment\n1,img.png,NAN_HERE,0,0,10,10,0.9,0,fish,0.9\n")

        result = run_viame(viame_env, "csv", "-i", str(work), "--increase-fid")

        assert result.returncode != 0
        assert "bad.csv:2" in result.stderr
        assert "NAN_HERE" in result.stderr

    def test_glob_matches_multiple_wildcards(self, viame_env, detections_csv, tmp_path):
        for name in ["det_a_tracks.csv", "det_b_tracks.csv", "unrelated.csv"]:
            shutil.copy(detections_csv, tmp_path / name)

        result = run_viame(
            viame_env, "csv", "-i", str(tmp_path / "det_*_tracks*.csv"), "--track-count"
        )

        assert result.returncode == 0
        assert result.stdout.count("Processing") == 2


class TestResampleTracksApplet:
    def test_doubling_the_rate_adds_states(self, viame_env, detections_csv, tmp_path):
        output = tmp_path / "resampled.csv"

        result = run_viame(
            viame_env,
            "resample-tracks",
            "-i", str(detections_csv),
            "-o", str(output),
            "--input-rate", "5",
            "--output-rate", "10",
        )

        assert result.returncode == 0
        assert output.exists()

        states = [
            line
            for line in output.read_text().splitlines()
            if line and not line.startswith("#")
        ]
        assert len(states) > 0

    def test_missing_rates_are_rejected(self, viame_env, detections_csv, tmp_path):
        result = run_viame(
            viame_env,
            "resample-tracks",
            "-i", str(detections_csv),
            "-o", str(tmp_path / "out.csv"),
        )

        assert result.returncode != 0


class TestScoreApplet:
    def test_writes_metrics_json(self, viame_env, scoring_data, tmp_path):
        computed, truth = scoring_data
        metrics = tmp_path / "metrics.json"

        result = run_viame(
            viame_env,
            "score",
            "-c", str(computed),
            "-t", str(truth),
            "-o", str(metrics),
        )

        assert result.returncode == 0
        assert metrics.exists()
        assert "precision" in metrics.read_text()


class TestPythonScriptApplets:
    def test_shim_runs_the_script(self, viame_env):
        result = run_viame(viame_env, "process-video", "--help")

        assert result.returncode == 0
        assert "usage: process_video.py" in result.stdout

    def test_help_subcommand_matches_script_help(self, viame_env):
        direct = run_viame(viame_env, "process-video", "--help")
        forwarded = run_viame(viame_env, "help", "process-video")

        assert forwarded.returncode == 0
        assert forwarded.stdout == direct.stdout

    def test_script_exit_code_is_propagated(self, viame_env):
        result = run_viame(viame_env, "process-video", "--not-a-real-flag")

        assert result.returncode != 0


class TestCompatibilityWrappers:
    @pytest.mark.parametrize(
        "wrapper,applet",
        [
            ("viame_score_results", "score"),
            ("viame_get_configs", "get-configs"),
            ("viame_resample_tracks", "resample-tracks"),
        ],
    )
    def test_wrapper_reaches_the_applet(self, viame_env, wrapper, applet):
        install = find_viame_install()
        path = install / "bin" / wrapper
        if not path.exists():
            pytest.skip(f"{wrapper} is not installed")

        result = subprocess.run(
            [str(path), "--help"],
            env=viame_env,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0
        assert f"viame {applet}" in result.stdout
