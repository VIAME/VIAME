# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Tests for the tools exposed as applets of the viame tool runner."""

import json
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
def a_pipeline():
    install = find_viame_install()
    if install is None:
        pytest.skip("No VIAME install found")
    path = install / "configs" / "pipelines" / "filter_enhance.pipe"
    if not path.exists():
        pytest.skip(f"Missing pipeline: {path}")
    return path


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


GEOMETRY_TRUTH = """\
1,f0.png,0,10,10,110,110,1.0,100,fish,1.0,(kp) head 60 10,(kp) tail 60 110,(poly) 60 10 110 60 60 110 10 60
2,f0.png,0,200,200,300,300,1.0,100,fish,1.0,(kp) head 200 250,(kp) tail 300 250,(poly) 200 200 300 200 300 300 200 300
"""

# Track 1 is a near-perfect match. Track 2 shares its box exactly but the
# computed outline covers only half of it, so polygon matching at IoU 0.6
# must reject the pair that box matching accepts.
GEOMETRY_COMPUTED = """\
1,f0.png,0,12,10,112,110,0.9,96,fish,0.9,(kp) head 62 12,(kp) tail 60 108,(poly) 62 10 112 60 62 110 12 60
2,f0.png,0,200,200,300,300,0.8,-1,fish,0.8,(kp) head 205 250,(kp) tail 295 250,(poly) 200 200 300 200 200 300
"""


@pytest.fixture
def geometry_data(tmp_path):
    truth = tmp_path / "truth.csv"
    computed = tmp_path / "computed.csv"
    truth.write_text(GEOMETRY_TRUTH)
    computed.write_text(GEOMETRY_COMPUTED)
    return computed, truth


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
        loaded = json.loads(metrics.read_text())
        assert "precision" in loaded
        assert loaded["config"]["match_mode"] == "box"
        assert "sweep" not in loaded

    def test_sweep_records_every_threshold(self, viame_env, scoring_data, tmp_path):
        computed, truth = scoring_data
        metrics = tmp_path / "metrics.json"
        sweep_dir = tmp_path / "sweep"

        result = run_viame(
            viame_env,
            "score",
            "-c", str(computed),
            "-t", str(truth),
            "-o", str(metrics),
            "--per-class",
            "--sweep-thresholds",
            "--sweep-interval", "5",
            "--output-sweep", str(sweep_dir),
            "--no-print",
        )

        assert result.returncode == 0
        loaded = json.loads(metrics.read_text())
        curves = loaded["sweep"]["curves"]
        assert "overall" in curves
        assert set(loaded["per_class"]) <= set(curves)
        overall = curves["overall"]
        assert overall["thresholds"] == [0.0, 0.2, 0.4, 0.6, 0.8]
        assert len(overall["mota"]) == 5
        assert set(overall["best"]) == {"idf1", "idf1_thresh", "mota", "mota_thresh"}
        assert (sweep_dir / "sweep_curves.csv").exists()
        # the aggregate curve must not leak into the DIVE filter
        assert "overall" not in (sweep_dir / "class_metrics.csv").read_text()

    def test_sweep_leaves_the_headline_matching_in_place(
        self, viame_env, scoring_data, tmp_path
    ):
        computed, truth = scoring_data
        plain = tmp_path / "plain.json"
        swept = tmp_path / "swept.json"

        run_viame(viame_env, "score", "-c", str(computed), "-t", str(truth),
                  "-o", str(plain), "--no-print")
        run_viame(viame_env, "score", "-c", str(computed), "-t", str(truth),
                  "-o", str(swept), "--sweep-thresholds", "--sweep-interval", "4",
                  "--output-sweep", str(tmp_path / "sweep"), "--no-print")

        a = json.loads(plain.read_text())["confusion_matrix"]
        b = json.loads(swept.read_text())["confusion_matrix"]
        assert a == b

    def test_matches_export_accounts_for_every_object(
        self, viame_env, scoring_data, tmp_path
    ):
        computed, truth = scoring_data
        metrics = tmp_path / "metrics.json"
        matches = tmp_path / "matches.json"

        result = run_viame(
            viame_env,
            "score",
            "-c", str(computed),
            "-t", str(truth),
            "-o", str(metrics),
            "--output-matches", str(matches),
            "--no-print",
        )

        assert result.returncode == 0
        loaded = json.loads(metrics.read_text())
        rows = json.loads(matches.read_text())["rows"]
        counts = {}
        for row in rows:
            counts[row[2]] = counts.get(row[2], 0) + 1
        assert counts["tp"] == loaded["true_positives"]
        assert counts["fp"] == loaded["false_positives"]
        assert counts["fn"] == loaded["false_negatives"]

    def test_polygon_matching_rejects_a_poor_outline(
        self, viame_env, geometry_data, tmp_path
    ):
        computed, truth = geometry_data

        def score(mode):
            out = tmp_path / f"{mode}.json"
            result = run_viame(
                viame_env, "score", "-c", str(computed), "-t", str(truth),
                "--iou", "0.6", "--match-mode", mode, "-o", str(out), "--no-print",
            )
            assert result.returncode == 0
            return json.loads(out.read_text())

        box = score("box")
        polygon = score("polygon")

        assert box["true_positives"] == 2
        assert polygon["true_positives"] == 1
        assert polygon["false_positives"] == 1
        assert polygon["false_negatives"] == 1
        assert box["polygon_pairs"] == 2
        assert box["mean_polygon_iou"] < box["mean_iou"]

    def test_keypoint_and_length_metrics(self, viame_env, geometry_data, tmp_path):
        computed, truth = geometry_data
        out = tmp_path / "metrics.json"

        result = run_viame(
            viame_env, "score", "-c", str(computed), "-t", str(truth),
            "-o", str(out), "--no-print",
        )

        assert result.returncode == 0
        loaded = json.loads(out.read_text())
        assert loaded["keypoint_pairs"] == 2
        assert loaded["keypoint_pck"] == 1.0
        assert loaded["head_mean_error"] == pytest.approx((8 ** 0.5 + 5) / 2, abs=1e-4)
        assert loaded["length_pairs"] == 2
        # 96 vs 100 from the length column, 90 vs 100 from the keypoints
        assert loaded["length_mae"] == pytest.approx(7.0, abs=1e-4)
        assert loaded["length_bias"] == pytest.approx(-7.0, abs=1e-4)

    def test_rejects_an_unknown_match_mode(self, viame_env, geometry_data, tmp_path):
        computed, truth = geometry_data
        result = run_viame(
            viame_env, "score", "-c", str(computed), "-t", str(truth),
            "--match-mode", "mask",
        )
        assert result.returncode != 0


class TestRunDispatch:
    """`run` covers batch processing and executing a single pipeline file."""

    @staticmethod
    def _mode(result):
        if "usage: process_video.py" in result.stdout:
            return "batch"
        if "pipe-file" in result.stdout:
            return "pipeline"
        return "unknown"

    def test_bare_pipe_file_selects_the_pipeline_runner(self, viame_env, a_pipeline):
        result = run_viame(viame_env, "run", str(a_pipeline), "--help")
        assert self._mode(result) == "pipeline"

    def test_pipe_file_after_a_setting_is_still_positional(self, viame_env, a_pipeline):
        result = run_viame(viame_env, "run", "-s", "k=v", str(a_pipeline), "--help")
        assert self._mode(result) == "pipeline"

    def test_flags_only_selects_the_batch_driver(self, viame_env, tmp_path):
        result = run_viame(viame_env, "run", "-d", str(tmp_path), "--help")
        assert self._mode(result) == "batch"

    def test_pipeline_given_as_a_flag_value_selects_the_batch_driver(
        self, viame_env, a_pipeline
    ):
        result = run_viame(viame_env, "run", "-p", str(a_pipeline), "--help")
        assert self._mode(result) == "batch"

    def test_setting_holding_a_pipe_path_selects_the_batch_driver(
        self, viame_env, a_pipeline
    ):
        result = run_viame(viame_env, "run", "-s", f"x={a_pipeline}", "--help")
        assert self._mode(result) == "batch"

    def test_help_describes_both_modes(self, viame_env):
        result = run_viame(viame_env, "run", "--help")

        assert result.returncode == 0
        assert "two modes" in result.stdout
        assert "viame help runner" in result.stdout

    def test_pipeline_mode_matches_the_runner_applet(self, viame_env, a_pipeline):
        through_run = run_viame(viame_env, "run", str(a_pipeline), "--help")
        through_runner = run_viame(viame_env, "runner", str(a_pipeline), "--help")

        assert through_run.stdout == through_runner.stdout


class TestPythonScriptApplets:
    def test_shim_runs_the_script(self, viame_env):
        result = run_viame(viame_env, "run", "--help")

        assert result.returncode == 0
        assert "usage: process_video.py" in result.stdout

    def test_help_subcommand_matches_script_help(self, viame_env):
        direct = run_viame(viame_env, "run", "--help")
        forwarded = run_viame(viame_env, "help", "run")

        assert forwarded.returncode == 0
        assert forwarded.stdout == direct.stdout

    def test_script_exit_code_is_propagated(self, viame_env):
        result = run_viame(viame_env, "run", "--not-a-real-flag")

        assert result.returncode != 0
