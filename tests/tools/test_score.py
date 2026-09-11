# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""`viame score --output-plots` writes the pictures it always wrote.

The drawing moved from `plugins/opencv/plot_metrics` to matplotlib in
`tools/plot.py` under P7-T07. The pictures are a different picture on
purpose, so what is held here is the set of them: every file name the
OpenCV renderer produced still has to appear, drawn from the CSVs that
`tests/plugins/core/test_plot_data_export.cxx` records.
"""
import sys

from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from viame_env import find_viame_install, get_sourced_env

from .test_viame_applets import run_viame


# Every image `metrics_plotter::render_all_plots` used to save
OPENCV_PLOTS = [
    "pr_curve_overall.png",
    "pr_curves_per_class.png",
    "roc_curve_overall.png",
    "confusion_matrix.png",
    "iou_histogram.png",
    "track_purity_histogram.png",
    "track_continuity_histogram.png",
    "track_length_histogram.png",
]

# The CSVs beside them, which are the data the images are drawn from
PLOT_DATA = [
    "pr_curve_overall.csv",
    "confusion_matrix.csv",
    "roc_curve_overall.csv",
    "histograms.csv",
]


TRUTH = [
    (1, 0, 10, 10, 30, 30, 1.0, "fish"),
    (1, 1, 12, 10, 32, 30, 1.0, "fish"),
    (1, 2, 14, 10, 34, 30, 1.0, "fish"),
    (1, 3, 16, 10, 36, 30, 1.0, "fish"),
    (2, 0, 100, 100, 120, 120, 1.0, "scallop"),
    (2, 1, 102, 100, 122, 120, 1.0, "scallop"),
    (2, 2, 104, 100, 124, 120, 1.0, "scallop"),
    (2, 3, 106, 100, 126, 120, 1.0, "scallop"),
    (2, 4, 108, 100, 128, 120, 1.0, "scallop"),
    (2, 5, 110, 100, 130, 120, 1.0, "scallop"),
    (3, 4, 200, 200, 220, 220, 1.0, "scallop"),
]

COMPUTED = [
    (1, 0, 10, 10, 30, 30, 0.95, "fish"),
    (1, 1, 13, 10, 33, 30, 0.90, "fish"),
    (1, 2, 16, 10, 36, 30, 0.85, "fish"),
    (1, 3, 20, 10, 40, 30, 0.55, "fish"),
    (2, 0, 100, 100, 120, 120, 0.80, "scallop"),
    (2, 1, 103, 100, 123, 120, 0.75, "scallop"),
    (2, 4, 109, 100, 129, 120, 0.70, "scallop"),
    (2, 5, 112, 100, 132, 120, 0.65, "scallop"),
    (3, 4, 200, 200, 220, 220, 0.60, "fish"),
    (4, 2, 300, 300, 320, 320, 0.50, "fish"),
    (4, 3, 302, 300, 322, 320, 0.45, "scallop"),
]

HEADER = (
    "# 1: Detection or Track Id, 2: Video or Image String, 3: Frame Number, "
    "4-7: Bounding Box, 8: Confidence, 9: Length, 10+: Class name / score pairs\n"
)


def write_csv(path, rows):
    with open(path, "w") as stream:
        stream.write(HEADER)
        for track, frame, x1, y1, x2, y2, score, label in rows:
            stream.write(
                f"{track},frame_{frame}.png,{frame},{x1},{y1},{x2},{y2},"
                f"{score},0,{label},{score}\n"
            )
    return str(path)


@pytest.fixture
def viame_env():
    install = find_viame_install()
    if install is None:
        pytest.skip("No VIAME install found")
    return get_sourced_env(install)


@pytest.fixture
def scored(viame_env, tmp_path):
    truth = write_csv(tmp_path / "truth.csv", TRUTH)
    computed = write_csv(tmp_path / "computed.csv", COMPUTED)

    plots = tmp_path / "plots"
    plots.mkdir()

    result = run_viame(
        viame_env, "score", "-c", computed, "-t", truth,
        "--output-plots", str(plots) )

    assert result.returncode == 0, result.stderr

    return plots, result


def test_plot_data_is_written(scored):
    plots, _ = scored

    for name in PLOT_DATA:
        assert (plots / name).is_file(), f"{name} missing from the plot directory"


def test_every_plot_the_opencv_renderer_drew_is_still_drawn(scored):
    plots, result = scored

    matplotlib_missing = "matplotlib is required" in result.stdout + result.stderr
    if matplotlib_missing:
        pytest.skip("matplotlib is not installed in this python")

    for name in OPENCV_PLOTS:
        image = plots / name
        assert image.is_file(), f"{name} was not rendered"

        # A png header and something after it, rather than a stub
        assert image.stat().st_size > 1024, f"{name} is suspiciously small"
        assert image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{name} is not a png"


def test_the_plots_can_be_redrawn_from_the_directory(viame_env, scored, tmp_path):
    """The data on disk is enough on its own.

    Scoring renders through `viame plot eval` over the directory it just
    wrote, so a user can rerun that by hand and get the same pictures. That
    only holds if the CSVs carry everything the renderer needs.
    """
    plots, result = scored

    if "matplotlib is required" in result.stdout + result.stderr:
        pytest.skip("matplotlib is not installed in this python")

    redrawn = tmp_path / "redrawn"

    again = run_viame(
        viame_env, "plot", "eval", "-i", str(plots), "-o", str(redrawn) )

    assert again.returncode == 0, again.stderr

    for name in OPENCV_PLOTS:
        assert (redrawn / name).is_file(), f"{name} could not be redrawn"
