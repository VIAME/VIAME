#!/usr/bin/env python3
"""Tests for `select_default_configs.py`.

    python3 cmake/wheel/test_select_default_configs.py      # no pytest needed
    pytest cmake/wheel/test_select_default_configs.py

Each case is a release defect written down. 0.23.4 was published without
twenty training configs that 0.23.3 had carried -- every tracker trainer,
every RF-DETR and MIT-YOLO one -- and nothing failed, because the selector
reported how many entries it left out and not which.
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import select_default_configs as sel  # noqa: E402


def _select(files):
    """Run the selector over `files`, {name: text}; return (shipped, output)."""
    with tempfile.TemporaryDirectory() as directory:
        prefix = Path(directory)
        root = prefix / "configs" / "pipelines"
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        manifest = prefix / "install_manifest.txt"
        manifest.write_text("\n".join(
            str((root / name).resolve()) for name in files) + "\n")
        output = prefix / "default-configs.txt"

        argv, sys.argv = sys.argv, [
            "select_default_configs.py", "--prefix", str(prefix),
            "--manifest", str(manifest), "--output", str(output)]
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed), \
                    contextlib.redirect_stderr(printed):
                sel.main()
        finally:
            sys.argv = argv

        shipped = {line.split("->")[1].strip().split("configs/pipelines/")[-1]
                   for line in output.read_text().splitlines()
                   if line.startswith("include")}
        return shipped, printed.getvalue()


PLAIN = "process input\n  :: frame_list_input\n"


def test_an_output_file_is_not_a_model():
    """`output_file = trained_model.zip` is what a trainer writes.

    It sits in `common_train_detector.conf`, which every tracker and detector
    trainer includes, and `.zip` reads as weights to the loose pattern. That
    alone took the tracker trainers out of 0.23.4.
    """
    shipped, _ = _select({
        "common_train_detector.conf": "output_file = trained_model.zip\n",
        "detector_plain.pipe":
            "include common_train_detector.conf\n" + PLAIN,
    })
    assert "detector_plain.pipe" in shipped, shipped


def test_a_tracker_trainer_ships():
    shipped, _ = _select({
        "common_train_detector.conf": "output_file = trained_model.zip\n",
        "train_tracker_bytetrack.conf":
            "include common_train_detector.conf\ntracker:type = bytetrack\n",
    })
    assert "train_tracker_bytetrack.conf" in shipped, shipped
    assert "common_train_detector.conf" in shipped, shipped


def test_a_trainer_ships_without_its_seed():
    """A trainer's `models/` path is a seed it can fetch, not a requirement."""
    shipped, _ = _select({
        "train_detector_rf_detr_default.conf":
            "relativepath trainer:rf_detr:seed_model = "
            "models/rf-detr-large-2026.pth\n"
            "trainer:rf_detr:seed_model_url_fallback = True\n",
    })
    assert "train_detector_rf_detr_default.conf" in shipped, shipped


def test_a_trainer_brings_its_template():
    shipped, _ = _select({
        "templates/embedded_default.pipe": PLAIN,
        "train_detector_default.conf":
            "relativepath pipeline_template = templates/embedded_default.pipe\n",
    })
    assert "templates/embedded_default.pipe" in shipped, shipped


def test_a_pipeline_that_runs_a_missing_model_is_still_left_out():
    """The rule the selector exists for, which the fix must not loosen."""
    shipped, _ = _select({
        "detector_plain.pipe": PLAIN,
        "detector_gfit.pipe":
            "relativepath weight = models/gfit_groups_rf_detr_1728.pth\n",
    })
    assert "detector_plain.pipe" in shipped, shipped
    assert "detector_gfit.pipe" not in shipped, shipped


def test_what_is_left_out_is_named():
    """A count is how twenty configs left a release without anyone knowing."""
    _, printed = _select({
        "detector_plain.pipe": PLAIN,
        "detector_gfit.pipe":
            "relativepath weight = models/gfit_groups_rf_detr_1728.pth\n",
    })
    assert "detector_gfit.pipe" in printed, printed
    assert "gfit_groups_rf_detr_1728.pth" in printed, printed


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except (AssertionError, SystemExit, Exception) as e:
            # SystemExit too: the selector exits when it selects nothing,
            # which is what it did for a lone trainer.
            failed += 1
            print(f"  FAIL  {t.__name__}: {str(e)[:120]}")
    print(f"  {len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
