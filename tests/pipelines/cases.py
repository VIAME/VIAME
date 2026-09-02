"""Test cases for the pipelines DIVE exposes.

The pipeline list is discovered from the install tree with the same rules as
dive_tasks.pipeline_discovery, so a renamed or newly added pipeline needs no
edit here. RULES attach a fixture and assertions to a whole family at once;
OVERRIDES replace them for individual pipelines, and a Case may carry its own
setup/check callables when a pipeline needs a routine of its own. Anything
discovered that matches neither is reported as an uncovered skip.
"""

import re

import pytest

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from viame_env import find_viame_install

from .validators import (check_csv, check_generated_chips, check_generated_frames,
                         check_generated_video)

# Mirrors dive_tasks.pipeline_discovery, minus its hough exclusion: hough is
# the one detector with a deterministic, model-free answer.
DISALLOWED = re.compile(
    r".*local.*|.*seagis.*|.*_svm_models\.pipe|detector_extract_chips\.pipe|"
    r"tracker_stabilized_iou\.pipe|tracker_short_term\.pipe"
)
# Multi-camera pipelines are their own DIVE category and have no fixture yet.
MULTICAM = re.compile(r"_[23][-_]cam$")

# kwiver ignores -s entries for processes a pipeline does not contain, so one
# union of writer/reader keys serves every mono-camera category.
MONO_PARAMS = {
    "input:video_filename": "image-manifest.txt",
    "input:video_reader:type": "image_list",
    "input:video_reader:image_list:image_reader:type": "vxl",
    "detection_reader:file_name": "groundtruth.csv",
    "track_reader:file_name": "groundtruth.csv",
    "detector_writer:file_name": "output/detector_output.csv",
    "track_writer:file_name": "output/track_output.csv",
    "kwa_writer:output_directory": "output/",
    "image_writer:file_name_prefix": "output/",
    "debayered_writer:file_name_prefix": "output/",
    "depth_map_writer:file_name_prefix": "output/",
    "video_writer:video_filename": "output/output.mp4",
    "track_resampler:track_file": "groundtruth.csv",
}

STEREO_PARAMS = {
    "input:video_reader:type": "image_list",
    "input:video_filename": "input1_images.txt",
    "input1:video_filename": "input1_images.txt",
    "input2:video_filename": "input2_images.txt",
    "detection_reader:file_name": "detections1.csv",
    "detection_reader1:file_name": "detections1.csv",
    "detection_reader2:file_name": "detections2.csv",
    "track_reader:file_name": "detections1.csv",
    "track_reader1:file_name": "detections1.csv",
    "track_reader2:file_name": "detections2.csv",
    "detector_writer1:file_name": "output/detector_output1.csv",
    "detector_writer2:file_name": "output/detector_output2.csv",
    "track_writer1:file_name": "output/track_output1.csv",
    "track_writer2:file_name": "output/track_output2.csv",
}

CALIBRATION_PARAMS = {
    "measurer:calibration_file": "calibration_matrices.json",
    "calibration_reader:file": "calibration_matrices.json",
}


def csv(**kwargs):
    return lambda env_dir: check_csv(env_dir, **kwargs)


def frames(**kwargs):
    return lambda env_dir: check_generated_frames(env_dir, **kwargs)


def video(**kwargs):
    return lambda env_dir: check_generated_video(env_dir, **kwargs)


def chips(**kwargs):
    return lambda env_dir: check_generated_chips(env_dir, **kwargs)


SMOKE = csv()
MIN_1 = csv(expected_detections=1, comparison_detection="min")
MIN_2 = csv(expected_detections=2, comparison_detection="min")
POLYGON = csv(all_types="polygon")
HEAD_TAIL = csv(all_types="head-tail")
STEREO = csv(is_stereo=True)
STEREO_MIN_2 = csv(expected_detections=2, comparison_detection="min", is_stereo=True)
# 9x6 inner corners over the 8 frames of the checkerboard sequence.
CHECKERBOARD_CORNERS = 9 * 6 * 8


@dataclass(frozen=True)
class Case:
    pipe: str = ""
    id: str = ""
    env: str = ""
    params: dict = field(default_factory=dict)
    setup: Callable = None
    check: Callable = None
    skip: str = ""


# Per-category defaults, refined by RULES and then by OVERRIDES.
DEFAULTS = {
    "detector": Case(env="env_fish", params=MONO_PARAMS, check=SMOKE),
    "tracker": Case(env="env_fish_sequence", params=MONO_PARAMS, check=SMOKE),
    "filter": Case(env="env_fish", params=MONO_PARAMS, check=frames()),
    "transcode": Case(env="env_fish", params=MONO_PARAMS, check=video(min_size=100)),
    "utility": Case(env="env_fish_with_detections", params=MONO_PARAMS, check=SMOKE),
    "measurement": Case(env="env_stereo_fish_with_polygons", params=STEREO_PARAMS, check=STEREO),
}

# Families of pipelines that share a fixture and expectation. Every matching
# rule is applied in order, so later entries refine earlier ones, and a new
# sibling pipeline shipped by an add-on is covered without an edit here.
RULES = (
    (r"^detector_(community_fish|default_fish|em_tuna|generic_proposals|grouper_moon"
     r"|huggingface_zeroshot|mouss_deep7|seamap_)", dict(check=MIN_1)),
    (r"^detector_(fish_with_motion|motion)", dict(env="env_fish_sequence", check=MIN_1)),
    (r"^detector_(arctic_seal|sea_lion|pengcam|swfsc)", dict(env="env_seal", check=MIN_1)),
    (r"^detector_penguin_aerial_", dict(env="env_seal", check=SMOKE)),
    (r"^tracker_(community_fish|default_fish|em_tuna|fish\.sfd|generic_proposals"
     r"|grouper_moon|motion|mouss_deep7|seamap_)", dict(check=MIN_2)),
    (r"^tracker_(sea_lion|penguin_aerial)", dict(env="env_seal", check=MIN_1)),
    # Registration-based suppression needs consecutive frames to align; the
    # fixture set has a single seal image and these deadlock without a second.
    (r"^tracker_sea_lion_(suppressor|tracker)_",
     dict(skip="needs a multi-frame seal sequence")),
    # These take the category default; the rule marks them covered rather than
    # letting them fall through to an uncovered skip.
    (r"^filter_(debayer|enhance|normalize|split)", {}),
    (r"^filter_(draw_dets|extract_chips)", dict(env="env_fish_with_detections")),
    (r"^transcode_", {}),
    (r"^utility_add_head_tail_keypoints", dict(check=HEAD_TAIL)),
    (r"^utility_add_segmentations", dict(check=POLYGON)),
    (r"^measurement_from_annotations", dict(params=CALIBRATION_PARAMS, check=STEREO_MIN_2)),
)


def _target(width: int, height: int) -> dict:
    block = "detector1:detector:ocv_detect_calibration_targets"
    return {f"{block}:target_width": width, f"{block}:target_height": height}


def _check_calibrated(env_dir: Path):
    check_csv(env_dir, expected_detections=CHECKERBOARD_CORNERS, is_stereo=True)
    assert (env_dir / "calibration_matrices.json").is_file()


def _check_kwa(env_dir: Path):
    for name, min_size in (("kwa.data", 20_000), ("kwa.index", 25), ("kwa.meta", 70)):
        path = env_dir / "output" / name
        assert path.is_file(), f"{name} not written"
        assert path.stat().st_size >= min_size, f"{name} is {path.stat().st_size} bytes"


def _check_homographies(env_dir: Path):
    lines = (env_dir / "output" / "homogs.txt").read_text().splitlines()
    assert len(lines) == 9
    # 3x3 homography, source frame, destination frame.
    assert all(len(line.split()) == 11 for line in lines)


def _check_debayer_and_depth_map(env_dir: Path):
    for name in ("frame000001.png", "depth_map000001.png"):
        assert (env_dir / "output" / name).is_file(), f"{name} not written"


def _check_depth_maps(env_dir: Path):
    depth_maps = env_dir / "output" / "depthMap"
    assert len(list(depth_maps.glob("*.png"))) == 2


# Pipelines needing config, fixtures or assertions of their own. A list value
# expands into several cases over the same pipeline.
OVERRIDES = {
    "detector_simple_hough": [
        Case(id="detector_simple_hough_empty", env="env_single_empty", check=csv(expected_detections=0)),
        Case(id="detector_simple_hough_circles", env="env_circles_3", check=csv(expected_detections=3)),
    ],
    "detector_calibration_target": [
        Case(id="detector_calibration_target_9_6", env="env_checkerboard_9_6",
             params=_target(9, 6), check=csv(expected_detections=54)),
        Case(id="detector_calibration_target_4_4", env="env_checkerboard_4_4",
             params=_target(4, 4), check=csv(expected_detections=16)),
    ],
    "tracker_calibration_target": Case(
        env="env_checkerboard_sequence", check=csv(expected_detections=CHECKERBOARD_CORNERS)),
    "filter_debayer_and_depth_map": Case(check=_check_debayer_and_depth_map),
    "filter_extract_chips": Case(check=chips()),
    "filter_stereo_depth_map": Case(check=frames(match_names=False)),
    "filter_to_kwa": Case(params={"kwa_writer:base_filename": "kwa"}, check=_check_kwa),
    "filter_to_video": Case(check=video(min_size=10_000)),
    "filter_tracks_only": Case(
        env="env_fish_sequence_with_detections", check=frames(match_names=False, delta=-2)),
    "filter_tracks_only_adjust_csv": Case(
        env="env_fish_sequence_with_detections", check=frames(match_names=False, delta=-2)),
    "transcode_native_fps": Case(env="env_fish_sequence_with_detections"),
    "transcode_tracks_only": Case(env="env_fish_sequence_with_detections"),
    "utility_add_head_tail_keypoints_from_dets": Case(env="env_fish_with_polygons"),
    # Both label every frame of the 9-frame fixture; the sequence is one
    # continuous shot, so auto finds no break and emits a single track.
    "utility_empty_frame_lbls_auto": Case(
        env="env_fish_sequence", check=csv(expected_detections=9)),
    "utility_empty_frame_lbls_fixed_interval": Case(
        env="env_fish_sequence", check=csv(expected_detections=9)),
    "utility_max_points_per_poly": Case(env="env_fish_with_polygons", check=POLYGON),
    "utility_register_frames": Case(
        env="env_fish_sequence", params={"homog_writer:output": "output/homogs.txt"},
        check=_check_homographies),
    "utility_remove_dets_in_ignore_regions": Case(env="env_fish_sequence"),
    "measurement_calibrate_cameras_default": Case(
        env="env_stereo_checkerboards", check=_check_calibrated),
    "measurement_calibrate_cameras_fast": Case(
        env="env_stereo_checkerboards", check=_check_calibrated),
    "measurement_detect_calibration_target": Case(
        env="env_stereo_checkerboards", check=csv(expected_detections=CHECKERBOARD_CORNERS, is_stereo=True)),
    "measurement_compute_rectified_disparity": Case(
        env="env_stereo_fish",
        params=CALIBRATION_PARAMS | {
            "depth_map:computer:ocv_stereo_disparity:calibration_file": "./",
            "output:file_name_template": "output/depthMap/depth_map%06d.png",
        },
        setup=lambda env_dir: (env_dir / "output" / "depthMap").mkdir(exist_ok=True),
        check=_check_depth_maps),
    "measurement_default_fish_fully_auto": Case(
        env="env_stereo_fish", params=CALIBRATION_PARAMS, check=STEREO_MIN_2),
    "measurement_fully_auto_gmm_motion": Case(
        env="env_stereo_fish", params=CALIBRATION_PARAMS,
        check=csv(expected_detections=0, comparison_detection="min", is_stereo=True)),
}


def _installed_pipelines(category: str) -> list[str]:
    install = find_viame_install()
    assert install is not None, "VIAME install not found; set VIAME_INSTALL"
    pipelines = install / "configs" / "pipelines"
    assert pipelines.is_dir(), f"{pipelines} does not exist"
    stems = []
    for path in sorted(pipelines.glob(f"{category}_*.pipe")):
        if DISALLOWED.match(path.name):
            continue
        if MULTICAM.search(path.stem):
            continue
        stems.append(path.stem)
    return stems


def _merge(base: Case, updates: dict, stem: str) -> Case:
    params = {**base.params, **updates.get("params", {})}
    merged = {
        "pipe": f"pipelines/{stem}.pipe",
        "id": updates.get("id") or base.id or stem,
        "env": updates.get("env") or base.env,
        "params": params,
        "setup": updates.get("setup") or base.setup,
        "check": updates.get("check") or base.check,
        "skip": updates.get("skip") or base.skip,
    }
    return Case(**merged)


_RELATIVEPATH = re.compile(r"^\s*relativepath\s+\S+\s*=\s*(\S+)", re.M)
_INCLUDE = re.compile(r"^\s*include\s+(\S+\.pipe)\s*$", re.M)


def _missing_model(stem: str) -> str:
    """First model path a pipe (or anything it includes) needs but the install
    lacks, or "". Model packs get slimmed over time and nothing reconciles the
    pipes they ship against the models they still carry, so a pipeline can be
    installed whose model directory no longer exists anywhere. Running it just
    documents the pack's inconsistency as a 900s failure; skipping names the
    missing file instead."""
    pipelines = find_viame_install() / "configs" / "pipelines"
    queue, seen = [pipelines / f"{stem}.pipe"], set()
    while queue:
        pipe = queue.pop()
        if pipe in seen or not pipe.is_file():
            continue
        seen.add(pipe)
        text = pipe.read_text(errors="replace")
        for rel in _RELATIVEPATH.findall(text):
            if not (pipe.parent / rel).exists():
                return rel
        for inc in _INCLUDE.findall(text):
            queue.append(pipe.parent / inc)
    return ""


def discover(category: str) -> list[Case]:
    """Every installed pipeline of a category, as a runnable Case."""
    cases = []
    for stem in _installed_pipelines(category):
        base, matched = DEFAULTS[category], False
        for pattern, updates in RULES:
            if re.match(pattern, stem):
                base, matched = _merge(base, updates, stem), True
        overrides = OVERRIDES.get(stem)
        if overrides is None:
            overrides = [Case()] if matched else [Case(skip="no test case defined")]
        elif not isinstance(overrides, list):
            overrides = [overrides]
        merged = [_merge(base, override.__dict__, stem) for override in overrides]
        absent = _missing_model(stem)
        if absent:
            merged = [case if case.skip else
                      replace(case, skip=f"missing model {absent}")
                      for case in merged]
        cases += merged
    ids = [case.id for case in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"duplicate case ids in {category}: {sorted(duplicates)}"
    return cases


def run_case(case: Case, runner, env_dir, request):
    if case.skip:
        pytest.skip(case.skip)
    request.getfixturevalue(case.env)
    if case.setup:
        case.setup(env_dir)
    result = runner.run(case.pipe, env_dir, overrides=case.params)
    assert result.returncode == 0, result.stderr[-4000:]
    case.check(env_dir)
