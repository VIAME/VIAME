import shutil

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from .viame_runner import ViameRunner


def pytest_sessionfinish(session, exitstatus):
    # ctest reads 5 as "skipped"; only claim that when nothing actually ran,
    # otherwise an uncovered pipeline would mask a real failure.
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if exitstatus == 0 and reporter.stats.get('skipped') and not reporter.stats.get('passed'):
        session.exitstatus = 5


@pytest.fixture
def runner():
    return ViameRunner()


@pytest.fixture
def data_path(request) -> Path:
    return request.path.parent / "pipelines_test_data"


@pytest.fixture
def env_dir(tmp_path, data_path):
    (tmp_path / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)
    shutil.copy(data_path / "labels" / "empty.csv", tmp_path / "groundtruth.csv")
    return tmp_path


@dataclass(frozen=True)
class Environment:
    """Input data laid out the way DIVE hands a dataset to a pipeline."""

    images: tuple             # globs under pipelines_test_data/images
    files: dict = field(default_factory=dict)   # dest name -> path under pipelines_test_data
    stereo: bool = False


FISH_DETECTIONS = {"groundtruth.csv": "labels/fish/fish_1_detections.csv"}
FISH_POLYGONS = {"groundtruth.csv": "labels/fish/fish_1_polygons.csv"}
STEREO_CALIBRATION = {
    "calibration_matrices.json": "labels/stereo/fish/calibration_matrices.json",
    "intrinsics.yml": "labels/stereo/fish/intrinsics.yml",
    "extrinsics.yml": "labels/stereo/fish/extrinsics.yml",
}

ENVIRONMENTS = {
    "env_single_empty": Environment(("empty_100_100.jpg",)),
    "env_circles_3": Environment(("circles_3.jpg",)),
    "env_seal": Environment(("seal_1.jpg",)),
    "env_checkerboard_9_6": Environment(("checkerboards/checkerboard_9_6.jpg",)),
    "env_checkerboard_4_4": Environment(("checkerboards/checkerboard_4_4.jpg",)),
    "env_checkerboard_sequence": Environment(("stereo/checkerboards/L_*.jpg",)),
    "env_fish": Environment(("fish/fish_1.jpg",)),
    "env_fish_with_detections": Environment(("fish/fish_1.jpg",), FISH_DETECTIONS),
    "env_fish_with_polygons": Environment(("fish/fish_1.jpg",), FISH_POLYGONS),
    "env_fish_sequence": Environment(("fish/fish_1_seq_*.jpg",)),
    "env_fish_sequence_with_detections": Environment(
        ("fish/fish_1_seq_*.jpg",), {"groundtruth.csv": "labels/fish/fish_1_seq_detections.csv"}),
    "env_fish_sequence_with_polygons": Environment(
        ("fish/fish_1_seq_*.jpg",), {"groundtruth.csv": "labels/fish/fish_1_seq_polygons.csv"}),
    "env_stereo_checkerboards": Environment(
        ("stereo/checkerboards/L_*.jpg", "stereo/checkerboards/R_*.jpg"), stereo=True),
    "env_stereo_fish": Environment(
        ("stereo/fish/L_*.jpg", "stereo/fish/R_*.jpg"), STEREO_CALIBRATION, stereo=True),
    "env_stereo_fish_with_polygons": Environment(
        ("stereo/fish/L_*.jpg", "stereo/fish/R_*.jpg"),
        STEREO_CALIBRATION | {
            "detections1.csv": "labels/stereo/fish/left-fish.csv",
            "detections2.csv": "labels/stereo/fish/right-fish.csv",
        },
        stereo=True),
}


def _write_manifest(path: Path, names: list[str]):
    path.write_text("".join(f"images/{name}\n" for name in names))


def _build(env_dir: Path, data_path: Path, spec: Environment) -> Path:
    for pattern in spec.images:
        for source in sorted((data_path / "images").glob(pattern)):
            shutil.copy(source, env_dir / "images")
    for destination, source in spec.files.items():
        shutil.copy(data_path / source, env_dir / destination)

    names = sorted(path.name for path in (env_dir / "images").glob("*"))
    if spec.stereo:
        _write_manifest(env_dir / "input1_images.txt", [n for n in names if n.startswith("L_")])
        _write_manifest(env_dir / "input2_images.txt", [n for n in names if n.startswith("R_")])
    else:
        _write_manifest(env_dir / "image-manifest.txt", names)
    return env_dir


def _env_fixture(spec: Environment):
    @pytest.fixture
    def fixture(env_dir, data_path):
        return _build(env_dir, data_path, spec)

    return fixture


for _name, _spec in ENVIRONMENTS.items():
    globals()[_name] = _env_fixture(_spec)
