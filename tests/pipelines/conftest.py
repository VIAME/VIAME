from pathlib import Path

import pytest
import cv2
import numpy as np
import shutil

from .viame_runner import ViameRunner

def pytest_sessionfinish(session, exitstatus):
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = len(reporter.stats.get('skipped', []))
    if skipped > 0:
        session.exitstatus = 5

@pytest.fixture
def runner(tmp_path):
    return ViameRunner(tmp_path)


@pytest.fixture
def data_path(request) -> Path:
    return request.path.parent / "pipelines_test_data"

@pytest.fixture
def output_path(tmp_path):
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return out_dir

@pytest.fixture
def images_path(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    return images_dir

@pytest.fixture
def env_dir(tmp_path, data_path):
    (tmp_path / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)
    shutil.copy(data_path / "labels" / "empty.csv", tmp_path / "groundtruth.csv")
    return tmp_path

@pytest.fixture
def env_single_empty(env_dir, data_path):
    shutil.copy(data_path / "images" / "empty_100_100.jpg", env_dir / "images")
    return _finalize_env(env_dir)

@pytest.fixture
def env_checkerboard_9_6(env_dir, data_path):
    shutil.copy(data_path / "images" / "checkerboards" / "checkerboard_9_6.jpg", env_dir / "images")
    return _finalize_env(env_dir)

@pytest.fixture
def env_checkerboard_4_4(env_dir, data_path):
    shutil.copy(data_path / "images" / "checkerboards" / "checkerboard_4_4.jpg", env_dir / "images")
    return _finalize_env(env_dir)

@pytest.fixture
def env_checkerboard_sequence(env_dir, data_path):
    images = sorted((data_path / "images" / "stereo" / "checkerboards").glob("L_*.jpg"))
    for image in images:
        shutil.copy(image, env_dir / "images")
    return _finalize_env(env_dir)

@pytest.fixture
def env_circles_3(env_dir, data_path):
    shutil.copy(data_path / "images" / "circles_3.jpg", env_dir / "images")
    return _finalize_env(env_dir)

@pytest.fixture
def env_fish(env_dir, data_path):
    shutil.copy(data_path / "images" / "fish" / "fish_1.jpg", env_dir / "images")
    return _finalize_env(env_dir)

@pytest.fixture
def env_fish_with_detections(env_dir, data_path):
    shutil.copy(data_path / "images" / "fish" / "fish_1_jpg", env_dir / "images")
    shutil.copy(data_path / "labels" / "fish" / "fish_1_detections.csv", env_dir / "groundtruth.csv")
    return _finalize_env(env_dir)

@pytest.fixture
def env_fish_with_polygons(env_dir, data_path):
    shutil.copy(data_path / "images" / "fish" / "fish_1.jpg", env_dir / "images")
    shutil.copy(data_path / "labels" / "fish" / "fish_1_polygons.csv", env_dir / "groundtruth.csv")
    return _finalize_env(env_dir)

@pytest.fixture
def env_fish_sequence(env_dir, data_path):
    for i in range(1, 10):
        shutil.copy(data_path / "images" / "fish" / f"fish_1_seq_{i:02}.jpg", env_dir / "images")
    return _finalize_env(env_dir)

@pytest.fixture
def env_fish_with_detections(env_dir, data_path):
    for i in range(1, 10):
        shutil.copy(data_path / "images" / "fish" / f"fish_1_seq_{i:02}.jpg", env_dir / "images")
    shutil.copy(data_path / "labels" / "fish" / "fish_1_seq_detections.csv", env_dir / "groundtruth.csv")
    return _finalize_env(env_dir)

@pytest.fixture
def env_fish_with_polygons(env_dir, data_path):
    for i in range(1, 10):
        shutil.copy(data_path / "images" / "fish" / f"fish_1_seq_{i:02}.jpg", env_dir / "images")
    shutil.copy(data_path / "labels" / "fish" / "fish_1_seq_polygons.csv", env_dir / "groundtruth.csv")
    return _finalize_env(env_dir)

@pytest.fixture
def env_seal(env_dir, data_path):
    shutil.copy(data_path / "images" / "seal_1.jpg", env_dir / "images")
    return _finalize_env(env_dir)

@pytest.fixture
def env_fish_sequence(env_dir, data_path):
    for i in range(1, 10):
        shutil.copy(data_path / "images" / "fish" / f"fish_1_seq_{i:02}.jpg", env_dir / "images")
    return _finalize_env(env_dir)

@pytest.fixture
def env_fish_with_detections(env_dir, data_path):
    for i in range(1, 10):
        shutil.copy(data_path / "images" / "fish" / f"fish_1_seq_{i:02}.jpg", env_dir / "images")
    shutil.copy(data_path / "labels" / "fish" / "fish_1_seq_detections.csv", env_dir / "groundtruth.csv")
    return _finalize_env(env_dir)

@pytest.fixture
def env_fish_with_polygons(env_dir, data_path):
    for i in range(1, 10):
        shutil.copy(data_path / "images" / "fish" / f"fish_1_seq_{i:02}.jpg", env_dir / "images")
    shutil.copy(data_path / "labels" / "fish" / "fish_1_seq_polygons.csv", env_dir / "groundtruth.csv")
    return _finalize_env(env_dir)

@pytest.fixture
def env_seal(env_dir, data_path):
    shutil.copy(data_path / "images" / "seal_1.jpg", env_dir / "images")
    return _finalize_env(env_dir)

@pytest.fixture
def env_stereo_checkerboards(env_dir, data_path):
    checkerboards_path = data_path / "images" / "stereo" / "checkerboards"
    left_checkerboards = sorted(checkerboards_path.glob("L_*.jpg"))
    right_checkerboards = sorted(checkerboards_path.glob("R_*.jpg"))
    for img_path in left_checkerboards + right_checkerboards:
        shutil.copy(img_path, env_dir / "images")
    return _finalize_stereo_env(env_dir)


@pytest.fixture
def env_stereo_fish(env_dir, data_path):
    fish_path = data_path / "images" / "stereo" / "fish"
    labels_path = data_path / "labels" / "stereo" / "fish"
    left_fish = sorted(fish_path.glob("L_*.jpg"))
    right_fish = sorted(fish_path.glob("R_*.jpg"))
    for img_path in left_fish + right_fish:
        shutil.copy(img_path, env_dir / "images")
    shutil.copy(labels_path / "calibration_matrices.json", env_dir)
    shutil.copy(labels_path / "intrinsics.yml", env_dir)
    shutil.copy(labels_path / "extrinsics.yml", env_dir)
    return _finalize_stereo_env(env_dir)


@pytest.fixture
def env_stereo_fish_with_polygons(env_dir, data_path):
    fish_path = data_path / "images" / "stereo" / "fish"
    labels_path = data_path / "labels" / "stereo" / "fish"
    left_fish = sorted(fish_path.glob("L_*.jpg"))
    right_fish = sorted(fish_path.glob("R_*.jpg"))
    for img_path in left_fish + right_fish:
        shutil.copy(img_path, env_dir / "images")
    shutil.copy(labels_path / "left-fish.csv", env_dir / "detections1.csv")
    shutil.copy(labels_path / "right-fish.csv", env_dir / "detections2.csv")
    shutil.copy(labels_path / "calibration_matrices.json", env_dir)
    shutil.copy(labels_path / "intrinsics.yml", env_dir)
    shutil.copy(labels_path / "extrinsics.yml", env_dir)
    return _finalize_stereo_env(env_dir)


def _finalize_env(env_path: Path):
    images_dir = env_path / "images"
    image_files = sorted([img.name for img in images_dir.glob("*")])

    with open(env_path / "image-manifest.txt", 'w') as f:
        for name in image_files:
            f.write(f"images/{name}\n")

    return env_path

def _finalize_stereo_env(env_path: Path):
    images_dir = env_path / "images"
    left_image_files = sorted([img.name for img in images_dir.glob("L_*")])
    right_image_files = sorted([img.name for img in images_dir.glob("R_*")])

    with open(env_path / "input1_images.txt", 'w') as f:
        for name in left_image_files:
            f.write(f"images/{name}\n")

    with open(env_path / "input2_images.txt", 'w') as f:
        for name in right_image_files:
            f.write(f"images/{name}\n")

    return env_path
