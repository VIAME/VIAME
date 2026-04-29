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
def data_path(request):
    return request.path.parent / "data"

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
def env_dir(tmp_path):
    (tmp_path / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)
    return tmp_path

@pytest.fixture
def env_single_empty(env_dir, data_path):
    shutil.copy(data_path / "images" / "empty_100_100.jpg", env_dir / "images")
    return _finalize_env(env_dir)

@pytest.fixture
def env_checkerboard_9_6(env_dir, data_path):
    shutil.copy(data_path / "images" / "checkerboard_9_6.jpg", env_dir / "images")
    return _finalize_env(env_dir)

@pytest.fixture
def env_checkerboard_4_4(env_dir, data_path):
    shutil.copy(data_path / "images" / "checkerboard_4_4.jpg", env_dir / "images")
    return _finalize_env(env_dir)

@pytest.fixture
def env_circles_3(env_dir, data_path):
    shutil.copy(data_path / "images" / "circles_3.jpg", env_dir / "images")
    return _finalize_env(env_dir)

@pytest.fixture
def env_fish(env_dir, data_path):
    shutil.copy(data_path / "images" / "fish_1.jpg", env_dir / "images")
    return _finalize_env(env_dir)

@pytest.fixture
def env_fish_with_detections(env_dir, data_path):
    shutil.copy(data_path / "images" / "fish_1_jpg", env_dir / "images")
    shutil.copy(data_path / "labels" / "fish_1_detections.csv", env_dir / "groundtruth.csv")
    return _finalize_env(env_dir)

@pytest.fixture
def env_fish_with_polygons(env_dir, data_path):
    shutil.copy(data_path / "images" / "fish_1.jpg", env_dir / "images")
    shutil.copy(data_path / "labels" / "fish_1_polygons.csv", env_dir / "groundtruth.csv")
    return _finalize_env(env_dir)

@pytest.fixture
def env_seal(env_dir, data_path):
    shutil.copy(data_path / "images" / "seal_1.jpg", env_dir / "images")
    return _finalize_env(env_dir)


def _finalize_env(env_path):
    images_dir = env_path / "images"
    image_files = sorted([img.name for img in images_dir.glob("*")])

    with open(env_path / "image-manifest.txt", 'w') as f:
        for name in image_files:
            f.write(f"images/{name}\n")

    return env_path
