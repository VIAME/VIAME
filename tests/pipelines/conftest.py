import pytest
import cv2
import numpy as np
import shutil

from .viame_runner import ViameRunner

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
def env_circles(env_dir):
    img = np.full((100, 100, 3), (20, 20, 20), dtype=np.uint8)
    circles = [(22, 10, 10), (56, 65, 20), (82, 20, 16)]
    for x, y, r in circles:
        cv2.circle(img, (x, y), r, (255, 255, 255), -1)
    cv2.imwrite(str(env_dir / "images" / "circles.jpg"), img)
    return _finalize_env(env_dir)

@pytest.fixture
def env_checkerboard(env_dir, request):
    grid_size = getattr(request, "param", (9, 6))
    width, height = grid_size

    square_size = 20
    img_size = ((width + 1) * square_size, (height + 1) * square_size)

    bg = np.zeros((height + 1, width + 1), dtype=np.uint8)
    bg[1::2, ::2] = 255
    bg[::2, 1::2] = 255

    img = cv2.resize(bg, img_size, interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(str(env_dir / "images" / "checkerboard.jpg"), img)

    return _finalize_env(env_dir)

@pytest.fixture
def env_single_empty(env_dir, request):
    img_path = env_dir / "images" / "test.jpg"
    cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
    return _finalize_env(env_dir)

@pytest.fixture
def env_empty(env_dir, request):
    count = getattr(request, "param", 10)

    images_dir = env_dir / "images"
    for i in range(count):
        img_path = images_dir / f"test{i:03}.jpg"
        cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))

    return _finalize_env(env_dir)

@pytest.fixture
def env_fish(env_dir, data_path):
    shutil.copy(data_path / "images" / "fish-1.jpg", env_dir / "images" / "fish-1.jpg")
    return _finalize_env(env_dir)

@pytest.fixture
def env_fish_with_detections(env_dir, data_path):
    shutil.copy(data_path / "images" / "fish-1.jpg", env_dir / "images" / "fish-1.jpg")
    shutil.copy(data_path / "labels" / "fish-1-detections.csv", env_dir / "groundtruth.csv")
    return _finalize_env(env_dir)

@pytest.fixture
def env_fish_with_polygons(env_dir, data_path):
    shutil.copy(data_path / "images" / "fish-1.jpg", env_dir / "images" / "fish-1.jpg")
    shutil.copy(data_path / "labels" / "fish-1-polygons.csv", env_dir / "groundtruth.csv")
    return _finalize_env(env_dir)


def _finalize_env(env_path):
    images_dir = env_path / "images"
    image_files = sorted([img.name for img in images_dir.glob("*")])

    with open(env_path / "image-manifest.txt", 'w') as f:
        for name in image_files:
            f.write(f"images/{name}\n")

    return env_path
