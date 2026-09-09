# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""Tests for annotation conversions through the viame convert applet."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from viame_env import find_viame_install, get_sourced_env, get_viame_source

CSV_HEADER = (
    "# 1: Detection or Track-id,2: Video or Image Identifier,"
    "3: Unique Frame Identifier,4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y),"
    "8: Detection or Length Confidence,9: Target Length (0 or -1 if invalid),"
    "10-11+: Repeated Species, Confidence Pairs or Attributes\n"
)

SPARSE_ROWS = [
    "1,img0.png,0,10,10,50,50,0.9,-1,fish,0.9",
    "1,img5.png,5,12,12,52,52,0.8,-1,fish,0.8",
    "2,img9.png,9,1,1,9,9,0.5,-1,rock,0.5,(kp) head 3 4,(kp) tail 7 8,"
    "(poly) 1 1 9 1 9 9,(atr) sex male,(note) hello",
]


@pytest.fixture(scope="module")
def viame_env():
    install = find_viame_install()
    if install is None:
        pytest.skip("No VIAME install found")
    return get_sourced_env(install)


@pytest.fixture(scope="module")
def mouss_data():
    folder = (get_viame_source() / "examples" / "object_detector_training"
              / "training_data_mouss")
    if not (folder / "seq1" / "groundtruth.csv").exists():
        pytest.skip("Missing mouss training data")
    return folder


def run_convert(env, *args, cwd=None):
    result = subprocess.run(
        ["viame", "convert", *[str(a) for a in args]],
        env=env, cwd=cwd, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def csv_rows(path):
    return [line for line in Path(path).read_text().splitlines()
            if line and not line.startswith("#")]


def write_sparse_csv(folder):
    path = folder / "sparse.csv"
    path.write_text(CSV_HEADER + "\n".join(SPARSE_ROWS) + "\n")
    return path


def test_lists_formats(viame_env):
    out = run_convert(viame_env, "--list-formats")
    assert "dive" in out and "coco" in out and "viame_csv" in out


def test_csv_to_dive_and_back(viame_env, tmp_path):
    source = write_sparse_csv(tmp_path)
    dive = tmp_path / "sparse.dive.json"
    run_convert(viame_env, source, dive)

    doc = json.loads(dive.read_text())
    assert doc["version"] == 2
    assert set(doc["tracks"]) == {"1", "2"}
    rock = doc["tracks"]["2"]
    assert rock["confidencePairs"][0][0] == "rock"
    feature = rock["features"][0]
    assert feature["frame"] == 9
    assert feature["head"] == [3.0, 4.0] and feature["tail"] == [7.0, 8.0]
    assert feature["attributes"] == {"sex": "male"}
    assert feature["notes"] == ["hello"]
    polygons = [f for f in feature["geometry"]["features"]
                if f["geometry"]["type"] == "Polygon"]
    assert polygons and polygons[0]["geometry"]["coordinates"][0][0] == [1.0, 1.0]

    back = tmp_path / "back.csv"
    run_convert(viame_env, dive, back)
    rows = csv_rows(back)
    assert len(rows) == 3
    assert "(kp) head 3 4" in rows[2] and "(poly) 1 1 9 1 9 9" in rows[2]
    assert "(atr) sex male" in rows[2] and "(note) hello" in rows[2]


def test_csv_to_coco_and_back(viame_env, tmp_path):
    source = write_sparse_csv(tmp_path)
    coco = tmp_path / "sparse.json"
    run_convert(viame_env, source, coco)

    doc = json.loads(coco.read_text())
    assert len(doc["annotations"]) == 3
    assert {c["name"] for c in doc["categories"]} == {"fish", "rock"}

    back = tmp_path / "back.csv"
    run_convert(viame_env, coco, back)
    rows = csv_rows(back)
    assert len(rows) == 3
    assert rows[0].split(",")[1] == "img0.png"


def test_folder_with_imagery_alongside(viame_env, mouss_data, tmp_path):
    out = run_convert(viame_env, mouss_data, tmp_path / "coco", "-o", "coco")
    assert "frames from seq1" in out
    doc = json.loads((tmp_path / "coco" / "seq1" / "groundtruth.json").read_text())
    assert doc["images"][0]["file_name"].endswith(".png")
    assert len(doc["annotations"]) == 4
    # groundtruth.kw18 next to groundtruth.csv must not clobber its output
    assert "Skipping" in out and "groundtruth.kw18" in out

    run_convert(viame_env, tmp_path / "coco", tmp_path / "csv", "-o", "viame_csv",
                "--no-images")
    rows = csv_rows(tmp_path / "csv" / "seq1" / "groundtruth.csv")
    assert len(rows) == 4 and rows[0].split(",")[1].endswith(".png")


def test_video_alongside_sets_frame_count(viame_env, mouss_data, tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    image = next((mouss_data / "seq1").glob("*.png"))
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-i", str(image),
         "-t", "2", "-r", "10", "-pix_fmt", "yuv420p", "-vf", "scale=320:240",
         str(tmp_path / "clip.mp4")],
        check=True,
    )
    (tmp_path / "clip.csv").write_text(
        CSV_HEADER + "1,clip.mp4,0,10,10,50,50,0.9,-1,fish,0.9\n"
        "2,clip.mp4,9,1,1,9,9,0.5,-1,rock,0.5\n")

    run_convert(viame_env, tmp_path / "clip.csv", tmp_path / "native.json")
    native = json.loads((tmp_path / "native.json").read_text())
    assert len(native["images"]) == 20

    run_convert(viame_env, tmp_path / "clip.csv", tmp_path / "half.json",
                "--frame-rate", "5")
    half = json.loads((tmp_path / "half.json").read_text())
    assert len(half["images"]) == 10


def test_calibration_inputs_go_to_the_script(viame_env, tmp_path):
    intrinsics = get_viame_source() / "tests" / "data" / "intrinsics.yml"
    if not intrinsics.exists():
        pytest.skip("Missing calibration test data")
    run_convert(viame_env, intrinsics, tmp_path / "calibration.json")
    doc = json.loads((tmp_path / "calibration.json").read_text())
    assert doc
