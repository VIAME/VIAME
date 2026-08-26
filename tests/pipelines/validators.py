from pathlib import Path
from typing import Literal

def get_viame_csv_lines(path: Path) -> list[str]:
    with open(path, 'r') as f:
        return [l for l in f.readlines() if l.strip() and not l.startswith('#')]


def _check_csv(
        csv_path: Path,
        expected_detections: int = None,
        comparison_detection: Literal['equal', 'min', 'max'] = 'equal',
        all_types: Literal['polygon', 'head-tail'] = None
):
    assert csv_path.is_file()

    lines = get_viame_csv_lines(csv_path)

    if expected_detections is not None:
        if comparison_detection == 'equal':
            assert len(lines) == expected_detections, f"Expected {expected_detections} detections, found {len(lines)}"
        elif comparison_detection == 'min':
            assert len(lines) >= expected_detections, f"Expected at least {expected_detections} detections, found {len(lines)}"
        elif comparison_detection == 'max':
            assert len(lines) <= expected_detections, f"Expected at most {expected_detections} detections, found {len(lines)}"
        else:
            raise ValueError(f"{comparison_detection} comparison method not supported")

    if all_types is not None:
        if all_types == 'polygon':
            search_for = '(poly)'
        elif all_types == 'head-tail':
            search_for = '(kp)'
        else:
            raise ValueError("Invalid type", all_types)
        for line in lines:
            assert search_for in line



def check_csv(
        env_dir: Path,
        expected_detections: int = None,
        comparison_detection: Literal['equal', 'min', 'max'] = 'equal',
        all_types: Literal['polygon', 'head-tail'] = None,
        is_stereo: bool = False
):
    """
    Assert the pipeline wrote a detection CSV, plus any optional expectations.

    Track output is preferred over detector output when both exist. Stereo
    pipelines write one CSV per camera, named <base>1.csv and <base>2.csv.
    """
    output = env_dir / "output"
    suffixes = ("1", "2") if is_stereo else ("",)
    base = ("track_output" if (output / f"track_output{suffixes[0]}.csv").is_file()
            else "detector_output")
    for suffix in suffixes:
        _check_csv(output / f"{base}{suffix}.csv", expected_detections,
                   comparison_detection, all_types)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pgm", ".ppm"}


def _image_stems(folder: Path) -> set[str]:
    return set(p.stem for p in folder.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def check_generated_frames(env_dir: Path, match_names: bool = True, delta: int = 0):
    input_folder = env_dir / "images"
    output_folder = env_dir / "output"

    input_image_names = _image_stems(input_folder)
    output_image_names = _image_stems(output_folder)

    if match_names:
        diff_set = input_image_names - output_image_names
        assert len(diff_set) == abs(delta)
    assert len(input_image_names) + delta == len(output_image_names)


def check_generated_chips(env_dir: Path, csv_name: str = "groundtruth.csv"):
    """One chip image per detection in the input CSV."""
    expected = len(get_viame_csv_lines(env_dir / csv_name))
    assert len(_image_stems(env_dir / "output")) == expected


def check_generated_video(env_dir: Path, file_name: str = 'output.mp4', min_size: int = 0):
    video_file = env_dir / "output" / file_name
    assert video_file.is_file()
    if min_size > 0:
        assert video_file.stat().st_size >= min_size
