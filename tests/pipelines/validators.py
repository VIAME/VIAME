from pathlib import Path
from typing import Literal

def get_viame_csv_lines(path: Path) -> list[str]:
    with open(path, 'r') as f:
        return [l for l in f.readlines() if l.strip() and not l.startswith('#')]


def check_csv(
        env_dir: Path,
        expected_detections: int = None,
        comparison_detection: Literal['equal', 'min', 'max'] = 'equal',
        all_types: Literal['polygon', 'head-tail'] = None
):
    """
    Assert that the detector_output.csv file is created.
    Additional checks can be provided using optional parameters
    :param env_dir: fixture to get test environment directory.
    :param expected_detections: Optional int, expected number of detections in the csv
    :param comparison_detection: Optional int, expected number of detections in the csv
    :param all_types: Optional str (polygon, head-tail), all detections must have the declared type
    """
    detector_csv_path = env_dir / "output" / "detector_output.csv"
    track_csv_path = env_dir / "output" / "track_output.csv"

    csv_path = detector_csv_path
    if track_csv_path.is_file():
        csv_path = track_csv_path

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


def check_generated_frames(env_dir: Path):
    pass


def check_generated_video(env_dir: Path, file_name: str = 'output.mp4', min_size: int = 0):
    video_file = env_dir / "output" / file_name
    assert video_file.is_file()
    if min_size > 0:
        assert video_file.stat().st_size >= min_size
