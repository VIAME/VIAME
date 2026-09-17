# This test is meant to run having the kwiver wheel installed.
# the dump_klv executable correponds to the dump_klv entrypoint which is bound to dump_klv.py
from pathlib import Path
import pytest
import subprocess
import filecmp

DATA_DIR = Path(__file__).parents[4] / "test_data"


def test_simple_run():
    config_file = DATA_DIR / "config_files" / "dump_klv_testing.conf"
    video_file = DATA_DIR / "videos" / "aphill_klv_10s.ts"
    args = ["--config", str(config_file), str(video_file)]
    try:
        result = subprocess.run(
            ["dump_klv"] + args, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as error:
        print(error.stdout)
        print(error.stderr)
        raise error


def test_with_baseline():
    config_file = DATA_DIR / "config_files" / "dump_klv_testing.conf"
    video_file = DATA_DIR / "videos" / "aphill_klv_10s.ts"
    other_args = ["--exporter=klv-json", "--compress"]
    output_file = "-l=out.json"
    args = ["--config", str(config_file), output_file, *other_args, str(video_file)]
    try:
        result = subprocess.run(
            ["dump_klv"] + args, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as error:
        print(error.stdout)
        print(error.stderr)
        raise error
    # compare against baseline
    baseline = DATA_DIR / "dump_klv_aphill_output.json"
    assert filecmp.cmp(str(baseline), "out.json"), "output file and basefile differ !"
