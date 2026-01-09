#!/usr/bin/env python

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Launch the VIAME timeline/track viewer GUI interface."""

import argparse
import atexit
import glob
import os
import shutil
import subprocess
import sys
import tempfile

DIV = '\\' if os.name == 'nt' else '/'

temp_dir = tempfile.mkdtemp(prefix='viqui-tmp')
atexit.register(lambda: shutil.rmtree(temp_dir))


def _list_files(folder, extension):
    return glob.glob(os.path.join(folder, f'*{extension}'))


def _get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def _get_gui_cmd():
    if os.name == 'nt':
        return ['vsPlay.exe']
    return ['vsPlay']


def _find_file(filename):
    if os.path.exists(filename):
        return os.path.abspath(filename)
    alt_path = os.path.join(_get_script_path(), filename)
    if os.path.exists(alt_path):
        return alt_path
    print(f"Unable to find {filename}")
    sys.exit(1)


def _select_option(option_list, display_str="Select File:"):
    print()
    for i, option in enumerate(option_list, 1):
        print(f"({i}) {option}")

    sys.stdout.write(f"\n{display_str} ")
    sys.stdout.flush()
    choice = input().lower()

    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(option_list):
        print("Invalid selection, must be a valid number")
        sys.exit(1)

    return option_list[int(choice) - 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch track viewer GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", dest="input_dir", default="database",
                        help="Input directory containing results")
    parser.add_argument("-t", dest="threshold", default="0.0",
                        help="Optional detection threshold to apply")
    parser.add_argument("-theme", dest="gui_theme_file",
                        default=f"gui-params{DIV}view_color_settings.ini",
                        help="GUI default theme settings")
    parser.add_argument("-filter", dest="filter_file",
                        default=f"gui-params{DIV}default_timeline_filter.vpefs",
                        help="GUI default filter settings")

    args = parser.parse_args()

    files = _list_files(args.input_dir, ".index")
    if not files:
        print("Error: No computed results in input directory")
        sys.exit(1)

    files.sort()
    filename = _select_option(files, "Select File:")
    base = os.path.splitext(filename)[0]

    # Find detection file
    detection_file = f"{base}_tracks.csv"
    if not os.path.isfile(detection_file):
        detection_file = f"{base}_detections.csv"
        if not os.path.isfile(detection_file):
            print("Error: Detection file does not exist")
            sys.exit(1)

    # Look for object category instances in file
    unique_ids = set()
    with open(detection_file) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 10 or (parts[0] and parts[0][0] == '#'):
                continue
            for i in range(9, len(parts), 2):
                unique_ids.add(parts[i])

    if not unique_ids:
        print("Error: Detection file contains no categories")
        sys.exit(1)

    sample = next(iter(unique_ids))
    all_category = "all_categories" if sample[0].islower() else "All Categories"
    category_list = [all_category] + list(unique_ids)
    category = _select_option(category_list, "Select Category:")

    # Open index file and get timestamp vector
    ts_vec = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if i < 5:
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                ts_vec.append(str(float(parts[0]) / 1e6))

    if not ts_vec:
        print("Error: Selected video file is empty")
        sys.exit(1)

    # Perform conversion and thresholding
    fd1, track_file = tempfile.mkstemp(prefix='vsplay-tmp-tracks-',
                                       suffix='.kw18', text=True, dir=temp_dir)
    fd2, class_file = tempfile.mkstemp(prefix='vsplay-tmp-tracks-',
                                       suffix='.fso.txt', text=True, dir=temp_dir)

    threshold = float(args.threshold)

    with os.fdopen(fd1, 'w') as ftrk, os.fdopen(fd2, 'w') as fcls:
        with open(detection_file) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 10 or (parts[0] and parts[0][0] == '#'):
                    continue

                # Find matching category and confidence
                confidence = 0.0
                use_detection = False
                for i in range(9, len(parts), 2):
                    if (category == all_category or parts[i] == category) and \
                            float(parts[i + 1]) >= threshold:
                        use_detection = True
                        confidence = float(parts[i + 1])
                        break

                if not use_detection:
                    continue

                # Calculate center point
                c_x = int((float(parts[3]) + float(parts[5])) / 2)
                c_y = int((float(parts[4]) + float(parts[6])) / 2)

                # Write track file (kw18 format)
                ftrk.write(f"{parts[0]} 1 {parts[2]} 0 0 0 0 {c_x} {c_y} ")
                ftrk.write(f"{parts[3]} {parts[4]} {parts[5]} {parts[6]} 0 0 0 0 ")
                ftrk.write(f"{ts_vec[int(parts[2])]} {confidence}\n")

                # Write class file
                fcls.write(f"{parts[0]} {confidence} 0 {1.0 - confidence}\n")

    print()

    cmd = _get_gui_cmd() + ["-tf", track_file, "-vf", filename, "-df", class_file]

    if args.gui_theme_file:
        cmd += ["--theme", _find_file(args.gui_theme_file)]
    if args.filter_file:
        cmd += ["-ff", _find_file(args.filter_file)]

    subprocess.call(cmd)
