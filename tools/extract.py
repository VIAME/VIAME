#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #


"""
Extract frames from video files using ffmpeg or KWIVER pipelines.
"""

import argparse
import os
import subprocess
import sys

sys.dont_write_bytecode = True

INVALID_TIME = "99:99:99.99"


def list_files_in_dir(folder):
    """List non-hidden files in a directory.

    Args:
        folder: Directory path to list

    Returns:
        List of full file paths
    """
    if not os.path.isdir(folder):
        raise ValueError(f"Input folder '{folder}' does not exist")
    return [
        os.path.join(folder, f) for f in sorted(os.listdir(folder))
        if not f.startswith('.') and os.path.isfile(os.path.join(folder, f))
    ]


def create_dir(dirname, logging=True):
    """Create a directory if it doesn't exist.

    Args:
        dirname: Directory path to create
        logging: Whether to print creation message
    """
    if not os.path.exists(dirname):
        if logging:
            print(f"Creating {dirname}")
        os.makedirs(dirname)


def get_ffmpeg_cmd():
    """Get the ffmpeg command based on platform."""
    if os.name == 'nt':
        return ['ffmpeg.exe']
    else:
        return ['ffmpeg']



def main():
    """Main entry point for video frame extraction."""
    parser = argparse.ArgumentParser(
        description="Extract frames from video files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-d", dest="input_dir", default="videos",
                        help="Input directory containing videos")
    parser.add_argument("-o", dest="output_dir", default="frames",
                        help="Output directory to put frames into")
    parser.add_argument("-r", dest="frame_rate", default="",
                        help="Video frame rate in Hz to extract at")
    parser.add_argument("-s", dest="start_time", default=INVALID_TIME,
                        help="Start time of frames to extract")
    parser.add_argument("-t", dest="duration", default=INVALID_TIME,
                        help="Duration of sequence to extract")
    parser.add_argument("-p", dest="pattern", default="frame%06d.png",
                        help="Frame pattern to dump frames into")
    parser.add_argument("-m", dest="method", default="kwiver",
                        choices=["kwiver", "ffmpeg"],
                        help="Extraction method to use")

    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite frames that already exist')
    args = parser.parse_args()
    failures = 0

    create_dir(args.output_dir)

    if args.method == "ffmpeg":
        files = list_files_in_dir(args.input_dir)
        for file_with_path in files:
            file_no_path = os.path.basename(file_with_path)
            output_folder = os.path.join(args.output_dir, file_no_path)
            create_dir(output_folder)

            cmd = get_ffmpeg_cmd() + ["-nostdin", "-y" if args.overwrite else "-n", "-i", file_with_path]
            if args.frame_rate:
                cmd += ["-r", args.frame_rate]
            if args.start_time and args.start_time != INVALID_TIME:
                cmd += ["-ss", args.start_time]
            if args.duration and args.duration != INVALID_TIME:
                cmd += ["-t", args.duration]
            cmd += [os.path.join(output_folder, args.pattern)]

            failures += subprocess.call(cmd) != 0
    else:
        cmd = ["viame", "run"]
        cmd += ["-d", args.input_dir]
        cmd += ["-o", args.output_dir]
        cmd += ["-p", "pipelines/filter_default.pipe"]
        cmd += ["--no-reset-prompt"]
        cmd += ["-pattern", args.pattern]
        if args.start_time and args.start_time != INVALID_TIME:
            cmd += ["-start-time", args.start_time]
        if args.duration and args.duration != INVALID_TIME:
            cmd += ["-duration", args.duration]

        if args.frame_rate:
            cmd += ["-frate", args.frame_rate]
        failures += subprocess.call(cmd) != 0

    if failures:
        print(f"Frame extraction failed for {failures} command(s)", file=sys.stderr)
        return 1
    print("\n\nFrame extraction complete, exiting.\n\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
