#!/usr/bin/env python

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Launch the VIAME annotation GUI interface."""

import argparse
import atexit
import glob
import os
import shutil
import subprocess
import sys
import tempfile

DIV = '\\' if os.name == 'nt' else '/'

temp_dir = tempfile.mkdtemp(prefix='vpview-tmp')
atexit.register(lambda: shutil.rmtree(temp_dir))


def _get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def _get_gui_cmd(debug=False):
    if os.name == 'nt':
        return ['vpView.exe']
    if debug:
        return ['gdb', '--args', 'vpView']
    return ['vpView']


def _execute_command(cmd, stdout=None, stderr=None):
    if os.name == 'nt' and stdout is None:
        with open(os.devnull, "w") as fnull:
            return subprocess.call(cmd, stdout=fnull, stderr=subprocess.STDOUT)
    return subprocess.call(cmd, stdout=stdout, stderr=stderr)


def _find_file(filename):
    if os.path.exists(filename):
        return os.path.abspath(filename)
    alt_path = os.path.join(_get_script_path(), filename)
    if os.path.exists(alt_path):
        return alt_path
    print(f"Unable to find {filename}")
    sys.exit(1)


def _create_pipelines_list(glob_str):
    fd, name = tempfile.mkstemp(prefix='vpview-pipelines-', suffix='.ini',
                                text=True, dir=temp_dir)
    search_str = os.path.join(_get_script_path(), glob_str)
    pipeline_files = sorted(glob.glob(search_str))

    with os.fdopen(fd, 'w') as f:
        f.write("[EmbeddedPipelines]\n")
        f.write(f"size={len(pipeline_files)}\n")
        for ind, full_path in enumerate(pipeline_files, 1):
            name_id = os.path.splitext(os.path.basename(full_path))[0]
            f.write(f'{ind}\\Name="{name_id}"\n')
            f.write(f'{ind}\\Path="{full_path.replace(chr(92), chr(92)*2)}"\n')

    return name


def _default_annotator_args(args):
    command_args = []
    if args.gui_theme:
        command_args += ["--theme", _find_file(args.gui_theme)]
    if args.pipelines:
        command_args += ["--import-config", _create_pipelines_list(args.pipelines)]
    return command_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch annotation GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-o", dest="output_directory", default="database",
                        help="Output directory to store files in")
    parser.add_argument("-v", dest="input_video", default="",
                        help="Input video file to run annotator on")
    parser.add_argument("-l", dest="input_list", default="",
                        help="Input image list file to run annotator on")
    parser.add_argument("-theme", dest="gui_theme",
                        default=f"gui-params{DIV}view_color_settings.ini",
                        help="GUI theme settings file")
    parser.add_argument("-pipelines", dest="pipelines",
                        default=f"pipelines{DIV}embedded_single_stream{DIV}*.pipe",
                        help="Glob pattern for runnable processing pipelines")
    parser.add_argument("--debug", dest="debug", action="store_true",
                        help="Run with debugger attached to process")

    args = parser.parse_args()

    if args.input_video:
        print("Function not yet implemented")
        sys.exit(1)
    elif args.input_list:
        print("Function not yet implemented")
        sys.exit(1)
    else:
        _execute_command(_get_gui_cmd(args.debug) + _default_annotator_args(args))
