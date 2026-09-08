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


def _list_files(folder):
    if os.path.exists(folder):
        return [
            os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if not f.startswith('.')
        ]
    return []


def _list_files_with_ext(folder, extension):
    return [f for f in _list_files(folder) if f.endswith(extension)]


def _glob_files(folder, prefix, extension):
    return glob.glob(os.path.join(folder, prefix) + "*" + extension)


def _multi_glob_files(folder, prefixes, extensions):
    output = []
    for prefix in prefixes:
        for extension in extensions:
            output.extend(glob.glob(os.path.join(folder, prefix) + "*" + extension))
    return output


def _get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def _create_dir(dirname):
    if not os.path.exists(dirname):
        print(f"Creating {dirname}")
        os.makedirs(dirname)
    if not os.path.exists(dirname):
        print(f"Unable to create {dirname}")
        sys.exit(1)


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


def _get_pipeline_cmd(debug=False):
    if os.name == 'nt':
        return ['kwiver.exe', 'runner']
    if debug:
        return ['gdb', '--args', 'kwiver', 'runner']
    return ['kwiver', 'runner']


def _generate_index_for_video(args, file_path, basename):
    if not os.path.isfile(file_path):
        print(f"Unable to find file: {file_path}")
        sys.exit(1)

    cmd = _get_pipeline_cmd() + [
        "-p", _find_file(args.cache_pipeline),
        "-s", f'input:video_filename={file_path}',
        "-s", 'input:video_reader:type=vidl_ffmpeg',
        "-s", f'kwa_writer:output_directory={args.cache_dir}',
        "-s", f'kwa_writer:base_filename={basename}',
        "-s", f'kwa_writer:stream_id={basename}'
    ]

    if args.frame_rate:
        cmd += ["-s", f'downsampler:target_frame_rate={args.frame_rate}']

    _execute_command(cmd)
    return os.path.join(args.cache_dir, f"{basename}.index")


def _select_option(option_list, display_str="Select Option:"):
    print()
    for i, option in enumerate(option_list, 1):
        print(f"({i}) {option}")

    sys.stdout.write(f"\n{display_str} ")
    sys.stdout.flush()
    choice = input().lower()

    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(option_list):
        print("Invalid selection, must be a valid number")
        sys.exit(1)

    return int(choice) - 1


def _process_video_dir(args):
    video_files = _list_files(args.video_dir)
    index_files = _list_files_with_ext(args.cache_dir, "index")

    video_files.sort()
    index_files.sort()

    video_names = [os.path.splitext(os.path.basename(f))[0] for f in video_files]
    index_names = [os.path.splitext(os.path.basename(f))[0] for f in index_files]

    net_files = video_names[:]
    net_full_paths = video_files[:]
    has_index = [False] * len(video_names)

    for fpath, fname in zip(index_files, index_names):
        if fname in net_files:
            index = net_files.index(fname)
            has_index[index] = True
            net_full_paths[index] = fpath
        else:
            net_files.append(fname)
            has_index.append(True)
            net_full_paths.append(fpath)

    if not net_files:
        print(f"\nError: No videos found in input directory: {args.video_dir}\n")
        print("If you want to load videos, not just images, make sure it is non-empty")
        sys.exit(1)

    # Build file list for selection
    file_list = []
    for fname, is_cached in zip(net_files, has_index):
        suffix = f" (cached in: {args.cache_dir})" if is_cached else ""
        file_list.append(fname + suffix)

    no_file = "with_no_imagery_loaded" if file_list[0].islower() else "With No Imagery Loaded"
    file_list = [no_file] + sorted(file_list)

    special_list_option = "input_list.txt"
    has_special = os.path.exists(special_list_option)
    if has_special:
        file_list.append(special_list_option)

    file_id = _select_option(file_list)

    if file_id == 0:
        _execute_command(_get_gui_cmd(args.debug) + _default_annotator_args(args))
        sys.exit(0)
    elif has_special and file_id == len(file_list) - 1:
        file_no_ext = special_list_option
        file_has_index = True
        file_path = special_list_option
    else:
        file_id -= 1
        file_no_ext = net_files[file_id]
        file_has_index = has_index[file_id]
        file_path = net_full_paths[file_id]

    # Scan for possible detection file
    detection_list = []
    detection_file = ""

    detection_search = _multi_glob_files('.', [file_no_ext], ["csv"])
    detection_list.extend(detection_search)

    if args.video_dir and args.video_dir != '.':
        detection_list.extend(_glob_files(args.video_dir, file_no_ext, "csv"))
    if args.cache_dir and args.cache_dir not in ('.', args.video_dir):
        detection_list.extend(_glob_files(args.cache_dir, file_no_ext, "csv"))

    detection_list = sorted(set(detection_list))

    if detection_list:
        no_det = "with_no_detections" if detection_list[0].islower() else "Launch Without Loading Detections"
        detection_list = [no_det] + detection_list
        detection_id = _select_option(detection_list)
        if detection_id != 0:
            detection_file = detection_list[detection_id]

    # Generate cache if needed
    if not file_has_index:
        _create_dir(args.cache_dir)
        if not os.path.isdir(file_path):
            print("Generating cache for video file, this may take up to a few minutes.\n")
            file_path = _generate_index_for_video(args, file_path, file_no_ext)
        else:
            from process_video import make_filelist_for_dir
            file_path = make_filelist_for_dir(file_path, args.cache_dir, file_no_ext)

    # Create project file
    fd, name = tempfile.mkstemp(prefix='vpview-project-', suffix='.prj',
                                text=True, dir=temp_dir)
    with os.fdopen(fd, 'w') as f:
        escaped_path = os.path.abspath(file_path).replace("\\", "\\\\")
        f.write(f"DataSetSpecifier={escaped_path}\n")
        if detection_file:
            escaped_det = os.path.abspath(detection_file).replace("\\", "\\\\")
            f.write(f"TracksFile={escaped_det}\n")

    _execute_command(_get_gui_cmd(args.debug) + ["-p", name] + _default_annotator_args(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch annotation GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", dest="video_dir", default="",
                        help="Input directory containing videos to run annotator on")
    parser.add_argument("-c", dest="cache_dir", default="",
                        help="Input directory containing cached video .index files")
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
    parser.add_argument("-cache-pipe", dest="cache_pipeline",
                        default=f"pipelines{DIV}filter_to_kwa.pipe",
                        help="Pipeline used for generating video .index files")
    parser.add_argument("-frate", dest="frame_rate", default="",
                        help="Frame rate override to process videos at")
    parser.add_argument("--debug", dest="debug", action="store_true",
                        help="Run with debugger attached to process")

    args = parser.parse_args()

    if args.video_dir or args.cache_dir:
        _process_video_dir(args)
    elif args.input_video:
        print("Function not yet implemented")
        sys.exit(1)
    elif args.input_list:
        print("Function not yet implemented")
        sys.exit(1)
    else:
        _execute_command(_get_gui_cmd(args.debug) + _default_annotator_args(args))
