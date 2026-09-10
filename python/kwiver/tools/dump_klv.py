import argparse
import sys
import os
from pathlib import Path
import threading

from kwiver.vital import plugin_management
from kwiver.vital.types import Timestamp, Metadata, SimpleMetadataMap
from kwiver.vital.config import ConfigBlockFormatter, read_config_file
from kwiver.vital.algo import VideoInput, MetadataMapIO, ImageIO
from kwiver.vital.types.metadata_traits import tag_traits_by_tag
from kwiver.vital.io import basename_from_metadata
import kwiver
from importlib.metadata import version

DEFAULT_CONFIG_PREFIX = (
    Path(kwiver.__file__).parents[0] / "share" / "kwiver" / version("kwiver") / "config"
)


def add_command_options():
    parser = argparse.ArgumentParser(
        prog="dump_klv",
        description="""[options]  video-file


     This program displays the KLV metadata packets that are embedded
     in a video file.""",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Configuration file for tool",
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Dump configuration to file and exit",
        type=argparse.FileType("w"),
    )
    parser.add_argument(
        "-l",
        "--log",
        help="""Log metadata to a file. This requires the JSON serialization plugin.
      The file is structured as an array of frames where each frame contains an array
      of metadata packets associated with that frame. Each packet is an
      array of metadata fields. Alternatively, the configuration file,
      dump_klv.conf, can be updated to use CSV instead.""",
        type=argparse.FileType("w"),
    )
    parser.add_argument(
        "-f",
        "--frames",
        help="Dump frames into the given image format.",
    )
    parser.add_argument(
        "--frames-dir",
        help="""Directory in which to dump frames.
      Defaults to current directory.""",
    )
    parser.add_argument(
        "-d",
        "--detail",
        help="Display a detailed description of the metadata",
        action="store_true",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help="Do not show metadata. Overrides -d/--detail.",
        action="store_true",
    )
    parser.add_argument(
        "-e",
        "--exporter",
        help="""Choose the format of the exported KLV data.
      Current options are: csv, json, klv-json.""",
        choices=["csv", "json", "klv-json"],
    )
    parser.add_argument(
        "--multithread",
        help="""Use multithreading to accelerate encoding of frame images.
      Number of worker threads is not configurable at this time.""",
        action="store_true",
    )
    parser.add_argument(
        "--compress",
        help="Compress output file. Only available for klv-json.",
        action="store_true",
    )
    # positional parameters
    parser.add_argument("video_file", help="Video input file", metavar="video-file")

    return parser


def run():
    vpm = plugin_management.plugin_manager_instance()
    vpm.load_all_plugins()
    parser = add_command_options()
    cmd_args = parser.parse_args()

    video_file: str = ""
    print(cmd_args)

    if cmd_args.video_file:
        video_file = cmd_args.video_file
    else:
        print("Missing video file name")
        parser.print_help()
        exit(1)

    video_reader = VideoInput()
    metadata_serializer = MetadataMapIO()
    image_writer = ImageIO()

    try:
        # the path exists when installed as a wheel
        config_path = DEFAULT_CONFIG_PREFIX / "applets" / "dump_klv.conf"
        config = read_config_file(str(config_path))
    except RuntimeError:
        # look into the parent directory for configuration files
        # this is the layout in the build/install tree
        prefix = os.path.dirname(sys.executable) + "/.."
        config = read_config_file("applets/dump_klv.conf", "", "", prefix)

    # If --config given, read in config file, merge in with default just generated
    if cmd_args.config:
        config.merge_config(read_config_file(cmd_args.config.name))
    print(f"CONFIG #### {config}")
    fmt = ConfigBlockFormatter(config)
    fmt.print(sys.stdout)

    # Output file extension configures exporter
    if cmd_args.log and cmd_args.exporter is None:
        # Is this correct ? if could be config instead of cmd_args!cmd_args.count( "metadata_serializer:type" ) &&
        filename = cmd_args.log.name
        _, extension = os.path.splitext(filename)
        if len(extension) > 0:
            EXTENSION_MAP = {
                ".JSON": "json",
                ".json": "json",
                ".CSV": "csv",
                ".csv": "csv",
            }
            serializer_type = EXTENSION_MAP.get(extension, "csv")
            config["metadata_serializer:type"] = serializer_type

    if cmd_args.exporter:
        config["metadata_serializer:type"] = cmd_args.exporter

    if cmd_args.compress:
        config["metadata_serializer:klv-json:compress"] = True

    video_reader = VideoInput.set_nested_algo_configuration("video_reader", config)
    VideoInput.get_nested_algo_configuration("video_reader", config, video_reader)

    metadata_serializer = MetadataMapIO.set_nested_algo_configuration(
        "metadata_serializer", config
    )
    MetadataMapIO.get_nested_algo_configuration(
        "metadata_serializer", config, metadata_serializer
    )

    if cmd_args.frames:
        image_writer = ImageIO.set_nested_algo_configuration("image_writer", config)
        ImageIO.get_nested_algo_configuration("image_writer", config, image_writer)

    # Check to see if we are to dump config
    if cmd_args.output:
        fmt = ConfigBlockFormatter(config)
        fmt.print(cmd_args.output)
        print(f"Wrote config to {cmd_args.output.name}. Exiting.")
        exit(0)
    if not VideoInput.check_nested_algo_configuration("video_reader", config):
        raise RuntimeError("Invalid video_reader config")

    if not MetadataMapIO.check_nested_algo_configuration("metadata_serializer", config):
        raise RuntimeError("Invalid metadata_serializer config")

    if cmd_args.frames and not ImageIO.check_nested_algo_configuration(
        "image_writer", config
    ):
        raise RuntimeError("Invalid image_writer config")

    video_reader.open(video_file)

    count = 1
    # kv::wrap_text_block wtb;
    frame_metadata = dict()
    image_write_threads = []

    # wtb.set_indent_string( "    " );

    # Avoid repeated dictionary access
    detail = cmd_args.detail
    quiet = cmd_args.quiet
    log = cmd_args.log is not None

    while video_reader.next_frame():
        ts = video_reader.frame_timestamp()
        if not quiet:
            print(f"========== Read frame {ts.get_frame()} (index {count}) ==========")

        metadata = video_reader.frame_metadata()

        if log:
            # Add the (frame number, vector of metadata packets) item
            frame_metadata[ts.get_frame()] = metadata

        if not quiet:
            for meta in metadata:
                print(f"\n\n---------------- Metadata from: {meta.timestamp}")
                print(type(meta))
                if detail:
                    for ix in meta:
                        name = ix[1].name
                        tag = ix[1].tag
                        descrip = tag_traits_by_tag(tag).description()
                        print(
                            f"""Metadata item: {name}
            {descrip}
            Data: < {ix[1].type} >: {Metadata.format_string(ix[1].as_string())}
            """
                        )
                else:
                    Metadata.print_metadata(sys.stdout, meta)

        if cmd_args.frames:
            directory = "."
            if cmd_args.frames_dir:
                directory = cmd_args.frames_dir
            name = basename_from_metadata(metadata, int(ts.get_frame()))
            extension = cmd_args.frames
            filename = name + "." + extension
            filepath = os.path.join(directory, filename)
            image = video_reader.frame_image()

            def write_image_task(filepath, image):
                image_writer.save(filepath, image)

            if cmd_args.multithread:
                thread = threading.Thread(
                    target=write_image_task, args=(filepath, image)
                )
                image_write_threads.append(thread)
                thread.start()
            else:
                write_image_task(filepath, image)

        count += 1
    # end while over video
    if log:
        mms = SimpleMetadataMap(frame_metadata)
        metadata_serializer.save(cmd_args.log.name, data=mms)
        print(f'Wrote KLV log to "{cmd_args.log}".')

    for t in image_write_threads:
        t.join()


if __name__ == "__main__":
    run()
