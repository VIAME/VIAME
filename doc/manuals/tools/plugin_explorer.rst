Plugin Explorer
===============

Plugin explorer is the tool to use to explore the available
plugins. Since kwiver relies heavily on dynamically loaded content
through plugins, this tool is essential to determine what is available
and to help diagnose plugin problems.

The ``-h`` option is used to display the built in help for the command
line options. Before we delve into the full set of options, there are
two common uses: locating processes and locating algorithms. This can
be done with the ``--proc opt`` option, for processes of the ``--algo
opt`` option for algorithms. The ``opt`` argument can be ``all`` to
list all plugins of that type. If the option is not ``all``, then it
is interpreted as a regular-expression and all plugins of the selected
type that match are listed.

Processes
---------

For example, it you are looking for processes that provides input, you
could enter the following query that looks for any process with
'input' in its type.::

  $ plugin_explorer --proc input

Generates the following output::

  Plugins that implement type "sprokit::process"

  ---------------------
  Process type: frame_list_input
  Description:  Reads a list of image file names and generates stream of images and
                associated time stamps

  ---------------------
  Process type: detected_object_input
  Description:  Reads detected object sets from an input file. Detections read from the
                input file are grouped into sets for each image and individually returned.

  ---------------------
  Process type: video_input
  Description:  Reads video files and produces sequential images with metadata per frame.

  ---------------------
  Process type: track_descriptor_input
  Description:  Reads track descriptor sets from an input file.

  ---------------------
  Process type: input_adapter
  Description:  Source process for pipeline. Pushes data items into pipeline ports. Ports
                are dynamically created as needed based on connections specified in the
                pipeline file.

After you have determined which process meets your needs, more
detailed information can be displayed by adding the ``-d`` option.::

  $ plugin_explorer --proc video_input -d

  Plugins that implement type "sprokit::process"

  ---------------------
  Process type: video_input
  Description:       Reads video files and produces sequential images with metadata per frame.

    Properties: _no_reentrant

    -- Configuration --
    Name       : frame_time
    Default    : 0.03333333
    Description: Inter frame time in seconds. If the input video stream does not supply
                 frame times, this value is used to create a default timestamp. If the
                 video stream has frame times, then those are used.
    Tunable    : no

    Name       : video_filename
    Default    :
    Description: Name of video file.
    Tunable    : no

    Name       : video_reader
    Default    :
    Description: Name of video input algorithm.  Name of the video reader algorithm plugin
                 is specified as video_reader:type = <algo-name>
    Tunable    : no

  -- Input ports --
    No input ports

  -- Output ports --
    Name       : image
    Data type  : kwiver:image
    Flags      :
    Description: Single frame image.

    Name       : timestamp
    Data type  : kwiver:timestamp
    Flags      :
    Description: Timestamp for input image.

    Name       : metadata
    Data type  : kwiver:metadata
    Flags      :
    Description: Video metadata vector for a frame.

The detailed information display shows the configuration parameters,
input and output ports.

Algorithms
----------

Algorithms can be querried in a similar manner. The algorithm query
lists all implementations for the selected algorithm type. We can bet
a list brief list of all algorithm type names that contain "input by
using the following command.::

  $ plugin_explorer --algo input -b

  Plugins that implement type "detected_object_set_input"
      Algorithm type: detected_object_set_input   Implementation: kw18
      Algorithm type: detected_object_set_input   Implementation: csv

  Plugins that implement type "video_input"
      Algorithm type: video_input   Implementation: filter
      Algorithm type: video_input   Implementation: image_list
      Algorithm type: video_input   Implementation: pos
      Algorithm type: video_input   Implementation: split
      Algorithm type: video_input   Implementation: vidl_ffmpeg

You can see that two algorithm types were found and their different
implementations are listed. We can further examine what
implementations are available for the "video_input" with the folloeing command.::

  $ plugin_explorer --algo video_input

The result is a brief listing of all algorithms that implement the
"video_input" algorithm.::

  Plugins that implement type "video_input"

  ---------------------
  Info on algorithm type "video_input" implementation "filter"
    Plugin name: filter      Version: 1.0
        A video input that calls another video input and filters the output on
        frame range and other parameters.

  ---------------------
  Info on algorithm type "video_input" implementation "image_list"
    Plugin name: image_list      Version: 1.0
        Read a list of images from a list of file names and presents them in the
        same way as reading a video.  The actual algorithm to read an image is
        specified in the "image_reader" config block.  Read an image list as a
        video stream.

  ---------------------
  Info on algorithm type "video_input" implementation "pos"
    Plugin name: pos      Version: 1.0
        Read video metadata in AFRL POS format. The algorithm takes configuration
        for a directory full of images and an associated directory name for the
        metadata files. These metadata files have the same base name as the image
        files. Each metadata file is associated with the image file.

  ---------------------
  Info on algorithm type "video_input" implementation "split"
    Plugin name: split      Version: 1.0
        Coordinate two video readers. One reader supplies the image/data stream.
        The other reader supplies the metadata stream.

  ---------------------
  Info on algorithm type "video_input" implementation "vidl_ffmpeg"
    Plugin name: vidl_ffmpeg      Version: 1.0
        Use VXL (vidl with FFmpeg) to read video files as a sequence of images.

A detailed description of an algorithm can be generated by adding the
``-d`` option to the command line. The detailed output for one of the
algorithms is shown below::

  ---------------------
  Info on algorithm type "video_input" implementation "vidl_ffmpeg"
  Plugin name: vidl_ffmpeg      Version: 1.0
      Use VXL (vidl with FFmpeg) to read video files as a sequence of images.
    -- Configuration --
    "absolute_time_source" = "none"
    Description:       List of sources for absolute frame time information. This entry specifies
      a comma separated list of sources that are tried in order until a valid
      time source is found. If an absolute time source is found, it is used in
      the output time stamp. Absolute times are derived from the metadata in the
      video stream. Valid source names are "none", "misp", "klv0601", "klv0104".
      Where:
          none - do not supply absolute time
          misp - use frame embedded time stamps.
          klv0601 - use klv 0601 format metadata for frame time
          klv0104 - use klv 0104 format metadata for frame time
      Note that when "none" is found in the list no further time sources will be
      evaluated, the output timestamp will be marked as invalid, and the
      HAS_ABSOLUTE_FRAME_TIME capability will be set to false.  The same
      behavior occurs when all specified sources are tried and no valid time
      source is found.

    "start_at_frame" = "0"
    Description:       Frame number (from 1) to start processing video input. If set to zero,
      start at the beginning of the video.

    "stop_after_frame" = "0"
    Description:       Number of frames to supply. If set to zero then supply all frames after
      start frame.

    "time_scan_frame_limit" = "100"
    Description:       Number of frames to be scanned searching input video for embedded time. If
      the value is zero, the whole video will be scanned.

Other Plugin Types
------------------

A summary of all plugin types that are available can be displayed
using the ``--summary`` command line option.::

  ----Summary of plugin types
    38 types of plugins registered.
        1 plugin(s) that create "sprokit::process_instrumentation"
        53 plugin(s) that create "sprokit::process"
        3 plugin(s) that create "sprokit::scheduler"
        1 plugin(s) that create "analyze_tracks"
        3 plugin(s) that create "bundle_adjust"
        5 plugin(s) that create "close_loops"
        1 plugin(s) that create "compute_ref_homography"
        1 plugin(s) that create "convert_image"
        11 plugin(s) that create "detect_features"
        1 plugin(s) that create "detected_object_filter"
        2 plugin(s) that create "detected_object_set_input"
        2 plugin(s) that create "detected_object_set_output"
        1 plugin(s) that create "draw_detected_object_set"
        1 plugin(s) that create "draw_tracks"
        1 plugin(s) that create "dynamic_configuration"
        2 plugin(s) that create "estimate_canonical_transform"
        1 plugin(s) that create "estimate_essential_matrix"
        2 plugin(s) that create "estimate_fundamental_matrix"
        2 plugin(s) that create "estimate_homography"
        1 plugin(s) that create "estimate_similarity_transform"
        9 plugin(s) that create "extract_descriptors"
        1 plugin(s) that create "feature_descriptor_io"
        2 plugin(s) that create "filter_features"
        1 plugin(s) that create "filter_tracks"
        1 plugin(s) that create "formulate_query"
        2 plugin(s) that create "image_io"
        2 plugin(s) that create "image_object_detector"
        1 plugin(s) that create "initialize_cameras_landmarks"
        5 plugin(s) that create "match_features"
        2 plugin(s) that create "optimize_cameras"
        1 plugin(s) that create "refine_detections"
        2 plugin(s) that create "split_image"
        1 plugin(s) that create "track_descriptor_set_output"
        1 plugin(s) that create "track_features"
        1 plugin(s) that create "train_detector"
        2 plugin(s) that create "triangulate_landmarks"
        5 plugin(s) that create "video_input"
    137 total plugins

This summary output can be used to get an overview of what algorithm
types are available.

A full list of the options
--------------------------

A full list of all program options can be displayed with the ``-h``
command line option.::

 $ plugin_explorer -h
 Usage for plugin_explorer
   Version: 1.1

  --algo opt        Display only algorithm type plugins. If type is specified
                    as "all", then all algorithms are listed. Otherwise, the
                    type will be treated as a regexp and only algorithm types
                    that match the regexp will be displayed.

  --algorithm opt   Display only algorithm type plugins. If type is specified
                    as "all", then all algorithms are listed. Otherwise, the
                    type will be treated as a regexp and only algorithm types
                    that match the regexp will be displayed.

  --all             Display all plugin types

  --attrs           Display raw attributes for plugins without calling any
                    category specific formatting

  --brief           Generate brief display

  --detail          Display detailed information about plugins

  --fact opt        Only display factories whose interface type matches
                    specified regexp

  --factory opt     Only display factories whose interface type matches
                    specified regexp

  --files           Display list of loaded files

  --filter opts     Filter factories based on attribute name and value. Only
                    two fields must follow: <attr-name> <attr-value>

  --fmt opt         Generate display using alternative format, such as 'rst' or
                    'pipe'

  --help            Display usage information

  --hidden          Display hidden properties and ports

  --load opt        Load only specified plugin file for inspection. No other
                    plugins are loaded.

  --mod             Display list of loaded modules

  --path            Display plugin search path

  --proc opt        Display only sprokit process type plugins. If type is
                    specified as "all", then all processes are listed.
                    Otherwise, the type will be treated as a regexp and only
                    processes names that match the regexp will be displayed.

  --process opt     Display only sprokit process type plugins. If type is
                    specified as "all", then all processes are listed.
                    Otherwise, the type will be treated as a regexp and only
                    processes names that match the regexp will be displayed.

  --scheduler       Displat scheduler type plugins

  --sep-proc-dir opt  Generate .rst output for processes as separate files in
                      specified directory.

  --summary         Display summary of all plugin types

  --type opt        Only display factories whose instance name matches the
                    specified regexp

  --version         Display program version

  -I opt            Add directory to plugin search path

  -b                Generate brief display

  -d                Display detailed information about plugins

  -h                Display usage information

  -v                Display program version


Debugging the Plugin Loading Process
------------------------------------

There are times when an expected plugin is not being found. The
plugin_explorer provides several options to assist in determining what
may be the problem. A plugin file may contain more than one plugin. A
common problem with loading plugins which results in the plugin not
being loaded is unresolved external references. A warning message is
displayed when the program starts indicating a problem loading a
plugin and indicating the problem. In this case, the plugin is not
loaded and not available for use.

The ``--files`` option is used to display a list of all plugin files
that have been found and successfully loaded.::

 $ plugin_explorer --files

 ---- Files Successfully Opened
  /disk2/projects/KWIVER/build/lib/modules/instrumentation_plugin.so
  /disk2/projects/KWIVER/build/lib/modules/kwiver_algo_ceres_plugin.so
  /disk2/projects/KWIVER/build/lib/modules/kwiver_algo_core_plugin.so
  /disk2/projects/KWIVER/build/lib/modules/kwiver_algo_darknet_plugin.so
  /disk2/projects/KWIVER/build/lib/modules/kwiver_algo_ocv_plugin.so
  /disk2/projects/KWIVER/build/lib/modules/kwiver_algo_proj_plugin.so
  /disk2/projects/KWIVER/build/lib/modules/kwiver_algo_vxl_plugin.so
  /disk2/projects/KWIVER/build/lib/sprokit/kwiver_processes.so
  /disk2/projects/KWIVER/build/lib/sprokit/kwiver_processes_adapter.so
  /disk2/projects/KWIVER/build/lib/sprokit/kwiver_processes_ocv.so
  /disk2/projects/KWIVER/build/lib/sprokit/kwiver_processes_vxl.so
  /disk2/projects/KWIVER/build/lib/sprokit/modules_python.so
  /disk2/projects/KWIVER/build/lib/sprokit/processes_clusters.so
  /disk2/projects/KWIVER/build/lib/sprokit/processes_examples.so
  /disk2/projects/KWIVER/build/lib/sprokit/processes_flow.so
  /disk2/projects/KWIVER/build/lib/sprokit/schedulers.so
  /disk2/projects/KWIVER/build/lib/sprokit/schedulers_examples.so
  /disk2/projects/KWIVER/build/lib/sprokit/template_processes.so

If a file was expected to be loaded and is not in the list, then it is
possible that the directory containing the file was not in the loading
path. The set of directories that are scanned for loadable plugins can
be displayed with the ``path`` command line option.::

 $ plugin_explorer --path

 ---- Plugin search path
    /disk2/projects/KWIVER/build/lib/modules
    /disk2/projects/KWIVER/build/lib/sprokit

    /usr/local/lib/sprokit
    /usr/local/lib/modules

Additional directories can be supplied to the plugin_explorer using
the ``-I dir`` command line option.
