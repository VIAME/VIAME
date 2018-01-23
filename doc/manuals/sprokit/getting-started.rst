Getting Started with sprokit
============================

The central component of KWIVER is vital which supplies basic data
types and fundimental alrogithms.  In addition, we use sprokit's
pipelining facilities to manage, integrate and run many of KWIVER's
modules and capabilities.  To see what modules (called processes in
sprockit) are available, run the following command::

    $ plugin_explorer --proc all -b

Here's a typical list of modules (note that as KWIVER expands, this
list is likely to grow):

---- All process Factories

Factories that create type "sprokit::process"
    Process type: frame_list_input          Reads a list of image file names and generates stream of images and
       associated time stamps

    Process type: stabilize_image          Generate current-to-reference image homographies

    Process type: detect_features          Detect features in an image that will be used for stabilization

    Process type: extract_descriptors          Extract descriptors from detected features

    Process type: feature_matcher          Match extracted descriptors and detected features

    Process type: compute_homography          Compute a frame to frame homography based on tracks

    Process type: compute_stereo_depth_map          Compute a stereo depth map given two frames

    Process type: draw_tracks          Draw feature tracks on image

    Process type: read_d_vector          Read vector of doubles

    Process type: refine_detections          Refines detections for a given frame

    Process type: image_object_detector          Apply selected image object detector algorithm to incoming images.

    Process type: image_filter          Apply selected image filter algorithm to incoming images.

    Process type: image_writer          Write image to disk.

    Process type: image_file_reader          Reads an image file given the file name.

    Process type: detected_object_input          Reads detected object sets from an input file. Detections read from the
       input file are grouped into sets for each image and individually
       returned.

    Process type: detected_object_output          Writes detected object sets to an output file. All detections are written
       to the same file.

    Process type: detected_object_filter          Filters sets of detected objects using the detected_object_filter
       algorithm.

    Process type: video_input          Reads video files and produces sequential images with metadata per frame.

    Process type: draw_detected_object_set          Draws border around detected objects in the set using the selected
       algorithm.

    Process type: track_descriptor_input          Reads track descriptor sets from an input file.

    Process type: track_descriptor_output          Writes track descriptor sets to an output file. All descriptors are
       written to the same file.

    Process type: image_viewer          Display input image and delay

    Process type: draw_detected_object_boxes          Draw detected object boxes on images.

    Process type: collate          Collates data from multiple worker processes

    Process type: distribute          Distributes data to multiple worker processes

    Process type: pass          Pass a data stream through

    Process type: sink          Ignores incoming data

    Process type: any_source          A process which creates arbitrary data

    Process type: const          A process wth a const flag

    Process type: const_number          Outputs a constant number

    Process type: data_dependent          A process with a data dependent type

    Process type: duplicate          A process which duplicates input

    Process type: expect          A process which expects some conditions

    Process type: feedback          A process which feeds data into itself

    Process type: flow_dependent          A process with a flow dependent type

    Process type: multiplication          Multiplies numbers

    Process type: multiplier_cluster          A constant factor multiplier cluster

    Process type: mutate          A process with a mutable flag

    Process type: numbers          Outputs numbers within a range

    Process type: orphan_cluster          A dummy cluster

    Process type: orphan          A dummy process

    Process type: print_number          Print numbers to a file

    Process type: shared          A process with the shared flag

    Process type: skip          A process which skips input data

    Process type: tagged_flow_dependent          A process with a tagged flow dependent types

    Process type: take_number          Print numbers to a file

    Process type: take_string          Print strings to a file

    Process type: tunable          A process with a tunable parameter

    Process type: input_adapter          Source process for pipeline. Pushes data items into pipeline ports. Ports
       are dynamically created as needed based on connections specified in the
       pipeline file.

    Process type: output_adapter          Sink process for pipeline. Accepts data items from pipeline ports. Ports
       are dynamically created as needed based on connections specified in the
       pipeline file.

    Process type: template          Description of process. Make as long as necessary to fully explain what
       the process does and how to use it. Explain specific algorithms used,
       etc.

    Process type: kw_archive_writer          Writes kw archives

    Process type: test_python_process          A test Python process

    Process type: pyprint_number          A Python process which prints numbers

This is the list of modules that can be included in a Sprokit
pipeline.  We're going to use the ``numbers`` module and the the
``print_number`` module to create a very simple pipeline.  To learn more
about the ``numbers`` module we'll again use ``plugin_explorer`` this time
to get details on a particular module.  For ``numbers`` we'll use the
following command::

    $ plugin_explorer --process --type numbers -d --config

    Factories that create type "sprokit::process"

      Process type: numbers
      Description:        Outputs numbers within a range

        Properties: _no_reentrant,
        -- Configuration --
        Name       : end
        Default    : 100
        Description: The value to stop counting at.
        Tunable    : no

        Name       : start
        Default    : 0
        Description: The value to start counting at.
        Tunable    : no

      Input ports:
      Output ports:
        Name       : number
        Type       : integer
        Flags      : _required,
        Description: Where the numbers will be available.

And for ``print_number``, we'll use::

    $ plugin_explorer --process --type print_number -d --config

    Factories that create type "sprokit::process"

      Process type: print_number
      Description:        Print numbers to a file

        Properties: _no_reentrant,
        -- Configuration --
        Name       : output
        Default    :
        Description: The path of the file to output to.
        Tunable    : no

      Input ports:
        Name       : number
        Type       : integer
        Flags      : _required,
        Description: Where numbers are read from.

      Output ports:


The output of these commands tells us enough about each process to
construct a Sprockit ".pipe" file that defines a processing pipeline.
In particular we'll need to know how to configure each process (the
"Configuration") and how they can be hooked together (the input and
output "Ports").

KWIVER comes with a sample
[sprokit/pipelines/number_flow.pipe](sprokit/pipelines/number_flow.pipe)
file that configures and connects the pipeline so that the ``numbers``
process will generate a set of integers from 1 to 99 and the
``print_number`` process will write those to a file called
``numbers.txt``.  Of particular interest is the section at the end of
the file that actually "hooks up" the pipeline.

To run the pipeline, we'll use the Sprokit ``pipeline_runner`` command::

    $ pipeline_runner -p </path/to/kwiver/source>/sprokit/pipelines/number_flow.pipe

After the pipeline completes, you should find a file, ``numbers.txt``, in your working directory.


Python Processes
----------------

One of KWIVER's great strengths (as provided by sprokit) is the
ability to create hybrid pipelines which combine C++ and Python
processes in the same pipeline.  This greatly facilitates prototyping
complex processing pipelines.  To test this out we'll still use the
``numbers`` process, but we'll use a Python version of the
``print_number`` process called ``kw_print_number_process`` the code for
which can be seen in
[sprokit/processes/python/kw_print_number_process.py](sprokit/processes/python/kw_print_number_process.py).
As usual, we can lean about this process with the following command::

    $ plugin_explorer --process --type kw_print_number_process -d --config

    Process type: kw_print_number_process
      Description: A Simple Kwiver Test Process
      Properties: _no_reentrant, _python
    Configuration:
      Name       : output
      Default    : .
      Description: The path for the output file.
      Tunable    : no

    Input ports:
      Name       : input
      Type       : integer
      Flags      : _required
      Description: Where numbers are read from.

    Output ports:

As you can see, the process is very similar to the C++ ``print_number``
process.  As a result, the [".pipe" file is very
similar](sprokit/pipelines/number_flow_python.pipe).

In order to get around limitations imposed by the Python Global
Interpreter Lock, we'll use a different Sprokit scheduler for this
pipeline.  The ``pythread_per_process`` scheduler which does essentially
what it says: it creates a Python thread for every process in the
pipeline::

	pipeline_runner -S pythread_per_process -p </path/to/kwiver/source>/sprokit/pipelines/number_flow_python.pipe>

As with the previous pipeline, the numbers will be written to an output file, this time ``numbers_from_python.txt``
