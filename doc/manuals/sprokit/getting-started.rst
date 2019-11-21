.. highlight:: none

Getting Started with Sprokit
============================

In computer vision applications, the interaction between  data structures
(expressed in KWIVER as VITAL types) and algorithms (expressed in KWIVER
as Arrows)  can frequently be expressed as a pipeline of processing steps:

#. Input processing to load images or video
#. Manipulation and/or analysis of the imagery
#. Output of resulting imagery and/or analytics in a useful format

A graphical representation of such a pipeline might look something
like this:

.. _processingpipelineblock:
.. figure:: /_images/processing_pipeline.png
   :align: center

The Manipulation/Analysis step, step 2, might be a collection of processing
operations that build on one another to achieve some result, such as repsented
in this graphical depiction of a more elaborate pipeline:

.. _complexpipelineblock:
.. figure:: /_images/complex_pipeline.png
   :align: center

Because this type of processing architecture is so common, KWIVER includes a
data flow architecture called Sprokit. Sprokit is a dynamic
pipeline configuration and execution framework that combines all of KWIVER’s
other components --  VITAL types, Configuration blocks, and Arrows to create a powerful,
dynamic system for expressing processing pipelines that address computer vision
problems.

Sprokit pipelines consist of a series of Sprokit processes that are connected
together through “ports” over which various VITAL types flow. A Sprokit
pipeline can be a straightforward sequence of steps as shown in the first
pipline Figure or can consist of many steps arranged with various branches into
a more sophisticated processing system as shown in the second pipeline figure.

A key benefit of Sprokit is that it provides algorithm-independent support for
system engineering issues. Much of the difficulty translating a system such as
the first figure from a conceptually simple diagram into a functioning system lies in
the mundane issues of data transport, buffering, synchronization, and error
handling. By providing a common representation for data (via VITAL) and
processing steps (via Sprokit), KWIVER allows the developer to focus on
foundational algorithmic research subject to the constraint of a well-defined
system interface.

Sprokit Pipeline Example
------------------------

The easiest way to understand Sprokit is to work through an example of building
and executing a pipeline using existing KWIVER Arrows.  For this example, we
will filter object detections using the confidence scores associated with the
detection and then write them back to disk. The pipeline accepts a collection
of bounding boxes as inputs. Every bounding box is characterized by the
coordinates for the box, the confidence score, and a class type. The VITAL type
detected_object_sets is used to represent these bounding boxes in the pipeline.
The plugin_explorer application can be used to help construct the pipeline.
After the pipeline is defined it can then be executed using kwiver runner.
Note that during this entire exercise no code is written or compiled.

Input
'''''

The first step is to define where the inputs come from and where they are
going.  We'll use KWIVER's ``plugin_explorer`` application to identify the
processes that we want.  The following command::

	plugin_explorer --proc all --brief

Generates the following output (abbreviated for clarity in this document)::

	.
	.
	.
	Process type: image_filter         			Apply selected image filter algorithm to incoming images.
	Process type: image_writer         			Write image to disk.
	Process type: image_file_reader         Reads an image file given the file name.
	Process type: detected_object_input     Reads detected object sets from an input file.
		Detections read from the input file are grouped into sets for each image
		and individually returned.
	Process type: detected_object_output    Writes detected object sets to an output file.
		All detections are written to the same file.
	Process type: detected_object_filter    Filters sets of detected objects using the detected_object_filter
		algorithm.
	Process type: video_input         			Reads video files and produces sequential images with metadata per frame.
	.
	.
	.

We see ``detected_object_input`` and use the following command::

	plugin_explorer --proc detected_object_input --detail

To get the following, more detailed information about ``detected_object_input``::


	Process type: detected_object_input
		Description:       Reads detected object sets from an input file.

				Detections read from the input file are grouped into sets for each image
				and individually returned.

			Properties: _no_reentrant

			-- Configuration --
			Name       : file_name
			Default    :
			Description:       Name of the detection set file to read.
			Tunable    : no

			Name       : reader
			Default    :
			Description:       Algorithm type to use as the reader.
			Tunable    : no

		-- Input ports --
			No input ports

		-- Output ports --
			Name       : detected_object_set
			Data type  : kwiver:detected_object_set
			Flags      :
			Description: Set of detected objects.

			Name       : image_file_name
			Data type  : kwiver:image_file_name
			Flags      :
			Description: Name of an image file. The file name may contain leading path components.


What this tells us is that

#. There is a ``detected_object_input`` process that takes a	``file_name`` and a ``reader``
   (more on that in a moment) as a configuration parameter,
#. That it has no input ports
#. That it produces a ``detected_object_set`` and an ``image_file_name`` on its output
   ports when it runs.

The *ports* in a process are the points at which one process can connect to
another.  Input ports of one type can be connected to output ports of the same
type from a an earlier process in the pipeline.  This particular process is
referred to as an *end cap*, specifcally an *input end cap* for the pipeline.
This is because it’s function is to load data external to the Sprokit pipeline
(for example from a CSV file)  and present it for processing on the Sprokit
pipeline.  Similarly, *output end caps* would have no output ports but would
convert their input data to some form external to the Sprokit pipeline.

Of particular interest is the ``reader`` parameter, which lets us select the
particular arrow that we want to use to obtain our detected_object_set for
reading.

We can use the following ``plugin_explorer`` command to see what is available
for the configuration parameter::

	plugin_explorer --algorithm detected_object_set_input --detail

Which results in the following output::

	Plugins that implement type "detected_object_set_input"
	---------------------
	Info on algorithm type "detected_object_set_input" implementation "csv"
		Plugin name: csv      Version: 1.0
				Detected object set reader using CSV format.

				 - 1: frame number
				 - 2: file name
				 - 3: TL-x
				 - 4: TL-y
				 - 5: BR-x
				 - 6: BR-y
				 - 7: confidence
				 - 8,9: class-name, score (this pair may be omitted or may repeat any
				number of times)

			-- Configuration --
	---------------------
	Info on algorithm type "detected_object_set_input" implementation "kw18"
		Plugin name: kw18      Version: 1.0
				Detected object set reader using kw18 format.

					- Column(s) 1: Track-id
					- Column(s) 2: Track-length (number of detections)
					- Column(s) 3: Frame-number (-1 if not available)
					- Column(s) 4-5: Tracking-plane-loc(x,y) (could be same as World-loc)
					- Column(s) 6-7: Velocity(x,y)
					- Column(s) 8-9: Image-loc(x,y)
					- Column(s) 10-13: Img-bbox(TL_x,TL_y,BR_x,BR_y) (location of top-left &
				bottom-right vertices)
					- Column(s) 14: Area
					- Column(s) 15-17: World-loc(x,y,z) (longitude, latitude, 0 - when
				available)
					- Column(s) 18: Timesetamp (-1 if not available)
					- Column(s) 19: Track-confidence (-1 if not available)

			-- Configuration --
	---------------------
	Info on algorithm type "detected_object_set_input" implementation "simulator"
		Plugin name: simulator      Version: 1.0
				Detected object set reader using SIMULATOR format.

				Detection are generated algorithmicly.
			-- Configuration --
			"center_x" = "100"
			Description:       Bounding box center x coordinate.

			"center_y" = "100"
			Description:       Bounding box center y coordinate.

			"detection_class" = "detection"
			Description:       Label for detection detected object type

			"dx" = "0"
			Description:       Bounding box x translation per frame.

			"dy" = "0"
			Description:       Bounding box y translation per frame.

			"height" = "200"
			Description:       Bounding box height.

			"max_sets" = "10"
			Description:       Number of detection sets to generate.

			"set_size" = "4"
			Description:       Number of detection in a set.

			"width" = "200"
			Description:       Bounding box width.

	---------------------
	Info on algorithm type "detected_object_set_input" implementation "kpf_input"
		Plugin name: kpf_input      Version: 1.0
				Detected object set reader using kpf format.
			-- Configuration --


As we can see, we have a number of choices including a CSV reader, a simulator,
and some others. For this example we’ll use the CSV reader when we construct
the pipeline.

Filter
''''''

Similarly, we can look at filters for ``detected_object_sets``::

	plugin_explorer --proc detected_object_input --detail

Which gives us::

	Process type: detected_object_filter
	Description:       Filters sets of detected objects using the detected_object_filter
			algorithm.

		Properties: _no_reentrant

		-- Configuration --
		Name       : filter
		Default    :
		Description:       Algorithm configuration subblock.
		Tunable    : no

	-- Input ports --
		Name       : detected_object_set
		Data type  : kwiver:detected_object_set
		Flags      : _required
		Description: Set of detected objects.

	-- Output ports --
		Name       : detected_object_set
		Data type  : kwiver:detected_object_set
		Flags      :
		Description: Set of detected objects.

And the associated Arrows::

	Plugins that implement type "detected_object_filter"
	---------------------
	Info on algorithm type "detected_object_filter" implementation "class_probablity_filter"
		Plugin name: class_probablity_filter      Version: 1.0
				Filters detections based on class probability.

				This algorithm filters out items that are less than the threshold. The
				following steps are applied to each input detected object set.

				1) Select all class names with scores greater than threshold.

				2) Create a new detected_object_type object with all selected class names
				from step 1. The class name can be selected individually or with the
				keep_all_classes option.

				3) The input detection_set is cloned and the detected_object_type from
				step 2 is attached.
			-- Configuration --
			"keep_all_classes" = "true"
			Description:       If this options is set to true, all classes are passed through this filter
				if they are above the selected threshold.

			"keep_classes" = ""
			Description:       A list of class names to pass through this filter. Multiple names are
				separated by a ';' character. The keep_all_classes parameter overrides
				this list of classes. So be sure to set that to false if you only want the
				listed classes.

			"threshold" = "0"
			Description:       Detections are passed through this filter if they have a selected
				classification that is above this threshold.

We will use the class_probability_filter to only pass detections from all
classes that are above a confidence value that we'll set in our pipeline
configuration file.

Output
''''''

Finally, we will select our output process, which has the following definition::

 Process type: detected_object_output
  Description:       Writes detected object sets to an output file.

      All detections are written to the same file.

    Properties: _no_reentrant

    -- Configuration --
    Name       : file_name
    Default    :
    Description:       Name of the detection set file to write.
    Tunable    : no

    Name       : writer
    Default    :
    Description:       Block name for algorithm parameters. e.g. writer:type would be used to
      specify the algorithm type.
    Tunable    : no

  -- Input ports --
    Name       : detected_object_set
    Data type  : kwiver:detected_object_set
    Flags      : _required
    Description: Set of detected objects.

    Name       : image_file_name
    Data type  : kwiver:image_file_name
    Flags      :
    Description: Name of an image file. The file name may contain leading path components.

  -- Output ports --

This output process accepts a detected_object_set and image_file_name as input
and writes out the result. We will look at our selection of arrows that we
could use::

	Plugins that implement type "detected_object_set_output"
	---------------------
	Info on algorithm type "detected_object_set_output" implementation "csv"
		Plugin name: csv      Version: 1.0
				Detected object set writer using CSV format.

				 - 1: frame number
				 - 2: file name
				 - 3: TL-x
				 - 4: TL-y
				 - 5: BR-x
				 - 6: BR-y
				 - 7: confidence
				 - 8,9: class-name, score (this pair may be omitted or may repeat any
				number of times)

			-- Configuration --
	---------------------
	Info on algorithm type "detected_object_set_output" implementation "kw18"
		Plugin name: kw18      Version: 1.0
				Detected object set writer using kw18 format.

					- Column(s) 1: Track-id
					- Column(s) 2: Track-length (number of detections)
					- Column(s) 3: Frame-number (-1 if not available)
					- Column(s) 4-5: Tracking-plane-loc(x,y) (could be same as World-loc)
					- Column(s) 6-7: Velocity(x,y)
					- Column(s) 8-9: Image-loc(x,y)
					- Column(s) 10-13: Img-bbox(TL_x,TL_y,BR_x,BR_y) (location of top-left &
				bottom-right vertices)
					- Column(s) 14: Area
					- Column(s) 15-17: World-loc(x,y,z) (longitude, latitude, 0 - when
				available)
					- Column(s) 18: Timestamp (-1 if not available)
					- Column(s) 19: Track-confidence (-1 if not available)

			-- Configuration --
			"tot_field1_ids" = ""
			Description:       Comma separated list of ids used for TOT field 1.

			"tot_field2_ids" = ""
			Description:       Comma separated list of ids used for TOT field 2.

			"write_tot" = "false"
			Description:       Write a file in the vpView TOT format alongside the computed tracks.
	---------------------
	Info on algorithm type "detected_object_set_output" implementation "kpf_output"
		Plugin name: kpf_output      Version: 1.0
				Detected object set writer using kpf format.t
			-- Configuration --


In this case, we’ll select the DIVA KPF writer when we assemble our pipeline.

Pipeline
''''''''

A text file is used to construct the pipeline processes, their input and output
port connections, and the configuration parameters.

We'll construct a pipeline that has the following structure based on the information
we obtained from using ``plugin_explorer``::



Here is the pipeline file that configures our selected input, filter, and
output:

.. _basicpipelineblock:
.. figure:: /_images/sprokit_basic_pipeline.png
	 :align: center

Which can be represented by the follwing pipeline file::

	# --------------------------------------------------
	process reader :: detected_object_input
					file_name = sample_detected_objects.csv
					reader:type = csv

	# --------------------------------------------------
	process filter :: detected_object_filter
					filter:type = class_probablity_filter
					filter:threshold = .5

	connect from reader.detected_object_set to filter.detected_object_set

	# --------------------------------------------------
	process writer :: detected_object_output
					file_name = sample_filtered_detected_objects.kpf
					writer:type = kpf

	connect from filter.detected_object_set to writer.detected_object_set

In this pipeline file we define three processes: reader, filter, and writer. We
connect the detected_object_set output of reader to the detected_object_set
input of filter. We configure filter to only pass detected_objects with a
confidence above a threshold of 0.5 and then we pass its detected_object_set
output port to our writer processes’ input port. We select a KPF writer for our
writer process.

We can run the pipeline with the following command::

	kwiver runner sample_reader_filter_writer.pipe

When the pipeline runs it will read a set of detected_objects from the file
sample_detected_objects.csv, filter out any that have a confidence less than
50%, and then write the remainder to a KPF file for further processing, etc.


PythonsProcesses
----------------

One of KWIVER's great strengths (as provided by sprokit) is the ability to
create hybrid pipelines which combine C++ and Python processes in the same
pipeline.  This greatly facilitates prototyping complex processing pipelines.
To test this out we'll  use a simple process called ``numbers`` which simply
generates numbers on a Sprokit port.  We'll also use a simple Python process
that prints the number  called ``kw_print_number_process`` the code for which
can be seen in
[sprokit/processes/python/kw_print_number_process.py](sprokit/processes/python/kw_print_number_process.py).

As usual, we can lean about this process with the following command::

  plugin_explorer --proc kw_print_number_process -d

Which produces the following output::

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

In order to get around limitations imposed by the Python Global
Interpreter Lock, we'll use a different Sprokit scheduler for this
pipeline.  The ``pythread_per_process`` scheduler which does essentially
what it says: it creates a Python thread for every process in the
pipeline::

	kwiver runner -S pythread_per_process </path/to/kwiver/source>/sprokit/pipelines/number_flow_python.pipe>

The previous pipeline, the numbers will be written to an output file,
this time ``numbers_from_python.txt``
