# KWIVER

Kitware Image and Video Exploitation and Retrieval

---

The KWIVER  toolkit is a collection of software tools designed to tackle challenging image and video analysis problems and other related challenges. Recently started by Kitware’s Computer Vision and Scientific Visualization teams, KWIVER is an ongoing effort to transition technology developed over multiple years to the open source domain to further research, collaboration, and product development.

The project is structured with the parent `kwiver` repository working as CMake "super-build" that pulls in a number of KWIVER and other open source projects.

## Building KWIVER

### Fletch

KWIVER has (and will have more) a number of dependencies on 3rd party Open Source libraries.  To make it easier to build KWIVER, especially on systems like Microsoft Windows that don't have package manager, [Fletch](https://github.com/Kitware/fletch) was developed to gather, configure and build those packages for use with KWIVER.  Fletch is a [CMake](www.cmake.org) based "super-build" that takes care of most of the build details for you.

To build Fletch, clone the Fletch repository:

	git clone https://github.com/Kitware/fletch.git

	git submodule update --init

Then, create a build directory and run the following `cmake` command:

	cmake -DFLETCH_BUILD_WITH_PYTHON:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Release -Dfletch_ENABLE_Boost:BOOL=TRUE -Dfletch_ENABLE_OpenCV:BOOL=TRUE  /path/to/fletch/source/directory

If you have more than one version of Python installed on your system and you want to be sure to use a particular one (for example we here at KWIVER development central use [Anaconda](https://store.continuum.io/cshop/anaconda/) fairly frequently) you'll want to add the following arguments to the `cmake` command:

* `-DPYTHON_INCLUDE_DIR=/path/to/python/include/directory`  For example, for a default Python 2.7 Anaconda install on Linux this would be `${HOME}/anaconda/include/python2.7`
* `-DPYTHON_EXECUTABLE=/path/to/executable/python` For example, for a default Python 2.7 Anaconda install on Linux this would `${HOME}/anaconda/bin/python`
* `-DPYTHON_LIBRARY=/path/to/python/library` For example, for a default Python 2.7 Anaconda install on Linux, this would be `${HOME}/anaconda/lib/libpython2.7.so`

Once your `cmake` command has completed, you can build with the following command

	make

### kwiver

Once Fletch has been built, it's possible to build the `kwiver` repository as well.  This repo is also a CMake super-build and can be fetched with this command:

	git clone https://github.com/Kitware/kwiver.git

The build can be configured with this command:

	cmake -DKWIVER_ENABLE_PYTHON:BOOL=ON -Dfletch_DIR:PATH=/path/to/fletch/build/directory /path/to/kwiver/source/directory

As with Fletch, if you want to specify a particular Python installation (such as Anaconda) use the the `-DPYTHON...` command arguments as outlined in the Fletch section.

Once your `cmake` command has completed, use `make` (on Linux) to build it.

## Running KWIVER

Once you've built KWIVER, you'll want to test that it's working on your system.  From a command prompt execute the following command:

	source </path/to/kwiver/build>/install/setup_KWIVER.sh

Where `</path/to/kwiver/build>` is the actual path of your KWIVER CMake build directory.

This will set up your PATH, PYTHONPATH and other environment variables to allow KWIVER to work conveniently.

The central component of KWIVER is [Sprokit](www.sprokit.org).  We use Sprokit's pipelining facilities to manage, integrate and run many of KWIVER's modules and capabilities.  To see what modules (called processes in Sprocket) are available, issue the `processopedia` command.  Here's a typical list of modules (note that as KWIVER expands, this list is likely to grow):

	any_source: A process which creates arbitrary data
	collate: Collates data from multiple worker processes
	const: A process with the const flag
	const_number: Outputs a constant number
	data_dependent: A process with a data dependent type
	distribute: Distributes data to multiple worker processes
	duplicate: A process which duplicates input
	expect: A process which expects some conditions
	feedback: A process which feeds data into itself
	flow_dependent: A process with a flow dependent type
	frame_list_process: A process that reads a list of image file names and generates stream of images and associated time stamps
	kw_archive_writer_process: A process to write kw archives
	kw_print_number_process: A Simple Kwiver Test Process
	multiplication: Multiplies numbers
	multiplier_cluster: A constant factor multiplier cluster
	mutate: A process with a mutable flag
	numbers: Outputs numbers within a range
	orphan: A dummy process
	orphan_cluster: A dummy cluster
	pass: Pass a data stream through
	print_number: Print numbers to a file
	shared: A process with the shared flag
	sink: Ignores incoming data
	skip: A process which skips input data
	stabilize_image_process: A process to generate current-to-reference image homographies
	tagged_flow_dependent: A process with a tagged flow dependent types
	take_number: Print numbers to a file
	take_string: Print strings to a file
	tunable: A process with a tunable parameter

This is the list of modules that can be included in a Sprokit pipeline.  We're going to use the `numbers` module and the the `print_number` module to create a very simple pipeline.  To learn more about the `numbers` module we'll again use `processopedia` this time to get details on a particular module.  For `numbers` we'll use the `processopedia -t numbers -d` command:

	Process type: numbers
	  Description: Outputs numbers within a range
	  Properties: _no_reentrant
	  Configuration:
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
	    Flags      : _required
	    Description: Where the numbers will be available.

And for `print_number`, we'll use `processopedia -t print_number -d`:

	Process type: print_number
	  Description: Print numbers to a file
	  Properties: _no_reentrant
	  Configuration:
	    Name       : output
	    Default    :
	    Description: The path of the file to output to.
	    Tunable    : no

	  Input ports:
	    Name       : number
	    Type       : integer
	    Flags      : _required
	    Description: Where numbers are read from.

	  Output ports:

The output of these commands tells us enough about each process to construct a Sprocket ".pipe" file that defines a processing pipeline.  In particular we'll need to know how to configure each process (the "Configuration") and how they can be hooked together (the input and output "Ports").

KWIVER comes with a sample [kwiver/pipeline_configs/number_flow.pipe](kwiver/pipeline_configs/number_flow.pipe) file that configures and connects the pipeline so that the `numbers` process will generate a set of integers from 1 to 99 and the `print_number` process will write those to a file called `numbers.txt`.  Of particular interest is the section at the end of the file that actually "hooks up" the pipeline.

To run the pipeline, we'll use the Sprokit `pipeline_runner` command:

	pipeline_runner -p </path/to/kwiver/source>/kwiver/pipeline_configs>/number_flow.pipe

After the pipeline completes, you should find a file, `numbers.txt`, in your working directory.

### Python Processes

One KWIVER's great strengths (as provided by Sprokit) is the ability to create hybrid pipelines which combine C++ and Python processes in the same pipeline.  This greatly facilitates prototyping complex processing pipelines.  To test this out we'll still use the `numbers` process, but we'll use a Python version of the `print_number` process called `kw_print_number_process` the code for which can be seen in [kwiver/processes/kw_print_number_process.py](kwiver/processes/kw_print_number_process.py).    As usual, we can lean about this process with a `processopedia` command: `processopedia -t kw_print_number_process -d`:

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

As you can see, the process is very similar to the C++ `print_number` process.  As a result, the [".pipe" file is very similar](kwiver/pipeline_configs/number_flow_python.pipe).

In order to get around limitations imposed by the Python Global Interpreter Lock, we'll use a different Sprokit scheduler for this pipeline.  The `pythread_per_process` scheduler which does essentially what it says: it creates a Python thread for every process in the pipeline:

	pipeline_runner -S pythread_per_process -p </path/to/kwiver/source>/kwiver/pipeline_configs>/number_flow_python.pipe

As with the previous pipeline, the numbers will be written to an output file, this time `numbers_from_python.txt`
