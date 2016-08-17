This directory contains a process template that can serve as a
starting point when starting a new set of processes. The build product
from a process or set of processes is a loadable module. A loadable
module looks very much the same as a shared library but the name does
not have the 'lib' prefix. This is done to differentiate these
loadable modules from regulat shared libraries.

Throughout the files in this directory are comments with "++" after
the comment delimeter. These are directed at the programmer and are
there to offer help on how to customize the template files. They
should be deleted from the final product.

This directory contains the following files:

CMakeLists.txt - Description of how to build the process(es) into a
    loadable module

README.rst - this file

register_processes.cxx - Process registration code. This file acts as
    the executable hook in the module. It contains an interface
    function with 'C' linkage that is invoked by the module loader.

template_process.[h,cxx] - These are the skelital process code. They
    contain more than is usually needed for a basic process, but it is
    all included for the sake of completeness. These files are a good
    place to start when creating a new process. Replace all
    occurrences of "template" with something more meaningful in the
    application context, including in the file name.

template_types_traits.h - This file is a minimal starter/place-holder
    for creating new type and port traits for a new application. If a
    new project has application specific data type semantics, then it
    is useful to set up type traits for these types.


A quick overview of processes
-----------------------------

A process is a unit of execution in the sprokit pipeline. Processes
typically have input and output ports that are connected to other
processes to form a pipeline. (Not a strict pipeline but more of a
directed graph.) Once a process has been initialized, it runs in a
loop, reading inputs, processing that data, and creating outputs.

Pipelines are created by sprokit from a pipeline configuration file
which processes to use and how they are connected. Sprokit collects a
set of known processes by dynamically loading a set of files (called
modules or plug-ins) from a set of directories. Each of these files
can contain one or more processes which register themselves with
sprokit process management.

When connecting processes, the ports must be declared as having the
same data type or an error will be thrown. This data type is a string
defining the logical or semantic data type that appears when creating
type traits (See template_type_traits.h file). It is important to make
the distinction between physical data types and logical data
types. Naming the data type based on its purpose will make for a more
robust application.
