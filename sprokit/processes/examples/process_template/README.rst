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

README.tsy - this file

register_processes.cxx - Process registration code. This file acts as
    the executable hook in the module. It contains an interface
    function with 'C' linkage that is invoked by the module loader.

template_process.[h,cxx] - These are the skelital process code. They
    contain more than is usually needed for a basic process, but it is
    all included for the sake of completeness. These files are a good
    place to start when creating a new process. Replace all
    occurrences of "template" with something more meaningful in the
    application context, including in the file name.

template_types_traits.h - This file is a minimal starter/place holder
    for creating new type andport traits for a new application. If a
    new project has application specific data type semantics, then it
    is useful to set up type traits for these types.
