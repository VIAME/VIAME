# Pybind11 generation of wrappers for vital/algo.

The contents of this directory together with `/scripts/cpp_to_pybind11.py` generate wrappers for the classes in `vital/algo`.

## Process
1. During cmake configuration time `config.ini` is generated in the current binary directory based on the template `./config.ini`.
2. The config file drives the `cpp_to_pybind11.py` which will generate three files per class:
  - `<algo>.cxx`, the wrappings themselves
  - `<algo>_trampoline.txx` , code to take care all pure virtual methods in the interface class
  - `<algo>.h`, a definition of the method that add `<algo>` in the python  modules that is generated out of this directory.
3. `CMakeLists.txt` in this directory will compose `algorithm_module.cxx` that create a module for all the wrapped classes in this directory.



# Notes:
- Generated headers can be inspected at `<build-dir>/python/kwiver/vital/algo`
- During compilation the macro `KWIVER_PYBIND11_INCLUDE` is defined. A common use case is to include pybind11 specific code. i.e.
```
#ifdef KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
PYBIND11_MAKE_OPAQUE( std::shared_ptr< std::string > );
#endif
```
- During parsing  the macro `KWIVER_PYBIND11_WRAPPING` is defined. So, to exclude a piece of code from the parsing of a header use :
```
#ifndef KWIVER_PYBIND11_WRAPPING
<code to exclude>
#endif
```
A common use case is to exclude methods from begin wrapped in Python.
