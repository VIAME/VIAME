###
# `viame.types`
#
# The bindings live beside the C++ they bind since P8-T01, rather than in
# `python/kwiver/vital/types`, which is what the copy from kwiver left. Each
# file keeps its name with a `_python` suffix -- the convention
# `library/file_io/opencv_yaml_python.cxx` already set -- because 47 of them
# would otherwise collide with the C++ source of the very type they bind.
#
# **The module path is unchanged.** `viame_add_python_library` takes it as an
# argument rather than deriving it from the source location, so
# `viame.types.bounding_box` is still `viame.types.bounding_box`
# and the four files in the tree that import a submodule by name keep working.
# `library/core_types/tests/test_python_types.py` holds that surface to what
# it was: 106 classes and 965 members, recorded before this moved.
##

# `${PYTHON_LIBRARIES}` goes on every module below, because VIAME links with
# `-Wl,--no-undefined` and an extension module leaves the interpreter's
# symbols to be resolved at import. In `python/` the flag was stripped from
# the directory's link flags instead; linking libpython is what
# `library/file_io` already does for `_opencv_yaml`, and it does not weaken
# the check for the C++ in this directory the way stripping the flag would.
#
# The package these install into used to be set here, to `kwiver`. Nothing
# sets it now: P11-T02 made the whole tree one package, derived from the
# project name.
set( THIS_MODULE types )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/types_init.py "${THIS_MODULE}" __init__ )

# ----------------------------------------------------------------------------
# One extension module, not fifty-six
#
# Each of these was its own `.so`, and fifty-six of them cost 68.6 MB because
# every module carries its own copy of the same instantiated pybind11 and STL
# templates. Linked together the code is 16.6 MB, 8.3 MB stripped: 83% of the
# exported symbols were duplicates. Measured with the linker on the existing
# object files before the change was written, not estimated.
#
# `viame.types.<name>` is unchanged. Each name is a submodule of `_types` and
# a one-line `viame/types/<name>.py` re-exports it, so the import path, the
# `from viame.types.X import *` lines in `types_init.py`, and the four files
# in the tree that import a submodule by name all keep working.
# `tests/test_python_types.py` holds the surface to 106 classes and 965
# members and is what says whether that is true.
#
# The order comes from `types_init.py` rather than a list kept here, because
# it is load-bearing -- `detected_object` references `detected_object_type`,
# `homography` references `transform_2d` -- and two lists that must agree
# eventually do not.
# ----------------------------------------------------------------------------

set( vital_python_headers
     image_python.h
     image_container_python.h
  )

# `types` is spelled `types_module_python.cxx`, the file having been renamed
# when it would otherwise have collided with the C++ it binds.
set( VIAME_FOLD_SOURCE_types types_module_python.cxx )

viame_fold_python_package( "${THIS_MODULE}" viame.types _types
  INIT types_init.py
  MIN_MODULES 50
  # `image` and `image_container` are translation units of the `types`
  # module rather than modules of their own.
  EXTRA_SOURCES ${vital_python_headers}
                image_python.cxx
                image_container_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          viame_algorithm_framework
  )

