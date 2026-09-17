###
# The python bindings for the algorithm framework
#
# `viame.config`, `viame.io`, `viame.util`, `viame.modules`,
# `viame.plugins`, `viame.plugin_management`, `viame.applets`,
# `viame.exceptions`, `viame.test_interface`, `viame.log`, and the root
# `viame` package itself. P11-T02b moved all of it out of
# `python/kwiver/vital/*` so that each binding sits beside the C++ it binds.
#
# `viame.algo` is not here -- it has been in `algo/python.cmake` since
# P8-T02, and `viame.types` in `library/core_types` since P8-T01. This file
# is the rest of what `python/` held.
#
# A binding whose name would collide with the C++ it binds carries a
# `_python` suffix, the convention the two files above already set:
# `camera_from_metadata_python.cxx` beside `camera_from_metadata.cxx`,
# `applet_python.cxx` beside `kwiver_applet.cxx`.
#
# `${PYTHON_LIBRARIES}` goes on every module. VIAME links with
# `-Wl,--no-undefined`, and an extension module leaves the interpreter's
# symbols to be resolved at import; `python/` stripped that flag from its
# own directory scope, which this directory cannot inherit.
##

set( _viame_af_python_common
  pybind11::pybind11
  ${PYTHON_LIBRARIES}
  )

# ---------------------------------------------------------------------------
# The root of the package
#
# `viame/__init__.py`: the logging level, the Windows DLL directories and
# `PROJ_LIB`, which have to be set before anything below is imported.
#
# It no longer imports its own subpackages. `kwiver/__init__.py` ended with
# `from . import tools` and `from . import vital`, so importing any part of
# the package imported all of it -- the cost P8-T10 spent its startup budget
# removing everywhere else.
##
viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/viame_init.py "" __init__ )

# `viame.log`, which was `kwiver.vital.vital_logging`.
viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/logger/log.py "" log )

# ---------------------------------------------------------------------------
# `viame_python_util`: the C++ helpers the bindings share
#
# A real library, linked by the bindings here and by the pipeline framework's.
##
viame_add_library( viame_python_util
  ${CMAKE_CURRENT_LIST_DIR}/util/python_exceptions.cxx
  ${CMAKE_CURRENT_LIST_DIR}/util/python_exceptions.h
  ${CMAKE_CURRENT_LIST_DIR}/util/python.h
  )
target_link_libraries( viame_python_util
  PRIVATE $<BUILD_INTERFACE:pybind11::embed> $<BUILD_INTERFACE:pybind11::module>
  )
viame_install_headers(
  util/python.h
  util/python_exceptions.h
  SUBDIR   viame/algorithm_framework
  )
# Republished under the prefix its own header includes it by. It is written
# to this directory's binary dir, which is on no include path as `viame/...`,
# so without this `python_exceptions.h` cannot find its own export macros.
viame_lite_generated( viame_python_util_export.h algorithm_framework/util )

# ---------------------------------------------------------------------------
# `viame.util`
##
set( THIS_MODULE util )
viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/util/util_init.py "${THIS_MODULE}" __init__ )
# `pil` was `VitalPIL`; the functions in it kept their names.
foreach( _viame_util_module IN ITEMS pil find_python_library entrypoint env )
  viame_add_python_module(
    ${CMAKE_CURRENT_LIST_DIR}/util/${_viame_util_module}.py
    "${THIS_MODULE}" ${_viame_util_module} )
endforeach()

# ---------------------------------------------------------------------------
# `viame.config`
##
set( THIS_MODULE config )
viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/config/config_init.py "${THIS_MODULE}" __init__ )
viame_add_python_library( _config "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/config/config_module_python.cxx
          ${CMAKE_CURRENT_LIST_DIR}/config/config_helpers_python.cxx
          ${CMAKE_CURRENT_LIST_DIR}/config/config_helpers_python.h
  PRIVATE ${_viame_af_python_common}
          viame_config
          viame_algorithm_framework
  )

# ---------------------------------------------------------------------------
# `viame.io`
##
set( THIS_MODULE io )
viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/io/io_init.py "${THIS_MODULE}" __init__ )
viame_add_python_library( metadata_io "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/io/metadata_io_python.cxx
  PRIVATE ${_viame_af_python_common}
          viame_algorithm_framework
  )
viame_add_python_library( camera_from_metadata "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/io/camera_from_metadata_python.cxx
  PRIVATE ${_viame_af_python_common}
          viame_algorithm_framework
  )

# ---------------------------------------------------------------------------
# `viame.exceptions`: python only.
##
set( THIS_MODULE exceptions )
foreach( _viame_exception IN ITEMS
    __init__ algorithm base config_block config_block_io eigen image math )
  viame_add_python_module(
    ${CMAKE_CURRENT_LIST_DIR}/exceptions/${_viame_exception}.py
    "${THIS_MODULE}" ${_viame_exception} )
endforeach()

# ---------------------------------------------------------------------------
# `viame.applets`
##
set( THIS_MODULE applets )
viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/applets/applets_init.py "${THIS_MODULE}" __init__ )
viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/applets/applet_utils.py
  "${THIS_MODULE}" applet_utils )
viame_add_python_library( _applets "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/applets/applet_python.cxx
          ${CMAKE_CURRENT_LIST_DIR}/applets/applets_module_python.cxx
          ${CMAKE_CURRENT_LIST_DIR}/applets/applet_trampoline_python.txx
  PRIVATE ${_viame_af_python_common}
          viame_applets
          viame_config
          viame_algorithm_framework
  )

# ---------------------------------------------------------------------------
# `viame.test_interface`
##
set( THIS_MODULE test_interface )
viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/test_interface/test_interface_init.py
  "${THIS_MODULE}" __init__ )
foreach( _viame_say IN ITEMS python_say python_they_say )
  viame_add_python_module(
    ${CMAKE_CURRENT_LIST_DIR}/test_interface/${_viame_say}.py
    "${THIS_MODULE}" ${_viame_say} )
endforeach()
viame_add_python_library( _interface "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_interface/interface_python.cxx
  PRIVATE ${_viame_af_python_common}
          viame_algorithm_framework
  )

# ---------------------------------------------------------------------------
# `viame.plugin_management`
##
set( THIS_MODULE plugin_management )
viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/plugin_management/__init__.py
  "${THIS_MODULE}" __init__ )
viame_add_python_library( _plugin_management "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/plugin_management/plugin_management_python.cxx
  PRIVATE ${_viame_af_python_common}
          viame_algorithm_framework
  )

# ---------------------------------------------------------------------------
# `viame.plugins`, and the C++ plugin that loads python plugins
##
set( _viame_python_embed pybind11::embed pybind11::pybind11 )
if( ( NOT SKBUILD ) OR MSVC )
  list( APPEND _viame_python_embed Python::Python )
endif()

set( THIS_MODULE plugins )
viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/plugins/__init__.py "${THIS_MODULE}" __init__ )
viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/plugins/discovery.py "${THIS_MODULE}" discovery )
viame_add_python_library( _pluggable "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/plugins/pluggable_python.cxx
  PRIVATE ${_viame_af_python_common}
          viame_algorithm_framework
  )
viame_add_plugin( plugins_from_python
  KIND      OTHERS
  SOURCES   ${CMAKE_CURRENT_LIST_DIR}/plugins/register_python.cxx
  PRIVATE   ${_viame_python_embed}
            viame_algorithm_framework
  )
viame_lite_generated( plugins_from_python_export.h algorithm_framework/plugins )

# ---------------------------------------------------------------------------
# `viame.modules`, and the C++ plugin that loads python modules
##
set( THIS_MODULE modules )
viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/modules/__init__.py "${THIS_MODULE}" __init__ )
foreach( _viame_module IN ITEMS module_loader loaders )
  viame_add_python_module(
    ${CMAKE_CURRENT_LIST_DIR}/modules/${_viame_module}.py
    "${THIS_MODULE}" ${_viame_module} )
endforeach()

viame_add_python_library( modules "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/modules/modules_python.cxx
  PRIVATE ${_viame_af_python_common}
          viame_python_util
          viame_plugin
  )

if( KWIVER_ENABLE_TOOLS )
  viame_add_plugin( modules_python
    KIND      OTHERS
    SOURCES   ${CMAKE_CURRENT_LIST_DIR}/modules/module_helpers_python.cxx
              ${CMAKE_CURRENT_LIST_DIR}/modules/module_helpers_python.h
              ${CMAKE_CURRENT_LIST_DIR}/modules/registration_python.cxx
    PRIVATE   ${_viame_python_embed}
              viame_python_util
              viame_logger
              viame_plugin
    )

  # On the target rather than the directory. `add_definitions` here would
  # reach every target in `library/algorithm_framework`, because this file is
  # `include()`d into that directory's scope rather than being a
  # subdirectory of its own as it was under `python/`.
  if( UNIX )
    target_compile_definitions( modules_python PRIVATE VITAL_LOAD_PYLIB_SYM )
  endif()

  viame_lite_generated( modules_python_export.h algorithm_framework/modules )
endif()
