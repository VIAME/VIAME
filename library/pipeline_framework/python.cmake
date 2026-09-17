###
# The python bindings for the pipeline framework
#
# `viame.pipeline`, `viame.pipeline.util`, `viame.adapters`,
# `viame.schedulers` and `viame.processes`. P11-T02b moved these out of
# `python/kwiver/sprokit/*` so that each binding sits beside the C++ it
# binds, which is the rule every other library in `library/` already
# follows.
#
# Every binding here carries a `_python` suffix because ten of them would
# otherwise collide outright with the source of the very class they bind:
# `datum.cxx`, `edge.cxx`, `pipeline.cxx`, `process.cxx` and the rest are
# all names this directory already used.
#
# `python_wrappers.cxx` is named by no target on purpose. It is `#include`d
# by seven of the bindings rather than compiled, and listing it would
# compile it a second time as a translation unit of its own.
#
# `${PYTHON_LIBRARIES}` goes on every module. VIAME links with
# `-Wl,--no-undefined`, and an extension module leaves the interpreter's
# symbols to be resolved at import; `python/` stripped that flag from its
# own directory scope, which is not something this directory can inherit.
# Linking libpython is what `library/core_types` already does, and it does
# not weaken the check for the C++ beside it.
##

set( _viame_pipeline_python_common
  pybind11::pybind11
  ${PYTHON_LIBRARIES}
  )

# ---------------------------------------------------------------------------
# The C++ helper that lets the bake and load bindings read a pipeline out of
# a python file object. A library rather than a module: the bindings link it.
#
# Not on Windows, as it was not before: `sprokit/CMakeLists.txt` guarded the
# whole directory.
if( NOT WIN32 )
  viame_add_library( viame_pipeline_python_util
    ${CMAKE_CURRENT_LIST_DIR}/pystream.cxx
    ${CMAKE_CURRENT_LIST_DIR}/pystream.h
    )
  target_link_libraries( viame_pipeline_python_util
    LINK_PUBLIC     ${Python_LIBRARIES}
    LINK_PRIVATE    pybind11::pybind11
                    viame_python_util
    )
  viame_install_headers(
    pystream.h
    SUBDIR   viame/pipeline_framework
    )
  # As above: `pystream.h` includes its own export header by the `viame/`
  # prefix, which only this republishes it under.
  viame_lite_generated( viame_pipeline_python_util_export.h pipeline_framework )
endif()

# ---------------------------------------------------------------------------
# `viame.pipeline`
##
set( THIS_MODULE pipeline )

viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/pipeline_init.py "${THIS_MODULE}" __init__ )

viame_add_python_library( datum "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/datum_python.cxx
  PRIVATE ${_viame_pipeline_python_common}
          viame_pipeline_framework
          viame_python_util
          viame_algorithm_framework
          viame_util
  )

viame_add_python_library( edge "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/edge_python.cxx
  PRIVATE ${_viame_pipeline_python_common}
          viame_pipeline_framework
          viame_python_util
  )

viame_add_python_library( pipeline "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/pipeline_python.cxx
  PRIVATE ${_viame_pipeline_python_common}
          viame_pipeline_framework
          viame_python_util
  )

viame_add_python_library( process "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/process_python.cxx
  PRIVATE ${_viame_pipeline_python_common}
          viame_pipeline_framework
          viame_python_util
          viame_util
  )

viame_add_python_library( process_factory "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/process_factory_python.cxx
  PRIVATE ${_viame_pipeline_python_common}
          viame_pipeline_framework
          viame_python_util
          viame_plugin
  )

viame_add_python_library( scheduler "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/scheduler_python.cxx
  PRIVATE ${_viame_pipeline_python_common}
          viame_pipeline_framework
          viame_python_util
  )

viame_add_python_library( scheduler_factory "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/scheduler_factory_python.cxx
  PRIVATE ${_viame_pipeline_python_common}
          viame_pipeline_framework
          viame_python_util
          viame_plugin
  )

viame_add_python_library( stamp "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/stamp_python.cxx
  PRIVATE ${_viame_pipeline_python_common}
          viame_pipeline_framework
          viame_python_util
  )

viame_add_python_library( utils "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/utils_python.cxx
  PRIVATE ${_viame_pipeline_python_common}
          viame_pipeline_framework
          viame_python_util
  )

viame_add_python_library( version "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/version_python.cxx
  PRIVATE ${_viame_pipeline_python_common}
          viame_pipeline_framework
          viame_python_util
  )

# ---------------------------------------------------------------------------
# `viame.pipeline.util` -- what `kwiver.sprokit.pipeline_util` was. It is the
# python side of `viame_pipeline_util`, so it sits under the pipeline package
# rather than beside it.
##
if( NOT WIN32 )
  set( THIS_MODULE pipeline/util )

  viame_add_python_module(
    ${CMAKE_CURRENT_LIST_DIR}/pipeline_util_init.py "${THIS_MODULE}" __init__ )

  viame_add_python_library( bake "${THIS_MODULE}"
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/bake_python.cxx
    PRIVATE ${_viame_pipeline_python_common}
            viame_pipeline_framework
            viame_pipeline_util
            viame_pipeline_python_util
            viame_python_util
            viame_plugin
            ${Boost_IOSTREAMS_LIBRARY}
            ${Boost_SYSTEM_LIBRARY}
    )

  viame_add_python_library( load "${THIS_MODULE}"
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/load_python.cxx
    PRIVATE ${_viame_pipeline_python_common}
            viame_pipeline_framework
            viame_pipeline_util
            viame_pipeline_python_util
            viame_python_util
            viame_util
            ${Boost_IOSTREAMS_LIBRARY}
    )
endif()

# ---------------------------------------------------------------------------
# `viame.adapters`
##
set( THIS_MODULE adapters )

viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/adapters/adapters_init.py "${THIS_MODULE}" __init__ )

viame_add_python_library( adapter_data_set "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/adapters/adapter_data_set_python.cxx
  PRIVATE ${_viame_pipeline_python_common}
          viame_adapter
          viame_pipeline_framework
          viame_python_util
  )

viame_add_python_library( embedded_pipeline "${THIS_MODULE}"
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/adapters/embedded_pipeline_python.cxx
  PRIVATE ${_viame_pipeline_python_common}
          viame_adapter
          viame_pipeline_util
          viame_python_util
  )

# ---------------------------------------------------------------------------
# `viame.schedulers` and `viame.processes`: python only.
##
viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/schedulers/schedulers_init.py schedulers __init__ )
viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/schedulers/pythread_per_process_scheduler.py
  schedulers pythread_per_process )

# `base` is what `kwiver_process.py` was: the class every python process in
# VIAME derives from, renamed with its class in P11-T02.
viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/processes/processes_init.py processes __init__ )
foreach( _viame_process IN ITEMS
    base apply_descriptor homography_writer kw_print_number_process
    process_image )
  viame_add_python_module(
    ${CMAKE_CURRENT_LIST_DIR}/processes/${_viame_process}.py
    processes ${_viame_process} )
endforeach()
