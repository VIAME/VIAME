// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/pipeline_framework/pipe_bakery.h>
#include <viame/pipeline_framework/pipe_bakery_exception.h>
#include <viame/pipeline_framework/pipeline_builder.h>

#include <viame/pipeline_framework/pipeline.h>
#include <viame/pipeline_framework/process_factory.h>

#include <python/kwiver/sprokit/util/pystream.h>

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

/**
 * \file bake.cxx
 *
 * \brief Python bindings for baking pipelines.
 */

using namespace pybind11;

namespace viame {

namespace pipeline {

namespace python {

static ::viame::pipeline::pipeline_t bake_pipe_file( std::string const& path );
static ::viame::pipeline::pipeline_t bake_pipe( object stream );

} // namespace python

} // namespace pipeline

} // namespace viame

using namespace viame::pipeline::python;
PYBIND11_MODULE( bake, m )
{
  m.def(
    "bake_pipe_file", &bake_pipe_file,
    call_guard< pybind11::gil_scoped_release >(),
    ( arg( "path" ) ),
    "Build a pipeline from a file." );
  m.def(
    "bake_pipe", &bake_pipe, call_guard< pybind11::gil_scoped_release >(),
    ( arg( "stream" ) ),
    "Build a pipeline from a stream." );
  m.def(
    "bake_pipe_blocks", &viame::pipeline::bake_pipe_blocks,
    call_guard< pybind11::gil_scoped_release >(),
    ( arg( "blocks" ) ),
    "Build a pipeline from pipe blocks." );
  m.def(
    "extract_configuration", &viame::pipeline::extract_configuration,
    call_guard< pybind11::gil_scoped_release >(),
    ( arg( "blocks" ) ),
    "Extract the configuration from pipe blocks." );
}

namespace viame {

namespace pipeline {

namespace python {

// ------------------------------------------------------------------
::viame::pipeline::pipeline_t
bake_pipe_file( std::string const& path )
{
  ::viame::pipeline::pipeline_builder builder;
  builder.load_pipeline( path );
  return builder.pipeline();
}

// ------------------------------------------------------------------
::viame::pipeline::pipeline_t
bake_pipe( object stream )
{
  ::viame::pipeline::python::pyistream istr( stream );
  ::viame::pipeline::pipeline_builder builder;
  builder.load_pipeline( istr );
  return builder.pipeline();
}

} // namespace python

} // namespace pipeline

} // namespace viame
