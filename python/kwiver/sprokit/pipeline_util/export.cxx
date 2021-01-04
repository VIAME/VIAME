// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <sprokit/pipeline/pipeline.h>

#include <sprokit/pipeline_util/export_dot.h>
#include <sprokit/pipeline_util/export_dot_exception.h>

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/sprokit/util/pystream.h>

#include <pybind11/pybind11.h>

#include <string>

/**
 * \file export.cxx
 *
 * \brief Python bindings for exporting functions.
 */

using namespace pybind11;

namespace kwiver{
namespace sprokit{
namespace python{
void export_dot(object const& stream, ::sprokit::pipeline_t const pipe, std::string const& graph_name);
}
}
}

using namespace kwiver::sprokit::python;
PYBIND11_MODULE(export_, m)
{
  m.def("export_dot", &export_dot, call_guard<kwiver::vital::python::gil_scoped_release>()
    , arg("stream"), arg("pipeline"), arg("name")
    , "Writes the pipeline to the stream in dot format.");
}
namespace kwiver{
namespace sprokit{
namespace python{
void
export_dot(object const& stream, ::sprokit::pipeline_t const pipe, std::string const& graph_name)
{
  ::sprokit::python::pyostream ostr(stream);

  return ::sprokit::export_dot(ostr, pipe, graph_name);
}
}}}
