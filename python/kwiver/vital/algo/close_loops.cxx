// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/close_loops_trampoline.txx>
#include <python/kwiver/vital/algo/close_loops.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
void close_loops(py::module &m)
{
  py::class_< kwiver::vital::algo::close_loops,
              std::shared_ptr<kwiver::vital::algo::close_loops>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::close_loops>,
              close_loops_trampoline<> >( m, "CloseLoops" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::close_loops::static_type_name)
    .def("stitch",
         &kwiver::vital::algo::close_loops::stitch);
}
}
}
}
