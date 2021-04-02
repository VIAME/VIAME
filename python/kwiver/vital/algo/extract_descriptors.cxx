// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/extract_descriptors_trampoline.txx>
#include <python/kwiver/vital/algo/extract_descriptors.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void extract_descriptors(py::module &m)
{
  py::class_< kwiver::vital::algo::extract_descriptors,
              std::shared_ptr<kwiver::vital::algo::extract_descriptors>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::extract_descriptors>,
              extract_descriptors_trampoline<> >( m, "ExtractDescriptors" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::extract_descriptors::static_type_name)
    .def("extract",
         &kwiver::vital::algo::extract_descriptors::extract);
}
}
}
}
