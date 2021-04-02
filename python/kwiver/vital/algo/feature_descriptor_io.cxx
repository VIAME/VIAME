// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/feature_descriptor_io.h>
#include <python/kwiver/vital/algo/trampoline/feature_descriptor_io_trampoline.txx>

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void feature_descriptor_io(py::module &m)
{
  py::class_< kwiver::vital::algo::feature_descriptor_io,
              std::shared_ptr<kwiver::vital::algo::feature_descriptor_io>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::feature_descriptor_io>,
              feature_descriptor_io_trampoline<> >( m, "FeatureDescriptorIO" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::feature_descriptor_io::static_type_name)
    .def("load",
         &kwiver::vital::algo::feature_descriptor_io::load)
    .def("save",
         &kwiver::vital::algo::feature_descriptor_io::save);
}
}
}
}
