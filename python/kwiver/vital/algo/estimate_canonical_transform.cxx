// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/estimate_canonical_transform_trampoline.txx>
#include <python/kwiver/vital/algo/estimate_canonical_transform.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void estimate_canonical_transform(py::module &m)
{
  py::class_< kwiver::vital::algo::estimate_canonical_transform,
              std::shared_ptr<kwiver::vital::algo::estimate_canonical_transform>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::estimate_canonical_transform>,
              estimate_canonical_transform_trampoline<> >( m, "EstimateCanonicalTransform" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::estimate_canonical_transform::static_type_name)
    .def("estimate_transform",
         &kwiver::vital::algo::estimate_canonical_transform::estimate_transform);
}
}
}
}
