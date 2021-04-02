// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/filter_features.h>
#include <python/kwiver/vital/algo/trampoline/filter_features_trampoline.txx>

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void filter_features(py::module &m)
{
  py::class_< kwiver::vital::algo::filter_features,
              std::shared_ptr<kwiver::vital::algo::filter_features>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::filter_features>,
              filter_features_trampoline<> >( m, "FilterFeatures" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::filter_features::static_type_name)
    .def("filter",
         ( kwiver::vital::feature_set_sptr
           ( kwiver::vital::algo::filter_features::* )
           ( kwiver::vital::feature_set_sptr ) const )
         &kwiver::vital::algo::filter_features::filter)
    .def("filter",
         ( std::pair< kwiver::vital::feature_set_sptr,
                      kwiver::vital::descriptor_set_sptr >
           ( kwiver::vital::algo::filter_features::* )
           ( kwiver::vital::feature_set_sptr,
             kwiver::vital::descriptor_set_sptr ) const )
         &kwiver::vital::algo::filter_features::filter);
}
}
}
}
