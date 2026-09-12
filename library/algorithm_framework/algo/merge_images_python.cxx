// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/merge_images.h>
#include "algorithm_python.txx"
#include "merge_images_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void merge_images(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::merge_images,
               std::shared_ptr<kwiver::vital::algo::merge_images>,
               kwiver::vital::algorithm,
               merge_images_trampoline<> > instance(m,  "MergeImages");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::merge_images::interface_name)
    .def("merge", &kwiver::vital::algo::merge_images::merge, py::doc(R"( Merge images)"), py::arg("image1"), py::arg("image2"))
    ;
  register_algorithm< kwiver::vital::algo::merge_images > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
