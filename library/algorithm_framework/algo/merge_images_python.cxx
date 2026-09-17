// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/merge_images.h>
#include "algorithm_python.txx"
#include "merge_images_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void merge_images(py::module& m)
{
  py::module::import("viame.config");
  py::module::import("viame.types");

    py::class_<viame::algo::merge_images,
               std::shared_ptr<viame::algo::merge_images>,
               viame::algorithm,
               merge_images_trampoline<> > instance(m,  "MergeImages");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::merge_images::interface_name)
    .def("merge", &viame::algo::merge_images::merge, py::doc(R"( Merge images)"), py::arg("image1"), py::arg("image2"))
    ;
  register_algorithm< viame::algo::merge_images > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
