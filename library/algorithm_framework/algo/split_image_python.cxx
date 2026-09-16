// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/split_image.h>
#include "algorithm_python.txx"
#include "split_image_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void split_image(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<viame::algo::split_image,
               std::shared_ptr<viame::algo::split_image>,
               viame::algorithm,
               split_image_trampoline<> > instance(m,  "SplitImage");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::split_image::interface_name)
    .def("split", &viame::algo::split_image::split, py::doc(R"( Split image)"), py::arg("img"))
    ;
  register_algorithm< viame::algo::split_image > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
