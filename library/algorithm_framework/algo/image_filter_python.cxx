// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/image_filter.h>
#include "algorithm_python.txx"
#include "image_filter_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void image_filter(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::image_filter,
               std::shared_ptr<kwiver::vital::algo::image_filter>,
               kwiver::vital::algorithm,
               image_filter_trampoline<> > instance(m,  "ImageFilter");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::image_filter::interface_name)
    .def("filter", &kwiver::vital::algo::image_filter::filter, py::doc(R"( Filter a  input image and return resulting image

 This method implements the filtering operation. The method does
 not modify the image in place. The resulting image must be a
 newly allocated image which is the same size as the input image.

 \param image_data Image to filter.
 \returns a filtered version of the input image)"), py::arg("image_data"))
    ;
  register_algorithm< kwiver::vital::algo::image_filter > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
