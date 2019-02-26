#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vital/algo/image_object_detector.h>
#include "trampoline/algorithm_trampoline.tcc"
#include "trampoline/image_object_detector_trampoline.tcc"

using namespace kwiver::vital::algo;
using namespace kwiver::vital;
namespace py = pybind11;

PYBIND11_MODULE(image_object_detector, m)
{
  py::class_<algorithm, std::shared_ptr<algorithm>, py_algorithm<>>(m, "_algorithm")
    .def("get_configuration", &algorithm::get_configuration)
    .def("set_configuration", &algorithm::set_configuration)
    .def("check_configuration", &algorithm::check_configuration);
  
  py::class_< algorithm_def<image_object_detector>, 
              algorithm,
              std::shared_ptr<algorithm_def<image_object_detector>>, 
              py_iod_algorithm_def<>>(m, "Algorithm")
    .def(py::init())
    .def_static("create", &algorithm_def<image_object_detector>::create)
    .def_static("registered_names", &algorithm_def<image_object_detector>::registered_names)
    .def_static("get_nested_algo_configuration", &algorithm_def<image_object_detector>::get_nested_algo_configuration)
    .def_static("set_nested_algo_configuration", &algorithm_def<image_object_detector>::set_nested_algo_configuration)
    .def_static("check_nested_algo_configuration", &algorithm_def<image_object_detector>::check_nested_algo_configuration);


  py::class_<image_object_detector, std::shared_ptr<image_object_detector>, 
            algorithm_def<image_object_detector>,
              py_image_object_detector<>>(m, "ImageObjectDetector")
    .def(py::init())
    .def_static("static_type_name", &image_object_detector::static_type_name)
    .def("detect", &image_object_detector::detect);
}
