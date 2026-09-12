// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/transform_2d_io.h>
#include "algorithm_python.txx"
#include "transform_2d_io_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void transform_2d_io(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::transform_2d_io,
               std::shared_ptr<kwiver::vital::algo::transform_2d_io>,
               kwiver::vital::algorithm,
               transform_2d_io_trampoline<> > instance(m,  "Transform2DIO");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::transform_2d_io::interface_name)
    .def("load", &kwiver::vital::algo::transform_2d_io::load, py::doc(R"( Load transform from the file

 \throws kwiver::vital::path_not_exists
   Thrown when the given path does not exist.

 \throws kwiver::vital::path_not_a_file
   Thrown when the given path does not point to a file (i.e. it points to a
   directory).

 \param filename the path to the file to load
 \returns a transform instance referring to the loaded transform)"), py::arg("filename"))
    .def("save", &kwiver::vital::algo::transform_2d_io::save, py::doc(R"( Save transform to a file

 Transform file format is based on the algorithm instance.

 \throws kwiver::vital::path_not_exists
   Thrown when the expected containing directory of the given path does not
   exist.

 \throws kwiver::vital::path_not_a_directory
   Thrown when the expected containing directory of the given path is not
   actually a directory.

 \throws kwiver::vital::invalid_data
   Thrown when the algorithm does not recognize the concrete type of the
   transformation instance.

 \param filename the path to the file to save
 \param data the transform instance referring to the transform to write)"), py::arg("filename"), py::arg("data"))
    ;
  register_algorithm< kwiver::vital::algo::transform_2d_io > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
