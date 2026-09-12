// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/image_io.h>
#include "algorithm_python.txx"
#include "image_io_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void image_io(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::image_io,
               std::shared_ptr<kwiver::vital::algo::image_io>,
               kwiver::vital::algorithm,
               image_io_trampoline<> > instance(m,  "ImageIO");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::image_io::interface_name)
    .def("load", &kwiver::vital::algo::image_io::load, py::doc(R"( Load image from the file

 \throws kwiver::vital::path_not_exists Thrown when the given path does not
 exist.

 \throws kwiver::vital::path_not_a_file Thrown when the given path does
    not point to a file (i.e. it points to a directory).

 \param filename the path to the file to load
 \returns an image container refering to the loaded image)"), py::arg("filename"))
    .def("save", &kwiver::vital::algo::image_io::save, py::doc(R"( Save image to a file

 Image file format is based on file extension.

 \throws kwiver::vital::path_not_exists Thrown when the expected
    containing directory of the given path does not exist.

 \throws kwiver::vital::path_not_a_directory Thrown when the expected
    containing directory of the given path is not actually a
    directory.

 \param filename the path to the file to save
 \param data the image container refering to the image to write)"), py::arg("filename"), py::arg("data"))
    .def("load_metadata", &kwiver::vital::algo::image_io::load_metadata, py::doc(R"( Get the image metadata

 \throws kwiver::vital::path_not_exists Thrown when the given path does not
 exist.

 \throws kwiver::vital::path_not_a_file Thrown when the given path does
    not point to a file (i.e. it points to a directory).

 \param filename the path to the file to read
 \returns pointer to the loaded metadata)"), py::arg("filename"))
    .def("get_implementation_capabilities", &kwiver::vital::algo::image_io::get_implementation_capabilities, py::doc(R"( \brief Return capabilities of concrete implementation.

 This method returns the capabilities for the current image reader/writer.

 \return Reference to supported image capabilities.)"))
    .def_readonly_static("HAS_TIME", &kwiver::vital::algo::image_io::HAS_TIME)
    ;
  register_algorithm< kwiver::vital::algo::image_io > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
