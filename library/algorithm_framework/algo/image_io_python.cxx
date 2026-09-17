// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/image_io.h>
#include "algorithm_python.txx"
#include "image_io_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void image_io(py::module& m)
{
  py::module::import("viame.config");
  py::module::import("viame.types");

    py::class_<viame::algo::image_io,
               std::shared_ptr<viame::algo::image_io>,
               viame::algorithm,
               image_io_trampoline<> > instance(m,  "ImageIO");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::image_io::interface_name)
    .def("load", &viame::algo::image_io::load, py::doc(R"( Load image from the file

 \throws viame::path_not_exists Thrown when the given path does not
 exist.

 \throws viame::path_not_a_file Thrown when the given path does
    not point to a file (i.e. it points to a directory).

 \param filename the path to the file to load
 \returns an image container refering to the loaded image)"), py::arg("filename"))
    .def("save", &viame::algo::image_io::save, py::doc(R"( Save image to a file

 Image file format is based on file extension.

 \throws viame::path_not_exists Thrown when the expected
    containing directory of the given path does not exist.

 \throws viame::path_not_a_directory Thrown when the expected
    containing directory of the given path is not actually a
    directory.

 \param filename the path to the file to save
 \param data the image container refering to the image to write)"), py::arg("filename"), py::arg("data"))
    .def("load_metadata", &viame::algo::image_io::load_metadata, py::doc(R"( Get the image metadata

 \throws viame::path_not_exists Thrown when the given path does not
 exist.

 \throws viame::path_not_a_file Thrown when the given path does
    not point to a file (i.e. it points to a directory).

 \param filename the path to the file to read
 \returns pointer to the loaded metadata)"), py::arg("filename"))
    .def("get_implementation_capabilities", &viame::algo::image_io::get_implementation_capabilities, py::doc(R"( \brief Return capabilities of concrete implementation.

 This method returns the capabilities for the current image reader/writer.

 \return Reference to supported image capabilities.)"))
    .def_readonly_static("HAS_TIME", &viame::algo::image_io::HAS_TIME)
    ;
  register_algorithm< viame::algo::image_io > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
