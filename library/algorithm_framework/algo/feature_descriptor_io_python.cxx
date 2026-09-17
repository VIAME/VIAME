// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/feature_descriptor_io.h>
#include "algorithm_python.txx"
#include "feature_descriptor_io_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void feature_descriptor_io(py::module& m)
{
  py::module::import("viame.config");
  py::module::import("viame.types");

    py::class_<viame::algo::feature_descriptor_io,
               std::shared_ptr<viame::algo::feature_descriptor_io>,
               viame::algorithm,
               feature_descriptor_io_trampoline<> > instance(m,  "FeatureDescriptorIO");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::feature_descriptor_io::interface_name)
    .def("load", &viame::algo::feature_descriptor_io::load, py::doc(R"( Load features and descriptors from a file

 \throws viame::path_not_exists Thrown when the given path does not
 exist.

 \throws viame::path_not_a_file Thrown when the given path does
    not point to a file (i.e. it points to a directory).

 \param filename the path to the file the load
 \param feat the set of features to load from the file
 \param desc the set of descriptors to load from the file)"), py::arg("filename"), py::arg("feat"), py::arg("desc"))
    .def("save", &viame::algo::feature_descriptor_io::save, py::doc(R"( Save features and descriptors to a file

 Saves features and/or descriptors to a file.  Either \p feat or \p desc
 may be Null, but not both.  If both \p feat and \p desc are provided then
 the must be of the same size.

 \throws viame::path_not_exists Thrown when the expected
    containing directory of the given path does not exist.

 \throws viame::path_not_a_directory Thrown when the expected
    containing directory of the given path is not actually a
    directory.

 \param filename the path to the file to save
 \param feat the set of features to write to the file
 \param desc the set of descriptors to write to the file)"), py::arg("filename"), py::arg("feat"), py::arg("desc"))
    ;
  register_algorithm< viame::algo::feature_descriptor_io > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
