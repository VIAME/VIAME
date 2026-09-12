// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/feature_descriptor_io.h>
#include "algorithm_python.txx"
#include "feature_descriptor_io_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void feature_descriptor_io(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::feature_descriptor_io,
               std::shared_ptr<kwiver::vital::algo::feature_descriptor_io>,
               kwiver::vital::algorithm,
               feature_descriptor_io_trampoline<> > instance(m,  "FeatureDescriptorIO");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::feature_descriptor_io::interface_name)
    .def("load", &kwiver::vital::algo::feature_descriptor_io::load, py::doc(R"( Load features and descriptors from a file

 \throws kwiver::vital::path_not_exists Thrown when the given path does not
 exist.

 \throws kwiver::vital::path_not_a_file Thrown when the given path does
    not point to a file (i.e. it points to a directory).

 \param filename the path to the file the load
 \param feat the set of features to load from the file
 \param desc the set of descriptors to load from the file)"), py::arg("filename"), py::arg("feat"), py::arg("desc"))
    .def("save", &kwiver::vital::algo::feature_descriptor_io::save, py::doc(R"( Save features and descriptors to a file

 Saves features and/or descriptors to a file.  Either \p feat or \p desc
 may be Null, but not both.  If both \p feat and \p desc are provided then
 the must be of the same size.

 \throws kwiver::vital::path_not_exists Thrown when the expected
    containing directory of the given path does not exist.

 \throws kwiver::vital::path_not_a_directory Thrown when the expected
    containing directory of the given path is not actually a
    directory.

 \param filename the path to the file to save
 \param feat the set of features to write to the file
 \param desc the set of descriptors to write to the file)"), py::arg("filename"), py::arg("feat"), py::arg("desc"))
    ;
  register_algorithm< kwiver::vital::algo::feature_descriptor_io > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
