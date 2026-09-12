// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/write_track_descriptor_set.h>
#include "algorithm_python.txx"
#include "write_track_descriptor_set_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void write_track_descriptor_set(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::write_track_descriptor_set,
               std::shared_ptr<kwiver::vital::algo::write_track_descriptor_set>,
               kwiver::vital::algorithm,
               write_track_descriptor_set_trampoline<> > instance(m,  "WriteTrackDescriptorSet");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::write_track_descriptor_set::interface_name)
    .def("open", &kwiver::vital::algo::write_track_descriptor_set::open, py::doc(R"( Open a file of track descriptor sets.

 This method opens a track descriptor set file for reading.

 \param filename Name of file to open

 \throws kwiver::vital::path_not_exists Thrown when the given path does not
 exist.

 \throws kwiver::vital::path_not_a_file Thrown when the given path does
    not point to a file (i.e. it points to a directory).)"), py::arg("filename"))
    .def("use_stream", &kwiver::vital::algo::write_track_descriptor_set::use_stream, py::doc(R"( Write track descriptors to an existing stream

 This method specifies the output stream to use for reading
 track descriptors. Using a stream is handy when the track descriptors are
 available in a stream format.

 @param strm output stream to use)"), py::arg("strm"))
    .def("close", &kwiver::vital::algo::write_track_descriptor_set::close, py::doc(R"( Close track descriptor set file.

 The currently open track descriptor set file is closed. If there is no
 currently open file, then this method does nothing.)"))
    .def("write_set", &kwiver::vital::algo::write_track_descriptor_set::write_set, py::doc(R"( Write detected object set.

 This method writes the specified detected object to file.

 \param set Detected object set
 \param source_id String source ID)"), py::arg("set"), py::arg("source_id"))
    ;
  register_algorithm< kwiver::vital::algo::write_track_descriptor_set > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
