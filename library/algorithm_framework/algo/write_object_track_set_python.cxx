// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/write_object_track_set.h>
#include "algorithm_python.txx"
#include "write_object_track_set_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void write_object_track_set(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<viame::algo::write_object_track_set,
               std::shared_ptr<viame::algo::write_object_track_set>,
               viame::algorithm,
               write_object_track_set_trampoline<> > instance(m,  "WriteObjectTrackSet");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::write_object_track_set::interface_name)
    .def("open", &viame::algo::write_object_track_set::open, py::doc(R"( Open a file of object track sets.

 This method opens a object track set file for reading.

 \param filename Name of file to open

 \throws viame::path_not_exists Thrown when the given path does not
 exist.

 \throws viame::path_not_a_file Thrown when the given path does
    not point to a file (i.e. it points to a directory).)"), py::arg("filename"))
    .def("use_stream", &viame::algo::write_object_track_set::use_stream, py::doc(R"( Write object tracks to an existing stream

 This method specifies the output stream to use for writing
 object tracks.

 @param strm output stream to use)"), py::arg("strm"))
    .def("close", &viame::algo::write_object_track_set::close, py::doc(R"( Close object track set file.

 The currently open object track set file is closed. If there is no
 currently open file, then this method does nothing.)"))
    .def("write_set", &viame::algo::write_object_track_set::write_set, py::doc(R"( Write object track set.

 This method writes the specified object track set and image
 name to the currently open file.

 \param set Track object set
 \param ts Timestamp for the current frame
 \param frame_identifier Identifier for the current frame (e.g. file name))"), py::arg("set"), py::arg("ts"), py::arg("frame_identifier"))
    ;
  register_algorithm< viame::algo::write_object_track_set > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
