// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/read_object_track_set.h>
#include "algorithm_python.txx"
#include "read_object_track_set_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void read_object_track_set(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::read_object_track_set,
               std::shared_ptr<kwiver::vital::algo::read_object_track_set>,
               kwiver::vital::algorithm,
               read_object_track_set_trampoline<> > instance(m,  "ReadObjectTrackSet");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::read_object_track_set::interface_name)
    .def("open", &kwiver::vital::algo::read_object_track_set::open, py::doc(R"( Open a file of object track sets.

 This method opens a object track set file for reading.

 \param filename Name of file to open

 \throws kwiver::vital::path_not_exists Thrown when the given path does not
    exist.

 \throws kwiver::vital::path_not_a_file Thrown when the given path does
    not point to a file (i.e. it points to a directory).

 \throws kwiver::vital::file_not_found_exception)"), py::arg("filename"))
    .def("use_stream", &kwiver::vital::algo::read_object_track_set::use_stream, py::doc(R"( Read object tracks from an existing stream

 This method specifies the input stream to use for reading
 object tracks. Using a stream is handy when the object tracks are
 available in a stream format.

 @param strm input stream to use)"), py::arg("strm"))
    .def("close", &kwiver::vital::algo::read_object_track_set::close, py::doc(R"( Close object track set file.

 The currently open object track set file is closed. If there is no
 currently open file, then this method does nothing.)"))
    .def("read_set", &kwiver::vital::algo::read_object_track_set::read_set, py::doc(R"( Read next object track set

 This method reads the next set of track objects from the
 file. \b False is returned when the end of file is reached.

 \param[out] set Pointer to the new set of object tracks. Set may be
 empty if there are no object tracks on an image.

 @return \b true if object tracks are returned, \b false if end of file.)"), py::arg("set"))
    .def("at_eof", &kwiver::vital::algo::read_object_track_set::at_eof, py::doc(R"( Determine if input file is at end of file.

 This method reports the end of file status for a file open for reading.

 @return \b true if file is at end.)"))
    .def("stream", &kwiver::vital::algo::read_object_track_set::stream)
    .def("new_stream", &kwiver::vital::algo::read_object_track_set::new_stream)
    ;
  register_algorithm< kwiver::vital::algo::read_object_track_set > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
