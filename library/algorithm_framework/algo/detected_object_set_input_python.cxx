// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/detected_object_set_input.h>
#include "algorithm_python.txx"
#include "detected_object_set_input_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void detected_object_set_input(py::module& m)
{
  py::module::import("viame.config");
  py::module::import("viame.types");

    py::class_<viame::algo::detected_object_set_input,
               std::shared_ptr<viame::algo::detected_object_set_input>,
               viame::algorithm,
               detected_object_set_input_trampoline<> > instance(m,  "DetectedObjectSetInput");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::detected_object_set_input::interface_name)
    .def("open", &viame::algo::detected_object_set_input::open, py::doc(R"( Open a file of detection sets.

 This method opens a detection set file for reading.

 \param filename Name of file to open

 \throws viame::path_not_exists Thrown when the given path does not
         exist.

 \throws viame::path_not_a_file Thrown when the given path does
         not point to a file (i.e. it points to a directory).

 \throws viame::file_not_found_exception)"), py::arg("filename"))
    .def("use_stream", &viame::algo::detected_object_set_input::use_stream, py::doc(R"( Read detections from an existing stream

 This method specifies the input stream to use for reading
 detections. Using a stream is handy when the detections are
 available in a stream format.

 @param strm input stream to use)"), py::arg("strm"))
    .def("close", &viame::algo::detected_object_set_input::close, py::doc(R"( Close detection set file.

 The currently open detection set file is closed. If there is no
 currently open file, then this method does nothing.)"))
    .def("read_set", (bool (viame::algo::detected_object_set_input::*)(::viame::detected_object_set_sptr &, ::std::string &)) &viame::algo::detected_object_set_input::read_set, py::doc(R"( Read next detected object set

 This method reads the next set of detected objects from the
 file. \b False is returned when the end of file is reached.

 \param[out] set Pointer to the new set of detections. Set may be
 empty if there are no detections on an image.

 \param[out] image_name Name of the image that goes with the
 detections. This string may be empty depending on the source
 format.

 @return \b true if detections are returned, \b false if end of file.)"), py::arg("set"), py::arg("image_name"))
    .def("read_set", (std::pair<std::shared_ptr<viame::detected_object_set>, std::basic_string<char> > (viame::algo::detected_object_set_input::*)()) &viame::algo::detected_object_set_input::read_set)
    .def("at_eof", &viame::algo::detected_object_set_input::at_eof, py::doc(R"( Determine if input file is at end of file.

 This method reports the end of file status for a file open for reading.

 @return \b true if file is at end.)"))
    ;
  register_algorithm< viame::algo::detected_object_set_input > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
