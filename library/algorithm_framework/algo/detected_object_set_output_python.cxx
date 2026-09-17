// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/detected_object_set_output.h>
#include "algorithm_python.txx"
#include "detected_object_set_output_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void detected_object_set_output(py::module& m)
{
  py::module::import("viame.config");
  py::module::import("viame.types");

    py::class_<viame::algo::detected_object_set_output,
               std::shared_ptr<viame::algo::detected_object_set_output>,
               viame::algorithm,
               detected_object_set_output_trampoline<> > instance(m,  "DetectedObjectSetOutput");
    
    instance
    .def(py::init<>())
    .def_property("filename", &viame::algo::detected_object_set_output::get_filename, &viame::algo::detected_object_set_output::set_filename)
    .def_static("interface_name", &viame::algo::detected_object_set_output::interface_name)
    .def("open", &viame::algo::detected_object_set_output::open, py::doc(R"( Open a file of detection sets.

 This method opens a detection set file for writing.

 \param filename Name of file to open

 \throws viame::path_not_exists Thrown when the given path does not
 exist.

 \throws viame::path_not_a_file Thrown when the given path does
    not point to a file (i.e. it points to a directory).)"), py::arg("filename"))
    .def("use_stream", &viame::algo::detected_object_set_output::use_stream, py::doc(R"( Write detections to an existing stream

 This method specifies the output stream to use for writing
 detections. Using a stream is handy when the detections output is
 available in a stream format.

 @param strm output stream to use)"), py::arg("strm"))
    .def("close", &viame::algo::detected_object_set_output::close, py::doc(R"( Close detection set file.

 The currently open detection set file is closed. If there is no
 currently open file, then this method does nothing.)"))
    .def("write_set", &viame::algo::detected_object_set_output::write_set, py::doc(R"( Write detected object set.

 This method writes the specified detected object set and image
 name to the currently open file.

 \param set Detected object set
 \param image_path File path to image associated with the detections.)"), py::arg("set"), py::arg("image_path"))
    .def("complete", &viame::algo::detected_object_set_output::complete, py::doc(R"( Perform end-of-stream actions.

 This method writes any necessary final data to the currently open file.)"))
    .def("get_filename", &viame::algo::detected_object_set_output::get_filename, py::doc(R"(@{
 Filename property
 @note  Required for accessing it as a python property)"))
    .def("set_filename", &viame::algo::detected_object_set_output::set_filename, py::arg("filename"))
    ;
  register_algorithm< viame::algo::detected_object_set_output > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
