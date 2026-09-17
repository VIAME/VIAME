// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/video_output.h>
#include "algorithm_python.txx"
#include "video_output_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void video_output(py::module& m)
{
  py::module::import("viame.config");
  py::module::import("viame.types");

    py::class_<viame::algo::video_output,
               std::shared_ptr<viame::algo::video_output>,
               viame::algorithm,
               video_output_trampoline<> > instance(m,  "VideoOutput");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::video_output::interface_name)
    .def("open", &viame::algo::video_output::open, py::doc(R"( Open a video stream.

 This method opens the specified video stream for writing. The format of
 the name depends on the concrete implementation. It could be a file name,
 a directory, or a URI.

 \param video_name Identifier of the video stream.
 \param settings
   Additional information used to configure the video output.

 \throws exception Thrown if opening the video stream failed.)"), py::arg("video_name"), py::arg("settings"))
    .def("close", &viame::algo::video_output::close, py::doc(R"( Close video stream.

 Close the currently opened stream and release resources. Closing a stream
 that is already closed does not cause a problem.)"))
    .def("good", &viame::algo::video_output::good, py::doc(R"( Check whether state of video stream is good.

 This method checks the current state of the video stream to see if it is
 good. A stream is good if it is ready to receive images and/or metadata.

 \return \c true if video stream is good, \c false if not good.)"))
    .def("add_image", (void (viame::algo::video_output::*)(::viame::image_container_sptr const &, ::viame::timestamp const &)) &viame::algo::video_output::add_image, py::doc(R"( Add a frame image to the video stream.

 This method writes the next frame image to the video stream. The
 timestamp should be greater than that of the previously written frame, as
 many implementations are unable to write frames out of order.

 \throws video_stream_exception
   Thrown if is an error in the video stream.)"), py::arg("image"), py::arg("ts"))
    .def("add_image", (void (viame::algo::video_output::*)(::viame::video_raw_image const &)) &viame::algo::video_output::add_image, py::doc(R"( Add a raw frame image to the video stream.

 This method writes the raw image to the video stream. There is no
 guarantee that this functions correctly when intermixed with non-raw
 images.)"), py::arg("image"))
    .def("add_metadata", (void (viame::algo::video_output::*)(::viame::metadata const &)) &viame::algo::video_output::add_metadata, py::doc(R"( Add metadata collection to the video stream.

 This method adds metadata to the video stream. Depending on the
 implementation, the metadata may be written immediately, or may be
 deferred until the next frame is written. For this reason, the metadata's
 timestamp should be greater than that of the previously written frame.

 For implementations that do not support metadata, this method does
 nothing.

 \throws video_stream_exception
   Thrown if is an error in the video stream.)"), py::arg("md"))
    .def("add_metadata", (void (viame::algo::video_output::*)(::viame::video_raw_metadata const &)) &viame::algo::video_output::add_metadata, py::doc(R"( Add a frame of raw metadata to the video stream.

 This method writes the raw metadata to the video stream. There is no
 guarantee that this functions correctly when intermixed with non-raw
 metadata.)"), py::arg("md"))
    .def("add_uninterpreted_data", &viame::algo::video_output::add_uninterpreted_data, py::doc(R"( Add a frame of uninterpreted data to the video stream.

 This method writes the uninterpreted data to the video stream.)"), py::arg("misc_data"))
    .def("implementation_settings", &viame::algo::video_output::implementation_settings, py::doc(R"( Extract implementation-specific video encoding settings.

 The returned structure is intended to be passed to a video encoder of
 similar implementation to produce similarly formatted output. The
 returned value may not be identical to the one passed to this object via
 open().

 \return Implementation video settings, or \c nullptr if none are needed.)"))
    .def("get_implementation_capabilities", &viame::algo::video_output::get_implementation_capabilities, py::doc(R"( Return capabilities of concrete implementation.

 This method returns the capabilities of the algorithm implementation.

 \return Reference to supported algorithm capabilities.)"))
    .def_readonly_static("SUPPORTS_FRAME_RATE", &viame::algo::video_output::SUPPORTS_FRAME_RATE)
    .def_readonly_static("SUPPORTS_FRAME_TIME", &viame::algo::video_output::SUPPORTS_FRAME_TIME)
    .def_readonly_static("SUPPORTS_METADATA", &viame::algo::video_output::SUPPORTS_METADATA)
    .def_readonly_static("SUPPORTS_UNINTERPRETED_DATA", &viame::algo::video_output::SUPPORTS_UNINTERPRETED_DATA)
    ;
  register_algorithm< viame::algo::video_output > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
