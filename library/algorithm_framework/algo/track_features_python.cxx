// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/track_features.h>
#include "algorithm_python.txx"
#include "track_features_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void track_features(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::track_features,
               std::shared_ptr<kwiver::vital::algo::track_features>,
               kwiver::vital::algorithm,
               track_features_trampoline<> > instance(m,  "TrackFeatures");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::track_features::interface_name)
    .def("track", &kwiver::vital::algo::track_features::track, py::doc(R"( Extend a previous set of feature tracks using the current frame

 \throws image_size_mismatch_exception
    When the given non-zero mask image does not match the size of the
    dimensions of the given image data.

 \param [in] prev_tracks the feature tracks from previous tracking steps
 \param [in] frame_number the frame number of the current frame
 \param [in] image_data the image pixels for the current frame
 \param [in] mask Optional mask image that uses positive values to denote
                  regions of the input image to consider for feature
                  tracking. An empty sptr indicates no mask (default
                  value).
 \returns an updated set of feature tracks including the current frame)"), py::arg("prev_tracks"), py::arg("frame_number"), py::arg("image_data"), py::arg("mask"))
    ;
  register_algorithm< kwiver::vital::algo::track_features > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
