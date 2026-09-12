// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/track_objects.h>
#include "algorithm_python.txx"
#include "track_objects_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void track_objects(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::track_objects,
               std::shared_ptr<kwiver::vital::algo::track_objects>,
               kwiver::vital::algorithm,
               track_objects_trampoline<> > instance(m,  "TrackObjects");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::track_objects::interface_name)
    .def("track", (kwiver::vital::object_track_set_sptr (kwiver::vital::algo::track_objects::*)(::kwiver::vital::timestamp, ::kwiver::vital::image_container_sptr, ::kwiver::vital::detected_object_set_sptr) const) &kwiver::vital::algo::track_objects::track, py::doc(R"( Track objects in a new frame.

 This is the primary tracking method that processes a single frame.
 It takes the current frame's detections and optional additional
 inputs, updates internal track state, and returns the current
 set of active tracks.

 \param ts Timestamp for the current frame
 \param image The input image for the current frame (may be null
              for trackers that don't require image data)
 \param detections Detected objects from the current frame
 \returns Updated object track set containing all active tracks)"), py::arg("ts"), py::arg("image"), py::arg("detections"))
    .def("track", (kwiver::vital::object_track_set_sptr (kwiver::vital::algo::track_objects::*)(::kwiver::vital::timestamp, ::kwiver::vital::image_container_sptr, ::kwiver::vital::detected_object_set_sptr, ::kwiver::vital::f2f_homography_sptr) const) &kwiver::vital::algo::track_objects::track, py::doc(R"( Track objects with homography support.

 This overload supports trackers that use frame-to-frame or
 frame-to-reference homographies for camera motion compensation.
 This is useful for aerial or moving camera scenarios.

 \param ts Timestamp for the current frame
 \param image The input image for the current frame
 \param detections Detected objects from the current frame
 \param src_to_ref Homography from source (current frame) to
                   reference coordinates
 \returns Updated object track set containing all active tracks)"), py::arg("ts"), py::arg("image"), py::arg("detections"), py::arg("src_to_ref"))
    .def("track", (kwiver::vital::object_track_set_sptr (kwiver::vital::algo::track_objects::*)(::kwiver::vital::timestamp, ::kwiver::vital::image_container_sptr, ::kwiver::vital::detected_object_set_sptr, ::kwiver::vital::object_track_set_sptr) const) &kwiver::vital::algo::track_objects::track, py::doc(R"( Track objects with existing tracks provided.

 This overload allows passing in existing tracks for continuation
 or re-initialization scenarios. Useful for multi-stage tracking
 pipelines or when tracks need to be initialized from external sources.

 \param ts Timestamp for the current frame
 \param image The input image for the current frame
 \param detections Detected objects from the current frame
 \param existing_tracks Previously computed tracks to continue
 \returns Updated object track set with both existing and new tracks)"), py::arg("ts"), py::arg("image"), py::arg("detections"), py::arg("existing_tracks"))
    .def("initialize", &kwiver::vital::algo::track_objects::initialize, py::doc(R"( Initialize the tracker for a new sequence.

 Called at the start of a new video sequence to reset internal
 state and prepare for tracking. Some trackers may require
 initialization with seed detections or bounding boxes.

 \param ts Initial timestamp
 \param image Initial frame image
 \param seed_detections Optional initial detections to seed tracks
 \returns Initial track set (may be empty if no seeds provided))"), py::arg("ts"), py::arg("image"), py::arg("seed_detections"))
    .def("finalize", &kwiver::vital::algo::track_objects::finalize, py::doc(R"( Finalize tracking and return all tracks.

 Called at the end of a sequence to perform any final processing
 and return the complete set of tracks. This may include tracks
 that were previously lost but should still be returned.

 \returns Final object track set with all tracks from the sequence)"))
    .def("reset", &kwiver::vital::algo::track_objects::reset, py::doc(R"( Reset the tracker state.

 Clears all internal state, active tracks, and cached data.
 After reset, the tracker is ready to begin a new sequence.)"))
    ;
  register_algorithm< kwiver::vital::algo::track_objects > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
