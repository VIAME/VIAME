// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/object_track_set.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

typedef kwiver::vital::object_track_state obj_track_state;
typedef kwiver::vital::object_track_set obj_track_set;

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {

std::shared_ptr<kwiver::vital::track>
get_track(std::shared_ptr<obj_track_set> &self, uint64_t id)
{
  auto track = self->get_track(id);
  if(!track)
  {
    throw py::index_error("Track does not exist in set");
  }
  return track;
}

}}}
using namespace kwiver::vital::python;
PYBIND11_MODULE(object_track_set, m)
{
  py::class_<obj_track_state, kwiver::vital::track_state, std::shared_ptr<obj_track_state>>(m, "ObjectTrackState")
  .def(py::init<int64_t, int64_t, std::shared_ptr<kwiver::vital::detected_object>>())
  .def_property_readonly("frame_id", &kwiver::vital::track_state::frame)
  .def_property_readonly("time_usec", &kwiver::vital::object_track_state::time)
  .def("detection", (kwiver::vital::detected_object_sptr(obj_track_state::*)())&obj_track_state::detection)
  .def("image_point", &obj_track_state::image_point)
  .def("track_point", &obj_track_state::track_point)
  ;

  py::class_<obj_track_set, kwiver::vital::track_set, std::shared_ptr<obj_track_set>>(m, "ObjectTrackSet")
  .def(py::init<>())
  .def(py::init<std::vector<std::shared_ptr<kwiver::vital::track>>>())
  .def("all_frame_ids", &obj_track_set::all_frame_ids)
  .def("get_track", &get_track,
    py::arg("id"))
  .def("first_frame", &obj_track_set::first_frame)
  .def("last_frame", &obj_track_set::last_frame)
  .def("size", &obj_track_set::size)
  .def("tracks", &obj_track_set::tracks)
  .def("__len__", &obj_track_set::size)
  ;
}
