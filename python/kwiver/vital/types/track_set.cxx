// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/track_set.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py=pybind11;
namespace kwiver {
namespace vital  {
namespace python {

std::shared_ptr<kwiver::vital::track>
get_track(std::shared_ptr<kwiver::vital::track_set> &self, uint64_t id)
{
  auto track = self->get_track(id);
  if(!track)
  {
    throw py::index_error("Track does not exist in set");
  }
  return track;
}
}
}
}

using namespace kwiver::vital::python;
PYBIND11_MODULE(track_set, m)
{
  py::class_<kwiver::vital::track_set, std::shared_ptr<kwiver::vital::track_set>>(m, "TrackSet")
  .def(py::init<>())
  .def(py::init<std::vector<std::shared_ptr<kwiver::vital::track>>>())
  .def("all_frame_ids", &kwiver::vital::track_set::all_frame_ids)
  .def("get_track", &get_track,
    py::arg("id"))
  .def("first_frame", &kwiver::vital::track_set::first_frame)
  .def("last_frame", &kwiver::vital::track_set::last_frame)
  .def("size", &kwiver::vital::track_set::size)
  .def("tracks", &kwiver::vital::track_set::tracks)
  .def("__len__", &kwiver::vital::track_set::size)
  ;
}
