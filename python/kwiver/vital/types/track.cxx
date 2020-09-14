// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/track.h>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py=pybind11;
namespace kwiver {
namespace vital  {
namespace python {

py::object
track_find_state(kwiver::vital::track &self, int64_t frame_id)
{
  auto frame_itr = self.find(frame_id);
  if(frame_itr == self.end())
  {
    throw py::index_error();
  }
  return py::cast<std::shared_ptr<kwiver::vital::track_state>>(*frame_itr);
}
}
}
}

using namespace kwiver::vital::python;
PYBIND11_MODULE(track, m)
{
  py::class_<kwiver::vital::track_state, std::shared_ptr<kwiver::vital::track_state>>(m, "TrackState")
  .def(py::init<int64_t>())
  .def(py::self == py::self)
  .def_property("frame_id", &kwiver::vital::track_state::frame, &kwiver::vital::track_state::set_frame)
  ;

  py::class_<kwiver::vital::track, std::shared_ptr<kwiver::vital::track>>(m, "Track")
  .def(py::init([](int64_t id)
    {
      auto track = kwiver::vital::track::create();
      track->set_id(id);
      return track;
    }),
    py::arg("id")=0)
  .def("all_frame_ids", &kwiver::vital::track::all_frame_ids)
  .def("append", [](kwiver::vital::track &self, std::shared_ptr<kwiver::vital::track_state> track_state)
    {
      return self.append(track_state);
    })
  .def("append", [](kwiver::vital::track &self, kwiver::vital::track &track)
    {
      return self.append(track);
    })
  .def("find_state", &track_find_state)
  .def("__iter__", [](const kwiver::vital::track &self)
    {
      return py::make_iterator(self.begin(), self.end());
    }, py::keep_alive<0,1>())
  .def("__len__", &kwiver::vital::track::size)
  .def("__getitem__", &track_find_state)
  .def_property("id", &kwiver::vital::track::id, &kwiver::vital::track::set_id)
  .def_property_readonly("size", &kwiver::vital::track::size)
  .def_property_readonly("is_empty", &kwiver::vital::track::empty)
  .def_property_readonly("first_frame", &kwiver::vital::track::first_frame)
  .def_property_readonly("last_frame", &kwiver::vital::track::last_frame)
  ;
}
