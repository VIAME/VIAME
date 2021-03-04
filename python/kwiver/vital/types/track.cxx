// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/track.h>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py=pybind11;
namespace kv=kwiver::vital;

namespace kwiver {
namespace vital  {
namespace python {

py::object
track_find_state(kv::track &self, int64_t frame_id)
{
  auto frame_itr = self.find(frame_id);
  if(frame_itr == self.end())
  {
    throw py::index_error();
  }
  return py::cast<std::shared_ptr<kv::track_state>>(*frame_itr);
}
}
}
}

using namespace kwiver::vital::python;
PYBIND11_MODULE(track, m)
{
  py::class_<kv::track_state, std::shared_ptr<kv::track_state>>(m, "TrackState")
  .def(py::init<int64_t>())
  .def(py::self == py::self)
  .def_property_readonly("frame_id", &kv::track_state::frame)
  ;

  py::class_<kv::track, std::shared_ptr<kv::track>>(m, "Track")
  .def(py::init([](int64_t id)
    {
      auto track = kv::track::create();
      track->set_id(id);
      return track;
    }),
    py::arg("id")=0)
  .def(py::init([](int64_t id, std::vector<std::shared_ptr<kv::track_state>> states)
    {
      auto track = kv::track::create();
      track->set_id(id);
      for(auto state : states)
      {
        track->append(state);
      }
      return track;
    }))
  .def("all_frame_ids", &kv::track::all_frame_ids)
  .def("front", [](kv::track &self)
    {
      return *self.front();
    })
  .def("back", [](kv::track &self)
    {
      return *self.back();
    })
  .def("append", [](kv::track &self, std::shared_ptr<kv::track_state> track_state)
    {
      return self.append(track_state);
    })
  .def("append", [](kv::track &self, kv::track &track)
    {
      return self.append(track);
    })
  .def("find_state", &track_find_state)
  .def("__iter__", [](const kv::track &self)
    {
      return py::make_iterator(self.begin(), self.end());
    }, py::keep_alive<0,1>())
  .def("__len__", &kv::track::size)
  .def("__getitem__", &track_find_state)
  .def_property("id", &kv::track::id, &kv::track::set_id)
  .def_property_readonly("size", &kv::track::size)
  .def_property_readonly("is_empty", &kv::track::empty)
  .def_property_readonly("first_frame", &kv::track::first_frame)
  .def_property_readonly("last_frame", &kv::track::last_frame)
  ;
}
