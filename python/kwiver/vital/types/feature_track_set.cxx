// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/feature_track_set.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {

typedef kwiver::vital::feature_track_state feat_track_state;
typedef kwiver::vital::feature_track_set feat_track_set;

std::shared_ptr<feat_track_state>
new_feat_track_state(int64_t frame,
                     std::shared_ptr<PyFeatureBase> py_feat,
                     std::shared_ptr<kwiver::vital::descriptor> d)
{
  std::shared_ptr<kwiver::vital::feature> f;
  f = std::shared_ptr<kwiver::vital::feature>(py_feat->get_feature());
  return std::shared_ptr<feat_track_state>(new feat_track_state(frame, f, d));
}

std::shared_ptr<kwiver::vital::track>
get_track(std::shared_ptr<feat_track_set> &self, uint64_t id)
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
PYBIND11_MODULE(feature_track_set, m)
{
  py::class_<feat_track_state, kwiver::vital::track_state, std::shared_ptr<feat_track_state>>(m, "FeatureTrackState")
  .def(py::init(&new_feat_track_state))
  .def_property_readonly("frame_id", &kwiver::vital::track_state::frame)
  .def_readwrite("feature", &feat_track_state::feature)
  .def_readwrite("descriptor", &feat_track_state::descriptor)
  ;

  py::class_<feat_track_set, kwiver::vital::track_set, std::shared_ptr<feat_track_set>>(m, "FeatureTrackSet")
  .def(py::init<>())
  .def(py::init<std::vector<std::shared_ptr<kwiver::vital::track>>>())
  .def("all_frame_ids", &feat_track_set::all_frame_ids)
  .def("get_track", &get_track,
    py::arg("id"))
  .def("first_frame", &feat_track_set::first_frame)
  .def("last_frame", &feat_track_set::last_frame)
  .def("size", &feat_track_set::size)
  .def("tracks", &feat_track_set::tracks)
  .def("__len__", &feat_track_set::size)
  ;
}
