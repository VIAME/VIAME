// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/track_interval.h>

#include <pybind11/pybind11.h>

#include <memory>

namespace py=pybind11;
namespace kv=kwiver::vital;

PYBIND11_MODULE(track_interval, m)
{
  py::class_<kv::track_interval, std::shared_ptr<kv::track_interval>>(m, "TrackInterval")
  .def(py::init<>())
  .def(py::init<kv::track_id_t, kv::timestamp const&, kv::timestamp const&>())
  .def_readwrite("track", &kv::track_interval::track)
  .def_readwrite("start", &kv::track_interval::start)
  .def_readwrite("stop", &kv::track_interval::stop)
  ;
}
