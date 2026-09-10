// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/camera_perspective.h>
#include <viame/core_types/camera_perspective_map.h>

#include <viame/algorithm_framework/overload.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace kv = kwiver::vital;

typedef std::map< kv::frame_id_t,
  std::shared_ptr< kv::camera_perspective > > frame_to_cp_sptr_map;
typedef std::map< kv::frame_id_t, kv::camera_sptr > map_camera_t;

class camera_perspective_map_trampoline
  : public kv::camera_map_of_< kv::camera_perspective >
{
  using kv::camera_map_of_< kv::camera_perspective >::camera_map_of_;
  virtual size_t size() const;
  virtual map_camera_t cameras() const;
  virtual std::set< kv::frame_id_t > get_frame_ids() const;
};

PYBIND11_MODULE( camera_perspective_map, m )
{
  using cam_sptr_t = std::shared_ptr< kv::camera_perspective >;

  py::class_< kv::camera_map_of_< kv::camera_perspective >,
    std::shared_ptr< kv::camera_map_of_< kv::camera_perspective > >,
    camera_perspective_map_trampoline >( m, "CameraPerspectiveMap" )
    .def( py::init<>() )
    .def( py::init< frame_to_cp_sptr_map >() )
    .def(
      "size",
      &kv::camera_map_of_< kv::camera_perspective >::size )
    .def(
      "cameras",
      &kv::camera_map_of_< kv::camera_perspective >::cameras )
    .def(
      "get_frame_ids",
      &kv::camera_map_of_< kv::camera_perspective >::get_frame_ids )
    .def(
      "find",
      &kv::camera_map_of_< kv::camera_perspective >::find )
    .def(
      "erase",
      &kv::camera_map_of_< kv::camera_perspective >::erase )
    .def(
      "insert",
      kv::overload< kv::frame_id_t, cam_sptr_t const& >(
        &kv::camera_map_of_< kv::camera_perspective >::insert ) )
    .def(
      "clear",
      &kv::camera_map_of_< kv::camera_perspective >::clear )
    .def(
      "set_from_base_camera_map",
      &kv::camera_map_of_< kv::camera_perspective >::set_from_base_camera_map )
    .def(
      "clone",
      &kv::camera_map_of_< kv::camera_perspective >::clone )
  ;
}

size_t
camera_perspective_map_trampoline
::size() const
{
  PYBIND11_OVERLOAD(
    size_t,
    kv::camera_map_of_< kv::camera_perspective >,
    size,
  );
}

map_camera_t
camera_perspective_map_trampoline
::cameras() const
{
  PYBIND11_OVERLOAD(
    map_camera_t,
    kv::camera_map_of_< kv::camera_perspective >,
    cameras,
  );
}

std::set< kv::frame_id_t >
camera_perspective_map_trampoline
::get_frame_ids() const
{
  PYBIND11_OVERLOAD(
    std::set< kv::frame_id_t >,
    kv::camera_map_of_< kv::camera_perspective >,
    get_frame_ids,

  );
}
