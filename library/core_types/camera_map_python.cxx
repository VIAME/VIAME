// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/camera_map.h>
#include <viame/core_types/camera_perspective.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE( camera_map, m )
{
  py::module::import( "viame.types.camera_perspective" );

  py::class_< viame::simple_camera_map,
    std::shared_ptr< viame::simple_camera_map > >( m, "CameraMap" )
    .def( py::init<>() )
    .def(
      py::init(
        [](py::dict dict){
          std::map< viame::frame_id_t, viame::camera_sptr > cm;
          for( auto item : dict )
          {
            auto const c = item.second.cast< viame::camera* >();
            auto const cp_ptr =
              dynamic_cast< viame::simple_camera_perspective& >( *c );
            auto c_ptr =
              std::make_shared< viame::simple_camera_perspective >(
                cp_ptr );
            cm.insert(
              std::make_pair(
                item.first.cast< viame::frame_id_t >(),
                c_ptr ) );
          }
          return viame::simple_camera_map( cm );
        } ), py::arg( "cameras" ) )
    .def_property_readonly( "size", &viame::simple_camera_map::size )
    .def(
      "as_dict", [](viame::simple_camera_map& cm){
        std::map< viame::frame_id_t,
          viame::simple_camera_perspective > dict;
        auto cam_list = cm.cameras();
        for( auto item : cam_list )
        {
          auto cam_ptr =
            std::dynamic_pointer_cast< viame::camera_perspective >(
              item.second );
          viame::simple_camera_perspective cam( *( cam_ptr ) );
          dict.insert( std::make_pair( item.first, cam ) );
        }
        return dict;
      } )
  ;
}
