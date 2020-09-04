// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/camera_map.h>
#include <vital/types/camera_perspective.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py=pybind11;

PYBIND11_MODULE(camera_map, m)
{
  py::class_<kwiver::vital::simple_camera_map, std::shared_ptr<kwiver::vital::simple_camera_map> >(m, "CameraMap")
    .def(py::init<>())
    .def(py::init([](py::dict dict)
                    {
                      std::map<kwiver::vital::frame_id_t, kwiver::vital::camera_sptr> cm;
                      for( auto item : dict)
                      {
                        auto const c = item.second.cast< kwiver::vital::camera* >();
                        auto const cp_ptr = dynamic_cast< kwiver::vital::simple_camera_perspective& >(*c);
                        auto c_ptr = std::make_shared<kwiver::vital::simple_camera_perspective>(cp_ptr);
                        cm.insert(std::make_pair(
                                    item.first.cast<kwiver::vital::frame_id_t>(),
                                    c_ptr));
                      }
                      return kwiver::vital::simple_camera_map(cm);
                    }))
    .def_property_readonly("size", &kwiver::vital::simple_camera_map::size)
    .def("as_dict", [](kwiver::vital::simple_camera_map &cm)
                      {
                        std::map<kwiver::vital::frame_id_t, kwiver::vital::simple_camera_perspective> dict;
                        auto cam_list = cm.cameras();
                        for( auto item : cam_list)
                        {
                          auto cam_ptr =
                            std::dynamic_pointer_cast<kwiver::vital::camera_perspective>( item.second );
                          kwiver::vital::simple_camera_perspective cam(*(cam_ptr));
                          dict.insert(std::make_pair(item.first, cam));
                        }
                        return dict;
                      })
  ;
}
