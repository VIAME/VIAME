/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
                        cm.insert(std::make_pair(
                                    item.first.cast<kwiver::vital::frame_id_t>(),
                                    item.second.cast<std::shared_ptr<kwiver::vital::simple_camera_perspective>>()));
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
