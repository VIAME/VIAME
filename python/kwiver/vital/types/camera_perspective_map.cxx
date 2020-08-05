/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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


#include <vital/types/camera_perspective.h>
#include <vital/types/camera_perspective_map.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <python/kwiver/vital/util/pybind11.h>
#include <pybind11/eigen.h>

namespace py=pybind11;
namespace kv=kwiver::vital;

typedef std::map< kv::frame_id_t, std::shared_ptr< kv::camera_perspective > > frame_to_cp_sptr_map;
typedef std::map< kv::frame_id_t, kv::camera_sptr > map_camera_t;

class camera_perspective_map_trampoline
  : public kv::camera_map_of_<kv::camera_perspective>
{
    using kv::camera_map_of_< kv::camera_perspective >::camera_map_of_;
    virtual size_t size() const;
    virtual map_camera_t cameras() const;
    virtual std::set<kv::frame_id_t> get_frame_ids() const;
};

PYBIND11_MODULE( camera_perspective_map, m)
{
    py::class_< kv::camera_map_of_< kv::camera_perspective >,
                std::shared_ptr< kv::camera_map_of_< kv::camera_perspective > >,
                camera_perspective_map_trampoline >(m, "CameraPerspectiveMap" )
    .def( py::init<>() )
    .def( py::init< frame_to_cp_sptr_map >() )
    .def( "size",                       &kv::camera_map_of_< kv::camera_perspective >::size )
    .def( "cameras",                    &kv::camera_map_of_< kv::camera_perspective >::cameras )
    .def( "get_frame_ids",              &kv::camera_map_of_< kv::camera_perspective >::get_frame_ids )
    .def( "find",                       &kv::camera_map_of_< kv::camera_perspective >::find )
    .def( "erase",                      &kv::camera_map_of_< kv::camera_perspective >::erase )
    .def( "insert",                     &kv::camera_map_of_< kv::camera_perspective >::insert )
    .def( "clear",                      &kv::camera_map_of_< kv::camera_perspective >::clear )
    .def( "set_from_base_camera_map",   &kv::camera_map_of_< kv::camera_perspective >::set_from_base_camera_map )
    .def( "clone",                      &kv::camera_map_of_< kv::camera_perspective >::clone )
    ;

}


size_t
camera_perspective_map_trampoline
::size() const
{
    VITAL_PYBIND11_OVERLOAD(
        size_t,
        kv::camera_map_of_< kv::camera_perspective >,
        size,
    );
}

map_camera_t
camera_perspective_map_trampoline
::cameras() const
{
    VITAL_PYBIND11_OVERLOAD(
        map_camera_t,
        kv::camera_map_of_< kv::camera_perspective >,
        cameras,
    );
}

std::set<kv::frame_id_t>
camera_perspective_map_trampoline
::get_frame_ids() const
{
    VITAL_PYBIND11_OVERLOAD(
        std::set<kv::frame_id_t>,
        kv::camera_map_of_< kv::camera_perspective >,
        get_frame_ids,

    );
}
