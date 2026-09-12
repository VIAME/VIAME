// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/video_settings.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace kwiver::vital;

// ----------------------------------------------------------------------------
PYBIND11_MODULE( video_settings, m )
{
  py::class_< video_settings, video_settings_sptr >( m, "BaseVideoSettings" )
    .def( "width", &video_settings::width )
    .def( "height", &video_settings::height )
    .def( "frame_rate", &video_settings::frame_rate )
  ;

  py::class_< simple_video_settings, video_settings,
    simple_video_settings_sptr >( m, "VideoSettings" )
    .def(
    py::init< size_t, size_t, double >(), py::arg( "width" ),
    py::arg( "height" ), py::arg( "frame_rate" ) = -1.0 )
    .def( "width", &video_settings::width )
    .def( "height", &video_settings::height )
    .def( "frame_rate", &video_settings::frame_rate )
  ;
}
