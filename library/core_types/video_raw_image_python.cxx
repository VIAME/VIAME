// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/video_raw_image.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE( video_raw_image, m )
{
  py::class_< kwiver::vital::video_raw_image,
    kwiver::vital::video_raw_image_sptr >( m, "VideoRawImage" )
    .def( py::init<>() );
}
