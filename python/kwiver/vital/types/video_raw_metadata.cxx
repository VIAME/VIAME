// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/video_raw_metadata.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE( video_raw_metadata, m )
{
  py::class_< kwiver::vital::video_raw_metadata,
    kwiver::vital::video_raw_metadata_sptr >( m, "VideoRawMetadata" )
    .def( py::init<>() );
}
