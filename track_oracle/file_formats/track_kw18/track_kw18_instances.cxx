// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_box_2d.h>

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_kw18/track_kw18_export.h>

#define TRACK_FIELD_IO_PROXY_EXPORT TRACK_KW18_EXPORT

#include <track_oracle/core/track_field_io_proxy_instantiation.h>

TRACK_FIELD_IO_PROXY_INSTANCES( int );
TRACK_FIELD_IO_PROXY_INSTANCES( vgl_point_2d<double> );
TRACK_FIELD_IO_PROXY_INSTANCES( vgl_box_2d<double> );
TRACK_FIELD_IO_PROXY_INSTANCES( double );
TRACK_FIELD_IO_PROXY_INSTANCES( unsigned );
TRACK_FIELD_IO_PROXY_INSTANCES( uint64_t );

#undef TRACK_FIELD_IO_PROXY_EXPORT
