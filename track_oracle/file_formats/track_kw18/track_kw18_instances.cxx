/*ckwg +5
 * Copyright 2017 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

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
TRACK_FIELD_IO_PROXY_INSTANCES( unsigned long long );

#undef TRACK_FIELD_IO_PROXY_EXPORT
