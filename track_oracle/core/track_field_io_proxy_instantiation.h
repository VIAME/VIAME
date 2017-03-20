/*ckwg +5
 * Copyright 2017 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_FIELD_IO_PROXY_INSTANTIATION_H
#define INCL_TRACK_FIELD_IO_PROXY_INSTANTIATION_H

#include <track_oracle/core/track_field_io_proxy.txx>

#define TRACK_FIELD_IO_PROXY_INSTANCES(T) \
  template TRACK_FIELD_IO_PROXY_EXPORT std::ostream& kwiver::track_oracle::operator<<( std::ostream& os, const kwiver::track_oracle::track_field_io_proxy<T>& iop );

#endif
