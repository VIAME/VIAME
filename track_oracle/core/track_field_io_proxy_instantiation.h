// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_TRACK_FIELD_IO_PROXY_INSTANTIATION_H
#define INCL_TRACK_FIELD_IO_PROXY_INSTANTIATION_H

#include <track_oracle/core/track_field_io_proxy.txx>

#define TRACK_FIELD_IO_PROXY_INSTANCES(T) \
  template TRACK_FIELD_IO_PROXY_EXPORT std::ostream& kwiver::track_oracle::operator<<( std::ostream& os, const kwiver::track_oracle::track_field_io_proxy<T>& iop );

#endif
