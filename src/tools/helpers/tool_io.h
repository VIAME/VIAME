/*ckwg +5
 * Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_TOOL_HELPERS_TOOL_IO_H
#define SPROKIT_TOOL_HELPERS_TOOL_IO_H

#include <sprokit/pipeline_util/path.h>

#include <boost/shared_ptr.hpp>

#include <istream>
#include <ostream>

typedef boost::shared_ptr<std::istream> istream_t;
typedef boost::shared_ptr<std::ostream> ostream_t;

istream_t open_istream(sprokit::path_t const& path);
ostream_t open_ostream(sprokit::path_t const& path);

#endif // SPROKIT_TOOL_HELPERS_TOOL_IO_H
