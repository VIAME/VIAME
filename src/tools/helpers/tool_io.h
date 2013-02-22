/*ckwg +5
 * Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_TOOL_HELPERS_TOOL_IO_H
#define VISTK_TOOL_HELPERS_TOOL_IO_H

#include <vistk/pipeline_util/path.h>

#include <boost/shared_ptr.hpp>

#include <istream>
#include <ostream>

typedef boost::shared_ptr<std::istream> istream_t;
typedef boost::shared_ptr<std::ostream> ostream_t;

istream_t open_istream(vistk::path_t const& path);
ostream_t open_ostream(vistk::path_t const& path);

#endif // VISTK_TOOL_HELPERS_TOOL_IO_H
