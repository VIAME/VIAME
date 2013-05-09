/*ckwg +5
 * Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_TOOLS_TOOL_IO_H
#define SPROKIT_TOOLS_TOOL_IO_H

#include "tools-config.h"

#include <sprokit/pipeline_util/path.h>

#include <boost/shared_ptr.hpp>

#include <istream>
#include <ostream>

namespace sprokit
{

typedef boost::shared_ptr<std::istream> istream_t;
typedef boost::shared_ptr<std::ostream> ostream_t;

SPROKIT_TOOLS_EXPORT istream_t open_istream(sprokit::path_t const& path);
SPROKIT_TOOLS_EXPORT ostream_t open_ostream(sprokit::path_t const& path);

}

#endif // SPROKIT_TOOLS_TOOL_IO_H
