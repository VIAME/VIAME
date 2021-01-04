// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_TOOLS_TOOL_IO_H
#define SPROKIT_TOOLS_TOOL_IO_H

#include <vital/vital_types.h>

#include <istream>
#include <ostream>
#include <memory>

namespace sprokit {

typedef std::shared_ptr<std::istream> istream_t;
typedef std::shared_ptr<std::ostream> ostream_t;

istream_t open_istream(kwiver::vital::path_t const& path);
ostream_t open_ostream(kwiver::vital::path_t const& path);

}

#endif // SPROKIT_TOOLS_TOOL_IO_H
