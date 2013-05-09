/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_TOOLS_TOOL_USAGE_H
#define SPROKIT_TOOLS_TOOL_USAGE_H

#include "tools-config.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

namespace sprokit
{

SPROKIT_TOOLS_EXPORT SPROKIT_NO_RETURN void tool_usage(int ret, boost::program_options::options_description const& options);
SPROKIT_TOOLS_EXPORT void tool_version_message();

SPROKIT_TOOLS_EXPORT boost::program_options::options_description tool_common_options();

SPROKIT_TOOLS_EXPORT boost::program_options::variables_map tool_parse(int argc, char const* argv[], boost::program_options::options_description const& desc);

}

#endif // SPROKIT_TOOLS_TOOL_USAGE_H
