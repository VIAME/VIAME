/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "tool_usage.h"

#include <iostream>

#include <cstdlib>

void
tool_usage(int ret, boost::program_options::options_description const& options)
{
  std::cerr << options << std::endl;

  exit(ret);
}
