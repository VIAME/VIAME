/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "tool_usage.h"

#include <boost/program_options/parsers.hpp>

#include <iostream>

#include <cstdlib>

void
tool_usage(int ret, boost::program_options::options_description const& options)
{
  std::cerr << options << std::endl;

  exit(ret);
}

boost::program_options::variables_map
tool_parse(int argc, char* argv[], boost::program_options::options_description const& desc)
{
  boost::program_options::variables_map vm;

  try
  {
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  }
  catch (boost::program_options::unknown_option const& e)
  {
    std::cerr << "Error: unknown option " << e.get_option_name() << std::endl;

    tool_usage(EXIT_FAILURE, desc);
  }

  boost::program_options::notify(vm);

  return vm;
}
