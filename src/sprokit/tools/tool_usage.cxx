/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "tool_usage.h"

#include <sprokit/version.h>

#include <boost/program_options/parsers.hpp>

#include <iostream>

#include <cstdlib>

namespace sprokit
{

void
tool_usage(int ret, boost::program_options::options_description const& options)
{
  std::cerr << options << std::endl;

  exit(ret);
}

void
tool_version_message()
{
  std::cout << "sprokit " SPROKIT_VERSION_FULL << std::endl;
  std::cout << "Built with sprokit: " SPROKIT_VERSION << std::endl;
  std::cout << "Built from git:     "
#ifdef SPROKIT_BUILT_FROM_GIT
    "yes"
#else
    "no"
#endif
    << std::endl;
  std::cout << "Git hash:           " SPROKIT_GIT_HASH << std::endl;

  char const* const dirty = SPROKIT_GIT_DIRTY;
  bool const dirty_is_empty = (*dirty == '\0');
  char const* const is_dirty = (dirty_is_empty ? "no" : "yes");

  std::cout << "Dirty:              " << is_dirty << std::endl;
}

boost::program_options::options_description
tool_common_options()
{
  boost::program_options::options_description desc("General options");

  desc.add_options()
    ("help,h", "output help message and quit")
    ("version,V", "output version information")
  ;

  return desc;
}

boost::program_options::variables_map
tool_parse(int argc, char const* argv[], boost::program_options::options_description const& desc)
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

  if (vm.count("help"))
  {
    tool_version_message();

    tool_usage(EXIT_SUCCESS, desc);
  }

  if (vm.count("version"))
  {
    tool_version_message();

    exit(EXIT_SUCCESS);
  }

  return vm;
}

}
