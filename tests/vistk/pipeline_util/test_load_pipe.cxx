/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/modules.h>

#include <boost/filesystem/path.hpp>

#include <exception>
#include <iostream>
#include <string>

static std::string const pipe_ext = ".pipe";

static void run_test(std::string const& test_name, boost::filesystem::path const& pipe_file);

int
main(int argc, char* argv[])
{
  if (argc != 3)
  {
    std::cerr << "Error: Expected two arguments" << std::endl;

    return 1;
  }

  vistk::load_known_modules();

  std::string const test_name = argv[1];
  boost::filesystem::path const pipe_dir = argv[2];

  boost::filesystem::path const pipe_file = pipe_dir / (test_name + pipe_ext);

  try
  {
    run_test(test_name, pipe_file);
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: Unexpected exception: " << e.what() << std::endl;

    return 1;
  }

  return 0;
}

void
run_test(std::string const& test_name, boost::filesystem::path const& pipe_file)
{
  //else
  {
    std::cerr << "Error: Unknown test: " << test_name << std::endl;
  }
}
