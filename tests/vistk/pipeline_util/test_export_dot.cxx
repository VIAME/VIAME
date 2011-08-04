/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline_util/export_dot.h>
#include <vistk/pipeline_util/export_dot_exception.h>
#include <vistk/pipeline_util/pipe_bakery.h>

#include <vistk/pipeline/modules.h>

#include <boost/filesystem/path.hpp>

#include <exception>
#include <iostream>
#include <sstream>
#include <string>

static std::string const pipe_ext = ".pipe";

static void run_test(std::string const& test_name, boost::filesystem::path const& pipe_file);

int
main(int argc, char* argv[])
{
  if (argc != 3)
  {
    std::cerr << "Error: Expected one argument" << std::endl;

    return 1;
  }

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

static void test_pipeline_null(boost::filesystem::path const& pipe_file);
static void test_simple_pipeline(boost::filesystem::path const& pipe_file);

void
run_test(std::string const& test_name, boost::filesystem::path const& pipe_file)
{
  if (test_name == "pipeline_null")
  {
    test_pipeline_null(pipe_file);
  }
  else if (test_name == "simple_pipeline")
  {
    test_simple_pipeline(pipe_file);
  }
  else
  {
    std::cerr << "Error: Unknown test: " << test_name << std::endl;
  }
}

void test_pipeline_null(boost::filesystem::path const& /*pipe_file*/)
{
  vistk::pipeline_t pipeline;

  std::ostringstream sstr;

  bool got_exception = false;

  try
  {
    vistk::export_dot(sstr, pipeline, "(unnamed)");
  }
  catch (vistk::null_pipeline_export_dot_exception& e)
  {
    got_exception = true;

    (void)e.what();
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: Unexpected exception: "
              << e.what() << std::endl;

    got_exception = true;
  }

  if (!got_exception)
  {
    std::cerr << "Error: Did not get expected exception "
              << "when exporting a NULL pipeline" << std::endl;
  }
}

void
test_simple_pipeline(boost::filesystem::path const& pipe_file)
{
  vistk::load_known_modules();

  vistk::pipeline_t const pipeline = vistk::bake_pipe_from_file(pipe_file);

  std::ostringstream sstr;

  vistk::export_dot(sstr, pipeline, "(unnamed)");
}
