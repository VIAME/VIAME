/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/process_cluster.h>
#include <vistk/pipeline/process_exception.h>

#include <boost/make_shared.hpp>

#include <exception>
#include <iostream>
#include <string>

#include <cstdlib>

static void run_test(std::string const& test_name);

int
main(int argc, char* argv[])
{
  if (argc != 2)
  {
    TEST_ERROR("Expected one argument");

    return EXIT_FAILURE;
  }

  std::string const test_name = argv[1];

  try
  {
    run_test(test_name);
  }
  catch (std::exception const& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static void test_configure();
static void test_init();
static void test_step();

void
run_test(std::string const& test_name)
{
  if (test_name == "configure")
  {
    test_configure();
  }
  else if (test_name == "init")
  {
    test_init();
  }
  else if (test_name == "step")
  {
    test_step();
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
  }
}

class empty_cluster
  : public vistk::process_cluster
{
  public:
    empty_cluster();
    ~empty_cluster();

    vistk::processes_t processes() const;
    connections_t input_mappings() const;
    connections_t output_mappings() const;
    connections_t internal_connections() const;
};

void
test_configure()
{
  vistk::process_cluster_t const cluster = boost::make_shared<empty_cluster>();

  cluster->configure();
}

void
test_init()
{
  vistk::process_cluster_t const cluster = boost::make_shared<empty_cluster>();

  cluster->configure();
  cluster->init();
}

void
test_step()
{
  vistk::process_cluster_t const cluster = boost::make_shared<empty_cluster>();

  cluster->configure();
}

empty_cluster
::empty_cluster()
  : vistk::process_cluster(vistk::config::empty_config())
{
}

empty_cluster
::~empty_cluster()
{
}

vistk::processes_t
empty_cluster
::processes() const
{
  return vistk::processes_t();
}

vistk::process::connections_t
empty_cluster
::input_mappings() const
{
  return connections_t();
}

vistk::process::connections_t
empty_cluster
::output_mappings() const
{
  return connections_t();
}

vistk::process::connections_t
empty_cluster
::internal_connections() const
{
  return connections_t();
}
