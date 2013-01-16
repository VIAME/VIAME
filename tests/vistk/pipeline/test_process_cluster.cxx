/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/pipeline_exception.h>
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
static void test_duplicate_name();
static void test_map_config_no_exist();
static void test_map_input_no_exist();
static void test_map_input_port_no_exist();
static void test_map_output_no_exist();
static void test_map_output_port_no_exist();

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
  else if (test_name == "duplicate_name")
  {
    test_duplicate_name();
  }
  else if (test_name == "map_config_no_exist")
  {
    test_map_config_no_exist();
  }
  else if (test_name == "map_input_no_exist")
  {
    test_map_input_no_exist();
  }
  else if (test_name == "map_input_port_no_exist")
  {
    test_map_input_port_no_exist();
  }
  else if (test_name == "map_output_no_exist")
  {
    test_map_output_no_exist();
  }
  else if (test_name == "map_output_port_no_exist")
  {
    test_map_output_port_no_exist();
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
  cluster->init();

  EXPECT_EXCEPTION(vistk::process_exception,
                   cluster->step(),
                   "stepping a cluster");
}

class sample_cluster
  : public vistk::process_cluster
{
  public:
    sample_cluster(vistk::config_t const& conf = vistk::config::empty_config());
    ~sample_cluster();

    void _map_config(vistk::config::key_t const& key, name_t const& name_, vistk::config::key_t const& mapped_key);
    void _add_process(name_t const& name_, type_t const& type_, vistk::config_t const& config = vistk::config::empty_config());
    void _input_map(port_t const& port, name_t const& name_, port_t const& mapped_port);
    void _output_map(port_t const& port, name_t const& name_, port_t const& mapped_port);
    void _connect(name_t const& upstream_name, port_t const& upstream_port,
                  name_t const& downstream_name, port_t const& downstream_port);
};

void
test_duplicate_name()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("orphan");

  vistk::load_known_modules();

  cluster->_add_process(name, type);

  EXPECT_EXCEPTION(vistk::duplicate_process_name_exception,
                   cluster->_add_process(name, type),
                   "adding a process with a duplicate name to a cluster");
}

void
test_map_config_no_exist()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::config::key_t const key = vistk::config::key_t("key");
  vistk::process::name_t const name = vistk::process::name_t("name");

  cluster->_map_config(key, name, key);
}

void
test_map_input_no_exist()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::port_t const port = vistk::process::port_t("port");
  vistk::process::name_t const name = vistk::process::name_t("name");

  EXPECT_EXCEPTION(vistk::no_such_process_exception,
                   cluster->_input_map(port, name, port),
                   "mapping an input to a non-existent process");
}

void
test_map_input_port_no_exist()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::port_t const port = vistk::process::port_t("no_such_port");
  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("orphan");

  vistk::load_known_modules();

  cluster->_add_process(name, type);

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   cluster->_input_map(port, name, port),
                   "mapping an input to a non-existent port");
}

void
test_map_output_no_exist()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::port_t const port = vistk::process::port_t("port");
  vistk::process::name_t const name = vistk::process::name_t("name");

  EXPECT_EXCEPTION(vistk::no_such_process_exception,
                   cluster->_output_map(port, name, port),
                   "mapping an output to a non-existent process");
}

void
test_map_output_port_no_exist()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::port_t const port = vistk::process::port_t("no_such_port");
  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("orphan");

  vistk::load_known_modules();

  cluster->_add_process(name, type);

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   cluster->_output_map(port, name, port),
                   "mapping an output to a non-existent port");
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

sample_cluster
::sample_cluster(vistk::config_t const& conf)
  : vistk::process_cluster(conf)
{
}

sample_cluster
::~sample_cluster()
{
}

void
sample_cluster
::_map_config(vistk::config::key_t const& key, name_t const& name_, vistk::config::key_t const& mapped_key)
{
  map_config(key, name_, mapped_key);
}

void
sample_cluster
::_add_process(name_t const& name_, type_t const& type_, vistk::config_t const& config)
{
  add_process(name_, type_, config);
}

void
sample_cluster
::_input_map(port_t const& port, name_t const& name_, port_t const& mapped_port)
{
  input_map(port, name_, mapped_port);
}

void
sample_cluster
::_output_map(port_t const& port, name_t const& name_, port_t const& mapped_port)
{
  output_map(port, name_, mapped_port);
}

void
sample_cluster
::_connect(name_t const& upstream_name, port_t const& upstream_port,
           name_t const& downstream_name, port_t const& downstream_port)
{
  connect(upstream_name, upstream_port, downstream_name, downstream_port);
}
