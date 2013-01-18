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
#include <vistk/pipeline/process_cluster_exception.h>
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
static void test_add_process();
static void test_duplicate_name();
static void test_map_config();
static void test_map_config_after_process();
static void test_map_config_no_exist();
static void test_map_input();
static void test_map_input_twice();
static void test_map_input_no_exist();
static void test_map_input_port_no_exist();
static void test_map_output();
static void test_map_output_twice();
static void test_map_output_no_exist();
static void test_map_output_port_no_exist();
static void test_connect();
static void test_connect_upstream_no_exist();
static void test_connect_upstream_port_no_exist();
static void test_connect_downstream_no_exist();
static void test_connect_downstream_port_no_exist();

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
  else if (test_name == "add_process")
  {
    test_add_process();
  }
  else if (test_name == "duplicate_name")
  {
    test_duplicate_name();
  }
  else if (test_name == "map_config")
  {
    test_map_config();
  }
  else if (test_name == "map_config_after_process")
  {
    test_map_config_after_process();
  }
  else if (test_name == "map_config_no_exist")
  {
    test_map_config_no_exist();
  }
  else if (test_name == "map_input")
  {
    test_map_input();
  }
  else if (test_name == "map_input_twice")
  {
    test_map_input_twice();
  }
  else if (test_name == "map_input_no_exist")
  {
    test_map_input_no_exist();
  }
  else if (test_name == "map_input_port_no_exist")
  {
    test_map_input_port_no_exist();
  }
  else if (test_name == "map_output")
  {
    test_map_output();
  }
  else if (test_name == "map_output_twice")
  {
    test_map_output_twice();
  }
  else if (test_name == "map_output_no_exist")
  {
    test_map_output_no_exist();
  }
  else if (test_name == "map_output_port_no_exist")
  {
    test_map_output_port_no_exist();
  }
  else if (test_name == "connect")
  {
    test_connect();
  }
  else if (test_name == "connect_upstream_no_exist")
  {
    test_connect_upstream_no_exist();
  }
  else if (test_name == "connect_upstream_port_no_exist")
  {
    test_connect_upstream_port_no_exist();
  }
  else if (test_name == "connect_downstream_no_exist")
  {
    test_connect_downstream_no_exist();
  }
  else if (test_name == "connect_downstream_port_no_exist")
  {
    test_connect_downstream_port_no_exist();
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
test_add_process()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("orphan");

  vistk::load_known_modules();

  cluster->_add_process(name, type);

  vistk::processes_t const procs = cluster->processes();

  if (procs.empty())
  {
    TEST_ERROR("A cluster does not contain a process after adding one");

    // The remaining code won't be happy with an empty vector.
    return;
  }

  if (procs.size() != 1)
  {
    TEST_ERROR("A cluster has more processes than declared");
  }

  vistk::process_t const& proc = procs[0];

  if (proc->type() != type)
  {
    TEST_ERROR("A cluster added a process of a different type than requested");
  }

  // TODO: Get the mangled name.
  if (proc->name() == name)
  {
    TEST_ERROR("A cluster did not mangle a processes name");
  }
}

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
test_map_config()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::config::key_t const key = vistk::config::key_t("key");
  vistk::process::name_t const name = vistk::process::name_t("name");

  cluster->_map_config(key, name, key);
}

void
test_map_config_after_process()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::config::key_t const key = vistk::config::key_t("key");
  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("orphan");

  vistk::load_known_modules();

  cluster->_add_process(name, type);

  EXPECT_EXCEPTION(vistk::mapping_after_process_exception,
                   cluster->_map_config(key, name, key),
                   "mapping a configuration after the process has been added");
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
test_map_input()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  vistk::config_t const conf = vistk::config::empty_config();

  vistk::process::name_t const cluster_name = vistk::process::name_t("cluster");

  conf->set_value(vistk::process::config_name, cluster_name);

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>(conf);

  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("print_number");
  vistk::process::port_t const port = vistk::process::port_t("cluster_number");
  vistk::process::port_t const mapped_port = vistk::process::port_t("number");

  vistk::load_known_modules();

  cluster->_add_process(name, type);

  cluster->_input_map(port, name, mapped_port);

  vistk::process::connections_t const mappings = cluster->input_mappings();

  if (mappings.empty())
  {
    TEST_ERROR("A cluster does not contain an input mapping after adding one");

    // The remaining code won't be happy with an empty vector.
    return;
  }

  if (mappings.size() != 1)
  {
    TEST_ERROR("A cluster has more input mappings than declared");
  }

  vistk::process::connection_t const& mapping = mappings[0];

  vistk::process::port_addr_t const& up_addr = mapping.first;
  vistk::process::name_t const& up_name = up_addr.first;
  vistk::process::port_t const& up_port = up_addr.second;

  if (up_name != cluster_name)
  {
    TEST_ERROR("A cluster input mapping\'s upstream name is not the cluster itself");
  }

  if (up_port != port)
  {
    TEST_ERROR("A cluster input mapping\'s upstream port is not the one requested");
  }

  vistk::process::port_addr_t const& down_addr = mapping.second;
  vistk::process::name_t const& down_name = down_addr.first;
  vistk::process::port_t const& down_port = down_addr.second;

  // TODO: Get the mangled name.
  if (down_name == name)
  {
    TEST_ERROR("A cluster input mapping\'s downstream name was not mangled");
  }

  if (down_port != mapped_port)
  {
    TEST_ERROR("A cluster input mapping\'s downstream port is not the one requested");
  }
}

void
test_map_input_twice()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("print_number");
  vistk::process::port_t const port1 = vistk::process::port_t("cluster_number1");
  vistk::process::port_t const port2 = vistk::process::port_t("cluster_number2");
  vistk::process::port_t const mapped_port = vistk::process::port_t("number");

  vistk::load_known_modules();

  cluster->_add_process(name, type);

  cluster->_input_map(port1, name, mapped_port);

  EXPECT_EXCEPTION(vistk::port_reconnect_exception,
                   cluster->_input_map(port2, name, mapped_port),
                   "mapping a second cluster port to a process input port");
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
test_map_output()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  vistk::config_t const conf = vistk::config::empty_config();

  vistk::process::name_t const cluster_name = vistk::process::name_t("cluster");

  conf->set_value(vistk::process::config_name, cluster_name);

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>(conf);

  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("numbers");
  vistk::process::port_t const port = vistk::process::port_t("cluster_number");
  vistk::process::port_t const mapped_port = vistk::process::port_t("number");

  vistk::load_known_modules();

  cluster->_add_process(name, type);

  cluster->_output_map(port, name, mapped_port);

  vistk::process::connections_t const mappings = cluster->output_mappings();

  if (mappings.empty())
  {
    TEST_ERROR("A cluster does not contain an output mapping after adding one");

    // The remaining code won't be happy with an empty vector.
    return;
  }

  if (mappings.size() != 1)
  {
    TEST_ERROR("A cluster has more output mappings than declared");
  }

  vistk::process::connection_t const& mapping = mappings[0];

  vistk::process::port_addr_t const& down_addr = mapping.second;
  vistk::process::name_t const& down_name = down_addr.first;
  vistk::process::port_t const& down_port = down_addr.second;

  if (down_name != cluster_name)
  {
    TEST_ERROR("A cluster output mapping\'s downstream name is not the cluster itself");
  }

  if (down_port != port)
  {
    TEST_ERROR("A cluster output mapping\'s downstream port is not the one requested");
  }

  vistk::process::port_addr_t const& up_addr = mapping.first;
  vistk::process::name_t const& up_name = up_addr.first;
  vistk::process::port_t const& up_port = up_addr.second;

  // TODO: Get the mangled name.
  if (up_name == name)
  {
    TEST_ERROR("A cluster output mapping\'s upstream name was not mangled");
  }

  if (up_port != mapped_port)
  {
    TEST_ERROR("A cluster output mapping\'s upstream port is not the one requested");
  }
}

void
test_map_output_twice()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::name_t const name1 = vistk::process::name_t("name1");
  vistk::process::name_t const name2 = vistk::process::name_t("name2");
  vistk::process::type_t const type = vistk::process::type_t("numbers");
  vistk::process::port_t const port = vistk::process::port_t("cluster_number");
  vistk::process::port_t const mapped_port = vistk::process::port_t("number");

  vistk::load_known_modules();

  cluster->_add_process(name1, type);
  cluster->_add_process(name2, type);

  cluster->_output_map(port, name1, mapped_port);

  EXPECT_EXCEPTION(vistk::port_reconnect_exception,
                   cluster->_output_map(port, name2, mapped_port),
                   "mapping a second port to a cluster output port");
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

void
test_connect()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::name_t const name1 = vistk::process::name_t("name1");
  vistk::process::name_t const name2 = vistk::process::name_t("name2");
  vistk::process::type_t const type1 = vistk::process::type_t("numbers");
  vistk::process::type_t const type2 = vistk::process::type_t("print_number");
  vistk::process::port_t const port = vistk::process::port_t("number");

  vistk::load_known_modules();

  cluster->_add_process(name1, type1);
  cluster->_add_process(name2, type2);

  cluster->_connect(name1, port, name2, port);

  vistk::process::connections_t const mappings = cluster->internal_connections();

  if (mappings.empty())
  {
    TEST_ERROR("A cluster does not contain an internal connection after adding one");

    // The remaining code won't be happy with an empty vector.
    return;
  }

  if (mappings.size() != 1)
  {
    TEST_ERROR("A cluster has more internal connections than declared");
  }

  vistk::process::connection_t const& mapping = mappings[0];

  vistk::process::port_addr_t const& down_addr = mapping.second;
  vistk::process::name_t const& down_name = down_addr.first;
  vistk::process::port_t const& down_port = down_addr.second;

  vistk::process::port_addr_t const& up_addr = mapping.first;
  vistk::process::name_t const& up_name = up_addr.first;
  vistk::process::port_t const& up_port = up_addr.second;

  // TODO: Get the mangled name.
  if (up_name == name1)
  {
    TEST_ERROR("A cluster internal connection\'s upstream name was not mangled");
  }

  if (up_port != port)
  {
    TEST_ERROR("A cluster internal connection\'s upstream port is not the one requested");
  }

  // TODO: Get the mangled name.
  if (down_name == name2)
  {
    TEST_ERROR("A cluster internal connection\'s downstream name is not the cluster itself");
  }

  if (down_port != port)
  {
    TEST_ERROR("A cluster internal connection\'s downstream port is not the one requested");
  }
}

void
test_connect_upstream_no_exist()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::name_t const name1 = vistk::process::name_t("name1");
  vistk::process::name_t const name2 = vistk::process::name_t("name2");
  vistk::process::type_t const type = vistk::process::type_t("print_number");
  vistk::process::port_t const port = vistk::process::port_t("number");

  vistk::load_known_modules();

  cluster->_add_process(name2, type);

  EXPECT_EXCEPTION(vistk::no_such_process_exception,
                   cluster->_connect(name1, port, name2, port),
                   "making a connection when the upstream process does not exist");
}

void
test_connect_upstream_port_no_exist()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::name_t const name1 = vistk::process::name_t("name1");
  vistk::process::name_t const name2 = vistk::process::name_t("name2");
  vistk::process::type_t const type1 = vistk::process::type_t("numbers");
  vistk::process::type_t const type2 = vistk::process::type_t("print_number");
  vistk::process::port_t const port1 = vistk::process::port_t("no_such_port");
  vistk::process::port_t const port2 = vistk::process::port_t("number");

  vistk::load_known_modules();

  cluster->_add_process(name1, type1);
  cluster->_add_process(name2, type2);

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   cluster->_connect(name1, port1, name2, port2),
                   "making a connection when the upstream port does not exist");
}

void
test_connect_downstream_no_exist()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::name_t const name1 = vistk::process::name_t("name1");
  vistk::process::name_t const name2 = vistk::process::name_t("name2");
  vistk::process::type_t const type = vistk::process::type_t("numbers");
  vistk::process::port_t const port = vistk::process::port_t("number");

  vistk::load_known_modules();

  cluster->_add_process(name1, type);

  EXPECT_EXCEPTION(vistk::no_such_process_exception,
                   cluster->_connect(name1, port, name2, port),
                   "making a connection when the upstream process does not exist");
}

void
test_connect_downstream_port_no_exist()
{
  typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::name_t const name1 = vistk::process::name_t("name1");
  vistk::process::name_t const name2 = vistk::process::name_t("name2");
  vistk::process::type_t const type1 = vistk::process::type_t("numbers");
  vistk::process::type_t const type2 = vistk::process::type_t("print_number");
  vistk::process::port_t const port1 = vistk::process::port_t("number");
  vistk::process::port_t const port2 = vistk::process::port_t("no_such_port");

  vistk::load_known_modules();

  cluster->_add_process(name1, type1);
  cluster->_add_process(name2, type2);

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   cluster->_connect(name1, port1, name2, port2),
                   "making a connection when the downstream port does not exist");
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
