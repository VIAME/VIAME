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

#define TEST_ARGS ()

DECLARE_TEST(configure);
DECLARE_TEST(init);
DECLARE_TEST(step);
DECLARE_TEST(add_process);
DECLARE_TEST(duplicate_name);
DECLARE_TEST(map_config);
DECLARE_TEST(map_config_after_process);
DECLARE_TEST(map_config_no_exist);
DECLARE_TEST(map_config_read_only);
DECLARE_TEST(map_input);
DECLARE_TEST(map_input_twice);
DECLARE_TEST(map_input_no_exist);
DECLARE_TEST(map_input_port_no_exist);
DECLARE_TEST(map_output);
DECLARE_TEST(map_output_twice);
DECLARE_TEST(map_output_no_exist);
DECLARE_TEST(map_output_port_no_exist);
DECLARE_TEST(connect);
DECLARE_TEST(connect_upstream_no_exist);
DECLARE_TEST(connect_upstream_port_no_exist);
DECLARE_TEST(connect_downstream_no_exist);
DECLARE_TEST(connect_downstream_port_no_exist);
DECLARE_TEST(reconfigure_pass_tunable_mappings);
DECLARE_TEST(reconfigure_no_pass_untunable_mappings);
DECLARE_TEST(reconfigure_pass_extra);
DECLARE_TEST(reconfigure_tunable_only_if_mapped);
DECLARE_TEST(reconfigure_mapped_untunable);

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, configure);
  ADD_TEST(tests, init);
  ADD_TEST(tests, step);
  ADD_TEST(tests, add_process);
  ADD_TEST(tests, duplicate_name);
  ADD_TEST(tests, map_config);
  ADD_TEST(tests, map_config_after_process);
  ADD_TEST(tests, map_config_no_exist);
  ADD_TEST(tests, map_config_read_only);
  ADD_TEST(tests, map_input);
  ADD_TEST(tests, map_input_twice);
  ADD_TEST(tests, map_input_no_exist);
  ADD_TEST(tests, map_input_port_no_exist);
  ADD_TEST(tests, map_output);
  ADD_TEST(tests, map_output_twice);
  ADD_TEST(tests, map_output_no_exist);
  ADD_TEST(tests, map_output_port_no_exist);
  ADD_TEST(tests, connect);
  ADD_TEST(tests, connect_upstream_no_exist);
  ADD_TEST(tests, connect_upstream_port_no_exist);
  ADD_TEST(tests, connect_downstream_no_exist);
  ADD_TEST(tests, connect_downstream_port_no_exist);
  ADD_TEST(tests, reconfigure_pass_tunable_mappings);
  ADD_TEST(tests, reconfigure_no_pass_untunable_mappings);
  ADD_TEST(tests, reconfigure_pass_extra);
  ADD_TEST(tests, reconfigure_tunable_only_if_mapped);
  ADD_TEST(tests, reconfigure_mapped_untunable);

  RUN_TEST(tests, testname);
}

class empty_cluster
  : public vistk::process_cluster
{
  public:
    empty_cluster();
    ~empty_cluster();
};

IMPLEMENT_TEST(configure)
{
  vistk::process_cluster_t const cluster = boost::make_shared<empty_cluster>();

  cluster->configure();
}

IMPLEMENT_TEST(init)
{
  vistk::process_cluster_t const cluster = boost::make_shared<empty_cluster>();

  cluster->configure();
  cluster->init();
}

IMPLEMENT_TEST(step)
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

    void _declare_configuration_key(vistk::config::key_t const& key,
                                    vistk::config::value_t const& def_,
                                    vistk::config::description_t const& description_,
                                    bool tunable_);

    void _map_config(vistk::config::key_t const& key, name_t const& name_, vistk::config::key_t const& mapped_key);
    void _add_process(name_t const& name_, type_t const& type_, vistk::config_t const& config = vistk::config::empty_config());
    void _map_input(port_t const& port, name_t const& name_, port_t const& mapped_port);
    void _map_output(port_t const& port, name_t const& name_, port_t const& mapped_port);
    void _connect(name_t const& upstream_name, port_t const& upstream_port,
                  name_t const& downstream_name, port_t const& downstream_port);
};
typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

IMPLEMENT_TEST(add_process)
{
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

IMPLEMENT_TEST(duplicate_name)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("orphan");

  vistk::load_known_modules();

  cluster->_add_process(name, type);

  EXPECT_EXCEPTION(vistk::duplicate_process_name_exception,
                   cluster->_add_process(name, type),
                   "adding a process with a duplicate name to a cluster");
}

IMPLEMENT_TEST(map_config)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::config::key_t const key = vistk::config::key_t("key");
  vistk::process::name_t const name = vistk::process::name_t("name");

  cluster->_map_config(key, name, key);
}

IMPLEMENT_TEST(map_config_after_process)
{
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

IMPLEMENT_TEST(map_config_no_exist)
{
  vistk::load_known_modules();

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::config::key_t const key = vistk::config::key_t("key");
  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("nnameame");

  cluster->_map_config(key, name, key);

  EXPECT_EXCEPTION(vistk::unknown_configuration_value_exception,
                   cluster->_add_process(name, type),
                   "mapping an unknown configuration on a cluster");
}

IMPLEMENT_TEST(map_config_read_only)
{
  vistk::load_known_modules();

  vistk::process::name_t const cluster_name = vistk::process::name_t("cluster");

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::config::key_t const key = vistk::config::key_t("key");

  cluster->_declare_configuration_key(
    key,
    vistk::config::value_t(),
    vistk::config::description_t(),
    true);

  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("orphan");

  vistk::config_t const conf = vistk::config::empty_config();

  vistk::config::key_t const mapped_key = vistk::config::key_t("mapped_key");

  cluster->_map_config(key, name, mapped_key);

  vistk::config::value_t const mapped_value = vistk::config::value_t("old_value");

  conf->set_value(mapped_key, mapped_value);
  conf->mark_read_only(mapped_key);

  EXPECT_EXCEPTION(vistk::mapping_to_read_only_value_exception,
                   cluster->_add_process(name, type, conf),
                   "when mapping to a value which already has a read-only value");
}

IMPLEMENT_TEST(map_input)
{
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

  cluster->_map_input(port, name, mapped_port);

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

IMPLEMENT_TEST(map_input_twice)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("print_number");
  vistk::process::port_t const port1 = vistk::process::port_t("cluster_number1");
  vistk::process::port_t const port2 = vistk::process::port_t("cluster_number2");
  vistk::process::port_t const mapped_port = vistk::process::port_t("number");

  vistk::load_known_modules();

  cluster->_add_process(name, type);

  cluster->_map_input(port1, name, mapped_port);

  EXPECT_EXCEPTION(vistk::port_reconnect_exception,
                   cluster->_map_input(port2, name, mapped_port),
                   "mapping a second cluster port to a process input port");
}

IMPLEMENT_TEST(map_input_no_exist)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::port_t const port = vistk::process::port_t("port");
  vistk::process::name_t const name = vistk::process::name_t("name");

  EXPECT_EXCEPTION(vistk::no_such_process_exception,
                   cluster->_map_input(port, name, port),
                   "mapping an input to a non-existent process");
}

IMPLEMENT_TEST(map_input_port_no_exist)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::port_t const port = vistk::process::port_t("no_such_port");
  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("orphan");

  vistk::load_known_modules();

  cluster->_add_process(name, type);

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   cluster->_map_input(port, name, port),
                   "mapping an input to a non-existent port");
}

IMPLEMENT_TEST(map_output)
{
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

  cluster->_map_output(port, name, mapped_port);

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

IMPLEMENT_TEST(map_output_twice)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::name_t const name1 = vistk::process::name_t("name1");
  vistk::process::name_t const name2 = vistk::process::name_t("name2");
  vistk::process::type_t const type = vistk::process::type_t("numbers");
  vistk::process::port_t const port = vistk::process::port_t("cluster_number");
  vistk::process::port_t const mapped_port = vistk::process::port_t("number");

  vistk::load_known_modules();

  cluster->_add_process(name1, type);
  cluster->_add_process(name2, type);

  cluster->_map_output(port, name1, mapped_port);

  EXPECT_EXCEPTION(vistk::port_reconnect_exception,
                   cluster->_map_output(port, name2, mapped_port),
                   "mapping a second port to a cluster output port");
}

IMPLEMENT_TEST(map_output_no_exist)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::port_t const port = vistk::process::port_t("port");
  vistk::process::name_t const name = vistk::process::name_t("name");

  EXPECT_EXCEPTION(vistk::no_such_process_exception,
                   cluster->_map_output(port, name, port),
                   "mapping an output to a non-existent process");
}

IMPLEMENT_TEST(map_output_port_no_exist)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  vistk::process::port_t const port = vistk::process::port_t("no_such_port");
  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("orphan");

  vistk::load_known_modules();

  cluster->_add_process(name, type);

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   cluster->_map_output(port, name, port),
                   "mapping an output to a non-existent port");
}

IMPLEMENT_TEST(connect)
{
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

IMPLEMENT_TEST(connect_upstream_no_exist)
{
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

IMPLEMENT_TEST(connect_upstream_port_no_exist)
{
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

IMPLEMENT_TEST(connect_downstream_no_exist)
{
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

IMPLEMENT_TEST(connect_downstream_port_no_exist)
{
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

IMPLEMENT_TEST(reconfigure_pass_tunable_mappings)
{
  vistk::load_known_modules();

  vistk::process::name_t const cluster_name = vistk::process::name_t("cluster");

  vistk::config_t const cluster_conf = vistk::config::empty_config();

  cluster_conf->set_value(vistk::process::config_name, cluster_name);

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>(cluster_conf);

  vistk::config::key_t const key = vistk::config::key_t("key");

  cluster->_declare_configuration_key(
    key,
    vistk::config::value_t(),
    vistk::config::description_t(),
    true);

  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("expect");

  vistk::config_t const conf = vistk::config::empty_config();

  vistk::config::key_t const key_tunable = vistk::config::key_t("tunable");
  vistk::config::key_t const key_expect = vistk::config::key_t("expect");

  cluster->_map_config(key, name, key_tunable);

  vistk::config::value_t const tunable_value = vistk::config::value_t("old_value");
  vistk::config::value_t const tuned_value = vistk::config::value_t("new_value");

  conf->set_value(key_tunable, tunable_value);
  conf->set_value(key_expect, tuned_value);

  cluster->_add_process(name, type, conf);

  vistk::pipeline_t const pipeline = boost::make_shared<vistk::pipeline>(vistk::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  vistk::config_t const new_conf = vistk::config::empty_config();

  new_conf->set_value(cluster_name + vistk::config::block_sep + key, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(reconfigure_no_pass_untunable_mappings)
{
  vistk::load_known_modules();

  vistk::process::name_t const cluster_name = vistk::process::name_t("cluster");

  vistk::config_t const cluster_conf = vistk::config::empty_config();

  cluster_conf->set_value(vistk::process::config_name, cluster_name);

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>(cluster_conf);

  vistk::config::key_t const key = vistk::config::key_t("key");

  cluster->_declare_configuration_key(
    key,
    vistk::config::value_t(),
    vistk::config::description_t(),
    false);

  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("expect");

  vistk::config_t const conf = vistk::config::empty_config();

  vistk::config::key_t const key_tunable = vistk::config::key_t("tunable");
  vistk::config::key_t const key_expect = vistk::config::key_t("expect");

  cluster->_map_config(key, name, key_tunable);

  vistk::config::value_t const tunable_value = vistk::config::value_t("old_value");

  conf->set_value(key_tunable, tunable_value);
  conf->set_value(key_expect, tunable_value);

  cluster->_add_process(name, type, conf);

  vistk::pipeline_t const pipeline = boost::make_shared<vistk::pipeline>(vistk::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  vistk::config_t const new_conf = vistk::config::empty_config();

  vistk::config::value_t const tuned_value = vistk::config::value_t("new_value");

  new_conf->set_value(cluster_name + vistk::config::block_sep + key, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(reconfigure_pass_extra)
{
  vistk::load_known_modules();

  vistk::process::name_t const cluster_name = vistk::process::name_t("cluster");

  vistk::config_t const cluster_conf = vistk::config::empty_config();

  cluster_conf->set_value(vistk::process::config_name, cluster_name);

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>(cluster_conf);

  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("expect");

  vistk::config_t const conf = vistk::config::empty_config();

  vistk::config::key_t const key_expect = vistk::config::key_t("expect");
  vistk::config::key_t const key_expect_key = vistk::config::key_t("expect_key");

  vistk::config::value_t const extra_key = vistk::config::value_t("new_key");

  conf->set_value(key_expect, extra_key);
  conf->set_value(key_expect_key, "true");

  cluster->_add_process(name, type, conf);

  vistk::pipeline_t const pipeline = boost::make_shared<vistk::pipeline>(vistk::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  vistk::config_t const new_conf = vistk::config::empty_config();

  new_conf->set_value(cluster_name + vistk::config::block_sep + name + vistk::config::block_sep + extra_key, extra_key);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(reconfigure_tunable_only_if_mapped)
{
  vistk::load_known_modules();

  vistk::process::name_t const cluster_name = vistk::process::name_t("cluster");

  vistk::config_t const cluster_conf = vistk::config::empty_config();

  cluster_conf->set_value(vistk::process::config_name, cluster_name);

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>(cluster_conf);

  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("expect");

  vistk::config_t const conf = vistk::config::empty_config();

  vistk::config::key_t const key_tunable = vistk::config::key_t("tunable");
  vistk::config::key_t const key_expect = vistk::config::key_t("expect");

  vistk::config::value_t const tunable_value = vistk::config::value_t("old_value");

  conf->set_value(key_tunable, tunable_value);
  conf->mark_read_only(key_tunable);
  conf->set_value(key_expect, tunable_value);

  cluster->_add_process(name, type, conf);

  vistk::pipeline_t const pipeline = boost::make_shared<vistk::pipeline>(vistk::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  vistk::config_t const new_conf = vistk::config::empty_config();

  vistk::config::value_t const tuned_value = vistk::config::value_t("new_value");

  new_conf->set_value(cluster_name + vistk::config::block_sep + name + vistk::config::block_sep + key_tunable, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(reconfigure_mapped_untunable)
{
  vistk::load_known_modules();

  vistk::process::name_t const cluster_name = vistk::process::name_t("cluster");

  vistk::config_t const cluster_conf = vistk::config::empty_config();

  cluster_conf->set_value(vistk::process::config_name, cluster_name);

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>(cluster_conf);

  vistk::config::key_t const key = vistk::config::key_t("key");

  vistk::config::value_t const tunable_value = vistk::config::value_t("old_value");

  cluster->_declare_configuration_key(
    key,
    tunable_value,
    vistk::config::description_t(),
    true);

  vistk::process::name_t const name = vistk::process::name_t("name");
  vistk::process::type_t const type = vistk::process::type_t("expect");

  vistk::config_t const conf = vistk::config::empty_config();

  vistk::config::key_t const key_tunable = vistk::config::key_t("tunable");
  vistk::config::key_t const key_expect = vistk::config::key_t("expect");

  cluster->_map_config(key, name, key_expect);

  vistk::config::value_t const tuned_value = vistk::config::value_t("new_value");

  conf->set_value(key_tunable, tunable_value);

  cluster->_add_process(name, type, conf);

  vistk::pipeline_t const pipeline = boost::make_shared<vistk::pipeline>(vistk::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  vistk::config_t const new_conf = vistk::config::empty_config();

  new_conf->set_value(cluster_name + vistk::config::block_sep + key, tuned_value);

  pipeline->reconfigure(new_conf);
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
::_declare_configuration_key(vistk::config::key_t const& key,
                             vistk::config::value_t const& def_,
                             vistk::config::description_t const& description_,
                             bool tunable_)
{
  declare_configuration_key(key, def_, description_, tunable_);
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
::_map_input(port_t const& port, name_t const& name_, port_t const& mapped_port)
{
  map_input(port, name_, mapped_port);
}

void
sample_cluster
::_map_output(port_t const& port, name_t const& name_, port_t const& mapped_port)
{
  map_output(port, name_, mapped_port);
}

void
sample_cluster
::_connect(name_t const& upstream_name, port_t const& upstream_port,
           name_t const& downstream_name, port_t const& downstream_port)
{
  connect(upstream_name, upstream_port, downstream_name, downstream_port);
}
