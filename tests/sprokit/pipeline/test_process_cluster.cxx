/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <test_common.h>

#include <sprokit/pipeline/config.h>
#include <sprokit/pipeline/edge.h>
#include <sprokit/pipeline/modules.h>
#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/pipeline_exception.h>
#include <sprokit/pipeline/process_cluster.h>
#include <sprokit/pipeline/process_cluster_exception.h>
#include <sprokit/pipeline/process_exception.h>

#include <boost/make_shared.hpp>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

class empty_cluster
  : public sprokit::process_cluster
{
  public:
    empty_cluster();
    ~empty_cluster();
};

IMPLEMENT_TEST(configure)
{
  sprokit::process_cluster_t const cluster = boost::make_shared<empty_cluster>();

  cluster->configure();
}

IMPLEMENT_TEST(init)
{
  sprokit::process_cluster_t const cluster = boost::make_shared<empty_cluster>();

  cluster->configure();
  cluster->init();
}

IMPLEMENT_TEST(step)
{
  sprokit::process_cluster_t const cluster = boost::make_shared<empty_cluster>();

  cluster->configure();
  cluster->init();

  EXPECT_EXCEPTION(sprokit::process_exception,
                   cluster->step(),
                   "stepping a cluster");
}

class sample_cluster
  : public sprokit::process_cluster
{
  public:
    sample_cluster(sprokit::config_t const& conf = sprokit::config::empty_config());
    ~sample_cluster();

    void _declare_configuration_key(sprokit::config::key_t const& key,
                                    sprokit::config::value_t const& def_,
                                    sprokit::config::description_t const& description_,
                                    bool tunable_);

    void _map_config(sprokit::config::key_t const& key, name_t const& name_, sprokit::config::key_t const& mapped_key);
    void _add_process(name_t const& name_, type_t const& type_, sprokit::config_t const& config = sprokit::config::empty_config());
    void _map_input(port_t const& port, name_t const& name_, port_t const& mapped_port);
    void _map_output(port_t const& port, name_t const& name_, port_t const& mapped_port);
    void _connect(name_t const& upstream_name, port_t const& upstream_port,
                  name_t const& downstream_name, port_t const& downstream_port);
};
typedef boost::shared_ptr<sample_cluster> sample_cluster_t;

IMPLEMENT_TEST(add_process)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("orphan");

  sprokit::load_known_modules();

  cluster->_add_process(name, type);

  sprokit::processes_t const procs = cluster->processes();

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

  sprokit::process_t const& proc = procs[0];

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

  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("orphan");

  sprokit::load_known_modules();

  cluster->_add_process(name, type);

  EXPECT_EXCEPTION(sprokit::duplicate_process_name_exception,
                   cluster->_add_process(name, type),
                   "adding a process with a duplicate name to a cluster");
}

IMPLEMENT_TEST(map_config)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  sprokit::config::key_t const key = sprokit::config::key_t("key");
  sprokit::process::name_t const name = sprokit::process::name_t("name");

  cluster->_map_config(key, name, key);
}

IMPLEMENT_TEST(map_config_after_process)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  sprokit::config::key_t const key = sprokit::config::key_t("key");
  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("orphan");

  sprokit::load_known_modules();

  cluster->_add_process(name, type);

  EXPECT_EXCEPTION(sprokit::mapping_after_process_exception,
                   cluster->_map_config(key, name, key),
                   "mapping a configuration after the process has been added");
}

IMPLEMENT_TEST(map_config_no_exist)
{
  sprokit::load_known_modules();

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  sprokit::config::key_t const key = sprokit::config::key_t("key");
  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("nnameame");

  cluster->_map_config(key, name, key);

  EXPECT_EXCEPTION(sprokit::unknown_configuration_value_exception,
                   cluster->_add_process(name, type),
                   "mapping an unknown configuration on a cluster");
}

IMPLEMENT_TEST(map_config_read_only)
{
  sprokit::load_known_modules();

  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  sprokit::config::key_t const key = sprokit::config::key_t("key");

  cluster->_declare_configuration_key(
    key,
    sprokit::config::value_t(),
    sprokit::config::description_t(),
    true);

  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("orphan");

  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::config::key_t const mapped_key = sprokit::config::key_t("mapped_key");

  cluster->_map_config(key, name, mapped_key);

  sprokit::config::value_t const mapped_value = sprokit::config::value_t("old_value");

  conf->set_value(mapped_key, mapped_value);
  conf->mark_read_only(mapped_key);

  EXPECT_EXCEPTION(sprokit::mapping_to_read_only_value_exception,
                   cluster->_add_process(name, type, conf),
                   "when mapping to a value which already has a read-only value");
}

IMPLEMENT_TEST(map_config_ignore_override)
{
  sprokit::load_known_modules();

  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::config_t const cluster_conf = sprokit::config::empty_config();

  cluster_conf->set_value(sprokit::process::config_name, cluster_name);

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>(cluster_conf);

  sprokit::config::key_t const key = sprokit::config::key_t("key");

  sprokit::config::value_t const tunable_value = sprokit::config::value_t("old_value");

  cluster->_declare_configuration_key(
    key,
    tunable_value,
    sprokit::config::description_t(),
    true);

  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("expect");

  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::config::key_t const key_tunable = sprokit::config::key_t("tunable");
  sprokit::config::key_t const key_expect = sprokit::config::key_t("expect");

  cluster->_map_config(key, name, key_expect);

  sprokit::config::value_t const tuned_value = sprokit::config::value_t("new_value");

  conf->set_value(key_tunable, tunable_value);
  // The setting should be used from the mapping, not here.
  conf->set_value(key_expect, tuned_value);

  cluster->_add_process(name, type, conf);

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  // Fill a block so that the expect process gets reconfigured to do its check;
  // if the block for it is empty, the check won't happen.
  new_conf->set_value(cluster_name + sprokit::config::block_sep + key, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(map_input)
{
  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  conf->set_value(sprokit::process::config_name, cluster_name);

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>(conf);

  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("print_number");
  sprokit::process::port_t const port = sprokit::process::port_t("cluster_number");
  sprokit::process::port_t const mapped_port = sprokit::process::port_t("number");

  sprokit::load_known_modules();

  cluster->_add_process(name, type);

  cluster->_map_input(port, name, mapped_port);

  sprokit::process::connections_t const mappings = cluster->input_mappings();

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

  sprokit::process::connection_t const& mapping = mappings[0];

  sprokit::process::port_addr_t const& up_addr = mapping.first;
  sprokit::process::name_t const& up_name = up_addr.first;
  sprokit::process::port_t const& up_port = up_addr.second;

  if (up_name != cluster_name)
  {
    TEST_ERROR("A cluster input mapping\'s upstream name is not the cluster itself");
  }

  if (up_port != port)
  {
    TEST_ERROR("A cluster input mapping\'s upstream port is not the one requested");
  }

  sprokit::process::port_addr_t const& down_addr = mapping.second;
  sprokit::process::name_t const& down_name = down_addr.first;
  sprokit::process::port_t const& down_port = down_addr.second;

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

  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("print_number");
  sprokit::process::port_t const port1 = sprokit::process::port_t("cluster_number1");
  sprokit::process::port_t const port2 = sprokit::process::port_t("cluster_number2");
  sprokit::process::port_t const mapped_port = sprokit::process::port_t("number");

  sprokit::load_known_modules();

  cluster->_add_process(name, type);

  cluster->_map_input(port1, name, mapped_port);

  EXPECT_EXCEPTION(sprokit::port_reconnect_exception,
                   cluster->_map_input(port2, name, mapped_port),
                   "mapping a second cluster port to a process input port");
}

IMPLEMENT_TEST(map_input_no_exist)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  sprokit::process::port_t const port = sprokit::process::port_t("port");
  sprokit::process::name_t const name = sprokit::process::name_t("name");

  EXPECT_EXCEPTION(sprokit::no_such_process_exception,
                   cluster->_map_input(port, name, port),
                   "mapping an input to a non-existent process");
}

IMPLEMENT_TEST(map_input_port_no_exist)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  sprokit::process::port_t const port = sprokit::process::port_t("no_such_port");
  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("orphan");

  sprokit::load_known_modules();

  cluster->_add_process(name, type);

  EXPECT_EXCEPTION(sprokit::no_such_port_exception,
                   cluster->_map_input(port, name, port),
                   "mapping an input to a non-existent port");
}

IMPLEMENT_TEST(map_output)
{
  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  conf->set_value(sprokit::process::config_name, cluster_name);

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>(conf);

  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("numbers");
  sprokit::process::port_t const port = sprokit::process::port_t("cluster_number");
  sprokit::process::port_t const mapped_port = sprokit::process::port_t("number");

  sprokit::load_known_modules();

  cluster->_add_process(name, type);

  cluster->_map_output(port, name, mapped_port);

  sprokit::process::connections_t const mappings = cluster->output_mappings();

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

  sprokit::process::connection_t const& mapping = mappings[0];

  sprokit::process::port_addr_t const& down_addr = mapping.second;
  sprokit::process::name_t const& down_name = down_addr.first;
  sprokit::process::port_t const& down_port = down_addr.second;

  if (down_name != cluster_name)
  {
    TEST_ERROR("A cluster output mapping\'s downstream name is not the cluster itself");
  }

  if (down_port != port)
  {
    TEST_ERROR("A cluster output mapping\'s downstream port is not the one requested");
  }

  sprokit::process::port_addr_t const& up_addr = mapping.first;
  sprokit::process::name_t const& up_name = up_addr.first;
  sprokit::process::port_t const& up_port = up_addr.second;

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

  sprokit::process::name_t const name1 = sprokit::process::name_t("name1");
  sprokit::process::name_t const name2 = sprokit::process::name_t("name2");
  sprokit::process::type_t const type = sprokit::process::type_t("numbers");
  sprokit::process::port_t const port = sprokit::process::port_t("cluster_number");
  sprokit::process::port_t const mapped_port = sprokit::process::port_t("number");

  sprokit::load_known_modules();

  cluster->_add_process(name1, type);
  cluster->_add_process(name2, type);

  cluster->_map_output(port, name1, mapped_port);

  EXPECT_EXCEPTION(sprokit::port_reconnect_exception,
                   cluster->_map_output(port, name2, mapped_port),
                   "mapping a second port to a cluster output port");
}

IMPLEMENT_TEST(map_output_no_exist)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  sprokit::process::port_t const port = sprokit::process::port_t("port");
  sprokit::process::name_t const name = sprokit::process::name_t("name");

  EXPECT_EXCEPTION(sprokit::no_such_process_exception,
                   cluster->_map_output(port, name, port),
                   "mapping an output to a non-existent process");
}

IMPLEMENT_TEST(map_output_port_no_exist)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  sprokit::process::port_t const port = sprokit::process::port_t("no_such_port");
  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("orphan");

  sprokit::load_known_modules();

  cluster->_add_process(name, type);

  EXPECT_EXCEPTION(sprokit::no_such_port_exception,
                   cluster->_map_output(port, name, port),
                   "mapping an output to a non-existent port");
}

IMPLEMENT_TEST(connect)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  sprokit::process::name_t const name1 = sprokit::process::name_t("name1");
  sprokit::process::name_t const name2 = sprokit::process::name_t("name2");
  sprokit::process::type_t const type1 = sprokit::process::type_t("numbers");
  sprokit::process::type_t const type2 = sprokit::process::type_t("print_number");
  sprokit::process::port_t const port = sprokit::process::port_t("number");

  sprokit::load_known_modules();

  cluster->_add_process(name1, type1);
  cluster->_add_process(name2, type2);

  cluster->_connect(name1, port, name2, port);

  sprokit::process::connections_t const mappings = cluster->internal_connections();

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

  sprokit::process::connection_t const& mapping = mappings[0];

  sprokit::process::port_addr_t const& down_addr = mapping.second;
  sprokit::process::name_t const& down_name = down_addr.first;
  sprokit::process::port_t const& down_port = down_addr.second;

  sprokit::process::port_addr_t const& up_addr = mapping.first;
  sprokit::process::name_t const& up_name = up_addr.first;
  sprokit::process::port_t const& up_port = up_addr.second;

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

  sprokit::process::name_t const name1 = sprokit::process::name_t("name1");
  sprokit::process::name_t const name2 = sprokit::process::name_t("name2");
  sprokit::process::type_t const type = sprokit::process::type_t("print_number");
  sprokit::process::port_t const port = sprokit::process::port_t("number");

  sprokit::load_known_modules();

  cluster->_add_process(name2, type);

  EXPECT_EXCEPTION(sprokit::no_such_process_exception,
                   cluster->_connect(name1, port, name2, port),
                   "making a connection when the upstream process does not exist");
}

IMPLEMENT_TEST(connect_upstream_port_no_exist)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  sprokit::process::name_t const name1 = sprokit::process::name_t("name1");
  sprokit::process::name_t const name2 = sprokit::process::name_t("name2");
  sprokit::process::type_t const type1 = sprokit::process::type_t("numbers");
  sprokit::process::type_t const type2 = sprokit::process::type_t("print_number");
  sprokit::process::port_t const port1 = sprokit::process::port_t("no_such_port");
  sprokit::process::port_t const port2 = sprokit::process::port_t("number");

  sprokit::load_known_modules();

  cluster->_add_process(name1, type1);
  cluster->_add_process(name2, type2);

  EXPECT_EXCEPTION(sprokit::no_such_port_exception,
                   cluster->_connect(name1, port1, name2, port2),
                   "making a connection when the upstream port does not exist");
}

IMPLEMENT_TEST(connect_downstream_no_exist)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  sprokit::process::name_t const name1 = sprokit::process::name_t("name1");
  sprokit::process::name_t const name2 = sprokit::process::name_t("name2");
  sprokit::process::type_t const type = sprokit::process::type_t("numbers");
  sprokit::process::port_t const port = sprokit::process::port_t("number");

  sprokit::load_known_modules();

  cluster->_add_process(name1, type);

  EXPECT_EXCEPTION(sprokit::no_such_process_exception,
                   cluster->_connect(name1, port, name2, port),
                   "making a connection when the upstream process does not exist");
}

IMPLEMENT_TEST(connect_downstream_port_no_exist)
{
  sample_cluster_t const cluster = boost::make_shared<sample_cluster>();

  sprokit::process::name_t const name1 = sprokit::process::name_t("name1");
  sprokit::process::name_t const name2 = sprokit::process::name_t("name2");
  sprokit::process::type_t const type1 = sprokit::process::type_t("numbers");
  sprokit::process::type_t const type2 = sprokit::process::type_t("print_number");
  sprokit::process::port_t const port1 = sprokit::process::port_t("number");
  sprokit::process::port_t const port2 = sprokit::process::port_t("no_such_port");

  sprokit::load_known_modules();

  cluster->_add_process(name1, type1);
  cluster->_add_process(name2, type2);

  EXPECT_EXCEPTION(sprokit::no_such_port_exception,
                   cluster->_connect(name1, port1, name2, port2),
                   "making a connection when the downstream port does not exist");
}

IMPLEMENT_TEST(reconfigure_pass_tunable_mappings)
{
  sprokit::load_known_modules();

  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::config_t const cluster_conf = sprokit::config::empty_config();

  cluster_conf->set_value(sprokit::process::config_name, cluster_name);

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>(cluster_conf);

  sprokit::config::key_t const key = sprokit::config::key_t("key");

  cluster->_declare_configuration_key(
    key,
    sprokit::config::value_t(),
    sprokit::config::description_t(),
    true);

  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("expect");

  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::config::key_t const key_tunable = sprokit::config::key_t("tunable");
  sprokit::config::key_t const key_expect = sprokit::config::key_t("expect");

  cluster->_map_config(key, name, key_tunable);

  sprokit::config::value_t const tunable_value = sprokit::config::value_t("old_value");
  sprokit::config::value_t const tuned_value = sprokit::config::value_t("new_value");

  conf->set_value(key_tunable, tunable_value);
  conf->set_value(key_expect, tuned_value);

  cluster->_add_process(name, type, conf);

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  new_conf->set_value(cluster_name + sprokit::config::block_sep + key, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(reconfigure_no_pass_untunable_mappings)
{
  sprokit::load_known_modules();

  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::config_t const cluster_conf = sprokit::config::empty_config();

  cluster_conf->set_value(sprokit::process::config_name, cluster_name);

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>(cluster_conf);

  sprokit::config::key_t const key = sprokit::config::key_t("key");

  cluster->_declare_configuration_key(
    key,
    sprokit::config::value_t(),
    sprokit::config::description_t(),
    false);

  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("expect");

  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::config::key_t const key_tunable = sprokit::config::key_t("tunable");
  sprokit::config::key_t const key_expect = sprokit::config::key_t("expect");

  cluster->_map_config(key, name, key_tunable);

  sprokit::config::value_t const tunable_value = sprokit::config::value_t("old_value");

  conf->set_value(key_tunable, tunable_value);
  conf->set_value(key_expect, tunable_value);

  cluster->_add_process(name, type, conf);

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  sprokit::config::value_t const tuned_value = sprokit::config::value_t("new_value");

  new_conf->set_value(cluster_name + sprokit::config::block_sep + key, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(reconfigure_pass_extra)
{
  sprokit::load_known_modules();

  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::config_t const cluster_conf = sprokit::config::empty_config();

  cluster_conf->set_value(sprokit::process::config_name, cluster_name);

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>(cluster_conf);

  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("expect");

  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::config::key_t const key_expect = sprokit::config::key_t("expect");
  sprokit::config::key_t const key_expect_key = sprokit::config::key_t("expect_key");

  sprokit::config::value_t const extra_key = sprokit::config::value_t("new_key");

  conf->set_value(key_expect, extra_key);
  conf->set_value(key_expect_key, "true");

  cluster->_add_process(name, type, conf);

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  new_conf->set_value(cluster_name + sprokit::config::block_sep + name + sprokit::config::block_sep + extra_key, extra_key);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(reconfigure_tunable_only_if_mapped)
{
  sprokit::load_known_modules();

  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::config_t const cluster_conf = sprokit::config::empty_config();

  cluster_conf->set_value(sprokit::process::config_name, cluster_name);

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>(cluster_conf);

  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("expect");

  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::config::key_t const key_tunable = sprokit::config::key_t("tunable");
  sprokit::config::key_t const key_expect = sprokit::config::key_t("expect");

  sprokit::config::value_t const tunable_value = sprokit::config::value_t("old_value");

  conf->set_value(key_tunable, tunable_value);
  conf->mark_read_only(key_tunable);
  conf->set_value(key_expect, tunable_value);

  cluster->_add_process(name, type, conf);

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  sprokit::config::value_t const tuned_value = sprokit::config::value_t("new_value");

  new_conf->set_value(cluster_name + sprokit::config::block_sep + name + sprokit::config::block_sep + key_tunable, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(reconfigure_mapped_untunable)
{
  sprokit::load_known_modules();

  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::config_t const cluster_conf = sprokit::config::empty_config();

  cluster_conf->set_value(sprokit::process::config_name, cluster_name);

  sample_cluster_t const cluster = boost::make_shared<sample_cluster>(cluster_conf);

  sprokit::config::key_t const key = sprokit::config::key_t("key");

  sprokit::config::value_t const tunable_value = sprokit::config::value_t("old_value");

  cluster->_declare_configuration_key(
    key,
    tunable_value,
    sprokit::config::description_t(),
    true);

  sprokit::process::name_t const name = sprokit::process::name_t("name");
  sprokit::process::type_t const type = sprokit::process::type_t("expect");

  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::config::key_t const key_tunable = sprokit::config::key_t("tunable");
  sprokit::config::key_t const key_expect = sprokit::config::key_t("expect");

  cluster->_map_config(key, name, key_expect);

  sprokit::config::value_t const tuned_value = sprokit::config::value_t("new_value");

  conf->set_value(key_tunable, tunable_value);

  cluster->_add_process(name, type, conf);

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  new_conf->set_value(cluster_name + sprokit::config::block_sep + key, tuned_value);

  pipeline->reconfigure(new_conf);
}

empty_cluster
::empty_cluster()
  : sprokit::process_cluster(sprokit::config::empty_config())
{
}

empty_cluster
::~empty_cluster()
{
}

sample_cluster
::sample_cluster(sprokit::config_t const& conf)
  : sprokit::process_cluster(conf)
{
}

sample_cluster
::~sample_cluster()
{
}

void
sample_cluster
::_declare_configuration_key(sprokit::config::key_t const& key,
                             sprokit::config::value_t const& def_,
                             sprokit::config::description_t const& description_,
                             bool tunable_)
{
  declare_configuration_key(key, def_, description_, tunable_);
}

void
sample_cluster
::_map_config(sprokit::config::key_t const& key, name_t const& name_, sprokit::config::key_t const& mapped_key)
{
  map_config(key, name_, mapped_key);
}

void
sample_cluster
::_add_process(name_t const& name_, type_t const& type_, sprokit::config_t const& config)
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
