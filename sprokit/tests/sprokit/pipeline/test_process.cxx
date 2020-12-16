/*ckwg +29
 * Copyright 2011-2016 by Kitware, Inc.
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

#include <vital/config/config_block.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <sprokit/pipeline/edge.h>
#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_exception.h>
#include <sprokit/pipeline/process_factory.h>

#include <memory>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

/**
 * @brief Create a process.
 *
 * This function creates a process from the registry.
 *
 * @param type Process type.
 * @param name Process name.
 * @param conf Config for process.
 *
 * @return Pointer to new process.
 */
static sprokit::process_t create_process(sprokit::process::type_t const& type,
                                         sprokit::process::name_t const& name = sprokit::process::name_t(),
                                         kwiver::vital::config_block_sptr const& conf = kwiver::vital::config_block::empty_config());

static sprokit::edge_t create_edge();


// ------------------------------------------------------------------
class remove_ports_process
  : public sprokit::process
{
  public:
    remove_ports_process(port_type_t const& port_type);
    ~remove_ports_process();

    void create_input_port(port_t const& port, port_type_t const& port_type);
    void create_output_port(port_t const& port, port_type_t const& port_type);

    void _remove_input_port(port_t const& port);
    void _remove_output_port(port_t const& port);

    static port_type_t const input_port;
    static port_type_t const output_port;
};


// ------------------------------------------------------------------
class null_config_process
  : public sprokit::process
{
  public:
    null_config_process(kwiver::vital::config_block_sptr const& config);
    ~null_config_process();
};


// ------------------------------------------------------------------
class null_input_info_process
  : public sprokit::process
{
  public:
    null_input_info_process(kwiver::vital::config_block_sptr const& config);
    ~null_input_info_process();
};


// ------------------------------------------------------------------
class null_output_info_process
  : public sprokit::process
{
  public:
    null_output_info_process(kwiver::vital::config_block_sptr const& config);
    ~null_output_info_process();
};


// ------------------------------------------------------------------
class null_conf_info_process
  : public sprokit::process
{
  public:
    null_conf_info_process(kwiver::vital::config_block_sptr const& config);
    ~null_conf_info_process();
};


// ==================================================================
IMPLEMENT_TEST(null_input_edge)
{
  const auto proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

  sprokit::edge_t const edge;

  EXPECT_EXCEPTION(sprokit::null_edge_port_connection_exception,
                   process->connect_input_port(sprokit::process::port_t(), edge),
                   "connecting a NULL edge to an input port");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(null_output_edge)
{
  const auto proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

  sprokit::edge_t const edge;

  EXPECT_EXCEPTION(sprokit::null_edge_port_connection_exception,
                   process->connect_output_port(sprokit::process::port_t(), edge),
                   "connecting a NULL edge to an output port");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(connect_after_init)
{
  const auto proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

  const auto config = kwiver::vital::config_block::empty_config();

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  sprokit::pipeline_t const pipe = std::make_shared<sprokit::pipeline>(config);

  // Only the pipeline can properly initialize a process.
  pipe->add_process(process);
  pipe->setup_pipeline();

  EXPECT_EXCEPTION(sprokit::connect_to_initialized_process_exception,
                   process->connect_input_port(sprokit::process::port_t(), edge),
                   "connecting an input edge after initialization");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(configure_twice)
{
  const auto proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

  process->configure();

  EXPECT_EXCEPTION(sprokit::reconfigured_exception,
                   process->configure(),
                   "reconfiguring a process");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(reinit)
{
  const auto proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

  process->configure();
  process->init();

  EXPECT_EXCEPTION(sprokit::reinitialization_exception,
                   process->init(),
                   "reinitializing a process");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(reset)
{
  const auto proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

  process->configure();

  process->reset();

  process->configure();
  process->init();

  process->reset();

  process->configure();
  process->init();
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(step_before_configure)
{
  const auto proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

  EXPECT_EXCEPTION(sprokit::unconfigured_exception,
                   process->step(),
                   "stepping before configuring");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(step_before_init)
{
  const auto proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

  process->configure();

  EXPECT_EXCEPTION(sprokit::uninitialized_exception,
                   process->step(),
                   "stepping before initialization");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(set_static_input_type)
{
  const auto proc_type = sprokit::process::type_t("multiplication");

  const auto port_name = sprokit::process::port_t("factor1");
  const auto port_type = sprokit::process::port_type_t("type");

  sprokit::process_t const process = create_process(proc_type);

  EXPECT_EXCEPTION(sprokit::static_type_reset_exception,
                   process->set_input_port_type(port_name, port_type),
                   "setting the type of a non-dependent input port");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(set_static_output_type)
{
  const auto proc_type = sprokit::process::type_t("multiplication");

  const auto port_name = sprokit::process::port_t("product");
  const auto port_type = sprokit::process::port_type_t("type");

  sprokit::process_t const process = create_process(proc_type);

  EXPECT_EXCEPTION(sprokit::static_type_reset_exception,
                   process->set_output_port_type(port_name, port_type),
                   "setting the type of a non-dependent output port");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(set_input_type_duplicate)
{
  const auto proc_type = sprokit::process::type_t("flow_dependent");

  const auto port_name = sprokit::process::port_t("input");
  const auto port_type = sprokit::process::port_type_t("type");

  sprokit::process_t const process = create_process(proc_type);

  process->set_input_port_type(port_name, port_type);

  sprokit::process::port_info_t const port_info_before = process->input_port_info(port_name);

  process->set_input_port_type(port_name, port_type);

  sprokit::process::port_info_t const port_info_duplicate = process->input_port_info(port_name);

  // If nothing actually changes, info pointers should still be valid.
  if (port_info_before != port_info_duplicate)
  {
    TEST_ERROR("Input port information changed without a data change");
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(set_output_type_duplicate)
{
  const auto proc_type = sprokit::process::type_t("flow_dependent");

  const auto port_name = sprokit::process::port_t("output");
  const auto port_type = sprokit::process::port_type_t("type");

  sprokit::process_t const process = create_process(proc_type);

  process->set_output_port_type(port_name, port_type);

  sprokit::process::port_info_t const port_info_before = process->output_port_info(port_name);

  process->set_output_port_type(port_name, port_type);

  sprokit::process::port_info_t const port_info_duplicate = process->output_port_info(port_name);

  // If nothing actually changes, info pointers should still be valid.
  if (port_info_before != port_info_duplicate)
  {
    TEST_ERROR("Output port information changed without a data change");
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(set_input_type_after_init)
{
  const auto proc_type = sprokit::process::type_t("flow_dependent");

  const auto port_name = sprokit::process::port_t("input");
  const auto port_type = sprokit::process::port_type_t("type");

  sprokit::process_t const process = create_process(proc_type);

  process->configure();
  process->init();

  EXPECT_EXCEPTION(sprokit::set_type_on_initialized_process_exception,
                   process->set_input_port_type(port_name, port_type),
                   "setting an input port type after initialization");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(set_output_type_after_init)
{
  const auto proc_type = sprokit::process::type_t("flow_dependent");

  const auto port_name = sprokit::process::port_t("output");
  const auto port_type = sprokit::process::port_type_t("type");

  sprokit::process_t const process = create_process(proc_type);

  process->configure();
  process->init();

  EXPECT_EXCEPTION(sprokit::set_type_on_initialized_process_exception,
                   process->set_output_port_type(port_name, port_type),
                   "setting an output port type after initialization");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(set_tagged_flow_dependent_port)
{
  const auto proc_type = sprokit::process::type_t("tagged_flow_dependent");

  const auto iport_name = sprokit::process::port_t("tagged_input");
  const auto oport_name = sprokit::process::port_t("tagged_output");

  const auto port_type = sprokit::process::port_type_t("type");

  sprokit::process_t const input_process = create_process(proc_type);

  if (!input_process->set_input_port_type(iport_name, port_type))
  {
    TEST_ERROR("Could not set the input port type");
  }

  sprokit::process::port_info_t const iiinfo = input_process->input_port_info(iport_name);
  sprokit::process::port_info_t const ioinfo = input_process->output_port_info(oport_name);

  if (iiinfo->type != port_type)
  {
    TEST_ERROR("Setting the input port type is not reflected in port info");
  }

  if (ioinfo->type != port_type)
  {
    TEST_ERROR("Setting the input port type did not also set the output port info");
  }

  sprokit::process_t const output_process = create_process(proc_type);

  if (!output_process->set_output_port_type(oport_name, port_type))
  {
    TEST_ERROR("Could not set the output port type");
  }

  sprokit::process::port_info_t const oiinfo = output_process->input_port_info(iport_name);
  sprokit::process::port_info_t const ooinfo = output_process->output_port_info(oport_name);

  if (ooinfo->type != port_type)
  {
    TEST_ERROR("Setting the output port type is not reflected in port info");
  }

  if (oiinfo->type != port_type)
  {
    TEST_ERROR("Setting the output port type did not also set the input port info");
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(set_tagged_flow_dependent_port_cascade)
{
  const auto proc_type = sprokit::process::type_t("tagged_flow_dependent");

  const auto iport_name = sprokit::process::port_t("tagged_input");
  const auto oport_name = sprokit::process::port_t("tagged_output");

  const auto tag_port_type = sprokit::process::type_flow_dependent + sprokit::process::port_type_t("other_tag");
  const auto port_type = sprokit::process::port_type_t("type");

  sprokit::process_t const input_process = create_process(proc_type);

  if (!input_process->set_input_port_type(iport_name, tag_port_type))
  {
    TEST_ERROR("Could not set the input port type");
  }

  if (!input_process->set_output_port_type(oport_name, port_type))
  {
    TEST_ERROR("Could not set the output port type");
  }

  sprokit::process::port_info_t const iinfo = input_process->input_port_info(iport_name);
  sprokit::process::port_info_t const oinfo = input_process->output_port_info(oport_name);

  if (iinfo->type != port_type)
  {
    TEST_ERROR("Setting the input port type is not reflected in port info");
  }

  if (oinfo->type != port_type)
  {
    TEST_ERROR("Setting the input port type did not also set the output port info");
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(set_tagged_flow_dependent_port_cascade_any)
{
  const auto proc_type = sprokit::process::type_t("tagged_flow_dependent");

  const auto iport_name = sprokit::process::port_t("tagged_input");
  const auto oport_name = sprokit::process::port_t("tagged_output");

  const auto port_type = sprokit::process::port_type_t("type");

  sprokit::process_t const input_process = create_process(proc_type);

  if (!input_process->set_input_port_type(iport_name, sprokit::process::type_any))
  {
    TEST_ERROR("Could not set the input port type");
  }

  if (!input_process->set_input_port_type(iport_name, port_type))
  {
    TEST_ERROR("Could not set the input port type after setting it to 'any'");
  }

  sprokit::process::port_info_t const info = input_process->input_port_info(iport_name);

  if (info->type != port_type)
  {
    TEST_ERROR("Setting the port type on a flow port set to 'any' did not apply");
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(add_input_port_after_type_pin)
{
  const auto proc_type = sprokit::process::type_t("collate");

  const auto status = sprokit::process::port_t("status/");
  const auto res = sprokit::process::port_t("res/");
  const auto coll = sprokit::process::port_t("coll/");

  const auto tag = sprokit::process::port_t("test");
  const auto group = sprokit::process::port_t("/group");

  const auto port_type = sprokit::process::port_type_t("type");

  sprokit::process_t const process = create_process(proc_type);
  sprokit::edge_t const edge = create_edge();

  (void)process->input_port_info(status + tag);

  EXPECT_EXCEPTION(sprokit::no_such_port_exception,
                   process->connect_input_port(coll + tag + group, edge),
                   "making sure the input port does not exist");

  if (!process->set_output_port_type(res + tag, port_type))
  {
    TEST_ERROR("Could not set the source port type");
  }

  sprokit::process::port_info_t const info = process->input_port_info(coll + tag + group);

  if (info->type != port_type)
  {
    TEST_ERROR("The port with an input tagged dependency port was not set to the new type automatically upon creation");
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(add_output_port_after_type_pin)
{
  const auto proc_type = sprokit::process::type_t("distribute");

  const auto status = sprokit::process::port_t("status/");
  const auto src = sprokit::process::port_t("src/");
  const auto dist = sprokit::process::port_t("dist/");

  const auto tag = sprokit::process::port_t("test");
  const auto group = sprokit::process::port_t("/group");

  const auto port_type = sprokit::process::port_type_t("type");

  sprokit::process_t const process = create_process(proc_type);
  sprokit::edge_t const edge = create_edge();

  (void)process->output_port_info(status + tag);

  EXPECT_EXCEPTION(sprokit::no_such_port_exception,
                   process->connect_output_port(dist + tag + group, edge),
                   "making sure the output port does not exist");

  if (!process->set_input_port_type(src + tag, port_type))
  {
    TEST_ERROR("Could not set the source port type");
  }

  sprokit::process::port_info_t const info = process->output_port_info(dist + tag + group);

  if (info->type != port_type)
  {
    TEST_ERROR("The port with an output tagged dependency port was not set to the new type automatically upon creation");
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(set_untagged_flow_dependent_port)
{
  const auto proc_type = sprokit::process::type_t("tagged_flow_dependent");

  const auto iport_name = sprokit::process::port_t("untagged_input");
  const auto oport_name = sprokit::process::port_t("untagged_output");

  const auto iport_type = sprokit::process::port_type_t("itype");
  const auto oport_type = sprokit::process::port_type_t("otype");

  sprokit::process_t const process = create_process(proc_type);

  if (!process->set_input_port_type(iport_name, iport_type))
  {
    TEST_ERROR("Could not set the input port type");
  }

  sprokit::process::port_info_t const iiinfo = process->input_port_info(iport_name);
  sprokit::process::port_info_t const ioinfo = process->output_port_info(oport_name);

  if (iiinfo->type != iport_type)
  {
    TEST_ERROR("Setting the input port type is not reflected in port info");
  }

  if (ioinfo->type == iport_type)
  {
    TEST_ERROR("Setting the input port type set the output port info");
  }

  if (!process->set_output_port_type(oport_name, oport_type))
  {
    TEST_ERROR("Could not set the output port type");
  }

  const auto oiinfo = process->input_port_info(iport_name);
  const auto ooinfo = process->output_port_info(oport_name);

  if (ooinfo->type != oport_type)
  {
    TEST_ERROR("Setting the output port type is not reflected in port info");
  }

  if (oiinfo->type == oport_type)
  {
    TEST_ERROR("Setting the output port type did not also set the input port info");
  }
}


sprokit::process::port_type_t const remove_ports_process::input_port = sprokit::process::port_type_t("input");
sprokit::process::port_type_t const remove_ports_process::output_port = sprokit::process::port_type_t("output");


// ------------------------------------------------------------------
IMPLEMENT_TEST(remove_input_port)
{
  const auto type = sprokit::process::port_type_t("type");
  std::unique_ptr<remove_ports_process> proc(new remove_ports_process(type));

  proc->_remove_input_port(remove_ports_process::input_port);

  EXPECT_EXCEPTION(sprokit::no_such_port_exception,
                   proc->input_port_info(remove_ports_process::input_port),
                   "after removing an input port");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(remove_output_port)
{
  const auto type = sprokit::process::port_type_t("type");
  std::unique_ptr<remove_ports_process> proc(new remove_ports_process(type));

  proc->_remove_output_port(remove_ports_process::output_port);

  EXPECT_EXCEPTION(sprokit::no_such_port_exception,
                   proc->output_port_info(remove_ports_process::output_port),
                   "after removing an output port");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(remove_non_exist_input_port)
{
  const auto port = sprokit::process::port_t("port");
  const auto type = sprokit::process::port_type_t("type");
  std::unique_ptr<remove_ports_process> proc(new remove_ports_process(type));

  EXPECT_EXCEPTION(sprokit::no_such_port_exception,
                   proc->_remove_input_port(port),
                   "after removing a non-existent input port");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(remove_non_exist_output_port)
{
  const auto port = sprokit::process::port_t("port");
  const auto type = sprokit::process::port_type_t("type");
  std::unique_ptr<remove_ports_process> proc(new remove_ports_process(type));

  EXPECT_EXCEPTION(sprokit::no_such_port_exception,
                   proc->_remove_output_port(port),
                   "after removing a non-existent output port");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(remove_only_tagged_flow_dependent_port)
{
  const auto port = sprokit::process::port_t("port");
  const auto flow_type = sprokit::process::type_flow_dependent + sprokit::process::port_type_t("tag");
  const auto type = sprokit::process::port_type_t("type");
  std::unique_ptr<remove_ports_process> proc(new remove_ports_process(flow_type));

  proc->_remove_input_port(remove_ports_process::input_port);
  proc->set_output_port_type(remove_ports_process::output_port, type);
  proc->_remove_output_port(remove_ports_process::output_port);
  proc->create_input_port(port, flow_type);

  sprokit::process::port_info_t const info = proc->input_port_info(port);
  sprokit::process::port_type_t const& new_type = info->type;

  if (new_type != flow_type)
  {
    TEST_ERROR("The flow type was not reset after all tagged ports were removed.");
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(remove_tagged_flow_dependent_port)
{
  const auto flow_type = sprokit::process::type_flow_dependent + sprokit::process::port_type_t("tag");
  const auto type = sprokit::process::port_type_t("type");
  std::unique_ptr<remove_ports_process> proc(new remove_ports_process(flow_type));

  proc->set_output_port_type(remove_ports_process::output_port, type);
  proc->_remove_output_port(remove_ports_process::output_port);

  sprokit::process::port_info_t const info = proc->input_port_info(remove_ports_process::input_port);
  sprokit::process::port_type_t const& new_type = info->type;

  if (new_type != type)
  {
    TEST_ERROR("The flow type was reset even when more were left.");
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(null_config)
{
  const auto proc_type = sprokit::process::type_t("null_config");

  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  auto fact = vpm.ADD_PROCESS( null_config_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, proc_type );

  const auto proc_name = sprokit::process::name_t(proc_type);

  EXPECT_EXCEPTION(sprokit::null_process_config_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as the configuration for a process");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(null_input_port_info)
{
  const auto proc_type = sprokit::process::type_t("null_input_port");

  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();

  auto fact = vpm.ADD_PROCESS( null_input_info_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, proc_type );

  const auto proc_name = sprokit::process::name_t(proc_type);

  EXPECT_EXCEPTION(sprokit::null_input_port_info_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as an input port info structure");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(null_output_port_info)
{
  const auto proc_type = sprokit::process::type_t("null_output_port");

  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();

  auto fact = vpm.ADD_PROCESS( null_output_info_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, proc_type );

  const auto proc_name = sprokit::process::name_t(proc_type);

  EXPECT_EXCEPTION(sprokit::null_output_port_info_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as an output port info structure");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(null_conf_info)
{
  const auto proc_type = sprokit::process::type_t("null_conf");

  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();

  auto fact = vpm.ADD_PROCESS( null_conf_info_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, proc_type );

  const auto proc_name = sprokit::process::name_t(proc_type);

  EXPECT_EXCEPTION(sprokit::null_conf_info_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as an configuration info structure");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(tunable_config)
{
  const auto proc_type = sprokit::process::type_t("tunable");

  const auto tunable_key = kwiver::vital::config_block_key_t("tunable");

  sprokit::process_t const proc = create_process(proc_type);

  kwiver::vital::config_block_keys_t const tunable = proc->available_tunable_config();

  if (tunable.size() != 1)
  {
    TEST_ERROR("Failed to get the expected number of tunable parameters");
  }

  if (tunable.empty())
  {
    return;
  }

  if (tunable[0] != tunable_key)
  {
    TEST_ERROR("Failed to get the expected tunable parameter");
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(tunable_config_read_only)
{
  const auto proc_type = sprokit::process::type_t("tunable");
  const auto proc_name = sprokit::process::name_t(proc_type);

  const auto conf = kwiver::vital::config_block::empty_config();

  const auto tunable_key = kwiver::vital::config_block_key_t("tunable");
  kwiver::vital::config_block_value_t const tunable_value = kwiver::vital::config_block_value_t("value");

  conf->set_value(tunable_key, tunable_value);
  conf->mark_read_only(tunable_key);

  sprokit::process_t const proc = create_process(proc_type, proc_name, conf);

  kwiver::vital::config_block_keys_t const tunable = proc->available_tunable_config();

  if (!tunable.empty())
  {
    TEST_ERROR("Failed to exclude read-only parameters as tunable");
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(reconfigure_tunable)
{
  const auto proc_type = sprokit::process::type_t("expect");
  const auto proc_name = sprokit::process::name_t("name");

  const auto conf = kwiver::vital::config_block::empty_config();

  const auto key_tunable = kwiver::vital::config_block_key_t("tunable");
  const auto key_expect = kwiver::vital::config_block_key_t("expect");

  const auto tunable_value = kwiver::vital::config_block_value_t("old_value");
  const auto tuned_value = kwiver::vital::config_block_value_t("new_value");

  conf->set_value(key_tunable, tunable_value);
  conf->set_value(key_expect, tuned_value);
  conf->set_value(sprokit::process::config_name, proc_name);

  sprokit::process_t const expect = create_process(proc_type, proc_name, conf);

  sprokit::pipeline_t const pipeline = std::make_shared<sprokit::pipeline>(kwiver::vital::config_block::empty_config());

  pipeline->add_process(expect);
  pipeline->setup_pipeline();

  kwiver::vital::config_block_sptr const new_conf = kwiver::vital::config_block::empty_config();

  new_conf->set_value(proc_name + kwiver::vital::config_block::block_sep() +
                      key_tunable, tuned_value);

  pipeline->reconfigure(new_conf);
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(reconfigure_non_tunable)
{
  const auto proc_type = sprokit::process::type_t("expect");
  const auto proc_name = sprokit::process::name_t("name");

  const auto conf = kwiver::vital::config_block::empty_config();

  const auto key_tunable = kwiver::vital::config_block_key_t("tunable");
  const auto key_expect = kwiver::vital::config_block_key_t("expect");

  const auto tunable_value = kwiver::vital::config_block_value_t("old_value");

  conf->set_value(key_tunable, tunable_value);
  conf->mark_read_only(key_tunable);
  conf->set_value(key_expect, tunable_value);
  conf->set_value(sprokit::process::config_name, proc_name);

  sprokit::process_t const expect = create_process(proc_type, proc_name, conf);

  sprokit::pipeline_t const pipeline = std::make_shared<sprokit::pipeline>(kwiver::vital::config_block::empty_config());

  pipeline->add_process(expect);
  pipeline->setup_pipeline();

  kwiver::vital::config_block_sptr const new_conf = kwiver::vital::config_block::empty_config();

  kwiver::vital::config_block_value_t const tuned_value = kwiver::vital::config_block_value_t("new_value");

  new_conf->set_value(proc_name + kwiver::vital::config_block::block_sep() +
                      key_tunable, tuned_value);

  pipeline->reconfigure(new_conf);
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(reconfigure_extra_parameters)
{
  const auto proc_type = sprokit::process::type_t("expect");
  const auto proc_name = sprokit::process::name_t("name");

  const auto conf = kwiver::vital::config_block::empty_config();

  const auto new_key = kwiver::vital::config_block_key_t("new_key");

  const auto key_expect = kwiver::vital::config_block_key_t("expect");
  const auto key_expect_key = kwiver::vital::config_block_key_t("expect_key");

  conf->set_value(key_expect, new_key);
  conf->set_value(key_expect_key, "true");
  conf->set_value(sprokit::process::config_name, proc_name);

  sprokit::process_t const expect = create_process(proc_type, proc_name, conf);

  sprokit::pipeline_t const pipeline = std::make_shared<sprokit::pipeline>(kwiver::vital::config_block::empty_config());

  pipeline->add_process(expect);
  pipeline->setup_pipeline();

  const auto new_conf = kwiver::vital::config_block::empty_config();

  const auto tunable_value = kwiver::vital::config_block_value_t("old_value");

  new_conf->set_value(proc_name + kwiver::vital::config_block::block_sep() +
                      new_key, tunable_value);

  pipeline->reconfigure(new_conf);
}


// ==================================================================
sprokit::process_t
create_process(sprokit::process::type_t const& type,
               sprokit::process::name_t const& name,
               kwiver::vital::config_block_sptr const& conf)
{
  static bool const modules_loaded = (kwiver::vital::plugin_manager::instance().load_all_plugins(), true);
  (void)modules_loaded;

  return sprokit::create_process(type, name, conf);
}


// ------------------------------------------------------------------
sprokit::edge_t
create_edge()
{
  const auto config = kwiver::vital::config_block::empty_config();
  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  return edge;
}


// ------------------------------------------------------------------
null_config_process
::null_config_process(kwiver::vital::config_block_sptr const& /*config*/)
  : sprokit::process(kwiver::vital::config_block_sptr())
{
}


// ------------------------------------------------------------------
null_config_process
::~null_config_process()
{
}


// ------------------------------------------------------------------
null_input_info_process
::null_input_info_process(kwiver::vital::config_block_sptr const& config)
  : sprokit::process(config)
{
  name_t const port = name_t("port");

  declare_input_port(port, port_info_t());
}

null_input_info_process
::~null_input_info_process()
{
}


// ------------------------------------------------------------------
null_output_info_process
::null_output_info_process(kwiver::vital::config_block_sptr const& config)
  : sprokit::process(config)
{
  name_t const port = name_t("port");

  declare_output_port(port, port_info_t());
}


// ------------------------------------------------------------------
null_output_info_process
::~null_output_info_process()
{
}


// ------------------------------------------------------------------
null_conf_info_process
::null_conf_info_process(kwiver::vital::config_block_sptr const& config)
  : sprokit::process(config)
{
  const auto key = kwiver::vital::config_block_key_t("key");

  declare_configuration_key(key, conf_info_t());
}


// ------------------------------------------------------------------
null_conf_info_process
::~null_conf_info_process()
{
}


// ------------------------------------------------------------------
remove_ports_process
::remove_ports_process(port_type_t const& port_type)
  : sprokit::process(kwiver::vital::config_block::empty_config())
{
  create_input_port(input_port, port_type);
  create_output_port(output_port, port_type);
}


// ------------------------------------------------------------------
remove_ports_process
::~remove_ports_process()
{
}


// ------------------------------------------------------------------
void
remove_ports_process
::create_input_port(port_t const& port, port_type_t const& port_type)
{
  declare_input_port(
    port,
    port_type,
    port_flags_t(),
    port_description_t("input port"));
}


// ------------------------------------------------------------------
void
remove_ports_process
::create_output_port(port_t const& port, port_type_t const& port_type)
{
  declare_output_port(
    port,
    port_type,
    port_flags_t(),
    port_description_t("output port"));
}


// ------------------------------------------------------------------
void
remove_ports_process
::_remove_input_port(port_t const& port)
{
  remove_input_port(port);
}


// ------------------------------------------------------------------
void
remove_ports_process
::_remove_output_port(port_t const& port)
{
  remove_output_port(port);
}
