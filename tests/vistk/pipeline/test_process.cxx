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
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/process_registry.h>

#include <boost/make_shared.hpp>

#define TEST_ARGS ()

DECLARE_TEST(null_input_edge);
DECLARE_TEST(null_output_edge);
DECLARE_TEST(connect_after_init);
DECLARE_TEST(configure_twice);
DECLARE_TEST(reinit);
DECLARE_TEST(reset);
DECLARE_TEST(step_before_configure);
DECLARE_TEST(step_before_init);
DECLARE_TEST(set_static_input_type);
DECLARE_TEST(set_static_output_type);
DECLARE_TEST(set_input_type_duplicate);
DECLARE_TEST(set_output_type_duplicate);
DECLARE_TEST(set_input_type_after_init);
DECLARE_TEST(set_output_type_after_init);
DECLARE_TEST(set_tagged_flow_dependent_port);
DECLARE_TEST(set_tagged_flow_dependent_port_cascade);
DECLARE_TEST(set_tagged_flow_dependent_port_cascade_any);
DECLARE_TEST(add_input_port_after_type_pin);
DECLARE_TEST(add_output_port_after_type_pin);
DECLARE_TEST(set_untagged_flow_dependent_port);
DECLARE_TEST(remove_input_port);
DECLARE_TEST(remove_output_port);
DECLARE_TEST(remove_non_exist_input_port);
DECLARE_TEST(remove_non_exist_output_port);
DECLARE_TEST(remove_only_tagged_flow_dependent_port);
DECLARE_TEST(remove_tagged_flow_dependent_port);
DECLARE_TEST(null_config);
DECLARE_TEST(null_input_port_info);
DECLARE_TEST(null_output_port_info);
DECLARE_TEST(null_conf_info);

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, null_input_edge);
  ADD_TEST(tests, null_output_edge);
  ADD_TEST(tests, connect_after_init);
  ADD_TEST(tests, configure_twice);
  ADD_TEST(tests, reinit);
  ADD_TEST(tests, reset);
  ADD_TEST(tests, step_before_configure);
  ADD_TEST(tests, step_before_init);
  ADD_TEST(tests, set_static_input_type);
  ADD_TEST(tests, set_static_output_type);
  ADD_TEST(tests, set_input_type_duplicate);
  ADD_TEST(tests, set_output_type_duplicate);
  ADD_TEST(tests, set_input_type_after_init);
  ADD_TEST(tests, set_output_type_after_init);
  ADD_TEST(tests, set_tagged_flow_dependent_port);
  ADD_TEST(tests, set_tagged_flow_dependent_port_cascade);
  ADD_TEST(tests, set_tagged_flow_dependent_port_cascade_any);
  ADD_TEST(tests, add_input_port_after_type_pin);
  ADD_TEST(tests, add_output_port_after_type_pin);
  ADD_TEST(tests, set_untagged_flow_dependent_port);
  ADD_TEST(tests, remove_input_port);
  ADD_TEST(tests, remove_output_port);
  ADD_TEST(tests, remove_non_exist_input_port);
  ADD_TEST(tests, remove_non_exist_output_port);
  ADD_TEST(tests, remove_only_tagged_flow_dependent_port);
  ADD_TEST(tests, remove_tagged_flow_dependent_port);
  ADD_TEST(tests, null_config);
  ADD_TEST(tests, null_input_port_info);
  ADD_TEST(tests, null_output_port_info);
  ADD_TEST(tests, null_conf_info);

  RUN_TEST(tests, testname);
}

static vistk::process_t create_process(vistk::process::type_t const& type, vistk::process::name_t const& name = vistk::process::name_t());
static vistk::edge_t create_edge();

class remove_ports_process
  : public vistk::process
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

class null_config_process
  : public vistk::process
{
  public:
    null_config_process(vistk::config_t const& config);
    ~null_config_process();
};

class null_input_info_process
  : public vistk::process
{
  public:
    null_input_info_process(vistk::config_t const& config);
    ~null_input_info_process();
};

class null_output_info_process
  : public vistk::process
{
  public:
    null_output_info_process(vistk::config_t const& config);
    ~null_output_info_process();
};

class null_conf_info_process
  : public vistk::process
{
  public:
    null_conf_info_process(vistk::config_t const& config);
    ~null_conf_info_process();
};

IMPLEMENT_TEST(null_input_edge)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("orphan");

  vistk::process_t const process = create_process(proc_type);

  vistk::edge_t const edge;

  EXPECT_EXCEPTION(vistk::null_edge_port_connection_exception,
                   process->connect_input_port(vistk::process::port_t(), edge),
                   "connecting a NULL edge to an input port");
}

IMPLEMENT_TEST(null_output_edge)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("orphan");

  vistk::process_t const process = create_process(proc_type);

  vistk::edge_t const edge;

  EXPECT_EXCEPTION(vistk::null_edge_port_connection_exception,
                   process->connect_output_port(vistk::process::port_t(), edge),
                   "connecting a NULL edge to an output port");
}

IMPLEMENT_TEST(connect_after_init)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("orphan");

  vistk::process_t const process = create_process(proc_type);

  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  vistk::pipeline_t const pipe = boost::make_shared<vistk::pipeline>(config);

  // Only the pipeline can properly initialize a process.
  pipe->add_process(process);
  pipe->setup_pipeline();

  EXPECT_EXCEPTION(vistk::connect_to_initialized_process_exception,
                   process->connect_input_port(vistk::process::port_t(), edge),
                   "connecting an input edge after initialization");
}

IMPLEMENT_TEST(configure_twice)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("orphan");

  vistk::process_t const process = create_process(proc_type);

  process->configure();

  EXPECT_EXCEPTION(vistk::reconfigured_exception,
                   process->configure(),
                   "reconfiguring a process");
}

IMPLEMENT_TEST(reinit)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("orphan");

  vistk::process_t const process = create_process(proc_type);

  process->configure();
  process->init();

  EXPECT_EXCEPTION(vistk::reinitialization_exception,
                   process->init(),
                   "reinitializing a process");
}

IMPLEMENT_TEST(reset)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("orphan");

  vistk::process_t const process = create_process(proc_type);

  process->configure();

  process->reset();

  process->configure();
  process->init();

  process->reset();

  process->configure();
  process->init();
}

IMPLEMENT_TEST(step_before_configure)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("orphan");

  vistk::process_t const process = create_process(proc_type);

  EXPECT_EXCEPTION(vistk::unconfigured_exception,
                   process->step(),
                   "stepping before configuring");
}

IMPLEMENT_TEST(step_before_init)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("orphan");

  vistk::process_t const process = create_process(proc_type);

  process->configure();

  EXPECT_EXCEPTION(vistk::uninitialized_exception,
                   process->step(),
                   "stepping before initialization");
}

IMPLEMENT_TEST(set_static_input_type)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("multiplication");

  vistk::process::port_t const port_name = vistk::process::port_t("factor1");
  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const process = create_process(proc_type);

  EXPECT_EXCEPTION(vistk::static_type_reset_exception,
                   process->set_input_port_type(port_name, port_type),
                   "setting the type of a non-dependent input port");
}

IMPLEMENT_TEST(set_static_output_type)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("multiplication");

  vistk::process::port_t const port_name = vistk::process::port_t("product");
  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const process = create_process(proc_type);

  EXPECT_EXCEPTION(vistk::static_type_reset_exception,
                   process->set_output_port_type(port_name, port_type),
                   "setting the type of a non-dependent output port");
}

IMPLEMENT_TEST(set_input_type_duplicate)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("flow_dependent");

  vistk::process::port_t const port_name = vistk::process::port_t("input");
  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const process = create_process(proc_type);

  process->set_input_port_type(port_name, port_type);

  vistk::process::port_info_t const port_info_before = process->input_port_info(port_name);

  process->set_input_port_type(port_name, port_type);

  vistk::process::port_info_t const port_info_duplicate = process->input_port_info(port_name);

  // If nothing actually changes, info pointers should still be valid.
  if (port_info_before != port_info_duplicate)
  {
    TEST_ERROR("Input port information changed without a data change");
  }
}

IMPLEMENT_TEST(set_output_type_duplicate)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("flow_dependent");

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const process = create_process(proc_type);

  process->set_output_port_type(port_name, port_type);

  vistk::process::port_info_t const port_info_before = process->output_port_info(port_name);

  process->set_output_port_type(port_name, port_type);

  vistk::process::port_info_t const port_info_duplicate = process->output_port_info(port_name);

  // If nothing actually changes, info pointers should still be valid.
  if (port_info_before != port_info_duplicate)
  {
    TEST_ERROR("Output port information changed without a data change");
  }
}

IMPLEMENT_TEST(set_input_type_after_init)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("flow_dependent");

  vistk::process::port_t const port_name = vistk::process::port_t("input");
  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const process = create_process(proc_type);

  process->configure();
  process->init();

  EXPECT_EXCEPTION(vistk::set_type_on_initialized_process_exception,
                   process->set_input_port_type(port_name, port_type),
                   "setting an input port type after initialization");
}

IMPLEMENT_TEST(set_output_type_after_init)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("flow_dependent");

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const process = create_process(proc_type);

  process->configure();
  process->init();

  EXPECT_EXCEPTION(vistk::set_type_on_initialized_process_exception,
                   process->set_output_port_type(port_name, port_type),
                   "setting an output port type after initialization");
}

IMPLEMENT_TEST(set_tagged_flow_dependent_port)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("tagged_flow_dependent");

  vistk::process::port_t const iport_name = vistk::process::port_t("tagged_input");
  vistk::process::port_t const oport_name = vistk::process::port_t("tagged_output");

  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const input_process = create_process(proc_type);

  if (!input_process->set_input_port_type(iport_name, port_type))
  {
    TEST_ERROR("Could not set the input port type");
  }

  vistk::process::port_info_t const iiinfo = input_process->input_port_info(iport_name);
  vistk::process::port_info_t const ioinfo = input_process->output_port_info(oport_name);

  if (iiinfo->type != port_type)
  {
    TEST_ERROR("Setting the input port type is not reflected in port info");
  }

  if (ioinfo->type != port_type)
  {
    TEST_ERROR("Setting the input port type did not also set the output port info");
  }

  vistk::process_t const output_process = create_process(proc_type);

  if (!output_process->set_output_port_type(oport_name, port_type))
  {
    TEST_ERROR("Could not set the output port type");
  }

  vistk::process::port_info_t const oiinfo = output_process->input_port_info(iport_name);
  vistk::process::port_info_t const ooinfo = output_process->output_port_info(oport_name);

  if (ooinfo->type != port_type)
  {
    TEST_ERROR("Setting the output port type is not reflected in port info");
  }

  if (oiinfo->type != port_type)
  {
    TEST_ERROR("Setting the output port type did not also set the input port info");
  }
}

IMPLEMENT_TEST(set_tagged_flow_dependent_port_cascade)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("tagged_flow_dependent");

  vistk::process::port_t const iport_name = vistk::process::port_t("tagged_input");
  vistk::process::port_t const oport_name = vistk::process::port_t("tagged_output");

  vistk::process::port_type_t const tag_port_type = vistk::process::type_flow_dependent + vistk::process::port_type_t("other_tag");
  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const input_process = create_process(proc_type);

  if (!input_process->set_input_port_type(iport_name, tag_port_type))
  {
    TEST_ERROR("Could not set the input port type");
  }

  if (!input_process->set_output_port_type(oport_name, port_type))
  {
    TEST_ERROR("Could not set the output port type");
  }

  vistk::process::port_info_t const iinfo = input_process->input_port_info(iport_name);
  vistk::process::port_info_t const oinfo = input_process->output_port_info(oport_name);

  if (iinfo->type != port_type)
  {
    TEST_ERROR("Setting the input port type is not reflected in port info");
  }

  if (oinfo->type != port_type)
  {
    TEST_ERROR("Setting the input port type did not also set the output port info");
  }
}

IMPLEMENT_TEST(set_tagged_flow_dependent_port_cascade_any)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("tagged_flow_dependent");

  vistk::process::port_t const iport_name = vistk::process::port_t("tagged_input");
  vistk::process::port_t const oport_name = vistk::process::port_t("tagged_output");

  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const input_process = create_process(proc_type);

  if (!input_process->set_input_port_type(iport_name, vistk::process::type_any))
  {
    TEST_ERROR("Could not set the input port type");
  }

  if (!input_process->set_input_port_type(iport_name, port_type))
  {
    TEST_ERROR("Could not set the input port type after setting it to 'any'");
  }

  vistk::process::port_info_t const info = input_process->input_port_info(iport_name);

  if (info->type != port_type)
  {
    TEST_ERROR("Setting the port type on a flow port set to 'any' did not apply");
  }
}

IMPLEMENT_TEST(add_input_port_after_type_pin)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("collate");

  vistk::process::port_t const status = vistk::process::port_t("status/");
  vistk::process::port_t const res = vistk::process::port_t("res/");
  vistk::process::port_t const coll = vistk::process::port_t("coll/");

  vistk::process::port_t const tag = vistk::process::port_t("test");
  vistk::process::port_t const group = vistk::process::port_t("/group");

  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const process = create_process(proc_type);
  vistk::edge_t const edge = create_edge();

  (void)process->input_port_info(status + tag);

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   process->connect_input_port(coll + tag + group, edge),
                   "making sure the input port does not exist");

  if (!process->set_output_port_type(res + tag, port_type))
  {
    TEST_ERROR("Could not set the source port type");
  }

  vistk::process::port_info_t const info = process->input_port_info(coll + tag + group);

  if (info->type != port_type)
  {
    TEST_ERROR("The port with an input tagged dependency port was not set to the new type automatically upon creation");
  }
}

IMPLEMENT_TEST(add_output_port_after_type_pin)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("distribute");

  vistk::process::port_t const status = vistk::process::port_t("status/");
  vistk::process::port_t const src = vistk::process::port_t("src/");
  vistk::process::port_t const dist = vistk::process::port_t("dist/");

  vistk::process::port_t const tag = vistk::process::port_t("test");
  vistk::process::port_t const group = vistk::process::port_t("/group");

  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const process = create_process(proc_type);
  vistk::edge_t const edge = create_edge();

  (void)process->output_port_info(status + tag);

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   process->connect_output_port(dist + tag + group, edge),
                   "making sure the output port does not exist");

  if (!process->set_input_port_type(src + tag, port_type))
  {
    TEST_ERROR("Could not set the source port type");
  }

  vistk::process::port_info_t const info = process->output_port_info(dist + tag + group);

  if (info->type != port_type)
  {
    TEST_ERROR("The port with an output tagged dependency port was not set to the new type automatically upon creation");
  }
}

IMPLEMENT_TEST(set_untagged_flow_dependent_port)
{
  vistk::process::type_t const proc_type = vistk::process::type_t("tagged_flow_dependent");

  vistk::process::port_t const iport_name = vistk::process::port_t("untagged_input");
  vistk::process::port_t const oport_name = vistk::process::port_t("untagged_output");

  vistk::process::port_type_t const iport_type = vistk::process::port_type_t("itype");
  vistk::process::port_type_t const oport_type = vistk::process::port_type_t("otype");

  vistk::process_t const process = create_process(proc_type);

  if (!process->set_input_port_type(iport_name, iport_type))
  {
    TEST_ERROR("Could not set the input port type");
  }

  vistk::process::port_info_t const iiinfo = process->input_port_info(iport_name);
  vistk::process::port_info_t const ioinfo = process->output_port_info(oport_name);

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

  vistk::process::port_info_t const oiinfo = process->input_port_info(iport_name);
  vistk::process::port_info_t const ooinfo = process->output_port_info(oport_name);

  if (ooinfo->type != oport_type)
  {
    TEST_ERROR("Setting the output port type is not reflected in port info");
  }

  if (oiinfo->type == oport_type)
  {
    TEST_ERROR("Setting the output port type did not also set the input port info");
  }
}

vistk::process::port_type_t const remove_ports_process::input_port = vistk::process::port_type_t("input");
vistk::process::port_type_t const remove_ports_process::output_port = vistk::process::port_type_t("output");

IMPLEMENT_TEST(remove_input_port)
{
  vistk::process::port_type_t const type = vistk::process::port_type_t("type");
  boost::scoped_ptr<remove_ports_process> proc(new remove_ports_process(type));

  proc->_remove_input_port(remove_ports_process::input_port);

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   proc->input_port_info(remove_ports_process::input_port),
                   "after removing an input port");
}

IMPLEMENT_TEST(remove_output_port)
{
  vistk::process::port_type_t const type = vistk::process::port_type_t("type");
  boost::scoped_ptr<remove_ports_process> proc(new remove_ports_process(type));

  proc->_remove_output_port(remove_ports_process::output_port);

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   proc->output_port_info(remove_ports_process::output_port),
                   "after removing an output port");
}

IMPLEMENT_TEST(remove_non_exist_input_port)
{
  vistk::process::port_t const port = vistk::process::port_t("port");
  vistk::process::port_type_t const type = vistk::process::port_type_t("type");
  boost::scoped_ptr<remove_ports_process> proc(new remove_ports_process(type));

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   proc->_remove_input_port(port),
                   "after removing a non-existent input port");
}

IMPLEMENT_TEST(remove_non_exist_output_port)
{
  vistk::process::port_t const port = vistk::process::port_t("port");
  vistk::process::port_type_t const type = vistk::process::port_type_t("type");
  boost::scoped_ptr<remove_ports_process> proc(new remove_ports_process(type));

  EXPECT_EXCEPTION(vistk::no_such_port_exception,
                   proc->_remove_output_port(port),
                   "after removing a non-existent output port");
}

IMPLEMENT_TEST(remove_only_tagged_flow_dependent_port)
{
  vistk::process::port_t const port = vistk::process::port_t("port");
  vistk::process::port_type_t const flow_type = vistk::process::type_flow_dependent + vistk::process::port_type_t("tag");
  vistk::process::port_type_t const type = vistk::process::port_type_t("type");
  boost::scoped_ptr<remove_ports_process> proc(new remove_ports_process(flow_type));

  proc->_remove_input_port(remove_ports_process::input_port);
  proc->set_output_port_type(remove_ports_process::output_port, type);
  proc->_remove_output_port(remove_ports_process::output_port);
  proc->create_input_port(port, flow_type);

  vistk::process::port_info_t const info = proc->input_port_info(port);
  vistk::process::port_type_t const& new_type = info->type;

  if (new_type != flow_type)
  {
    TEST_ERROR("The flow type was not reset after all tagged ports were removed.");
  }
}

IMPLEMENT_TEST(remove_tagged_flow_dependent_port)
{
  vistk::process::port_type_t const flow_type = vistk::process::type_flow_dependent + vistk::process::port_type_t("tag");
  vistk::process::port_type_t const type = vistk::process::port_type_t("type");
  boost::scoped_ptr<remove_ports_process> proc(new remove_ports_process(flow_type));

  proc->set_output_port_type(remove_ports_process::output_port, type);
  proc->_remove_output_port(remove_ports_process::output_port);

  vistk::process::port_info_t const info = proc->input_port_info(remove_ports_process::input_port);
  vistk::process::port_type_t const& new_type = info->type;

  if (new_type != type)
  {
    TEST_ERROR("The flow type was reset even when more were left.");
  }
}

IMPLEMENT_TEST(null_config)
{
  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::process::type_t const proc_type = vistk::process::type_t("null_config");

  reg->register_process(proc_type, vistk::process_registry::description_t(), vistk::create_process<null_config_process>);

  vistk::process::name_t const proc_name = vistk::process::name_t(proc_type);

  EXPECT_EXCEPTION(vistk::null_process_config_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as the configuration for a process");
}

IMPLEMENT_TEST(null_input_port_info)
{
  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::process::type_t const proc_type = vistk::process::type_t("null_input_port");

  reg->register_process(proc_type, vistk::process_registry::description_t(), vistk::create_process<null_input_info_process>);

  vistk::process::name_t const proc_name = vistk::process::name_t(proc_type);

  EXPECT_EXCEPTION(vistk::null_input_port_info_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as an input port info structure");
}

IMPLEMENT_TEST(null_output_port_info)
{
  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::process::type_t const proc_type = vistk::process::type_t("null_output_port");

  reg->register_process(proc_type, vistk::process_registry::description_t(), vistk::create_process<null_output_info_process>);

  vistk::process::name_t const proc_name = vistk::process::name_t(proc_type);

  EXPECT_EXCEPTION(vistk::null_output_port_info_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as an output port info structure");
}

IMPLEMENT_TEST(null_conf_info)
{
  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::process::type_t const proc_type = vistk::process::type_t("null_conf");

  reg->register_process(proc_type, vistk::process_registry::description_t(), vistk::create_process<null_conf_info_process>);

  vistk::process::name_t const proc_name = vistk::process::name_t(proc_type);

  EXPECT_EXCEPTION(vistk::null_conf_info_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as an configuration info structure");
}

vistk::process_t
create_process(vistk::process::type_t const& type, vistk::process::name_t const& name)
{
  static bool const modules_loaded = (vistk::load_known_modules(), true);
  static vistk::process_registry_t const reg = vistk::process_registry::self();

  (void)modules_loaded;

  return reg->create_process(type, name);
}

vistk::edge_t
create_edge()
{
  vistk::config_t const config = vistk::config::empty_config();
  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  return edge;
}

null_config_process
::null_config_process(vistk::config_t const& /*config*/)
  : vistk::process(vistk::config_t())
{
}

null_config_process
::~null_config_process()
{
}

null_input_info_process
::null_input_info_process(vistk::config_t const& config)
  : vistk::process(config)
{
  name_t const port = name_t("port");

  declare_input_port(port, port_info_t());
}

null_input_info_process
::~null_input_info_process()
{
}

null_output_info_process
::null_output_info_process(vistk::config_t const& config)
  : vistk::process(config)
{
  name_t const port = name_t("port");

  declare_output_port(port, port_info_t());
}

null_output_info_process
::~null_output_info_process()
{
}

null_conf_info_process
::null_conf_info_process(vistk::config_t const& config)
  : vistk::process(config)
{
  vistk::config::key_t const key = vistk::config::key_t("key");

  declare_configuration_key(key, conf_info_t());
}

null_conf_info_process
::~null_conf_info_process()
{
}

remove_ports_process
::remove_ports_process(port_type_t const& port_type)
  : vistk::process(vistk::config::empty_config())
{
  create_input_port(input_port, port_type);
  create_output_port(output_port, port_type);
}

remove_ports_process
::~remove_ports_process()
{
}

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

void
remove_ports_process
::_remove_input_port(port_t const& port)
{
  remove_input_port(port);
}

void
remove_ports_process
::_remove_output_port(port_t const& port)
{
  remove_output_port(port);
}
