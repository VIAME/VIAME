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
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_exception.h>
#include <sprokit/pipeline/process_registry.h>

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
DECLARE_TEST(tunable_config);
DECLARE_TEST(tunable_config_read_only);
DECLARE_TEST(reconfigure_tunable);
DECLARE_TEST(reconfigure_non_tunable);
DECLARE_TEST(reconfigure_extra_parameters);

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
  ADD_TEST(tests, tunable_config);
  ADD_TEST(tests, tunable_config_read_only);
  ADD_TEST(tests, reconfigure_tunable);
  ADD_TEST(tests, reconfigure_non_tunable);
  ADD_TEST(tests, reconfigure_extra_parameters);

  RUN_TEST(tests, testname);
}

static sprokit::process_t create_process(sprokit::process::type_t const& type, sprokit::process::name_t const& name = sprokit::process::name_t(), sprokit::config_t const& conf = sprokit::config::empty_config());
static sprokit::edge_t create_edge();

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

class null_config_process
  : public sprokit::process
{
  public:
    null_config_process(sprokit::config_t const& config);
    ~null_config_process();
};

class null_input_info_process
  : public sprokit::process
{
  public:
    null_input_info_process(sprokit::config_t const& config);
    ~null_input_info_process();
};

class null_output_info_process
  : public sprokit::process
{
  public:
    null_output_info_process(sprokit::config_t const& config);
    ~null_output_info_process();
};

class null_conf_info_process
  : public sprokit::process
{
  public:
    null_conf_info_process(sprokit::config_t const& config);
    ~null_conf_info_process();
};

IMPLEMENT_TEST(null_input_edge)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

  sprokit::edge_t const edge;

  EXPECT_EXCEPTION(sprokit::null_edge_port_connection_exception,
                   process->connect_input_port(sprokit::process::port_t(), edge),
                   "connecting a NULL edge to an input port");
}

IMPLEMENT_TEST(null_output_edge)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

  sprokit::edge_t const edge;

  EXPECT_EXCEPTION(sprokit::null_edge_port_connection_exception,
                   process->connect_output_port(sprokit::process::port_t(), edge),
                   "connecting a NULL edge to an output port");
}

IMPLEMENT_TEST(connect_after_init)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

  sprokit::config_t const config = sprokit::config::empty_config();

  sprokit::edge_t const edge = boost::make_shared<sprokit::edge>(config);

  sprokit::pipeline_t const pipe = boost::make_shared<sprokit::pipeline>(config);

  // Only the pipeline can properly initialize a process.
  pipe->add_process(process);
  pipe->setup_pipeline();

  EXPECT_EXCEPTION(sprokit::connect_to_initialized_process_exception,
                   process->connect_input_port(sprokit::process::port_t(), edge),
                   "connecting an input edge after initialization");
}

IMPLEMENT_TEST(configure_twice)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

  process->configure();

  EXPECT_EXCEPTION(sprokit::reconfigured_exception,
                   process->configure(),
                   "reconfiguring a process");
}

IMPLEMENT_TEST(reinit)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

  process->configure();
  process->init();

  EXPECT_EXCEPTION(sprokit::reinitialization_exception,
                   process->init(),
                   "reinitializing a process");
}

IMPLEMENT_TEST(reset)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

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
  sprokit::process::type_t const proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

  EXPECT_EXCEPTION(sprokit::unconfigured_exception,
                   process->step(),
                   "stepping before configuring");
}

IMPLEMENT_TEST(step_before_init)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("orphan");

  sprokit::process_t const process = create_process(proc_type);

  process->configure();

  EXPECT_EXCEPTION(sprokit::uninitialized_exception,
                   process->step(),
                   "stepping before initialization");
}

IMPLEMENT_TEST(set_static_input_type)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("multiplication");

  sprokit::process::port_t const port_name = sprokit::process::port_t("factor1");
  sprokit::process::port_type_t const port_type = sprokit::process::port_type_t("type");

  sprokit::process_t const process = create_process(proc_type);

  EXPECT_EXCEPTION(sprokit::static_type_reset_exception,
                   process->set_input_port_type(port_name, port_type),
                   "setting the type of a non-dependent input port");
}

IMPLEMENT_TEST(set_static_output_type)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("multiplication");

  sprokit::process::port_t const port_name = sprokit::process::port_t("product");
  sprokit::process::port_type_t const port_type = sprokit::process::port_type_t("type");

  sprokit::process_t const process = create_process(proc_type);

  EXPECT_EXCEPTION(sprokit::static_type_reset_exception,
                   process->set_output_port_type(port_name, port_type),
                   "setting the type of a non-dependent output port");
}

IMPLEMENT_TEST(set_input_type_duplicate)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("flow_dependent");

  sprokit::process::port_t const port_name = sprokit::process::port_t("input");
  sprokit::process::port_type_t const port_type = sprokit::process::port_type_t("type");

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

IMPLEMENT_TEST(set_output_type_duplicate)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("flow_dependent");

  sprokit::process::port_t const port_name = sprokit::process::port_t("output");
  sprokit::process::port_type_t const port_type = sprokit::process::port_type_t("type");

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

IMPLEMENT_TEST(set_input_type_after_init)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("flow_dependent");

  sprokit::process::port_t const port_name = sprokit::process::port_t("input");
  sprokit::process::port_type_t const port_type = sprokit::process::port_type_t("type");

  sprokit::process_t const process = create_process(proc_type);

  process->configure();
  process->init();

  EXPECT_EXCEPTION(sprokit::set_type_on_initialized_process_exception,
                   process->set_input_port_type(port_name, port_type),
                   "setting an input port type after initialization");
}

IMPLEMENT_TEST(set_output_type_after_init)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("flow_dependent");

  sprokit::process::port_t const port_name = sprokit::process::port_t("output");
  sprokit::process::port_type_t const port_type = sprokit::process::port_type_t("type");

  sprokit::process_t const process = create_process(proc_type);

  process->configure();
  process->init();

  EXPECT_EXCEPTION(sprokit::set_type_on_initialized_process_exception,
                   process->set_output_port_type(port_name, port_type),
                   "setting an output port type after initialization");
}

IMPLEMENT_TEST(set_tagged_flow_dependent_port)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("tagged_flow_dependent");

  sprokit::process::port_t const iport_name = sprokit::process::port_t("tagged_input");
  sprokit::process::port_t const oport_name = sprokit::process::port_t("tagged_output");

  sprokit::process::port_type_t const port_type = sprokit::process::port_type_t("type");

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

IMPLEMENT_TEST(set_tagged_flow_dependent_port_cascade)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("tagged_flow_dependent");

  sprokit::process::port_t const iport_name = sprokit::process::port_t("tagged_input");
  sprokit::process::port_t const oport_name = sprokit::process::port_t("tagged_output");

  sprokit::process::port_type_t const tag_port_type = sprokit::process::type_flow_dependent + sprokit::process::port_type_t("other_tag");
  sprokit::process::port_type_t const port_type = sprokit::process::port_type_t("type");

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

IMPLEMENT_TEST(set_tagged_flow_dependent_port_cascade_any)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("tagged_flow_dependent");

  sprokit::process::port_t const iport_name = sprokit::process::port_t("tagged_input");
  sprokit::process::port_t const oport_name = sprokit::process::port_t("tagged_output");

  sprokit::process::port_type_t const port_type = sprokit::process::port_type_t("type");

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

IMPLEMENT_TEST(add_input_port_after_type_pin)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("collate");

  sprokit::process::port_t const status = sprokit::process::port_t("status/");
  sprokit::process::port_t const res = sprokit::process::port_t("res/");
  sprokit::process::port_t const coll = sprokit::process::port_t("coll/");

  sprokit::process::port_t const tag = sprokit::process::port_t("test");
  sprokit::process::port_t const group = sprokit::process::port_t("/group");

  sprokit::process::port_type_t const port_type = sprokit::process::port_type_t("type");

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

IMPLEMENT_TEST(add_output_port_after_type_pin)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("distribute");

  sprokit::process::port_t const status = sprokit::process::port_t("status/");
  sprokit::process::port_t const src = sprokit::process::port_t("src/");
  sprokit::process::port_t const dist = sprokit::process::port_t("dist/");

  sprokit::process::port_t const tag = sprokit::process::port_t("test");
  sprokit::process::port_t const group = sprokit::process::port_t("/group");

  sprokit::process::port_type_t const port_type = sprokit::process::port_type_t("type");

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

IMPLEMENT_TEST(set_untagged_flow_dependent_port)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("tagged_flow_dependent");

  sprokit::process::port_t const iport_name = sprokit::process::port_t("untagged_input");
  sprokit::process::port_t const oport_name = sprokit::process::port_t("untagged_output");

  sprokit::process::port_type_t const iport_type = sprokit::process::port_type_t("itype");
  sprokit::process::port_type_t const oport_type = sprokit::process::port_type_t("otype");

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

  sprokit::process::port_info_t const oiinfo = process->input_port_info(iport_name);
  sprokit::process::port_info_t const ooinfo = process->output_port_info(oport_name);

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

IMPLEMENT_TEST(remove_input_port)
{
  sprokit::process::port_type_t const type = sprokit::process::port_type_t("type");
  boost::scoped_ptr<remove_ports_process> proc(new remove_ports_process(type));

  proc->_remove_input_port(remove_ports_process::input_port);

  EXPECT_EXCEPTION(sprokit::no_such_port_exception,
                   proc->input_port_info(remove_ports_process::input_port),
                   "after removing an input port");
}

IMPLEMENT_TEST(remove_output_port)
{
  sprokit::process::port_type_t const type = sprokit::process::port_type_t("type");
  boost::scoped_ptr<remove_ports_process> proc(new remove_ports_process(type));

  proc->_remove_output_port(remove_ports_process::output_port);

  EXPECT_EXCEPTION(sprokit::no_such_port_exception,
                   proc->output_port_info(remove_ports_process::output_port),
                   "after removing an output port");
}

IMPLEMENT_TEST(remove_non_exist_input_port)
{
  sprokit::process::port_t const port = sprokit::process::port_t("port");
  sprokit::process::port_type_t const type = sprokit::process::port_type_t("type");
  boost::scoped_ptr<remove_ports_process> proc(new remove_ports_process(type));

  EXPECT_EXCEPTION(sprokit::no_such_port_exception,
                   proc->_remove_input_port(port),
                   "after removing a non-existent input port");
}

IMPLEMENT_TEST(remove_non_exist_output_port)
{
  sprokit::process::port_t const port = sprokit::process::port_t("port");
  sprokit::process::port_type_t const type = sprokit::process::port_type_t("type");
  boost::scoped_ptr<remove_ports_process> proc(new remove_ports_process(type));

  EXPECT_EXCEPTION(sprokit::no_such_port_exception,
                   proc->_remove_output_port(port),
                   "after removing a non-existent output port");
}

IMPLEMENT_TEST(remove_only_tagged_flow_dependent_port)
{
  sprokit::process::port_t const port = sprokit::process::port_t("port");
  sprokit::process::port_type_t const flow_type = sprokit::process::type_flow_dependent + sprokit::process::port_type_t("tag");
  sprokit::process::port_type_t const type = sprokit::process::port_type_t("type");
  boost::scoped_ptr<remove_ports_process> proc(new remove_ports_process(flow_type));

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

IMPLEMENT_TEST(remove_tagged_flow_dependent_port)
{
  sprokit::process::port_type_t const flow_type = sprokit::process::type_flow_dependent + sprokit::process::port_type_t("tag");
  sprokit::process::port_type_t const type = sprokit::process::port_type_t("type");
  boost::scoped_ptr<remove_ports_process> proc(new remove_ports_process(flow_type));

  proc->set_output_port_type(remove_ports_process::output_port, type);
  proc->_remove_output_port(remove_ports_process::output_port);

  sprokit::process::port_info_t const info = proc->input_port_info(remove_ports_process::input_port);
  sprokit::process::port_type_t const& new_type = info->type;

  if (new_type != type)
  {
    TEST_ERROR("The flow type was reset even when more were left.");
  }
}

IMPLEMENT_TEST(null_config)
{
  sprokit::process_registry_t const reg = sprokit::process_registry::self();

  sprokit::process::type_t const proc_type = sprokit::process::type_t("null_config");

  reg->register_process(proc_type, sprokit::process_registry::description_t(), sprokit::create_process<null_config_process>);

  sprokit::process::name_t const proc_name = sprokit::process::name_t(proc_type);

  EXPECT_EXCEPTION(sprokit::null_process_config_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as the configuration for a process");
}

IMPLEMENT_TEST(null_input_port_info)
{
  sprokit::process_registry_t const reg = sprokit::process_registry::self();

  sprokit::process::type_t const proc_type = sprokit::process::type_t("null_input_port");

  reg->register_process(proc_type, sprokit::process_registry::description_t(), sprokit::create_process<null_input_info_process>);

  sprokit::process::name_t const proc_name = sprokit::process::name_t(proc_type);

  EXPECT_EXCEPTION(sprokit::null_input_port_info_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as an input port info structure");
}

IMPLEMENT_TEST(null_output_port_info)
{
  sprokit::process_registry_t const reg = sprokit::process_registry::self();

  sprokit::process::type_t const proc_type = sprokit::process::type_t("null_output_port");

  reg->register_process(proc_type, sprokit::process_registry::description_t(), sprokit::create_process<null_output_info_process>);

  sprokit::process::name_t const proc_name = sprokit::process::name_t(proc_type);

  EXPECT_EXCEPTION(sprokit::null_output_port_info_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as an output port info structure");
}

IMPLEMENT_TEST(null_conf_info)
{
  sprokit::process_registry_t const reg = sprokit::process_registry::self();

  sprokit::process::type_t const proc_type = sprokit::process::type_t("null_conf");

  reg->register_process(proc_type, sprokit::process_registry::description_t(), sprokit::create_process<null_conf_info_process>);

  sprokit::process::name_t const proc_name = sprokit::process::name_t(proc_type);

  EXPECT_EXCEPTION(sprokit::null_conf_info_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as an configuration info structure");
}

IMPLEMENT_TEST(tunable_config)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("tunable");

  sprokit::config::key_t const tunable_key = sprokit::config::key_t("tunable");

  sprokit::process_t const proc = create_process(proc_type);

  sprokit::config::keys_t const tunable = proc->available_tunable_config();

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

IMPLEMENT_TEST(tunable_config_read_only)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("tunable");
  sprokit::process::name_t const proc_name = sprokit::process::name_t(proc_type);

  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::config::key_t const tunable_key = sprokit::config::key_t("tunable");
  sprokit::config::value_t const tunable_value = sprokit::config::value_t("value");

  conf->set_value(tunable_key, tunable_value);
  conf->mark_read_only(tunable_key);

  sprokit::process_t const proc = create_process(proc_type, proc_name, conf);

  sprokit::config::keys_t const tunable = proc->available_tunable_config();

  if (!tunable.empty())
  {
    TEST_ERROR("Failed to exclude read-only parameters as tunable");
  }
}

IMPLEMENT_TEST(reconfigure_tunable)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("expect");
  sprokit::process::name_t const proc_name = sprokit::process::name_t("name");

  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::config::key_t const key_tunable = sprokit::config::key_t("tunable");
  sprokit::config::key_t const key_expect = sprokit::config::key_t("expect");

  sprokit::config::value_t const tunable_value = sprokit::config::value_t("old_value");
  sprokit::config::value_t const tuned_value = sprokit::config::value_t("new_value");

  conf->set_value(key_tunable, tunable_value);
  conf->set_value(key_expect, tuned_value);
  conf->set_value(sprokit::process::config_name, proc_name);

  sprokit::process_t const expect = create_process(proc_type, proc_name, conf);

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(expect);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  new_conf->set_value(proc_name + sprokit::config::block_sep + key_tunable, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(reconfigure_non_tunable)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("expect");
  sprokit::process::name_t const proc_name = sprokit::process::name_t("name");

  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::config::key_t const key_tunable = sprokit::config::key_t("tunable");
  sprokit::config::key_t const key_expect = sprokit::config::key_t("expect");

  sprokit::config::value_t const tunable_value = sprokit::config::value_t("old_value");

  conf->set_value(key_tunable, tunable_value);
  conf->mark_read_only(key_tunable);
  conf->set_value(key_expect, tunable_value);
  conf->set_value(sprokit::process::config_name, proc_name);

  sprokit::process_t const expect = create_process(proc_type, proc_name, conf);

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(expect);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  sprokit::config::value_t const tuned_value = sprokit::config::value_t("new_value");

  new_conf->set_value(proc_name + sprokit::config::block_sep + key_tunable, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(reconfigure_extra_parameters)
{
  sprokit::process::type_t const proc_type = sprokit::process::type_t("expect");
  sprokit::process::name_t const proc_name = sprokit::process::name_t("name");

  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::config::key_t const new_key = sprokit::config::key_t("new_key");

  sprokit::config::key_t const key_expect = sprokit::config::key_t("expect");
  sprokit::config::key_t const key_expect_key = sprokit::config::key_t("expect_key");

  conf->set_value(key_expect, new_key);
  conf->set_value(key_expect_key, "true");
  conf->set_value(sprokit::process::config_name, proc_name);

  sprokit::process_t const expect = create_process(proc_type, proc_name, conf);

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(expect);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  sprokit::config::value_t const tunable_value = sprokit::config::value_t("old_value");

  new_conf->set_value(proc_name + sprokit::config::block_sep + new_key, tunable_value);

  pipeline->reconfigure(new_conf);
}

sprokit::process_t
create_process(sprokit::process::type_t const& type, sprokit::process::name_t const& name, sprokit::config_t const& conf)
{
  static bool const modules_loaded = (sprokit::load_known_modules(), true);
  static sprokit::process_registry_t const reg = sprokit::process_registry::self();

  (void)modules_loaded;

  return reg->create_process(type, name, conf);
}

sprokit::edge_t
create_edge()
{
  sprokit::config_t const config = sprokit::config::empty_config();
  sprokit::edge_t const edge = boost::make_shared<sprokit::edge>(config);

  return edge;
}

null_config_process
::null_config_process(sprokit::config_t const& /*config*/)
  : sprokit::process(sprokit::config_t())
{
}

null_config_process
::~null_config_process()
{
}

null_input_info_process
::null_input_info_process(sprokit::config_t const& config)
  : sprokit::process(config)
{
  name_t const port = name_t("port");

  declare_input_port(port, port_info_t());
}

null_input_info_process
::~null_input_info_process()
{
}

null_output_info_process
::null_output_info_process(sprokit::config_t const& config)
  : sprokit::process(config)
{
  name_t const port = name_t("port");

  declare_output_port(port, port_info_t());
}

null_output_info_process
::~null_output_info_process()
{
}

null_conf_info_process
::null_conf_info_process(sprokit::config_t const& config)
  : sprokit::process(config)
{
  sprokit::config::key_t const key = sprokit::config::key_t("key");

  declare_configuration_key(key, conf_info_t());
}

null_conf_info_process
::~null_conf_info_process()
{
}

remove_ports_process
::remove_ports_process(port_type_t const& port_type)
  : sprokit::process(sprokit::config::empty_config())
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
