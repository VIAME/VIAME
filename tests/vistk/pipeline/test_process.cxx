/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/process_registry.h>

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
  catch (std::exception& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static void test_null_input_edge();
static void test_null_output_edge();
static void test_connect_after_init();
static void test_reanalyze();
static void test_reinit();
static void test_step_before_analyze();
static void test_step_before_init();
static void test_set_static_input_type();
static void test_set_static_output_type();
static void test_set_input_type_duplicate();
static void test_set_output_type_duplicate();
static void test_set_input_type_after_init();
static void test_set_output_type_after_init();
static void test_set_tagged_flow_dependent_port();
static void test_set_tagged_flow_dependent_port_cascade();
static void test_add_input_port_after_type_pin();
static void test_add_output_port_after_type_pin();
static void test_set_untagged_flow_dependent_port();
static void test_null_config();
static void test_null_input_port_info();
static void test_null_output_port_info();
static void test_null_conf_info();

void
run_test(std::string const& test_name)
{
  if (test_name == "null_input_edge")
  {
    test_null_input_edge();
  }
  else if (test_name == "null_output_edge")
  {
    test_null_output_edge();
  }
  else if (test_name == "connect_after_init")
  {
    test_connect_after_init();
  }
  else if (test_name == "reanalyze")
  {
    test_reanalyze();
  }
  else if (test_name == "reinit")
  {
    test_reinit();
  }
  else if (test_name == "step_before_analyze")
  {
    test_step_before_analyze();
  }
  else if (test_name == "step_before_init")
  {
    test_step_before_init();
  }
  else if (test_name == "set_static_input_type")
  {
    test_set_static_input_type();
  }
  else if (test_name == "set_static_output_type")
  {
    test_set_static_output_type();
  }
  else if (test_name == "set_input_type_duplicate")
  {
    test_set_input_type_duplicate();
  }
  else if (test_name == "set_output_type_duplicate")
  {
    test_set_output_type_duplicate();
  }
  else if (test_name == "set_input_type_after_init")
  {
    test_set_input_type_after_init();
  }
  else if (test_name == "set_output_type_after_init")
  {
    test_set_output_type_after_init();
  }
  else if (test_name == "set_tagged_flow_dependent_port")
  {
    test_set_tagged_flow_dependent_port();
  }
  else if (test_name == "set_tagged_flow_dependent_port_cascade")
  {
    test_set_tagged_flow_dependent_port_cascade();
  }
  else if (test_name == "add_input_port_after_type_pin")
  {
    test_add_input_port_after_type_pin();
  }
  else if (test_name == "add_output_port_after_type_pin")
  {
    test_add_output_port_after_type_pin();
  }
  else if (test_name == "set_untagged_flow_dependent_port")
  {
    test_set_untagged_flow_dependent_port();
  }
  else if (test_name == "null_config")
  {
    test_null_config();
  }
  else if (test_name == "null_input_port_info")
  {
    test_null_input_port_info();
  }
  else if (test_name == "null_output_port_info")
  {
    test_null_output_port_info();
  }
  else if (test_name == "null_conf_info")
  {
    test_null_conf_info();
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
  }
}

static vistk::process_t create_process(vistk::process_registry::type_t const& type, vistk::process::name_t const& name = vistk::process::name_t());
static vistk::edge_t create_edge();
static vistk::process_t create_null_config_process(vistk::config_t const& config);
static vistk::process_t create_null_input_port_info_process(vistk::config_t const& config);
static vistk::process_t create_null_output_port_info_process(vistk::config_t const& config);
static vistk::process_t create_null_conf_info_process(vistk::config_t const& config);

void
test_null_input_edge()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process_t const process = create_process(proc_type);

  vistk::edge_t const edge;

  EXPECT_EXCEPTION(vistk::null_edge_port_connection_exception,
                   process->connect_input_port(vistk::process::port_t(), edge),
                   "connecting a NULL edge to an input port");
}

void
test_null_output_edge()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process_t const process = create_process(proc_type);

  vistk::edge_t const edge;

  EXPECT_EXCEPTION(vistk::null_edge_port_connection_exception,
                   process->connect_output_port(vistk::process::port_t(), edge),
                   "connecting a NULL edge to an output port");
}

void
test_connect_after_init()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process_t const process = create_process(proc_type);

  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  process->analyze();
  process->init();

  EXPECT_EXCEPTION(vistk::connect_to_initialized_process_exception,
                   process->connect_input_port(vistk::process::port_t(), edge),
                   "connecting an input edge after initialization");
}

void
test_reanalyze()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process_t const process = create_process(proc_type);

  process->analyze();

  EXPECT_EXCEPTION(vistk::reanalyzed_exception,
                   process->analyze(),
                   "reanalyzing a process");
}

void
test_reinit()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process_t const process = create_process(proc_type);

  process->analyze();
  process->init();

  EXPECT_EXCEPTION(vistk::reinitialization_exception,
                   process->init(),
                   "reinitializing a process");
}

void
test_step_before_analyze()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process_t const process = create_process(proc_type);

  EXPECT_EXCEPTION(vistk::unanalyzed_exception,
                   process->step(),
                   "stepping before analysis");
}

void
test_step_before_init()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process_t const process = create_process(proc_type);

  process->analyze();

  EXPECT_EXCEPTION(vistk::uninitialized_exception,
                   process->step(),
                   "stepping before initialization");
}

void
test_set_static_input_type()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("multiplication");

  vistk::process::port_t const port_name = vistk::process::port_t("factor1");
  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const process = create_process(proc_type);

  EXPECT_EXCEPTION(vistk::static_type_reset_exception,
                   process->set_input_port_type(port_name, port_type),
                   "setting the type of a non-dependent input port");
}

void
test_set_static_output_type()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("multiplication");

  vistk::process::port_t const port_name = vistk::process::port_t("product");
  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const process = create_process(proc_type);

  EXPECT_EXCEPTION(vistk::static_type_reset_exception,
                   process->set_output_port_type(port_name, port_type),
                   "setting the type of a non-dependent output port");
}

void
test_set_input_type_duplicate()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("flow_dependent");

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

void
test_set_output_type_duplicate()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("flow_dependent");

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

void
test_set_input_type_after_init()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("flow_dependent");

  vistk::process::port_t const port_name = vistk::process::port_t("input");
  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const process = create_process(proc_type);

  process->analyze();
  process->init();

  EXPECT_EXCEPTION(vistk::set_type_on_initialized_process_exception,
                   process->set_input_port_type(port_name, port_type),
                   "setting an input port type after initialization");
}

void
test_set_output_type_after_init()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("flow_dependent");

  vistk::process::port_t const port_name = vistk::process::port_t("output");
  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const process = create_process(proc_type);

  process->analyze();
  process->init();

  EXPECT_EXCEPTION(vistk::set_type_on_initialized_process_exception,
                   process->set_output_port_type(port_name, port_type),
                   "setting an output port type after initialization");
}

void
test_set_tagged_flow_dependent_port()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("tagged_flow_dependent");

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

void
test_set_tagged_flow_dependent_port_cascade()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("tagged_flow_dependent");

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

void
test_add_input_port_after_type_pin()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("collate");

  vistk::process::port_t const color = vistk::process::port_t("color/");
  vistk::process::port_t const res = vistk::process::port_t("res/");
  vistk::process::port_t const coll = vistk::process::port_t("coll/");

  vistk::process::port_t const tag = vistk::process::port_t("test");
  vistk::process::port_t const group = vistk::process::port_t("/group");

  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const process = create_process(proc_type);
  vistk::edge_t const edge = create_edge();

  (void)process->input_port_info(color + tag);

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

void
test_add_output_port_after_type_pin()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("distribute");

  vistk::process::port_t const color = vistk::process::port_t("color/");
  vistk::process::port_t const src = vistk::process::port_t("src/");
  vistk::process::port_t const dist = vistk::process::port_t("dist/");

  vistk::process::port_t const tag = vistk::process::port_t("test");
  vistk::process::port_t const group = vistk::process::port_t("/group");

  vistk::process::port_type_t const port_type = vistk::process::port_type_t("type");

  vistk::process_t const process = create_process(proc_type);
  vistk::edge_t const edge = create_edge();

  (void)process->output_port_info(color + tag);

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

void
test_set_untagged_flow_dependent_port()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("tagged_flow_dependent");

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

void
test_null_config()
{
  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("null_config");

  reg->register_process(proc_type, vistk::process_registry::description_t(), create_null_config_process);

  vistk::process::name_t const proc_name = vistk::process::name_t(proc_type);

  EXPECT_EXCEPTION(vistk::null_process_config_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as the configuration for a process");
}

void
test_null_input_port_info()
{
  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("null_input_port");

  reg->register_process(proc_type, vistk::process_registry::description_t(), create_null_input_port_info_process);

  vistk::process::name_t const proc_name = vistk::process::name_t(proc_type);

  EXPECT_EXCEPTION(vistk::null_input_port_info_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as an input port info structure");
}

void
test_null_output_port_info()
{
  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("null_output_port");

  reg->register_process(proc_type, vistk::process_registry::description_t(), create_null_output_port_info_process);

  vistk::process::name_t const proc_name = vistk::process::name_t(proc_type);

  EXPECT_EXCEPTION(vistk::null_output_port_info_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as an output port info structure");
}

void
test_null_conf_info()
{
  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("null_conf");

  reg->register_process(proc_type, vistk::process_registry::description_t(), create_null_conf_info_process);

  vistk::process::name_t const proc_name = vistk::process::name_t(proc_type);

  EXPECT_EXCEPTION(vistk::null_conf_info_exception,
                   create_process(proc_type, proc_name),
                   "passing NULL as an configuration info structure");
}

vistk::process_t
create_process(vistk::process_registry::type_t const& type, vistk::process::name_t const& name)
{
  static bool const modules_loaded = (vistk::load_known_modules(), true);
  static vistk::process_registry_t const reg = vistk::process_registry::self();

  (void)modules_loaded;

  vistk::config_t config = vistk::config::empty_config();

  config->set_value(vistk::process::config_name, vistk::config::value_t(name));

  return reg->create_process(type, config);
}

vistk::edge_t
create_edge()
{
  vistk::config_t const config = vistk::config::empty_config();
  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  return edge;
}

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

vistk::process_t
create_null_config_process(vistk::config_t const& config)
{
  return boost::make_shared<null_config_process>(config);
}

vistk::process_t
create_null_input_port_info_process(vistk::config_t const& config)
{
  return boost::make_shared<null_input_info_process>(config);
}

vistk::process_t
create_null_output_port_info_process(vistk::config_t const& config)
{
  return boost::make_shared<null_output_info_process>(config);
}

vistk::process_t
create_null_conf_info_process(vistk::config_t const& config)
{
  return boost::make_shared<null_conf_info_process>(config);
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
