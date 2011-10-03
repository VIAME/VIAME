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

static void run_test(std::string const& test_name);

int
main(int argc, char* argv[])
{
  if (argc != 2)
  {
    TEST_ERROR("Expected one argument");

    return 1;
  }

  std::string const test_name = argv[1];

  try
  {
    run_test(test_name);
  }
  catch (std::exception& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return 1;
  }

  return 0;
}

static void test_null_input_edge();
static void test_null_output_edge();
static void test_connect_after_init();
static void test_reinit();
static void test_step_before_init();
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
  else if (test_name == "reinit")
  {
    test_reinit();
  }
  else if (test_name == "step_before_init")
  {
    test_step_before_init();
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

static vistk::process_t create_process(vistk::process_registry::type_t const& type, vistk::process::name_t const& name);
static vistk::process_t create_null_config_process(vistk::config_t const& config);
static vistk::process_t create_null_input_port_info_process(vistk::config_t const& config);
static vistk::process_t create_null_output_port_info_process(vistk::config_t const& config);
static vistk::process_t create_null_conf_info_process(vistk::config_t const& config);

void
test_null_input_edge()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process_t const process = create_process(proc_type, vistk::process::name_t());

  vistk::edge_t const edge;

  EXPECT_EXCEPTION(vistk::null_edge_port_connection_exception,
                   process->connect_input_port(vistk::process::port_t(), edge),
                   "connecting a NULL edge to an input port");
}

void
test_null_output_edge()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process_t const process = create_process(proc_type, vistk::process::name_t());

  vistk::edge_t const edge;

  EXPECT_EXCEPTION(vistk::null_edge_port_connection_exception,
                   process->connect_output_port(vistk::process::port_t(), edge),
                   "connecting a NULL edge to an output port");
}

void
test_connect_after_init()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process_t const process = create_process(proc_type, vistk::process::name_t());

  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  process->init();

  EXPECT_EXCEPTION(vistk::connect_to_initialized_process_exception,
                   process->connect_input_port(vistk::process::port_t(), edge),
                   "connecting an input edge after initialization");
}

void
test_reinit()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process_t const process = create_process(proc_type, vistk::process::name_t());

  process->init();

  EXPECT_EXCEPTION(vistk::reinitialization_exception,
                   process->init(),
                   "reinitializing a process");
}

void
test_step_before_init()
{
  vistk::process_registry::type_t const proc_type = vistk::process_registry::type_t("orphan");

  vistk::process_t const process = create_process(proc_type, vistk::process::name_t());

  EXPECT_EXCEPTION(vistk::uninitialized_exception,
                   process->step(),
                   "stepping before initialization");
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
