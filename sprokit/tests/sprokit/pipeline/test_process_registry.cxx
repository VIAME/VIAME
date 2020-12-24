// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_common.h>

#include <vital/config/config_block.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <sprokit/pipeline/process_cluster.h>
#include <sprokit/pipeline/process_factory.h>
#include <sprokit/pipeline/process_registry_exception.h>
#include <sprokit/pipeline/types.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(null_config)
{

  kwiver::vital::config_block_sptr const config;

  EXPECT_EXCEPTION(sprokit::null_process_registry_config_exception,
                   sprokit::create_process(sprokit::process::type_t(), sprokit::process::name_t(), config),
                   "requesting a NULL config to a process");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(load_processes)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  auto factories =  kwiver::vital::plugin_manager::instance().get_factories<sprokit::process>();

  for( auto fact : factories )
  {
    sprokit::process::type_t type; // process name
    if ( ! fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, type ) )
    {
      TEST_ERROR( "Process factory does not have process name attribute" );
      continue;
    }

    sprokit::process_t process;

    try
    {
      process = sprokit::create_process(type, sprokit::process::name_t());
    }
    catch (sprokit::no_such_process_type_exception const& e)
    {
      TEST_ERROR("Failed to create process: " << e.what());
      continue;
    }
    catch (std::exception const& e)
    {
      TEST_ERROR("Unexpected exception when creating process: " << e.what());
      continue;
    }

    if (!process)
    {
      TEST_ERROR("Received NULL process (" << type << ")");

      continue;
    }

    sprokit::process::description_t descrip;
    if ( ! fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descrip ) || descrip.empty() )
    {
      TEST_ERROR("The description for " << type << " is empty");
    }
  } // end foreach
}

// ------------------------------------------------------------------
class null_process
  : public sprokit::process
{
public:
  null_process(kwiver::vital::config_block_sptr const& config)
    : process( config )
  { }

  virtual ~null_process() {}
};

// ------------------------------------------------------------------
IMPLEMENT_TEST(duplicate_types)
{
  sprokit::process::type_t const non_existent_process = sprokit::process::type_t("no_such_process");

  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  vpm.ADD_PROCESS( null_process );

  EXPECT_EXCEPTION(kwiver::vital::plugin_already_exists,
                   vpm.ADD_PROCESS( null_process ),
                   "adding duplicate process type");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(unknown_types)
{
  sprokit::process::type_t const non_existent_process = sprokit::process::type_t("no_such_process");

  EXPECT_EXCEPTION(sprokit::no_such_process_type_exception,
                   sprokit::create_process(non_existent_process, sprokit::process::name_t()),
                   "requesting an non-existent process type");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(register_cluster)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  sprokit::process::type_t const cluster_type = sprokit::process::type_t("orphan_cluster");
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::process_t const cluster_from_reg = sprokit::create_process(cluster_type, sprokit::process::name_t(), config);

  sprokit::process_cluster_t const cluster = std::dynamic_pointer_cast<sprokit::process_cluster>(cluster_from_reg);

  if (!cluster)
  {
    TEST_ERROR("Failed to turn a process back into a cluster");
  }

  sprokit::process::type_t const type = sprokit::process::type_t("orphan");

  sprokit::process_t const not_a_cluster_from_reg = sprokit::create_process(type, sprokit::process::name_t(), config);

  sprokit::process_cluster_t const not_a_cluster = std::dynamic_pointer_cast<sprokit::process_cluster>(not_a_cluster_from_reg);

  if (not_a_cluster)
  {
    TEST_ERROR("Turned a non-cluster into a cluster");
  }
}
