// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_common.h>

#include <vital/config/config_block.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_registry.h>
#include <sprokit/pipeline/scheduler_registry_exception.h>
#include <sprokit/pipeline/types.h>

int
main()
{
  sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

  try
  {
    sprokit::load_known_modules();
  }
  catch (sprokit::scheduler_type_already_exists_exception const& e)
  {
    TEST_ERROR("Duplicate scheduler names: " << e.what());
  }
  catch (sprokit::pipeline_exception const& e)
  {
    TEST_ERROR("Failed to load modules: " << e.what());
  }
  catch (std::exception const& e)
  {
    TEST_ERROR("Unexpected exception when loading modules: " << e.what());

    return EXIT_FAILURE;
  }

  sprokit::scheduler_registry::types_t const types = reg->types();

  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::pipeline_t const pipe = sprokit::pipeline_t(new sprokit::pipeline(config));

  for (sprokit::scheduler_registry::type_t const& type : types)
  {
    sprokit::scheduler_t scheduler;

    try
    {
      scheduler = reg->create_scheduler(type, config, pipe);
    }
    catch (sprokit::no_such_scheduler_type_exception const& e)
    {
      TEST_ERROR("Failed to create scheduler: " << e.what());

      continue;
    }
    catch (std::exception const& e)
    {
      TEST_ERROR("Unexpected exception when creating scheduler: " << e.what());
    }

    if (!scheduler)
    {
      TEST_ERROR("Received NULL scheduler (" << type << ")");

      continue;
    }

    if (reg->description(type).empty())
    {
      TEST_ERROR("The description is empty");
    }
  }

  // Check exceptions for unknown types.
  {
    sprokit::scheduler_registry::type_t const non_existent_scheduler = sprokit::scheduler_registry::type_t("no_such_scheduler");

    EXPECT_EXCEPTION(sprokit::no_such_scheduler_type_exception,
                     reg->create_scheduler(non_existent_scheduler, config, pipe),
                     "requesting an non-existent scheduler type");

    EXPECT_EXCEPTION(sprokit::no_such_scheduler_type_exception,
                     reg->description(non_existent_scheduler),
                     "requesting an non-existent scheduler type");
  }

  return EXIT_SUCCESS;
}
