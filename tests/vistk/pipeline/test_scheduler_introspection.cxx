/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/scheduler.h>
#include <vistk/pipeline/scheduler_registry.h>
#include <vistk/pipeline/scheduler_registry_exception.h>
#include <vistk/pipeline/types.h>

#include <boost/foreach.hpp>

#include <exception>
#include <iostream>

#include <cstdlib>

int
main()
{
  vistk::scheduler_registry_t reg = vistk::scheduler_registry::self();

  try
  {
    vistk::load_known_modules();
  }
  catch (vistk::scheduler_type_already_exists_exception const& e)
  {
    TEST_ERROR("Duplicate scheduler names: " << e.what());
  }
  catch (vistk::pipeline_exception const& e)
  {
    TEST_ERROR("Failed to load modules: " << e.what());
  }
  catch (std::exception const& e)
  {
    TEST_ERROR("Unexpected exception when loading modules: " << e.what());

    return EXIT_FAILURE;
  }

  vistk::scheduler_registry::types_t const types = reg->types();

  vistk::config_t const config = vistk::config::empty_config();

  vistk::pipeline_t const pipe = vistk::pipeline_t(new vistk::pipeline(config));

  BOOST_FOREACH (vistk::scheduler_registry::type_t const& type, types)
  {
    vistk::scheduler_t scheduler;

    try
    {
      scheduler = reg->create_scheduler(type, config, pipe);
    }
    catch (vistk::no_such_scheduler_type_exception const& e)
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
    vistk::scheduler_registry::type_t const non_existent_scheduler = vistk::scheduler_registry::type_t("no_such_scheduler");

    EXPECT_EXCEPTION(vistk::no_such_scheduler_type_exception,
                     reg->create_scheduler(non_existent_scheduler, config, pipe),
                     "requesting an non-existent scheduler type");

    EXPECT_EXCEPTION(vistk::no_such_scheduler_type_exception,
                     reg->description(non_existent_scheduler),
                     "requesting an non-existent scheduler type");
  }

  return EXIT_SUCCESS;
}
