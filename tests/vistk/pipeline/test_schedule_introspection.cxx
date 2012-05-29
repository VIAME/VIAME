/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/schedule.h>
#include <vistk/pipeline/schedule_registry.h>
#include <vistk/pipeline/schedule_registry_exception.h>
#include <vistk/pipeline/types.h>

#include <boost/foreach.hpp>

#include <exception>
#include <iostream>

#include <cstdlib>

int
main()
{
  vistk::schedule_registry_t reg = vistk::schedule_registry::self();

  try
  {
    vistk::load_known_modules();
  }
  catch (vistk::schedule_type_already_exists_exception& e)
  {
    TEST_ERROR("Duplicate schedule names: " << e.what());
  }
  catch (vistk::pipeline_exception& e)
  {
    TEST_ERROR("Failed to load modules: " << e.what());
  }
  catch (std::exception& e)
  {
    TEST_ERROR("Unexpected exception when loading modules: " << e.what());

    return EXIT_FAILURE;
  }

  vistk::schedule_registry::types_t const types = reg->types();

  vistk::config_t config = vistk::config::empty_config();

  vistk::pipeline_t pipe = vistk::pipeline_t(new vistk::pipeline(config));

  BOOST_FOREACH (vistk::schedule_registry::type_t const& type, types)
  {
    vistk::schedule_t schedule;

    try
    {
      schedule = reg->create_schedule(type, config, pipe);
    }
    catch (vistk::no_such_schedule_type_exception& e)
    {
      TEST_ERROR("Failed to create schedule: " << e.what());

      continue;
    }
    catch (std::exception& e)
    {
      TEST_ERROR("Unexpected exception when creating schedule: " << e.what());
    }

    if (!schedule)
    {
      TEST_ERROR("Received NULL schedule (" << type << ")");

      continue;
    }

    if (reg->description(type).empty())
    {
      TEST_ERROR("The description is empty");
    }
  }

  // Check exceptions for unknown types.
  {
    vistk::schedule_registry::type_t const non_existent_schedule = vistk::schedule_registry::type_t("no_such_schedule");

    EXPECT_EXCEPTION(vistk::no_such_schedule_type_exception,
                     reg->create_schedule(non_existent_schedule, config, pipe),
                     "requesting an non-existent schedule type");

    EXPECT_EXCEPTION(vistk::no_such_schedule_type_exception,
                     reg->description(non_existent_schedule),
                     "requesting an non-existent schedule type");
  }

  return EXIT_SUCCESS;
}
