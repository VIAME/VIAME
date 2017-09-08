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
