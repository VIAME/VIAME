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
#include <sprokit/pipeline/modules.h>
#include <sprokit/pipeline/process_cluster.h>
#include <sprokit/pipeline/process_registry.h>
#include <sprokit/pipeline/process_registry_exception.h>
#include <sprokit/pipeline/types.h>

#include <boost/foreach.hpp>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

IMPLEMENT_TEST(get_twice)
{
  sprokit::process_registry_t const reg1 = sprokit::process_registry::self();
  sprokit::process_registry_t const reg2 = sprokit::process_registry::self();

  if (reg1 != reg2)
  {
    TEST_ERROR("Received two different registries");
  }
}

IMPLEMENT_TEST(null_config)
{
  sprokit::process_registry_t const reg = sprokit::process_registry::self();

  kwiver::vital::config_block_sptr const config;

  EXPECT_EXCEPTION(sprokit::null_process_registry_config_exception,
                   reg->create_process(sprokit::process::type_t(), sprokit::process::name_t(), config),
                   "requesting a NULL config to a process");
}

IMPLEMENT_TEST(load_processes)
{
  sprokit::load_known_modules();

  sprokit::process_registry_t const reg = sprokit::process_registry::self();

  sprokit::process::types_t const types = reg->types();

  BOOST_FOREACH (sprokit::process::type_t const& type, types)
  {
    sprokit::process_t process;

    try
    {
      process = reg->create_process(type, sprokit::process::name_t());
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

    if (reg->description(type).empty())
    {
      TEST_ERROR("The description for "
                 << type << " is empty");
    }
  }
}

IMPLEMENT_TEST(null_ctor)
{
  sprokit::process_registry_t const reg = sprokit::process_registry::self();

  EXPECT_EXCEPTION(sprokit::null_process_ctor_exception,
                   reg->register_process(sprokit::process::type_t(), sprokit::process_registry::description_t(), sprokit::process_ctor_t()),
                   "requesting an non-existent process type");
}

static sprokit::process_t null_process(kwiver::vital::config_block_sptr const& config);

IMPLEMENT_TEST(duplicate_types)
{
  sprokit::process_registry_t const reg = sprokit::process_registry::self();

  sprokit::process::type_t const non_existent_process = sprokit::process::type_t("no_such_process");

  reg->register_process(non_existent_process, sprokit::process_registry::description_t(), null_process);

  EXPECT_EXCEPTION(sprokit::process_type_already_exists_exception,
                   reg->register_process(non_existent_process, sprokit::process_registry::description_t(), null_process),
                   "requesting an non-existent process type");
}

IMPLEMENT_TEST(unknown_types)
{
  sprokit::process_registry_t const reg = sprokit::process_registry::self();

  sprokit::process::type_t const non_existent_process = sprokit::process::type_t("no_such_process");

  EXPECT_EXCEPTION(sprokit::no_such_process_type_exception,
                   reg->create_process(non_existent_process, sprokit::process::name_t()),
                   "requesting an non-existent process type");

  EXPECT_EXCEPTION(sprokit::no_such_process_type_exception,
                   reg->description(non_existent_process),
                   "requesting an non-existent process type");
}

IMPLEMENT_TEST(module_marking)
{
  sprokit::process_registry_t const reg = sprokit::process_registry::self();

  sprokit::process_registry::module_t const module = sprokit::process_registry::module_t("module");

  if (reg->is_module_loaded(module))
  {
    TEST_ERROR("The module \'" << module << "\' is "
               "already marked as loaded");
  }

  reg->mark_module_as_loaded(module);

  if (!reg->is_module_loaded(module))
  {
    TEST_ERROR("The module \'" << module << "\' is "
               "not marked as loaded");
  }
}

IMPLEMENT_TEST(register_cluster)
{
  sprokit::load_known_modules();

  sprokit::process_registry_t const reg = sprokit::process_registry::self();

  sprokit::process::type_t const cluster_type = sprokit::process::type_t("orphan_cluster");
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::process_t const cluster_from_reg = reg->create_process(cluster_type, sprokit::process::name_t(), config);

  sprokit::process_cluster_t const cluster = boost::dynamic_pointer_cast<sprokit::process_cluster>(cluster_from_reg);

  if (!cluster)
  {
    TEST_ERROR("Failed to turn a process back into a cluster");
  }

  sprokit::process::type_t const type = sprokit::process::type_t("orphan");

  sprokit::process_t const not_a_cluster_from_reg = reg->create_process(type, sprokit::process::name_t(), config);

  sprokit::process_cluster_t const not_a_cluster = boost::dynamic_pointer_cast<sprokit::process_cluster>(not_a_cluster_from_reg);

  if (not_a_cluster)
  {
    TEST_ERROR("Turned a non-cluster into a cluster");
  }
}

sprokit::process_t
null_process(kwiver::vital::config_block_sptr const& /*config*/)
{
  return sprokit::process_t();
}
