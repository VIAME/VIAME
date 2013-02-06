/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_registry.h>

#define TEST_ARGS ()

DECLARE_TEST(load);
DECLARE_TEST(multiple_load);
DECLARE_TEST(envvar);

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, load);
  ADD_TEST(tests, multiple_load);
  ADD_TEST(tests, envvar);

  RUN_TEST(tests, testname);
}

IMPLEMENT_TEST(load)
{
  vistk::load_known_modules();
}

IMPLEMENT_TEST(multiple_load)
{
  vistk::load_known_modules();
  vistk::load_known_modules();
}

IMPLEMENT_TEST(envvar)
{
  vistk::load_known_modules();

  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::process::type_t const proc_type = vistk::process::type_t("test");

  reg->create_process(proc_type, vistk::process::name_t());
}
