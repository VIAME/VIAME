/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <sprokit/pipeline/config.h>
#include <sprokit/pipeline/modules.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_registry.h>

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
  sprokit::load_known_modules();
}

IMPLEMENT_TEST(multiple_load)
{
  sprokit::load_known_modules();
  sprokit::load_known_modules();
}

IMPLEMENT_TEST(envvar)
{
  sprokit::load_known_modules();

  sprokit::process_registry_t const reg = sprokit::process_registry::self();

  sprokit::process::type_t const proc_type = sprokit::process::type_t("test");

  reg->create_process(proc_type, sprokit::process::name_t());
}
