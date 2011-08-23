/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/modules.h>

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

static void test_load();
static void test_multiple_load();

void
run_test(std::string const& test_name)
{
  if (test_name == "load")
  {
    test_load();
  }
  else if (test_name == "multiple_load")
  {
    test_multiple_load();
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
  }
}

void
test_load()
{
  vistk::load_known_modules();
}

void
test_multiple_load()
{
  vistk::load_known_modules();
  vistk::load_known_modules();
}
