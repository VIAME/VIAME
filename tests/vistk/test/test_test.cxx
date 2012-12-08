/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/utils.h>

#include <iostream>
#include <stdexcept>

#define TEST_ARGS ()

DECLARE_TEST(return_code);
DECLARE_TEST(error_string);
DECLARE_TEST(error_string_mid);
DECLARE_TEST(error_string_stdout);
DECLARE_TEST(error_string_second_line);
DECLARE_TEST(expected_exception);
DECLARE_TEST(unexpected_exception);
DECLARE_TEST(environment);

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, return_code);
  ADD_TEST(tests, error_string);
  ADD_TEST(tests, error_string_mid);
  ADD_TEST(tests, error_string_stdout);
  ADD_TEST(tests, error_string_second_line);
  ADD_TEST(tests, expected_exception);
  ADD_TEST(tests, unexpected_exception);
  ADD_TEST(tests, environment);

  RUN_TEST(tests, testname);
}

IMPLEMENT_TEST(return_code)
{
  exit(EXIT_FAILURE);
}

IMPLEMENT_TEST(error_string)
{
  TEST_ERROR("an error");
}

IMPLEMENT_TEST(error_string_mid)
{
  std::cerr << "Test";
  TEST_ERROR("an error");
}

IMPLEMENT_TEST(error_string_stdout)
{
  std::cout << "Error: an error" << std::endl;
}

IMPLEMENT_TEST(error_string_second_line)
{
  std::cerr << "Not an error" << std::endl;
  TEST_ERROR("an error");
}

IMPLEMENT_TEST(expected_exception)
{
  EXPECT_EXCEPTION(std::logic_error,
                    throw std::logic_error("reason"),
                    "when throwing an exception");
}

IMPLEMENT_TEST(unexpected_exception)
{
  EXPECT_EXCEPTION(std::runtime_error,
                    throw std::logic_error("reason"),
                    "when throwing an unexpected exception");
}

IMPLEMENT_TEST(environment)
{
  vistk::envvar_name_t const envvar = "TEST_ENVVAR";

  vistk::envvar_value_t const envvalue = vistk::get_envvar(envvar);

  if (!envvalue)
  {
    TEST_ERROR("failed to get environment from CTest");
  }
  else
  {
    vistk::envvar_value_t const expected = vistk::envvar_value_t("test_value");

    if (*envvalue != *expected)
    {
      TEST_ERROR("Did not get expected value: "
                  "Expected: " << expected << " "
                  "Received: " << envvalue);
    }
  }
}
