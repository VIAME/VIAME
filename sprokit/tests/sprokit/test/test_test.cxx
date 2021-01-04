// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_common.h>

#include <sprokit/pipeline/utils.h>

#include <kwiversys/SystemTools.hxx>

#include <iostream>
#include <stdexcept>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

TEST_PROPERTY(WILL_FAIL, TRUE)
IMPLEMENT_TEST(return_code)
{
  exit(EXIT_FAILURE);
}

TEST_PROPERTY(WILL_FAIL, TRUE)
IMPLEMENT_TEST(error_string)
{
  TEST_ERROR("an error");
}

IMPLEMENT_TEST(error_string_mid)
{
  std::cerr << "Test";
  TEST_ERROR("an error");
}

TEST_PROPERTY(WILL_FAIL, TRUE)
IMPLEMENT_TEST(error_string_stdout)
{
  std::cout << "Error: an error" << std::endl;
}

TEST_PROPERTY(WILL_FAIL, TRUE)
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

TEST_PROPERTY(WILL_FAIL, TRUE)
IMPLEMENT_TEST(unexpected_exception)
{
  EXPECT_EXCEPTION(std::runtime_error,
                    throw std::logic_error("reason"),
                    "when throwing an unexpected exception");
}

TEST_PROPERTY(ENVIRONMENT, TEST_ENVVAR=test_value)
IMPLEMENT_TEST(environment)
{
  const std::string envvar = "TEST_ENVVAR";

  std::string envvalue;
  kwiversys::SystemTools::GetEnv( envvar, envvalue );

  if ( envvalue.empty() )
  {
    TEST_ERROR("failed to get environment from CTest");
  }
  else
  {
    const std::string expected = std::string("test_value");

    if (envvalue != expected)
    {
      TEST_ERROR("Did not get expected value: "
                  "Expected: " << expected << " "
                  "Received: " << envvalue);
    }
  }
}
