/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <stdexcept>
#include <iostream>
#include <string>

#include <cstdlib>

int
main(int argc, char* argv[])
{
  if (argc != 2)
  {
    TEST_ERROR("Expected one argument");

    return EXIT_FAILURE;
  }

  std::string const test_name = argv[1];

  if (test_name == "return_code")
  {
    return EXIT_FAILURE;
  }
  else if (test_name == "error_string")
  {
    TEST_ERROR("an error");
  }
  else if (test_name == "error_string_mid")
  {
    std::cerr << "Test";
    TEST_ERROR("an error");
  }
  else if (test_name == "error_string_stdout")
  {
    std::cout << "Error: an error" << std::endl;
  }
  else if (test_name == "error_string_second_line")
  {
    std::cerr << "Not an error" << std::endl;
    TEST_ERROR("an error");
  }
  else if (test_name == "expected_exception")
  {
    EXPECT_EXCEPTION(std::logic_error,
                     throw std::logic_error("reason"),
                     "when throwing an exception");
  }
  else if (test_name == "unexpected_exception")
  {
    EXPECT_EXCEPTION(std::runtime_error,
                     throw std::logic_error("reason"),
                     "when throwing an unexpected exception");
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
