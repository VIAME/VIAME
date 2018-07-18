/*ckwg +29
 * Copyright 2012-2018 by Kitware, Inc.
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
