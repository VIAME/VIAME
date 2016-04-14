/*ckwg +29
 * Copyright 2011-2015 by Kitware, Inc.
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

/**
 * \file
 * \brief core config_block tests
 */

#include <tests/test_common.h>

#include <vital/config/config_block.h>
#include <vital/io/eigen_io.h>
#include <vital/types/vector.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(value_conversion)
{
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();
  kwiver::vital::config_block_key_t const key = kwiver::vital::config_block_key_t("key");

  {
    config->set_value(key, 123.456);
    double val = config->get_value<double>(key);

    TEST_EQUAL("A double value is not converted to a config value and back again",
               val, 123.456);
  }
  {
    config->set_value(key, 1234567);
    unsigned int val = config->get_value<unsigned int>(key);

    TEST_EQUAL("An unsigned int value is not converted to a config value and back again",
               val, 1234567);
  }

  {
    kwiver::vital::vector_2d in_val(2.34, 0.0567);
    config->set_value(key, in_val);
    kwiver::vital::vector_2d val = config->get_value<kwiver::vital::vector_2d>(key);

    TEST_EQUAL("A vector_2d value is not converted to a config value and back again",
               val, in_val);
  }

  {
    config->set_value(key, "some string");
    std::string val = config->get_value<std::string>(key);
    TEST_EQUAL("A std::string value was not converted to a config value and back again",
               val, "some string");
  }
  {
    kwiver::vital::config_block_value_t in_val("Some value string");
    config->set_value(key, in_val);
    kwiver::vital::config_block_value_t val = config->get_value<kwiver::vital::config_block_key_t>(key);
    TEST_EQUAL("A cb_value_t value was not converted to a config value and back again",
               val, in_val);
  }
}
