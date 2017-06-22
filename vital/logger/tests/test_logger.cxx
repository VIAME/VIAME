/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS [yas] elisp error!AS IS''
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

#include <tests/test_common.h>

#include <vital/logger/logger.h>

#include <functional>
#include <algorithm>
#include <stdlib.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

//
// Assume basic logger
//

// ------------------------------------------------------------------
IMPLEMENT_TEST(check_levels)
{
  kwiver::vital::logger_handle_t log2 = kwiver::vital::get_logger( "main.logger2" );

  log2->set_level(kwiver::vital::kwiver_logger::LEVEL_WARN);

  TEST_EQUAL ("Trace level enabled",  IS_TRACE_ENABLED(log2), false );
  TEST_EQUAL ("Debug level enabled",  IS_DEBUG_ENABLED(log2), false );
  TEST_EQUAL ("Info level enabled",   IS_INFO_ENABLED(log2), false );
  TEST_EQUAL ("Warn level enabled",   IS_WARN_ENABLED(log2), true );
  TEST_EQUAL ("Error level  enabled", IS_ERROR_ENABLED(log2), true );
  TEST_EQUAL ("Fatal level enabled",  IS_FATAL_ENABLED(log2), true );

  TEST_EQUAL ("Logger name", log2->get_name(), "main.logger2" );

  TEST_EQUAL ("Get logger2 level(1)", log2->get_level(), kwiver::vital::kwiver_logger::LEVEL_WARN );

  log2->set_level(kwiver::vital::kwiver_logger::LEVEL_DEBUG);

  TEST_EQUAL ("Trace level enabled",  IS_TRACE_ENABLED(log2), false );
  TEST_EQUAL ("Debug level enabled",  IS_DEBUG_ENABLED(log2), true );
  TEST_EQUAL ("Info level enabled",   IS_INFO_ENABLED(log2), true );
  TEST_EQUAL ("Warn level enabled",   IS_WARN_ENABLED(log2), true );
  TEST_EQUAL ("Error level  enabled", IS_ERROR_ENABLED(log2), true );
  TEST_EQUAL ("Fatal level enabled",  IS_FATAL_ENABLED(log2), true );
/*
  // a compiler check, mostly

  LOG_ASSERT( log2, true, "This should compile." );
  LOG_ASSERT( log2, false, "This should generate ERROR message." );
*/
  TEST_EQUAL ("Get logger2 level(2)", log2->get_level(), kwiver::vital::kwiver_logger::LEVEL_DEBUG );

  // test to see if we get the same logger back
  kwiver::vital::logger_handle_t log = kwiver::vital::get_logger( "main.logger2" );

  TEST_EQUAL ("Get logger level(3)", log->get_level(), kwiver::vital::kwiver_logger::LEVEL_DEBUG );

}


// ------------------------------------------------------------------
IMPLEMENT_TEST(logger_factory)
{
  // Need to unset any logger factory specification so we will use the
  // default factory
#if defined _WIN32
  _putenv( "VITAL_LOGGER_FACTORY=" );
#else
  unsetenv( "VITAL_LOGGER_FACTORY" );
#endif

  kwiver::vital::logger_handle_t log2 = kwiver::vital::get_logger( "main.logger" );

  TEST_EQUAL( "default logger factory name", log2->get_factory_name(), "default_logger factory" );
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(logger_output)
{
  kwiver::vital::logger_handle_t log2 = kwiver::vital::get_logger( "main.logger2" );

  log2->set_level(kwiver::vital::kwiver_logger::LEVEL_TRACE);

  LOG_DEBUG( log2, "Test message" );

}


//
// Need to test
//
// - loadable logger modules
// set env to bad module - get default logger
//  "default_logger factory"
// set env to test logger plugin - get that one.
//
// - logger output.
