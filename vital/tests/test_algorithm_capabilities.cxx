/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \brief test capabilities class
 */

#include <test_common.h>
#include <vital/algorithm_capabilities.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

IMPLEMENT_TEST(test_api)
{
  kwiver::vital::algorithm_capabilities cap;

  TEST_EQUAL( "Empty capabilities cap", cap.has_capability("test"), false );

  kwiver::vital::algorithm_capabilities::capability_list_t cap_list = cap.capability_list();
  TEST_EQUAL( "Empty cap list", cap_list.empty(), true );

  cap.set_capability( "cap1", true );
  cap.set_capability( "cap2", false );

  cap_list = cap.capability_list();
  TEST_EQUAL( "Empty cap list 2", cap_list.size(), 2 );

  TEST_EQUAL( "cap1", cap.capability( "cap1"), true );
  TEST_EQUAL( "cap2", cap.capability( "cap2"), false );

  kwiver::vital::algorithm_capabilities cap_2( cap );

  cap_list = cap_2.capability_list();
  TEST_EQUAL( "Empty cap list 2", cap_list.size(), 2 );

  TEST_EQUAL( "cap1", cap_2.capability( "cap1"), true );
  TEST_EQUAL( "cap2", cap_2.capability( "cap2"), false );

  kwiver::vital::algorithm_capabilities cap_3;
  cap_3 = cap;

  cap_list = cap_3.capability_list();
  TEST_EQUAL( "Empty cap list 2", cap_list.size(), 2 );

  TEST_EQUAL( "cap1", cap_3.capability( "cap1"), true );
  TEST_EQUAL( "cap2", cap_3.capability( "cap2"), false );
}
