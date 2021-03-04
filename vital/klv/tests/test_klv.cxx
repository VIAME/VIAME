// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test KLV classes
 */

#include <test_common.h>

#include <vital/klv/klv_parse.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

// data fields

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

IMPLEMENT_TEST(klv_api)
{
  // coming soon
}
