/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
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

#include <sprokit/pipeline/stamp.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

IMPLEMENT_TEST(equality)
{
  sprokit::stamp::increment_t const inca = sprokit::stamp::increment_t(1);
  sprokit::stamp::increment_t const incb = 2 * inca;

  sprokit::stamp_t const stampa = sprokit::stamp::new_stamp(inca);
  sprokit::stamp_t const stampb = sprokit::stamp::new_stamp(incb);

  if (*stampa != *stampb)
  {
    TEST_ERROR("New stamps are not equal");
  }

  sprokit::stamp_t const stampc = sprokit::stamp::incremented_stamp(stampa);

  if (*stampa == *stampc)
  {
    TEST_ERROR("An incremented stamp equals the original stamp");
  }

  sprokit::stamp_t const stampd = sprokit::stamp::incremented_stamp(stampc);
  sprokit::stamp_t const stampe = sprokit::stamp::incremented_stamp(stampb);

  if (*stampd != *stampe)
  {
    TEST_ERROR("Stamps with different rates do not compare as equal");
  }
}

IMPLEMENT_TEST(ordering)
{
  sprokit::stamp::increment_t const inc = sprokit::stamp::increment_t(1);

  sprokit::stamp_t const stampa = sprokit::stamp::new_stamp(inc);
  sprokit::stamp_t const stampb = sprokit::stamp::new_stamp(inc);

  if ((*stampa < *stampb) ||
      (*stampb < *stampa))
  {
    TEST_ERROR("New stamps compare");
  }

  if (*stampa < *stampa)
  {
    TEST_ERROR("A stamp is less than itself");
  }

  if (*stampa > *stampa)
  {
    TEST_ERROR("A stamp is greater than itself");
  }

  sprokit::stamp_t const stampe = sprokit::stamp::incremented_stamp(stampa);

  if (*stampe < *stampa)
  {
    TEST_ERROR("An incremented stamp is greater than the original stamp");
  }

  if (!(*stampe > *stampa))
  {
    TEST_ERROR("An incremented stamp is greater than the original stamp");
  }
}

IMPLEMENT_TEST(increment_null)
{
  EXPECT_EXCEPTION(std::runtime_error,
                   sprokit::stamp::incremented_stamp(sprokit::stamp_t()),
                   "incrementing a NULL stamp");
}

IMPLEMENT_TEST(increment)
{
  sprokit::stamp::increment_t const inca = sprokit::stamp::increment_t(2);
  sprokit::stamp::increment_t const incb = sprokit::stamp::increment_t(3);

  sprokit::stamp_t const stampa = sprokit::stamp::new_stamp(inca);
  sprokit::stamp_t const stampb = sprokit::stamp::new_stamp(incb);

  if (*stampa != *stampb)
  {
    TEST_ERROR("New stamps with different increments are different");
  }

  sprokit::stamp_t const stampa2 = sprokit::stamp::incremented_stamp(stampa);
  sprokit::stamp_t const stampb3 = sprokit::stamp::incremented_stamp(stampb);

  if (*stampb3 < *stampa2)
  {
    TEST_ERROR("A stamp with step 3 is not less than than a stamp with step 2 after one step");
  }

  sprokit::stamp_t const stampa4 = sprokit::stamp::incremented_stamp(stampa2);

  if (*stampa4 < *stampb3)
  {
    TEST_ERROR("A stamp with step 2 stepped twice is not less than than a stamp with step 3 after one step");
  }

  sprokit::stamp_t const stampa6 = sprokit::stamp::incremented_stamp(stampa4);
  sprokit::stamp_t const stampb6 = sprokit::stamp::incremented_stamp(stampb3);

  if (*stampa6 != *stampb6)
  {
    TEST_ERROR("A stamp with step 2 stepped thrice is not equal than than a stamp with step 3 after two steps");
  }
}
