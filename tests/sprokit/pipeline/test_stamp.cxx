/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <sprokit/pipeline/stamp.h>

#define TEST_ARGS ()

DECLARE_TEST(equality);
DECLARE_TEST(ordering);
DECLARE_TEST(increment_null);

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, equality);
  ADD_TEST(tests, ordering);
  ADD_TEST(tests, increment_null);

  RUN_TEST(tests, testname);
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
