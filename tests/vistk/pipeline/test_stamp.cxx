/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/stamp.h>

#define TEST_ARGS ()

DECLARE_TEST(equality);
DECLARE_TEST(ordering);

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, equality);
  ADD_TEST(tests, ordering);

  RUN_TEST(tests, testname);
}

IMPLEMENT_TEST(equality)
{
  vistk::stamp::increment_t const inca = vistk::stamp::increment_t(1);
  vistk::stamp::increment_t const incb = 2 * inca;

  vistk::stamp_t const stampa = vistk::stamp::new_stamp(inca);
  vistk::stamp_t const stampb = vistk::stamp::new_stamp(incb);

  if (*stampa != *stampb)
  {
    TEST_ERROR("New stamps are not equal");
  }

  vistk::stamp_t const stampc = vistk::stamp::incremented_stamp(stampa);

  if (*stampa == *stampc)
  {
    TEST_ERROR("An incremented stamp equals the original stamp");
  }

  vistk::stamp_t const stampd = vistk::stamp::incremented_stamp(stampc);
  vistk::stamp_t const stampe = vistk::stamp::incremented_stamp(stampb);

  if (*stampd != *stampe)
  {
    TEST_ERROR("Stamps with different rates do not compare as equal");
  }
}

IMPLEMENT_TEST(ordering)
{
  vistk::stamp::increment_t const inc = vistk::stamp::increment_t(1);

  vistk::stamp_t const stampa = vistk::stamp::new_stamp(inc);
  vistk::stamp_t const stampb = vistk::stamp::new_stamp(inc);

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

  vistk::stamp_t const stampe = vistk::stamp::incremented_stamp(stampa);

  if (*stampe < *stampa)
  {
    TEST_ERROR("An incremented stamp is greater than the original stamp");
  }

  if (!(*stampe > *stampa))
  {
    TEST_ERROR("An incremented stamp is greater than the original stamp");
  }
}
