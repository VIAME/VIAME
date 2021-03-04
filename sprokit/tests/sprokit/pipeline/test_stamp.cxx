// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_common.h>

#include <sprokit/pipeline/stamp.h>

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
