/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/stamp.h>

#include <exception>
#include <iostream>
#include <string>

#include <cstdlib>

static void run_test(std::string const& test_name);

int
main(int argc, char* argv[])
{
  if (argc != 2)
  {
    TEST_ERROR("Expected one argument");

    return EXIT_FAILURE;
  }

  std::string const test_name = argv[1];

  try
  {
    run_test(test_name);
  }
  catch (std::exception const& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static void test_equality();
static void test_ordering();

void
run_test(std::string const& test_name)
{
  if (test_name == "equality")
  {
    return test_equality();
  }
  else if (test_name == "ordering")
  {
    return test_ordering();
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
  }
}

void
test_equality()
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

void
test_ordering()
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
