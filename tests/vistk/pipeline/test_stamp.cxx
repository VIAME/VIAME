/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/stamp.h>

#include <exception>
#include <iostream>
#include <string>

static void run_test(std::string const& test_name);

int
main(int argc, char* argv[])
{
  if (argc != 2)
  {
    TEST_ERROR("Expected one argument");

    return 1;
  }

  std::string const test_name = argv[1];

  try
  {
    run_test(test_name);
  }
  catch (std::exception& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return 1;
  }

  return 0;
}

static void test_coloring();
static void test_equality();
static void test_ordering();

void
run_test(std::string const& test_name)
{
  if (test_name == "coloring")
  {
    return test_coloring();
  }
  else if (test_name == "equality")
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
test_coloring()
{
  vistk::stamp_t const stampa = vistk::stamp::new_stamp();
  vistk::stamp_t const stampb = vistk::stamp::new_stamp();
  vistk::stamp_t const stampc = vistk::stamp::incremented_stamp(stampa);

  if (stampa->is_same_color(stampb))
  {
    TEST_ERROR("New stamps have the same color");
  }

  if (!stampa->is_same_color(stampc))
  {
    TEST_ERROR("An incremented stamp changed color");
  }
}

void
test_equality()
{
  vistk::stamp_t const stampa = vistk::stamp::new_stamp();
  vistk::stamp_t const stampb = vistk::stamp::new_stamp();

  if (*stampa == *stampb)
  {
    TEST_ERROR("New stamps are equal");
  }

  vistk::stamp_t const stampc = vistk::stamp::copied_stamp(stampa);

  if (*stampa != *stampc)
  {
    TEST_ERROR("A copied stamp is not the same");
  }

  vistk::stamp_t const stampd = vistk::stamp::recolored_stamp(stampb, stampa);

  if (*stampa != *stampd)
  {
    TEST_ERROR("A recolored new stamp does not equal a new stamp");
  }

  vistk::stamp_t const stampe = vistk::stamp::incremented_stamp(stampa);

  if (*stampa == *stampe)
  {
    TEST_ERROR("An incremented stamp equals the original stamp");
  }
}

void
test_ordering()
{
  vistk::stamp_t const stampa = vistk::stamp::new_stamp();
  vistk::stamp_t const stampb = vistk::stamp::new_stamp();

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
