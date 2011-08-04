/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

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
    std::cerr << "Error: Expected one argument" << std::endl;

    return 1;
  }

  std::string const test_name = argv[1];

  try
  {
    run_test(test_name);
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: Unexpected exception: " << e.what() << std::endl;

    return 1;
  }

  return 0;
}

static void test_coloring();
static void test_equality();

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
  else
  {
    std::cerr << "Error: Unknown test: " << test_name << std::endl;
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
    std::cerr << "Error: New stamps have the same color" << std::endl;
  }

  if (!stampa->is_same_color(stampc))
  {
    std::cerr << "Error: An incremented stamp changed color" << std::endl;
  }
}

void
test_equality()
{
  vistk::stamp_t const stampa = vistk::stamp::new_stamp();
  vistk::stamp_t const stampb = vistk::stamp::new_stamp();

  if (*stampa == *stampb)
  {
    std::cerr << "Error: New stamps are equal" << std::endl;
  }

  vistk::stamp_t const stampc = vistk::stamp::copied_stamp(stampa);

  if (*stampa != *stampc)
  {
    std::cerr << "Error: A copied stamp is not the same" << std::endl;
  }

  vistk::stamp_t const stampd = vistk::stamp::recolored_stamp(stampb, stampa);

  if (*stampa != *stampd)
  {
    std::cerr << "Error: A recolored new stamp does not equal a new stamp" << std::endl;
  }

  vistk::stamp_t const stampe = vistk::stamp::incremented_stamp(stampa);

  if (*stampa == *stampe)
  {
    std::cerr << "Error: An incremented stamp equals the original stamp" << std::endl;
  }
}
