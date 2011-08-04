/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/edge.h>

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

static void test_makes_dependency();
static void test_new_has_no_data();
static void test_new_is_not_full();

void
run_test(std::string const& test_name)
{
  if (test_name == "makes_dependency")
  {
    test_makes_dependency();
  }
  else if (test_name == "new_has_no_data")
  {
    test_new_has_no_data();
  }
  else if (test_name == "new_is_not_full")
  {
    test_new_is_not_full();
  }
  else
  {
    std::cerr << "Error: Unknown test: " << test_name << std::endl;
  }
}

void
test_makes_dependency()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t edge = vistk::edge_t(new vistk::edge(config));

  edge->makes_dependency();
}

void
test_new_has_no_data()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t edge = vistk::edge_t(new vistk::edge(config));

  if (edge->has_data())
  {
    std::cerr << "Error: A new edge has data in it" << std::endl;
  }
}

void
test_new_is_not_full()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t edge = vistk::edge_t(new vistk::edge(config));

  if (edge->full_of_data())
  {
    std::cerr << "Error: A new edge is full of data" << std::endl;
  }
}
