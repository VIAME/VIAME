/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/config.h>

#include <exception>
#include <iostream>

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

static void test_has_value();
static void test_get_value();
static void test_get_value_no_exist();

void
run_test(std::string const& test_name)
{
  if (test_name == "has_value")
  {
    test_has_value();
  }
  else if (test_name == "get_value")
  {
    test_get_value();
  }
  else if (test_name == "get_value_no_exist")
  {
    test_get_value_no_exist();
  }
  else
  {
    std::cerr << "Error: Unknown test: " << test_name << std::endl;
  }
}

void
test_has_value()
{
  vistk::config_t config = vistk::config::empty_config();

  vistk::config::key_t const keya = vistk::config::key_t("keya");
  vistk::config::key_t const keyb = vistk::config::key_t("keyb");

  vistk::config::value_t const valuea = vistk::config::value_t("value_a");

  config->set_value(keya, valuea);

  if (!config->has_value(keya))
  {
    std::cerr << "Error: Block does not have value which set" << std::endl;
  }

  if (config->has_value(keyb))
  {
    std::cerr << "Error: Block has value which was not set" << std::endl;
  }
}

void
test_get_value()
{
  vistk::config_t config = vistk::config::empty_config();

  vistk::config::key_t const keya = vistk::config::key_t("keya");

  vistk::config::value_t const valuea = vistk::config::value_t("value_a");

  config->set_value(keya, valuea);

  vistk::config::value_t const get_valuea = config->get_value<vistk::config::value_t>(keya);

  if (valuea != get_valuea)
  {
    std::cerr << "Error: Did not retrieve value that was set" << std::endl;
  }
}

void
test_get_value_no_exist()
{
  vistk::config_t config = vistk::config::empty_config();

  vistk::config::key_t const keya = vistk::config::key_t("keya");
  vistk::config::key_t const keyb = vistk::config::key_t("keyb");

  vistk::config::value_t const valuea = vistk::config::value_t("value_a");
  vistk::config::value_t const valueb = vistk::config::value_t("value_b");

  bool got_exception = false;

  try
  {
    config->get_value<vistk::config::value_t>(keya);
  }
  catch (vistk::no_such_configuration_value_exception&)
  {
    got_exception = true;
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: Unexpected exception: "
              << e.what() << std::endl;

    got_exception = true;
  }

  if (!got_exception)
  {
    std::cerr << "Error: Did not get expected exception "
              << "when retrieving an unset value" << std::endl;
  }

  vistk::config::value_t const get_valueb = config->get_value<vistk::config::value_t>(keyb, valueb);

  if (valueb != get_valueb)
  {
    std::cerr << "Error: Did not retrieve default when requesting unset value" << std::endl;
  }
}
