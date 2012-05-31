/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>

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
  catch (std::exception& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static void test_has_value();
static void test_get_value();
static void test_get_value_no_exist();
static void test_get_value_type_mismatch();
static void test_bool_conversion();
static void test_unset_value();
static void test_available_values();
static void test_read_only();
static void test_read_only_unset();
static void test_subblock();
static void test_subblock_view();
static void test_merge_config();

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
  else if (test_name == "get_value_type_mismatch")
  {
    test_get_value_type_mismatch();
  }
  else if (test_name == "bool_conversion")
  {
    test_bool_conversion();
  }
  else if (test_name == "unset_value")
  {
    test_unset_value();
  }
  else if (test_name == "available_values")
  {
    test_available_values();
  }
  else if (test_name == "read_only")
  {
    test_read_only();
  }
  else if (test_name == "read_only_unset")
  {
    test_read_only_unset();
  }
  else if (test_name == "subblock")
  {
    test_subblock();
  }
  else if (test_name == "subblock_view")
  {
    test_subblock_view();
  }
  else if (test_name == "merge_config")
  {
    test_merge_config();
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
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
    TEST_ERROR("Block does not have value which was set");
  }

  if (config->has_value(keyb))
  {
    TEST_ERROR("Block has value which was not set");
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
    TEST_ERROR("Did not retrieve value that was set");
  }
}

void
test_get_value_no_exist()
{
  vistk::config_t config = vistk::config::empty_config();

  vistk::config::key_t const keya = vistk::config::key_t("keya");
  vistk::config::key_t const keyb = vistk::config::key_t("keyb");

  vistk::config::value_t const valueb = vistk::config::value_t("value_b");

  EXPECT_EXCEPTION(vistk::no_such_configuration_value_exception,
                   config->get_value<vistk::config::value_t>(keya),
                   "retrieving an unset value");

  vistk::config::value_t const get_valueb = config->get_value<vistk::config::value_t>(keyb, valueb);

  if (valueb != get_valueb)
  {
    TEST_ERROR("Did not retrieve default when requesting unset value");
  }
}

void
test_get_value_type_mismatch()
{
  vistk::config_t config = vistk::config::empty_config();

  vistk::config::key_t const keya = vistk::config::key_t("keya");

  vistk::config::value_t const valuea = vistk::config::value_t("value_a");
  int const valueb = 100;

  config->set_value(keya, valuea);

  EXPECT_EXCEPTION(vistk::bad_configuration_cast_exception,
                   config->get_value<int>(keya),
                   "doing an invalid cast");

  int const get_valueb = config->get_value<int>(keya, valueb);

  if (valueb != get_valueb)
  {
    TEST_ERROR("Did not retrieve default when requesting a bad cast");
  }
}

void
test_bool_conversion()
{
  vistk::config_t config = vistk::config::empty_config();

  vistk::config::key_t const key = vistk::config::key_t("key");

  vistk::config::value_t const lit_true = vistk::config::value_t("true");
  vistk::config::value_t const lit_false = vistk::config::value_t("false");
  vistk::config::value_t const lit_True = vistk::config::value_t("True");
  vistk::config::value_t const lit_False = vistk::config::value_t("False");
  vistk::config::value_t const lit_1 = vistk::config::value_t("1");
  vistk::config::value_t const lit_0 = vistk::config::value_t("0");

  bool val;

  config->set_value(key, lit_true);
  val = config->get_value<bool>(key);

  if (!val)
  {
    TEST_ERROR("The value \'true\' did not get converted to true when read as a boolean");
  }

  config->set_value(key, lit_false);
  val = config->get_value<bool>(key);

  if (val)
  {
    TEST_ERROR("The value \'false\' did not get converted to false when read as a boolean");
  }

  config->set_value(key, lit_True);
  val = config->get_value<bool>(key);

  if (!val)
  {
    TEST_ERROR("The value \'True\' did not get converted to true when read as a boolean");
  }

  config->set_value(key, lit_False);
  val = config->get_value<bool>(key);

  if (val)
  {
    TEST_ERROR("The value \'False\' did not get converted to false when read as a boolean");
  }

  config->set_value(key, lit_1);
  val = config->get_value<bool>(key);

  if (!val)
  {
    TEST_ERROR("The value \'1\' did not get converted to true when read as a boolean");
  }

  config->set_value(key, lit_0);
  val = config->get_value<bool>(key);

  if (val)
  {
    TEST_ERROR("The value \'0\' did not get converted to true when read as a boolean");
  }
}

void
test_unset_value()
{
  vistk::config_t config = vistk::config::empty_config();

  vistk::config::key_t const keya = vistk::config::key_t("keya");
  vistk::config::key_t const keyb = vistk::config::key_t("keyb");

  vistk::config::value_t const valuea = vistk::config::value_t("value_a");
  vistk::config::value_t const valueb = vistk::config::value_t("value_b");

  config->set_value(keya, valuea);
  config->set_value(keyb, valueb);

  config->unset_value(keya);

  EXPECT_EXCEPTION(vistk::no_such_configuration_value_exception,
                   config->get_value<vistk::config::value_t>(keya),
                   "retrieving an unset value");

  vistk::config::value_t const get_valueb = config->get_value<vistk::config::value_t>(keyb);

  if (valueb != get_valueb)
  {
    TEST_ERROR("Did not retrieve value when requesting after an unrelated unset");
  }
}

void
test_available_values()
{
  vistk::config_t config = vistk::config::empty_config();

  vistk::config::key_t const keya = vistk::config::key_t("keya");
  vistk::config::key_t const keyb = vistk::config::key_t("keyb");

  vistk::config::value_t const valuea = vistk::config::value_t("value_a");
  vistk::config::value_t const valueb = vistk::config::value_t("value_b");

  config->set_value(keya, valuea);
  config->set_value(keyb, valueb);

  vistk::config::keys_t keys;

  keys.push_back(keya);
  keys.push_back(keyb);

  vistk::config::keys_t const get_keys = config->available_values();

  if (keys.size() != get_keys.size())
  {
    TEST_ERROR("Did not retrieve correct number of keys");
  }
}

void
test_read_only()
{
  vistk::config_t config = vistk::config::empty_config();

  vistk::config::key_t const keya = vistk::config::key_t("keya");

  vistk::config::value_t const valuea = vistk::config::value_t("value_a");
  vistk::config::value_t const valueb = vistk::config::value_t("value_b");

  config->set_value(keya, valuea);

  config->mark_read_only(keya);

  EXPECT_EXCEPTION(vistk::set_on_read_only_value_exception,
                   config->set_value(keya, valueb),
                   "setting a read only value");

  vistk::config::value_t const get_valuea = config->get_value<vistk::config::value_t>(keya);

  if (valuea != get_valuea)
  {
    TEST_ERROR("Read only value changed");
  }
}

void
test_read_only_unset()
{
  vistk::config_t config = vistk::config::empty_config();

  vistk::config::key_t const keya = vistk::config::key_t("keya");

  vistk::config::value_t const valuea = vistk::config::value_t("value_a");

  config->set_value(keya, valuea);

  config->mark_read_only(keya);

  EXPECT_EXCEPTION(vistk::unset_on_read_only_value_exception,
                   config->unset_value(keya),
                   "unsetting a read only value");

  vistk::config::value_t const get_valuea = config->get_value<vistk::config::value_t>(keya);

  if (valuea != get_valuea)
  {
    TEST_ERROR("Read only value was unset");
  }
}

void
test_subblock()
{
  vistk::config_t config = vistk::config::empty_config();

  vistk::config::key_t const block_name = vistk::config::key_t("block");
  vistk::config::key_t const other_block_name = vistk::config::key_t("other_block");

  vistk::config::key_t const keya = vistk::config::key_t("keya");
  vistk::config::key_t const keyb = vistk::config::key_t("keyb");
  vistk::config::key_t const keyc = vistk::config::key_t("keyc");

  vistk::config::value_t const valuea = vistk::config::value_t("value_a");
  vistk::config::value_t const valueb = vistk::config::value_t("value_b");
  vistk::config::value_t const valuec = vistk::config::value_t("value_c");

  config->set_value(block_name + vistk::config::block_sep + keya, valuea);
  config->set_value(block_name + vistk::config::block_sep + keyb, valueb);
  config->set_value(other_block_name + vistk::config::block_sep + keyc, valuec);

  vistk::config_t const subblock = config->subblock(block_name);

  vistk::config::value_t const get_valuea = subblock->get_value<vistk::config::value_t>(keya);

  if (valuea != get_valuea)
  {
    TEST_ERROR("Subblock did not inherit expected keys");
  }

  vistk::config::value_t const get_valueb = subblock->get_value<vistk::config::value_t>(keyb);

  if (valueb != get_valueb)
  {
    TEST_ERROR("Subblock did not inherit expected keys");
  }

  if (subblock->has_value(keyc))
  {
    TEST_ERROR("Subblock inherited unrelated key");
  }
}

void
test_subblock_view()
{
  vistk::config_t config = vistk::config::empty_config();

  vistk::config::key_t const block_name = vistk::config::key_t("block");
  vistk::config::key_t const other_block_name = vistk::config::key_t("other_block");

  vistk::config::key_t const keya = vistk::config::key_t("keya");
  vistk::config::key_t const keyb = vistk::config::key_t("keyb");
  vistk::config::key_t const keyc = vistk::config::key_t("keyc");

  vistk::config::value_t const valuea = vistk::config::value_t("value_a");
  vistk::config::value_t const valueb = vistk::config::value_t("value_b");
  vistk::config::value_t const valuec = vistk::config::value_t("value_c");

  config->set_value(block_name + vistk::config::block_sep + keya, valuea);
  config->set_value(block_name + vistk::config::block_sep + keyb, valueb);
  config->set_value(other_block_name + vistk::config::block_sep + keyc, valuec);

  vistk::config_t subblock = config->subblock_view(block_name);

  if (!subblock->has_value(keya))
  {
    TEST_ERROR("Subblock view did not inherit key");
  }

  if (subblock->has_value(keyc))
  {
    TEST_ERROR("Subblock view inherited unrelated key");
  }

  config->set_value(block_name + vistk::config::block_sep + keya, valueb);

  vistk::config::value_t const get_valuea1 = subblock->get_value<vistk::config::value_t>(keya);

  if (valueb != get_valuea1)
  {
    TEST_ERROR("Subblock view persisted a changed value");
  }

  subblock->set_value(keya, valuea);

  vistk::config::value_t const get_valuea2 = config->get_value<vistk::config::value_t>(block_name + vistk::config::block_sep + keya);

  if (valuea != get_valuea2)
  {
    TEST_ERROR("Subblock view set value was not changed in parent");
  }

  subblock->unset_value(keyb);

  if (config->has_value(block_name + vistk::config::block_sep + keyb))
  {
    TEST_ERROR("Unsetting from a subblock view did not unset in parent view");
  }

  config->set_value(block_name + vistk::config::block_sep + keyc, valuec);

  vistk::config::keys_t keys;

  keys.push_back(keya);
  keys.push_back(keyc);

  vistk::config::keys_t const get_keys = subblock->available_values();

  if (keys.size() != get_keys.size())
  {
    TEST_ERROR("Did not retrieve correct number of keys from the subblock");
  }
}

void
test_merge_config()
{
  vistk::config_t configa = vistk::config::empty_config();
  vistk::config_t configb = vistk::config::empty_config();

  vistk::config::key_t const keya = vistk::config::key_t("keya");
  vistk::config::key_t const keyb = vistk::config::key_t("keyb");
  vistk::config::key_t const keyc = vistk::config::key_t("keyc");

  vistk::config::value_t const valuea = vistk::config::value_t("value_a");
  vistk::config::value_t const valueb = vistk::config::value_t("value_b");
  vistk::config::value_t const valuec = vistk::config::value_t("value_c");

  configa->set_value(keya, valuea);
  configa->set_value(keyb, valuea);

  configb->set_value(keyb, valueb);
  configb->set_value(keyc, valuec);

  configa->merge_config(configb);

  vistk::config::value_t const get_valuea = configa->get_value<vistk::config::value_t>(keya);

  if (valuea != get_valuea)
  {
    TEST_ERROR("Unmerged key changed");
  }

  vistk::config::value_t const get_valueb = configa->get_value<vistk::config::value_t>(keyb);

  if (valueb != get_valueb)
  {
    TEST_ERROR("Conflicting key was not overwritten");
  }

  vistk::config::value_t const get_valuec = configa->get_value<vistk::config::value_t>(keyc);

  if (valuec != get_valuec)
  {
    TEST_ERROR("New key did not appear");
  }
}
