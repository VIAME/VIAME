/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>

#define TEST_ARGS ()

DECLARE_TEST(has_value);
DECLARE_TEST(get_value);
DECLARE_TEST(get_value_nested);
DECLARE_TEST(get_value_no_exist);
DECLARE_TEST(get_value_type_mismatch);
DECLARE_TEST(bool_conversion);
DECLARE_TEST(unset_value);
DECLARE_TEST(available_values);
DECLARE_TEST(read_only);
DECLARE_TEST(read_only_unset);
DECLARE_TEST(subblock);
DECLARE_TEST(subblock_view);
DECLARE_TEST(merge_config);

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, has_value);
  ADD_TEST(tests, get_value);
  ADD_TEST(tests, get_value_nested);
  ADD_TEST(tests, get_value_no_exist);
  ADD_TEST(tests, get_value_type_mismatch);
  ADD_TEST(tests, bool_conversion);
  ADD_TEST(tests, unset_value);
  ADD_TEST(tests, available_values);
  ADD_TEST(tests, read_only);
  ADD_TEST(tests, read_only_unset);
  ADD_TEST(tests, subblock);
  ADD_TEST(tests, subblock_view);
  ADD_TEST(tests, merge_config);

  RUN_TEST(tests, testname);
}

IMPLEMENT_TEST(has_value)
{
  vistk::config_t const config = vistk::config::empty_config();

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

IMPLEMENT_TEST(get_value)
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::config::key_t const keya = vistk::config::key_t("keya");

  vistk::config::value_t const valuea = vistk::config::value_t("value_a");

  config->set_value(keya, valuea);

  vistk::config::value_t const get_valuea = config->get_value<vistk::config::value_t>(keya);

  if (valuea != get_valuea)
  {
    TEST_ERROR("Did not retrieve value that was set");
  }
}

IMPLEMENT_TEST(get_value_nested)
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::config::key_t const keya = vistk::config::key_t("keya");
  vistk::config::key_t const keyb = vistk::config::key_t("keyb");

  vistk::config::value_t const valuea = vistk::config::value_t("value_a");

  config->set_value(keya + vistk::config::block_sep + keyb, valuea);

  vistk::config_t const nested_config = config->subblock(keya);

  vistk::config::value_t const get_valuea = nested_config->get_value<vistk::config::value_t>(keyb);

  if (valuea != get_valuea)
  {
    TEST_ERROR("Did not retrieve value that was set");
  }
}

IMPLEMENT_TEST(get_value_no_exist)
{
  vistk::config_t const config = vistk::config::empty_config();

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

IMPLEMENT_TEST(get_value_type_mismatch)
{
  vistk::config_t const config = vistk::config::empty_config();

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

IMPLEMENT_TEST(bool_conversion)
{
  vistk::config_t const config = vistk::config::empty_config();

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

IMPLEMENT_TEST(unset_value)
{
  vistk::config_t const config = vistk::config::empty_config();

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

IMPLEMENT_TEST(available_values)
{
  vistk::config_t const config = vistk::config::empty_config();

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

IMPLEMENT_TEST(read_only)
{
  vistk::config_t const config = vistk::config::empty_config();

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

IMPLEMENT_TEST(read_only_unset)
{
  vistk::config_t const config = vistk::config::empty_config();

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

IMPLEMENT_TEST(subblock)
{
  vistk::config_t const config = vistk::config::empty_config();

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

IMPLEMENT_TEST(subblock_view)
{
  vistk::config_t const config = vistk::config::empty_config();

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

  vistk::config_t const subblock = config->subblock_view(block_name);

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

IMPLEMENT_TEST(merge_config)
{
  vistk::config_t const configa = vistk::config::empty_config();
  vistk::config_t const configb = vistk::config::empty_config();

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
