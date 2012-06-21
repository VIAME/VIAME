/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/edge_exception.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/stamp.h>

#include <boost/make_shared.hpp>

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

static void test_null_config();
static void test_makes_dependency();
static void test_new_has_no_data();
static void test_new_is_not_full();
static void test_new_has_count_zero();
static void test_push_datum();
static void test_peek_datum();
static void test_pop_datum();
static void test_get_datum();
static void test_null_upstream_process();
static void test_null_downstream_process();
static void test_set_upstream_process();
static void test_set_downstream_process();
static void test_push_data_into_complete();
static void test_get_data_from_complete();

void
run_test(std::string const& test_name)
{
  if (test_name == "null_config")
  {
    test_null_config();
  }
  else if (test_name == "makes_dependency")
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
  else if (test_name == "new_has_count_zero")
  {
    test_new_has_count_zero();
  }
  else if (test_name == "push_datum")
  {
    test_push_datum();
  }
  else if (test_name == "peek_datum")
  {
    test_peek_datum();
  }
  else if (test_name == "pop_datum")
  {
    test_pop_datum();
  }
  else if (test_name == "get_datum")
  {
    test_get_datum();
  }
  else if (test_name == "null_upstream_process")
  {
    test_null_upstream_process();
  }
  else if (test_name == "null_downstream_process")
  {
    test_null_downstream_process();
  }
  else if (test_name == "set_upstream_process")
  {
    test_set_upstream_process();
  }
  else if (test_name == "set_downstream_process")
  {
    test_set_downstream_process();
  }
  else if (test_name == "push_data_into_complete")
  {
    test_push_data_into_complete();
  }
  else if (test_name == "get_data_from_complete")
  {
    test_get_data_from_complete();
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
  }
}

void
test_null_config()
{
  vistk::config_t const config;

  EXPECT_EXCEPTION(vistk::null_edge_config_exception,
                   boost::make_shared<vistk::edge>(config),
                   "when passing a NULL config to an edge");
}

void
test_makes_dependency()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t edge = boost::make_shared<vistk::edge>(config);

  if (!edge->makes_dependency())
  {
    TEST_ERROR("A default edge does not imply a dependency");
  }

  config->set_value(vistk::edge::config_dependency, "false");

  edge = boost::make_shared<vistk::edge>(config);

  if (edge->makes_dependency())
  {
    TEST_ERROR("Setting the dependency config to \'false\' "
               "was not reflected in the result");
  }
}

void
test_new_has_no_data()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t edge = boost::make_shared<vistk::edge>(config);

  if (edge->has_data())
  {
    TEST_ERROR("A new edge has data in it");
  }
}

void
test_new_is_not_full()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t edge = boost::make_shared<vistk::edge>(config);

  if (edge->full_of_data())
  {
    TEST_ERROR("A new edge is full of data");
  }
}

void
test_new_has_count_zero()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t edge = boost::make_shared<vistk::edge>(config);

  if (edge->datum_count())
  {
    TEST_ERROR("A new edge has a count");
  }
}

void
test_push_datum()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t edge = boost::make_shared<vistk::edge>(config);

  vistk::datum_t const dat = vistk::datum::complete_datum();
  vistk::stamp_t const stamp = vistk::stamp::new_stamp();

  vistk::edge_datum_t const edat = vistk::edge_datum_t(dat, stamp);

  edge->push_datum(edat);

  if (edge->datum_count() != 1)
  {
    TEST_ERROR("An edge with a pushed datum does not have a count of one");
  }

  edge->push_datum(edat);

  if (edge->datum_count() != 2)
  {
    TEST_ERROR("An edge with two pushed data does not have a count of two");
  }
}

void
test_peek_datum()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t edge = boost::make_shared<vistk::edge>(config);

  vistk::datum_t const dat = vistk::datum::complete_datum();
  vistk::stamp_t const stamp = vistk::stamp::new_stamp();

  vistk::edge_datum_t const edat = vistk::edge_datum_t(dat, stamp);

  edge->push_datum(edat);

  vistk::edge_datum_t const get_edat = edge->peek_datum();

  if (edge->datum_count() != 1)
  {
    TEST_ERROR("An edge removed a datum on an peek");
  }

  if (*get_edat.get<1>() != *stamp)
  {
    TEST_ERROR("The edge modified a stamp on a peek");
  }
}

void
test_pop_datum()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t edge = boost::make_shared<vistk::edge>(config);

  vistk::datum_t const dat = vistk::datum::complete_datum();
  vistk::stamp_t const stamp = vistk::stamp::new_stamp();

  vistk::edge_datum_t const edat = vistk::edge_datum_t(dat, stamp);

  edge->push_datum(edat);

  edge->pop_datum();

  if (edge->datum_count())
  {
    TEST_ERROR("An edge did not remove a datum on a pop");
  }
}

void
test_get_datum()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t edge = boost::make_shared<vistk::edge>(config);

  vistk::datum_t const dat = vistk::datum::complete_datum();
  vistk::stamp_t const stamp = vistk::stamp::new_stamp();

  vistk::edge_datum_t const edat = vistk::edge_datum_t(dat, stamp);

  edge->push_datum(edat);

  vistk::edge_datum_t const get_edat = edge->get_datum();

  if (edge->datum_count() != 0)
  {
    TEST_ERROR("An edge did not remove a datum on a get");
  }

  if (*get_edat.get<1>() != *stamp)
  {
    TEST_ERROR("The edge modified a stamp on a get");
  }
}

void
test_null_upstream_process()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t edge = boost::make_shared<vistk::edge>(config);

  vistk::process_t const process;

  EXPECT_EXCEPTION(vistk::null_process_connection_exception,
                   edge->set_upstream_process(process),
                   "setting a NULL process as upstream");
}

void
test_null_downstream_process()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t edge = boost::make_shared<vistk::edge>(config);

  vistk::process_t const process;

  EXPECT_EXCEPTION(vistk::null_process_connection_exception,
                   edge->set_downstream_process(process),
                   "setting a NULL process as downstream");
}

void
test_set_upstream_process()
{
  vistk::load_known_modules();

  vistk::process_registry_t const reg = vistk::process_registry::self();
  vistk::process::type_t const proc_type = vistk::process::type_t("numbers");

  vistk::process_t const process = reg->create_process(proc_type, vistk::process::name_t());

  vistk::edge_t edge = boost::make_shared<vistk::edge>();

  edge->set_upstream_process(process);

  EXPECT_EXCEPTION(vistk::input_already_connected_exception,
                   edge->set_upstream_process(process),
                   "setting a second process as upstream");
}

void
test_set_downstream_process()
{
  vistk::load_known_modules();

  vistk::process_registry_t const reg = vistk::process_registry::self();
  vistk::process::type_t const proc_type = vistk::process::type_t("numbers");

  vistk::process_t const process = reg->create_process(proc_type, vistk::process::name_t());

  vistk::edge_t edge = boost::make_shared<vistk::edge>();

  edge->set_downstream_process(process);

  EXPECT_EXCEPTION(vistk::output_already_connected_exception,
                   edge->set_downstream_process(process),
                   "setting a second process as downstream");
}

void
test_push_data_into_complete()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t edge = boost::make_shared<vistk::edge>(config);

  vistk::datum_t const dat = vistk::datum::complete_datum();
  vistk::stamp_t const stamp = vistk::stamp::new_stamp();

  vistk::edge_datum_t const edat = vistk::edge_datum_t(dat, stamp);

  edge->push_datum(edat);

  edge->mark_downstream_as_complete();

  if (edge->datum_count())
  {
    TEST_ERROR("A complete edge did not flush data");
  }

  edge->push_datum(edat);

  if (edge->datum_count())
  {
    TEST_ERROR("A complete edge accepted data");
  }
}

void
test_get_data_from_complete()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t edge = boost::make_shared<vistk::edge>(config);

  edge->mark_downstream_as_complete();

  EXPECT_EXCEPTION(vistk::datum_requested_after_complete,
                   edge->peek_datum(),
                   "peeking at a complete edge");

  EXPECT_EXCEPTION(vistk::datum_requested_after_complete,
                   edge->get_datum(),
                   "getting data from a complete edge");

  EXPECT_EXCEPTION(vistk::datum_requested_after_complete,
                   edge->pop_datum(),
                   "popping data from a complete edge");
}
