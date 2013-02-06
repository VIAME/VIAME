/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
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

#include <boost/chrono/chrono_io.hpp>
#include <boost/chrono/duration.hpp>
#include <boost/chrono/process_cpu_clocks.hpp>
#if BOOST_VERSION < 105000
#include <boost/date_time/posix_time/posix_time.hpp>
#endif
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/make_shared.hpp>

#define TEST_ARGS ()

DECLARE_TEST(null_config);
DECLARE_TEST(makes_dependency);
DECLARE_TEST(new_has_no_data);
DECLARE_TEST(new_is_not_full);
DECLARE_TEST(new_has_count_zero);
DECLARE_TEST(push_datum);
DECLARE_TEST(peek_datum);
DECLARE_TEST(peek_datum_index);
DECLARE_TEST(pop_datum);
DECLARE_TEST(get_datum);
DECLARE_TEST(null_upstream_process);
DECLARE_TEST(null_downstream_process);
DECLARE_TEST(set_upstream_process);
DECLARE_TEST(set_downstream_process);
DECLARE_TEST(push_data_into_complete);
DECLARE_TEST(get_data_from_complete);
DECLARE_TEST(capacity);

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, null_config);
  ADD_TEST(tests, makes_dependency);
  ADD_TEST(tests, new_has_no_data);
  ADD_TEST(tests, new_is_not_full);
  ADD_TEST(tests, new_has_count_zero);
  ADD_TEST(tests, push_datum);
  ADD_TEST(tests, peek_datum);
  ADD_TEST(tests, peek_datum_index);
  ADD_TEST(tests, pop_datum);
  ADD_TEST(tests, get_datum);
  ADD_TEST(tests, null_upstream_process);
  ADD_TEST(tests, null_downstream_process);
  ADD_TEST(tests, set_upstream_process);
  ADD_TEST(tests, set_downstream_process);
  ADD_TEST(tests, push_data_into_complete);
  ADD_TEST(tests, get_data_from_complete);
  ADD_TEST(tests, capacity);

  RUN_TEST(tests, testname);
}

IMPLEMENT_TEST(null_config)
{
  vistk::config_t const config;

  EXPECT_EXCEPTION(vistk::null_edge_config_exception,
                   boost::make_shared<vistk::edge>(config),
                   "when passing a NULL config to an edge");
}

IMPLEMENT_TEST(makes_dependency)
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  if (!edge->makes_dependency())
  {
    TEST_ERROR("A default edge does not imply a dependency");
  }

  config->set_value(vistk::edge::config_dependency, "false");

  vistk::edge_t const edge2 = boost::make_shared<vistk::edge>(config);

  if (edge2->makes_dependency())
  {
    TEST_ERROR("Setting the dependency config to \'false\' "
               "was not reflected in the result");
  }
}

IMPLEMENT_TEST(new_has_no_data)
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  if (edge->has_data())
  {
    TEST_ERROR("A new edge has data in it");
  }
}

IMPLEMENT_TEST(new_is_not_full)
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  if (edge->full_of_data())
  {
    TEST_ERROR("A new edge is full of data");
  }
}

IMPLEMENT_TEST(new_has_count_zero)
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  if (edge->datum_count())
  {
    TEST_ERROR("A new edge has a count");
  }
}

IMPLEMENT_TEST(push_datum)
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  vistk::stamp::increment_t const inc = vistk::stamp::increment_t(1);

  vistk::datum_t const dat = vistk::datum::complete_datum();
  vistk::stamp_t const stamp = vistk::stamp::new_stamp(inc);

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

IMPLEMENT_TEST(peek_datum)
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  vistk::stamp::increment_t const inc = vistk::stamp::increment_t(1);

  vistk::datum_t const dat = vistk::datum::complete_datum();
  vistk::stamp_t const stamp = vistk::stamp::new_stamp(inc);

  vistk::edge_datum_t const edat = vistk::edge_datum_t(dat, stamp);

  edge->push_datum(edat);

  vistk::edge_datum_t const get_edat = edge->peek_datum();

  if (edge->datum_count() != 1)
  {
    TEST_ERROR("An edge removed a datum on an peek");
  }

  vistk::stamp_t const& estamp = get_edat.stamp;

  if (*estamp != *stamp)
  {
    TEST_ERROR("The edge modified a stamp on a peek");
  }
}

IMPLEMENT_TEST(peek_datum_index)
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  vistk::stamp::increment_t const inc = vistk::stamp::increment_t(1);

  vistk::datum_t const dat1 = vistk::datum::empty_datum();
  vistk::datum_t const dat2 = vistk::datum::complete_datum();
  vistk::stamp_t const stamp1 = vistk::stamp::new_stamp(inc);
  vistk::stamp_t const stamp2 = vistk::stamp::incremented_stamp(stamp1);

  vistk::edge_datum_t const edat1 = vistk::edge_datum_t(dat1, stamp1);
  vistk::edge_datum_t const edat2 = vistk::edge_datum_t(dat2, stamp2);

  edge->push_datum(edat1);
  edge->push_datum(edat2);

  vistk::edge_datum_t const get_edat = edge->peek_datum(1);

  if (edge->datum_count() != 2)
  {
    TEST_ERROR("An edge removed a datum on an indexed peek");
  }

  vistk::stamp_t const& estamp = get_edat.stamp;

  if (*estamp != *stamp2)
  {
    TEST_ERROR("The edge modified a stamp on a peek");
  }

  vistk::datum_t const& edatum = get_edat.datum;

  if (edatum != dat2)
  {
    TEST_ERROR("The edge modified a datum on a peek");
  }
}

IMPLEMENT_TEST(pop_datum)
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  vistk::stamp::increment_t const inc = vistk::stamp::increment_t(1);

  vistk::datum_t const dat = vistk::datum::complete_datum();
  vistk::stamp_t const stamp = vistk::stamp::new_stamp(inc);

  vistk::edge_datum_t const edat = vistk::edge_datum_t(dat, stamp);

  edge->push_datum(edat);

  edge->pop_datum();

  if (edge->datum_count())
  {
    TEST_ERROR("An edge did not remove a datum on a pop");
  }
}

IMPLEMENT_TEST(get_datum)
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  vistk::stamp::increment_t const inc = vistk::stamp::increment_t(1);

  vistk::datum_t const dat = vistk::datum::complete_datum();
  vistk::stamp_t const stamp = vistk::stamp::new_stamp(inc);

  vistk::edge_datum_t const edat = vistk::edge_datum_t(dat, stamp);

  edge->push_datum(edat);

  vistk::edge_datum_t const get_edat = edge->get_datum();

  if (edge->datum_count() != 0)
  {
    TEST_ERROR("An edge did not remove a datum on a get");
  }

  vistk::stamp_t const& estamp = get_edat.stamp;

  if (*estamp != *stamp)
  {
    TEST_ERROR("The edge modified a stamp on a get");
  }
}

IMPLEMENT_TEST(null_upstream_process)
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  vistk::process_t const process;

  EXPECT_EXCEPTION(vistk::null_process_connection_exception,
                   edge->set_upstream_process(process),
                   "setting a NULL process as upstream");
}

IMPLEMENT_TEST(null_downstream_process)
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  vistk::process_t const process;

  EXPECT_EXCEPTION(vistk::null_process_connection_exception,
                   edge->set_downstream_process(process),
                   "setting a NULL process as downstream");
}

IMPLEMENT_TEST(set_upstream_process)
{
  vistk::load_known_modules();

  vistk::process_registry_t const reg = vistk::process_registry::self();
  vistk::process::type_t const proc_type = vistk::process::type_t("numbers");

  vistk::process_t const process = reg->create_process(proc_type, vistk::process::name_t());

  vistk::edge_t const edge = boost::make_shared<vistk::edge>();

  edge->set_upstream_process(process);

  EXPECT_EXCEPTION(vistk::input_already_connected_exception,
                   edge->set_upstream_process(process),
                   "setting a second process as upstream");
}

IMPLEMENT_TEST(set_downstream_process)
{
  vistk::load_known_modules();

  vistk::process_registry_t const reg = vistk::process_registry::self();
  vistk::process::type_t const proc_type = vistk::process::type_t("numbers");

  vistk::process_t const process = reg->create_process(proc_type, vistk::process::name_t());

  vistk::edge_t const edge = boost::make_shared<vistk::edge>();

  edge->set_downstream_process(process);

  EXPECT_EXCEPTION(vistk::output_already_connected_exception,
                   edge->set_downstream_process(process),
                   "setting a second process as downstream");
}

IMPLEMENT_TEST(push_data_into_complete)
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  vistk::stamp::increment_t const inc = vistk::stamp::increment_t(1);

  vistk::datum_t const dat = vistk::datum::complete_datum();
  vistk::stamp_t const stamp = vistk::stamp::new_stamp(inc);

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

IMPLEMENT_TEST(get_data_from_complete)
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

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

#define SECONDS_TO_WAIT 1
#define WAIT_DURATION boost::chrono::seconds(SECONDS_TO_WAIT)

static void push_datum(vistk::edge_t edge, vistk::edge_datum_t edat);

void
test_capacity()
{
  vistk::config_t const config = vistk::config::empty_config();

  vistk::config::value_t const value_capacity = boost::lexical_cast<vistk::config::value_t>(1);

  config->set_value(vistk::edge::config_capacity, value_capacity);

  vistk::edge_t const edge = boost::make_shared<vistk::edge>(config);

  vistk::stamp::increment_t const inc = vistk::stamp::increment_t(1);

  vistk::datum_t const dat1 = vistk::datum::empty_datum();
  vistk::datum_t const dat2 = vistk::datum::complete_datum();
  vistk::stamp_t const stamp1 = vistk::stamp::new_stamp(inc);
  vistk::stamp_t const stamp2 = vistk::stamp::incremented_stamp(stamp1);

  vistk::edge_datum_t const edat1 = vistk::edge_datum_t(dat1, stamp1);
  vistk::edge_datum_t const edat2 = vistk::edge_datum_t(dat2, stamp2);

  // Fill the edge.
  edge->push_datum(edat1);

  boost::thread thread = boost::thread(boost::bind(&push_datum, edge, edat2));

  // Give the other thread some time.
#if BOOST_VERSION < 105000
  boost::this_thread::sleep(boost::posix_time::seconds(SECONDS_TO_WAIT));
#else
  boost::this_thread::sleep_for(WAIT_DURATION);
#endif

  // Make sure the edge still is at capacity.
  if (edge->datum_count() != 1)
  {
    TEST_ERROR("A datum was pushed into a full edge");
  }

  // Let the other thread go (it should have been blocking).
  edge->get_datum();

  // Make sure the other thread completes.
  thread.join();

  // Make sure the edge still is at capacity.
  if (edge->datum_count() != 1)
  {
    TEST_ERROR("The other thread did not push into the edge");
  }
}

void
push_datum(vistk::edge_t edge, vistk::edge_datum_t edat)
{
  // This clock is used because it is both steady (which rules out system_clock)
  // and uses the wall time (which rules out thread_clock).
  typedef boost::chrono::process_real_cpu_clock time_clock_t;
  typedef time_clock_t::time_point time_point_t;
  typedef time_clock_t::duration duration_t;

  time_point_t const start = time_clock_t::now();

  // This should be blocking.
  edge->push_datum(edat);

  time_point_t const end = time_clock_t::now();

  duration_t const duration = end - start;

  static double const tolerance = 0.75;

  if (duration < (tolerance * WAIT_DURATION))
  {
    TEST_ERROR("It seems as though blocking did not "
               "occur when pushing into a full edge: "
               "expected to wait between "
               << tolerance * WAIT_DURATION << " and "
               << WAIT_DURATION << ", but waited for "
               << duration << " instead");
  }

  if (edge->datum_count() != 1)
  {
    TEST_ERROR("A datum was pushed into a full edge");
  }
}
