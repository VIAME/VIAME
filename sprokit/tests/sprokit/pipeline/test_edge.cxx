// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_common.h>

#include <vital/config/config_block.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/edge.h>
#include <sprokit/pipeline/edge_exception.h>
#include <sprokit/pipeline/process_factory.h>
#include <sprokit/pipeline/stamp.h>

#include <boost/chrono/chrono_io.hpp>
#include <boost/chrono/duration.hpp>
#include <boost/chrono/process_cpu_clocks.hpp>
// XXX(boost): 1.50.0
#if BOOST_VERSION < 105000
#include <boost/date_time/posix_time/posix_time.hpp>
#endif

#include <boost/thread/thread.hpp>

#include <memory>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

IMPLEMENT_TEST(edge_datum_equal)
{
  sprokit::edge_datum_t edat1 = sprokit::edge_datum_t();
  sprokit::edge_datum_t edat2 = sprokit::edge_datum_t();

  if (edat1 != edat2)
  {
    TEST_ERROR("Empty edge data are not equivalent");
  }

  edat1.stamp = sprokit::stamp::new_stamp(1);
  edat2.stamp = sprokit::stamp::new_stamp(1);

  if (edat1 != edat2)
  {
    TEST_ERROR("Edge data with just a new stamp are not equivalent");
  }

  edat1.stamp = sprokit::stamp_t();
  edat2.stamp = sprokit::stamp_t();

  sprokit::datum_t const dat = sprokit::datum::complete_datum();

  edat1.datum = dat;
  edat2.datum = dat;

  if (edat1 != edat2)
  {
    TEST_ERROR("Edge data with just the same datum are not equivalent");
  }

  edat1.stamp = sprokit::stamp_t();
  edat2.stamp = sprokit::stamp_t();

  if (edat1 != edat2)
  {
    TEST_ERROR("Edge data with just the same datum and new stamps are not equivalent");
  }

  edat1.stamp = sprokit::stamp::new_stamp(1);
  edat1.stamp = sprokit::stamp::incremented_stamp(edat1.stamp);

  if (edat1 == edat2)
  {
    TEST_ERROR("Edge data with the same datum, but different stamps are equivalent");
  }

  edat1.stamp = sprokit::stamp::new_stamp(1);

  sprokit::datum_t const dat2 = sprokit::datum::complete_datum();

  edat1.datum = dat2;

  if (edat1 == edat2)
  {
    TEST_ERROR("Edge data with the same stamp, but different data (of the same type) are equivalent");
  }

  edat1.stamp = sprokit::stamp::incremented_stamp(edat1.stamp);

  if (edat1 == edat2)
  {
    TEST_ERROR("Edge data with different stamps and data are equivalent");
  }
}
IMPLEMENT_TEST(null_config)
{
  kwiver::vital::config_block_sptr const config;

  EXPECT_EXCEPTION(sprokit::null_edge_config_exception,
                   std::make_shared<sprokit::edge>(config),
                   "when passing a NULL config to an edge");
}

IMPLEMENT_TEST(makes_dependency)
{
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  if (!edge->makes_dependency())
  {
    TEST_ERROR("A default edge does not imply a dependency");
  }

  config->set_value(sprokit::edge::config_dependency, "false");

  sprokit::edge_t const edge2 = std::make_shared<sprokit::edge>(config);

  if (edge2->makes_dependency())
  {
    TEST_ERROR("Setting the dependency config to \'false\' "
               "was not reflected in the result");
  }
}

IMPLEMENT_TEST(new_has_no_data)
{
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  if (edge->has_data())
  {
    TEST_ERROR("A new edge has data in it");
  }
}

IMPLEMENT_TEST(new_is_not_full)
{
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  if (edge->full_of_data())
  {
    TEST_ERROR("A new edge is full of data");
  }
}

IMPLEMENT_TEST(new_has_count_zero)
{
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  if (edge->datum_count())
  {
    TEST_ERROR("A new edge has a count");
  }
}

IMPLEMENT_TEST(push_datum)
{
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  sprokit::stamp::increment_t const inc = sprokit::stamp::increment_t(1);

  sprokit::datum_t const dat = sprokit::datum::complete_datum();
  sprokit::stamp_t const stamp = sprokit::stamp::new_stamp(inc);

  sprokit::edge_datum_t const edat = sprokit::edge_datum_t(dat, stamp);

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
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  sprokit::stamp::increment_t const inc = sprokit::stamp::increment_t(1);

  sprokit::datum_t const dat = sprokit::datum::complete_datum();
  sprokit::stamp_t const stamp = sprokit::stamp::new_stamp(inc);

  sprokit::edge_datum_t const edat = sprokit::edge_datum_t(dat, stamp);

  edge->push_datum(edat);

  sprokit::edge_datum_t const get_edat = edge->peek_datum();

  if (edge->datum_count() != 1)
  {
    TEST_ERROR("An edge removed a datum on an peek");
  }

  sprokit::stamp_t const& estamp = get_edat.stamp;

  if (*estamp != *stamp)
  {
    TEST_ERROR("The edge modified a stamp on a peek");
  }
}

IMPLEMENT_TEST(peek_datum_index)
{
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  sprokit::stamp::increment_t const inc = sprokit::stamp::increment_t(1);

  sprokit::datum_t const dat1 = sprokit::datum::empty_datum();
  sprokit::datum_t const dat2 = sprokit::datum::complete_datum();
  sprokit::stamp_t const stamp1 = sprokit::stamp::new_stamp(inc);
  sprokit::stamp_t const stamp2 = sprokit::stamp::incremented_stamp(stamp1);

  sprokit::edge_datum_t const edat1 = sprokit::edge_datum_t(dat1, stamp1);
  sprokit::edge_datum_t const edat2 = sprokit::edge_datum_t(dat2, stamp2);

  edge->push_datum(edat1);
  edge->push_datum(edat2);

  sprokit::edge_datum_t const get_edat = edge->peek_datum(1);

  if (edge->datum_count() != 2)
  {
    TEST_ERROR("An edge removed a datum on an indexed peek");
  }

  sprokit::stamp_t const& estamp = get_edat.stamp;

  if (*estamp != *stamp2)
  {
    TEST_ERROR("The edge modified a stamp on a peek");
  }

  sprokit::datum_t const& edatum = get_edat.datum;

  if (edatum != dat2)
  {
    TEST_ERROR("The edge modified a datum on a peek");
  }
}

IMPLEMENT_TEST(pop_datum)
{
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  sprokit::stamp::increment_t const inc = sprokit::stamp::increment_t(1);

  sprokit::datum_t const dat = sprokit::datum::complete_datum();
  sprokit::stamp_t const stamp = sprokit::stamp::new_stamp(inc);

  sprokit::edge_datum_t const edat = sprokit::edge_datum_t(dat, stamp);

  edge->push_datum(edat);

  edge->pop_datum();

  if (edge->datum_count())
  {
    TEST_ERROR("An edge did not remove a datum on a pop");
  }
}

IMPLEMENT_TEST(get_datum)
{
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  sprokit::stamp::increment_t const inc = sprokit::stamp::increment_t(1);

  sprokit::datum_t const dat = sprokit::datum::complete_datum();
  sprokit::stamp_t const stamp = sprokit::stamp::new_stamp(inc);

  sprokit::edge_datum_t const edat = sprokit::edge_datum_t(dat, stamp);

  edge->push_datum(edat);

  sprokit::edge_datum_t const get_edat = edge->get_datum();

  if (edge->datum_count() != 0)
  {
    TEST_ERROR("An edge did not remove a datum on a get");
  }

  sprokit::stamp_t const& estamp = get_edat.stamp;

  if (*estamp != *stamp)
  {
    TEST_ERROR("The edge modified a stamp on a get");
  }
}

IMPLEMENT_TEST(null_upstream_process)
{
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  sprokit::process_t const process;

  EXPECT_EXCEPTION(sprokit::null_process_connection_exception,
                   edge->set_upstream_process(process),
                   "setting a NULL process as upstream");
}

IMPLEMENT_TEST(null_downstream_process)
{
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  sprokit::process_t const process;

  EXPECT_EXCEPTION(sprokit::null_process_connection_exception,
                   edge->set_downstream_process(process),
                   "setting a NULL process as downstream");
}

IMPLEMENT_TEST(set_upstream_process)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  sprokit::process::type_t const proc_type = sprokit::process::type_t("numbers");

  sprokit::process_t const process = sprokit::create_process(proc_type, sprokit::process::name_t());

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>();

  edge->set_upstream_process(process);

  EXPECT_EXCEPTION(sprokit::input_already_connected_exception,
                   edge->set_upstream_process(process),
                   "setting a second process as upstream");
}

IMPLEMENT_TEST(set_downstream_process)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  sprokit::process::type_t const proc_type = sprokit::process::type_t("numbers");

  sprokit::process_t const process = sprokit::create_process(proc_type, sprokit::process::name_t());

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>();

  edge->set_downstream_process(process);

  EXPECT_EXCEPTION(sprokit::output_already_connected_exception,
                   edge->set_downstream_process(process),
                   "setting a second process as downstream");
}

IMPLEMENT_TEST(push_data_into_complete)
{
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  sprokit::stamp::increment_t const inc = sprokit::stamp::increment_t(1);

  sprokit::datum_t const dat = sprokit::datum::complete_datum();
  sprokit::stamp_t const stamp = sprokit::stamp::new_stamp(inc);

  sprokit::edge_datum_t const edat = sprokit::edge_datum_t(dat, stamp);

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
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  edge->mark_downstream_as_complete();

  EXPECT_EXCEPTION(sprokit::datum_requested_after_complete,
                   edge->peek_datum(),
                   "peeking at a complete edge");

  EXPECT_EXCEPTION(sprokit::datum_requested_after_complete,
                   edge->get_datum(),
                   "getting data from a complete edge");

  EXPECT_EXCEPTION(sprokit::datum_requested_after_complete,
                   edge->pop_datum(),
                   "popping data from a complete edge");
}

namespace
{

// This clock is used because it is both steady (which rules out system_clock)
// and uses the wall time (which rules out thread_clock).
typedef boost::chrono::process_real_cpu_clock time_clock_t;
typedef time_clock_t::time_point time_point_t;
typedef time_clock_t::duration duration_t;

}
#define SECONDS_TO_WAIT 1
#define WAIT_DURATION boost::chrono::seconds(SECONDS_TO_WAIT)

static void push_datum(sprokit::edge_t edge, sprokit::edge_datum_t edat);

IMPLEMENT_TEST(capacity)
{
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  kwiver::vital::config_block_value_t const value_capacity = "1";

  config->set_value(sprokit::edge::config_capacity, value_capacity);

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  sprokit::stamp::increment_t const inc = sprokit::stamp::increment_t(1);

  sprokit::datum_t const dat1 = sprokit::datum::empty_datum();
  sprokit::datum_t const dat2 = sprokit::datum::complete_datum();
  sprokit::stamp_t const stamp1 = sprokit::stamp::new_stamp(inc);
  sprokit::stamp_t const stamp2 = sprokit::stamp::incremented_stamp(stamp1);

  sprokit::edge_datum_t const edat1 = sprokit::edge_datum_t(dat1, stamp1);
  sprokit::edge_datum_t const edat2 = sprokit::edge_datum_t(dat2, stamp2);

  // Fill the edge.
  edge->push_datum(edat1);

  boost::thread thread = boost::thread(std::bind(&push_datum, edge, edat2));

  // Give the other thread some time.
  // XXX(boost): 1.50.0
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

static void check_time(duration_t const& actual, duration_t const& expected, char const* const message);

IMPLEMENT_TEST(try_push_datum)
{
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();

  config->set_value(sprokit::edge::config_capacity, 1);

  sprokit::edge_t const edge = std::make_shared<sprokit::edge>(config);

  sprokit::stamp::increment_t const inc = sprokit::stamp::increment_t(1);

  sprokit::datum_t const dat1 = sprokit::datum::empty_datum();
  sprokit::datum_t const dat2 = sprokit::datum::complete_datum();
  sprokit::stamp_t const stamp1 = sprokit::stamp::new_stamp(inc);
  sprokit::stamp_t const stamp2 = sprokit::stamp::incremented_stamp(stamp1);

  sprokit::edge_datum_t const edat1 = sprokit::edge_datum_t(dat1, stamp1);
  sprokit::edge_datum_t const edat2 = sprokit::edge_datum_t(dat2, stamp2);

  // Fill the edge.
  edge->push_datum(edat1);

  time_point_t const start = time_clock_t::now();

  // This should be blocking.
  bool const pushed = edge->try_push_datum(edat2, WAIT_DURATION);

  time_point_t const end = time_clock_t::now();
  if (pushed)
  {
    TEST_ERROR("Returned true when a push should have timed out");
  }

  duration_t const duration = end - start;

  check_time(duration, WAIT_DURATION, "trying to get a datum from an edge");

  // Make sure the edge still is at capacity.
  if (edge->datum_count() != 1)
  {
    TEST_ERROR("A datum was pushed into a full edge");
  }
}

IMPLEMENT_TEST(try_get_datum)
{
  sprokit::edge_t const edge = std::make_shared<sprokit::edge>();

  time_point_t const start = time_clock_t::now();

  // This should be blocking.
  kwiver::vital::optional<sprokit::edge_datum_t> const opt_datum = edge->try_get_datum(WAIT_DURATION);

  time_point_t const end = time_clock_t::now();

  if (opt_datum)
  {
    TEST_ERROR("Returned a datum from an empty edge");
  }

  duration_t const duration = end - start;

  check_time(duration, WAIT_DURATION, "trying to get a datum from an edge");
}

void
push_datum(sprokit::edge_t edge, sprokit::edge_datum_t edat)
{
  time_point_t const start = time_clock_t::now();

  // This should be blocking.
  edge->push_datum(edat);

  time_point_t const end = time_clock_t::now();

  duration_t const duration = end - start;

  check_time(duration, WAIT_DURATION, "pushing into a full edge");

  if (edge->datum_count() != 1)
  {
    TEST_ERROR("A datum was pushed into a full edge");
  }
}

void
check_time(duration_t const& actual, duration_t const& expected, char const* const message)
{
  static double const tolerance = 0.75;
  boost::chrono::duration<double> const allowed = tolerance * WAIT_DURATION;

  if (actual < allowed)
  {
    TEST_ERROR("It seems as though blocking did not "
               "occur when " << message << ": "
               "expected to wait between "
               << allowed << " and "
               << expected << ", but waited for "
               << actual << " instead");
  }
}
