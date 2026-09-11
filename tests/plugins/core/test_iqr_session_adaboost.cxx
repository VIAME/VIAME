/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief What the IQR AdaBoost session learns and how it ranks
///
/// `iqr_session_adaboost` is the last caller of `library/opencv_bridge`'s
/// dependency -- `cv::ml::Boost` -- and P7-T09 replaces it with sklearn.
/// That is a **different algorithm**, not a port: OpenCV boosts CART trees
/// with DISCRETE, REAL, LOGIT or GENTLE AdaBoost, and sklearn's
/// `AdaBoostClassifier` is SAMME. The numbers will move and the model blob
/// changes format. The decision to accept that is the user's, recorded in
/// `design/STATUS.md`.
///
/// So this file is two things at once. The **assertions** are what has to
/// survive either implementation: a session that has been refined has a
/// model, ranks its own positives above its own negatives, round trips
/// through its serialised bytes without changing a prediction, and falls
/// back to similarity when it has no model. The **recording** beside it is
/// the numbers, and the diff in that file when the implementation changes is
/// the behaviour change, in the open, in the commit that makes it.
///
///     VIAME_RECORD_IQR_ADABOOST=1 ./tests/bin/test-viame_core-iqr_session_adaboost
///
/// which writes the file and fails, so a regeneration is never accidental.

#include <gtest/gtest.h>

#include "core/iqr_session_adaboost.h"

#include <pybind11/embed.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ----------------------------------------------------------------------------
std::string
expected_path()
{
#ifdef VIAME_IQR_ADABOOST_DIR
  return std::string( VIAME_IQR_ADABOOST_DIR ) + "/iqr_session_adaboost.txt";
#else
  return "iqr_session_adaboost.txt";
#endif
}

// The descriptors are sixteen dimensional, which is small enough to read in
// a recording and wide enough that a stump has a choice of feature.
constexpr size_t DIMENSION = 16;
constexpr unsigned SEED = 20260911;

// Half a standard deviation on six of the sixteen dimensions: the classes
// genuinely overlap, so the weak learners disagree and the margin is a range
// rather than its two endpoints.
constexpr double OFFSET = 0.5;

// Per side. Twenty a side is enough for the area under the ROC curve below
// to mean something and few enough to read in the recording.
constexpr size_t PROBES = 20;

// ----------------------------------------------------------------------------
/// Two overlapping clusters: a class boundary a booster can find and not
/// trivially, which is what makes the ranking say something.
///
/// The mean differs on the first six dimensions only, so ten of the sixteen
/// are noise a split on them would be wrong to take.
///
/// The offset is small on purpose. Clusters far enough apart to be linearly
/// separable make **every** weak learner agree on **every** sample, and then
/// the ensemble margin saturates at its extreme for all of them -- which
/// looks exactly like the two-valued score this test exists to rule out. A
/// fixture that cannot tell a graded score from a constant one is no test of
/// a ranking.
std::vector< viame::iqr::descriptor_element >
make_cluster( const std::string& prefix, size_t count, double offset,
              std::mt19937& rng )
{
  std::normal_distribution< double > noise( 0.0, 1.0 );

  std::vector< viame::iqr::descriptor_element > out;

  for( size_t i = 0; i < count; ++i )
  {
    std::vector< double > vector( DIMENSION );

    for( size_t j = 0; j < DIMENSION; ++j )
    {
      vector[j] = noise( rng ) + ( j < 6 ? offset : 0.0 );
    }

    out.emplace_back( prefix + "_" + std::to_string( i ), vector );
  }

  return out;
}

// ----------------------------------------------------------------------------
struct scenario
{
  std::vector< viame::iqr::descriptor_element > positives;
  std::vector< viame::iqr::descriptor_element > negatives;
  std::vector< viame::iqr::descriptor_element > probes;
};

// ----------------------------------------------------------------------------
scenario
build_scenario()
{
  std::mt19937 rng( SEED );

  scenario out;
  out.positives = make_cluster( "pos", 40, OFFSET, rng );
  out.negatives = make_cluster( "neg", 40, -OFFSET, rng );

  // Held-out probes, positives first, so a recording of the scores says
  // which way the ranking points as well as what the numbers are
  auto probe_positive = make_cluster( "probe_pos", PROBES, OFFSET, rng );
  auto probe_negative = make_cluster( "probe_neg", PROBES, -OFFSET, rng );

  out.probes = probe_positive;
  out.probes.insert( out.probes.end(),
                     probe_negative.begin(), probe_negative.end() );

  return out;
}

// ----------------------------------------------------------------------------
/// One configured session, refined on the scenario.
std::unique_ptr< viame::iqr::iqr_session_adaboost >
refined_session( const scenario& data, const std::string& boost_type,
                 int weak_count, int max_depth, double trim_rate )
{
  auto session =
    std::make_unique< viame::iqr::iqr_session_adaboost >( 0u );

  session->set_boost_type( boost_type );
  session->set_weak_count( weak_count );
  session->set_max_depth( max_depth );
  session->set_weight_trim_rate( trim_rate );

  session->adjudicate( data.positives, data.negatives );

  return session;
}

// ----------------------------------------------------------------------------
/// Six significant figures, which is more than a ranking needs and few
/// enough that a recording is not a record of the last bit of a double.
std::string
format( double value )
{
  std::ostringstream out;
  out << std::fixed << std::setprecision( 6 ) << value;
  return out.str();
}

// ----------------------------------------------------------------------------
struct variant
{
  std::string name;
  std::string boost_type;
  int weak_count;
  int max_depth;
  double trim_rate;
};

// The shipped pipeline's configuration is `gentle`, 100, 1, 0.95.
const std::vector< variant > VARIANTS = {
  { "shipped", "gentle", 100, 1, 0.95 },
  { "discrete", "discrete", 100, 1, 0.95 },
  { "real", "real", 100, 1, 0.95 },
  { "logit", "logit", 100, 1, 0.95 },
  { "weak_count_10", "gentle", 10, 1, 0.95 },
  { "max_depth_3", "gentle", 100, 3, 0.95 },
  { "no_trim", "gentle", 100, 1, 0.0 },
};

// ----------------------------------------------------------------------------
/// The session trains and scores through `viame.core.iqr_adaboost`, so the
/// test has to provide an interpreter the way a VIAME pipeline does -- there
/// the plugin loader brings python up before any process runs, and a bare
/// gtest has nobody to do that. Without one the session reports no model and
/// every assertion below falls through to the similarity fallback, which is
/// correct behaviour and no test of anything.
class python_environment : public ::testing::Environment
{
public:
  void SetUp() override
  {
    if( Py_IsInitialized() == 0 )
    {
      m_interpreter =
        std::make_unique< pybind11::scoped_interpreter >();
    }

#ifdef VIAME_PYTHON_PACKAGES
    // ctest runs a discovered gtest straight, without sourcing the install's
    // setup script, so `viame.core` is not on the path the way it is for
    // every python test and for anything running in a VIAME pipeline. Put
    // the install's site-packages on it rather than teach the whole gtest
    // harness about environments for one test's sake.
    pybind11::module_::import( "sys" ).attr( "path" ).attr( "insert" )(
      0, VIAME_PYTHON_PACKAGES );
#endif
  }

  void TearDown() override
  {
    // Deliberately **not** finalised. `Py_Finalize` with numpy and
    // scikit-learn's extension modules loaded segfaults on the way out --
    // the test passes and the process dies afterwards, which ctest reports
    // as a failure of the test. Leaking an interpreter that is about to be
    // torn down by exit costs nothing.
    ( void ) m_interpreter.release();
  }

private:
  std::unique_ptr< pybind11::scoped_interpreter > m_interpreter;
};

} // namespace

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  ::testing::AddGlobalTestEnvironment( new python_environment() );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
class iqr_adaboost_test : public ::testing::Test
{
protected:
  scenario m_data = build_scenario();
};

// ----------------------------------------------------------------------------
// The assertions: true of any implementation of this interface
// ----------------------------------------------------------------------------

TEST_F( iqr_adaboost_test, an_unrefined_session_has_no_model )
{
  viame::iqr::iqr_session_adaboost session( 0u );

  EXPECT_FALSE( session.is_model_valid() );

  // With no model it falls back to similarity against the positives, which
  // is the base class's behaviour and has to keep working: a query with no
  // negatives adjudicated yet takes this path every time.
  session.adjudicate( m_data.positives, {} );

  const double near = session.predict_score( m_data.probes.front().vector );
  const double far = session.predict_score( m_data.probes.back().vector );

  EXPECT_FALSE( session.is_model_valid() );
  EXPECT_GT( near, far );
}

TEST_F( iqr_adaboost_test, refining_trains_a_model )
{
  auto session = refined_session( m_data, "gentle", 100, 1, 0.95 );

  ASSERT_TRUE( session->refine() );
  EXPECT_TRUE( session->is_model_valid() );
}

// ----------------------------------------------------------------------------
/// The area under the ROC curve of a ranking: the probability that a
/// randomly chosen positive outranks a randomly chosen negative.
///
/// The right measure for this session, because the process sorts by score
/// and returns the top of the list -- what matters is the order, not where
/// the scores sit.
double
ranking_auc( const std::vector< double >& positives,
             const std::vector< double >& negatives )
{
  size_t better = 0;
  size_t tied = 0;

  for( const double p : positives )
  {
    for( const double n : negatives )
    {
      if( p > n ) { ++better; }
      else if( p == n ) { ++tied; }
    }
  }

  const double pairs =
    static_cast< double >( positives.size() * negatives.size() );

  return ( static_cast< double >( better ) + 0.5 * tied ) / pairs;
}

TEST_F( iqr_adaboost_test, the_model_ranks_positives_above_negatives )
{
  auto session = refined_session( m_data, "gentle", 100, 1, 0.95 );
  ASSERT_TRUE( session->refine() );

  std::vector< double > positives;
  std::vector< double > negatives;

  for( size_t i = 0; i < PROBES; ++i )
  {
    positives.push_back( session->predict_score( m_data.probes[i].vector ) );
  }

  for( size_t i = PROBES; i < m_data.probes.size(); ++i )
  {
    negatives.push_back( session->predict_score( m_data.probes[i].vector ) );
  }

  // Not "every positive beats every negative": the clusters overlap on
  // purpose and a probe drawn from the wrong tail is the data, not a defect.
  // What has to hold is that the ordering is strongly right.
  EXPECT_GT( ranking_auc( positives, negatives ), 0.9 );
}

// The defect this port fixes, pinned as a test.
//
// `cv::ml::Boost::predict` was asked for `cv::ml::StatModel::RAW_OUTPUT`
// meaning to get the weighted sum over the weak classifiers. That flag alone
// returns the **class label** instead -- `DTrees::PREDICT_SUM` is what asks
// for the sum, and nothing passed it. So `predict_distance` returned 0 or 1,
// `predict_score` returned 0.5 or 0.731059, and `ordered_results()`, whose
// whole job is to score the working index and sort it, had two values to
// sort by. The order within each half was whatever the hash map gave.
//
// scikit-learn's `decision_function` returns a real margin.
TEST_F( iqr_adaboost_test, the_score_is_graded_rather_than_two_valued )
{
  auto session = refined_session( m_data, "gentle", 100, 1, 0.95 );
  ASSERT_TRUE( session->refine() );

  std::set< double > distinct;

  for( const auto& probe : m_data.probes )
  {
    distinct.insert( session->predict_score( probe.vector ) );
  }

  // Forty probes over overlapping classes: a ranking worth the name puts
  // them at many different heights, not two.
  EXPECT_GT( distinct.size(), 2u );
  EXPECT_GT( distinct.size(), m_data.probes.size() / 2 );
}

TEST_F( iqr_adaboost_test, a_model_round_trips_through_its_bytes )
{
  auto session = refined_session( m_data, "gentle", 100, 1, 0.95 );
  ASSERT_TRUE( session->refine() );

  const auto bytes = session->get_model_bytes();
  ASSERT_FALSE( bytes.empty() );

  std::vector< double > before;
  for( const auto& probe : m_data.probes )
  {
    before.push_back( session->predict_distance( probe.vector ) );
  }

  // A second session, given only the bytes, predicts the same thing. This is
  // what the process relies on to carry a model between query iterations.
  viame::iqr::iqr_session_adaboost restored( 0u );
  ASSERT_TRUE( restored.load_model_from_bytes( bytes ) );
  ASSERT_TRUE( restored.is_model_valid() );

  for( size_t i = 0; i < m_data.probes.size(); ++i )
  {
    EXPECT_DOUBLE_EQ( before[i],
      restored.predict_distance( m_data.probes[i].vector ) )
      << "probe " << i;
  }
}

TEST_F( iqr_adaboost_test, freeing_a_model_returns_to_the_fallback )
{
  auto session = refined_session( m_data, "gentle", 100, 1, 0.95 );
  ASSERT_TRUE( session->refine() );
  ASSERT_TRUE( session->is_model_valid() );

  session->free_model();

  EXPECT_FALSE( session->is_model_valid() );
  EXPECT_TRUE( session->get_model_bytes().empty() );
  EXPECT_DOUBLE_EQ( 0.0,
    session->predict_distance( m_data.probes.front().vector ) );
}

TEST_F( iqr_adaboost_test, training_twice_gives_the_same_model )
{
  auto first = refined_session( m_data, "gentle", 100, 1, 0.95 );
  auto second = refined_session( m_data, "gentle", 100, 1, 0.95 );

  ASSERT_TRUE( first->refine() );
  ASSERT_TRUE( second->refine() );

  // Determinism, without which the recording below would be meaningless
  for( const auto& probe : m_data.probes )
  {
    EXPECT_DOUBLE_EQ( first->predict_distance( probe.vector ),
                      second->predict_distance( probe.vector ) );
  }
}

// ----------------------------------------------------------------------------
// The recording: the numbers, so a change of implementation shows its work
// ----------------------------------------------------------------------------

TEST_F( iqr_adaboost_test, the_scores_are_what_they_were )
{
  std::ostringstream actual;

  for( const auto& config : VARIANTS )
  {
    auto session = refined_session( m_data, config.boost_type,
      config.weak_count, config.max_depth, config.trim_rate );

    const bool refined = session->refine();

    actual << "=== " << config.name << " ===\n";
    actual << "boost_type=" << config.boost_type
           << " weak_count=" << config.weak_count
           << " max_depth=" << config.max_depth
           << " weight_trim_rate=" << format( config.trim_rate ) << "\n";
    actual << "refined=" << ( refined ? "true" : "false" )
           << " model_valid=" << ( session->is_model_valid() ? "true" : "false" )
           << "\n";

    actual << "probe,distance,score\n";

    for( const auto& probe : m_data.probes )
    {
      actual << probe.uid << ","
             << format( session->predict_distance( probe.vector ) ) << ","
             << format( session->predict_score( probe.vector ) ) << "\n";
    }

    actual << "\n";
  }

  if( std::getenv( "VIAME_RECORD_IQR_ADABOOST" ) )
  {
    std::ofstream file( expected_path() );
    ASSERT_TRUE( file.is_open() ) << "cannot write " << expected_path();
    file << actual.str();
    file.close();

    FAIL() << "Recorded " << expected_path() << ". Unset "
              "VIAME_RECORD_IQR_ADABOOST and run again.";
  }

  std::ifstream expected_file( expected_path() );
  ASSERT_TRUE( expected_file.is_open() )
    << "no recording at " << expected_path() << "; regenerate with "
       "VIAME_RECORD_IQR_ADABOOST=1";

  std::ostringstream expected;
  expected << expected_file.rdbuf();

  EXPECT_EQ( expected.str(), actual.str() );
}
