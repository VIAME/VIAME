// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Tests involving OCV nested algorithms and their parameter
 *        interactions with kwiver::vital::config_block objects.
 */

#include <test_gtest.h>

// Get headers of optional algos for ``MAPTK_OCV_HAS_*`` defines
#include <arrows/ocv/detect_features_AGAST.h>
#include <arrows/ocv/detect_features_MSD.h>
#include <arrows/ocv/detect_features_STAR.h>
#include <arrows/ocv/extract_descriptors_BRIEF.h>
#include <arrows/ocv/extract_descriptors_DAISY.h>
#include <arrows/ocv/extract_descriptors_FREAK.h>
#include <arrows/ocv/extract_descriptors_LATCH.h>
#include <arrows/ocv/extract_descriptors_LUCID.h>
#include <arrows/ocv/feature_detect_extract_SIFT.h>
#include <arrows/ocv/feature_detect_extract_SURF.h>

#include <arrows/ocv/detect_features.h>
#include <arrows/ocv/extract_descriptors.h>
#include <arrows/ocv/match_features.h>

#include <vital/exceptions.h>
#include <vital/logger/logger.h>
#include <vital/vital_types.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <opencv2/core/core.hpp>

#include <iostream>

using namespace kwiver::vital;
using namespace kwiver::arrows;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  TEST_LOAD_PLUGINS();
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
struct algorithm_test
{
  char const* const name;
  std::function<algorithm_sptr ()> factory;
};

// ----------------------------------------------------------------------------
void
PrintTo( algorithm_test const& v, ::std::ostream* os )
{
  (*os) << v.name;
}

// ----------------------------------------------------------------------------
class algo_config : public ::testing::TestWithParam<algorithm_test>
{
};

// ----------------------------------------------------------------------------
// Test that we can get, set and check the configurations for OCV feature
// detector implementations
TEST_P(algo_config, defaults)
{
  auto a = GetParam().factory();

  logger_handle_t log = get_logger(
    "arrows.test.plugins.ocv.ocv_algo_config_defaults" );

  LOG_INFO(log, "Testing configuration for algorithm instance @" << a.get() );
  LOG_INFO(log, "-- Algorithm info: " << a->type_name() << "::"
                << a->impl_name() );
  kwiver::vital::config_block_sptr c = a->get_configuration();
  LOG_INFO(log, "-- default config:");
  for ( auto const& key : c->available_values() )
  {
    LOG_INFO( log, "\t// " << c->get_description(key) << "\n" <<
                   "\t" << key << " = " <<
                   c->get_value<kwiver::vital::config_block_key_t>( key ) );
  }

  // Checking and setting the config of algo. Default should always be valid
  // thus passing check.
  LOG_INFO(log, "-- checking default config");
  EXPECT_TRUE( a->check_configuration( c ) );

  LOG_INFO(log, "-- Setting default config and checking again");
  a->set_configuration( c );
  EXPECT_TRUE( a->check_configuration( c ) );
}

// ----------------------------------------------------------------------------
// Test that setting and checking and empty configuration block is a valid
// operation
TEST_P(algo_config, empty_config)
{
  auto a = GetParam().factory();

  logger_handle_t log = get_logger(
    "arrows.test.plugins.ocv.algo_empty_config" );

  // Checking an empty config. Since there is literally nothing in the config,
  // we should pass here, as the default configuration should be used which
  // should pass (see test "ocv_algo_config_defaults")
  LOG_INFO(log, "Checking empty config for algorithm instance @" << a.get() );
  LOG_INFO(log, "-- Algorithm info: " << a->type_name() << "::"
                << a->impl_name() );
  kwiver::vital::config_block_sptr
    empty_conf = kwiver::vital::config_block::empty_config();
  EXPECT_TRUE( a->check_configuration( empty_conf ) )
    << a->type_name() + "::" + a->impl_name();

  // Should be able to set an empty config as defaults should take over.
  LOG_INFO(log, "-- setting empty config");
  a->set_configuration(empty_conf);

  // This should also pass as we take an empty type as a "use the default"
  // message
  EXPECT_TRUE( a->check_configuration( a->get_configuration() ) );
}

// ----------------------------------------------------------------------------
#define ALGORITHM( t, n ) \
  algorithm_test{ n, []{ return algo::t::create( n ); } }

auto detect_features_algorithms = []()
{
  return ::testing::Values(
      ALGORITHM( detect_features, "ocv_BRISK" )
    , ALGORITHM( detect_features, "ocv_FAST" )
    , ALGORITHM( detect_features, "ocv_GFTT" )
    , ALGORITHM( detect_features, "ocv_MSER" )
    , ALGORITHM( detect_features, "ocv_ORB" )
    , ALGORITHM( detect_features, "ocv_simple_blob" )

#ifdef KWIVER_OCV_HAS_AGAST
    , ALGORITHM( detect_features, "ocv_AGAST" )
#endif

#ifdef KWIVER_OCV_HAS_SIFT
    , ALGORITHM( detect_features, "ocv_SIFT" )
#endif

#ifdef KWIVER_OCV_HAS_STAR
    , ALGORITHM( detect_features, "ocv_STAR" )
#endif

#ifdef KWIVER_OCV_HAS_SURF
    , ALGORITHM( detect_features, "ocv_SURF" )
#endif
  );
};

INSTANTIATE_TEST_CASE_P(
  detect_features,
  algo_config,
  detect_features_algorithms()
);

INSTANTIATE_TEST_CASE_P(
  match_features,
  algo_config,
  ::testing::Values(
      ALGORITHM( match_features, "ocv_brute_force" )
    , ALGORITHM( match_features, "ocv_flann_based" )
  )
);

auto extract_descriptors_algorithms = []()
{
  return ::testing::Values(
      ALGORITHM( extract_descriptors, "ocv_BRISK" )
    , ALGORITHM( extract_descriptors, "ocv_ORB" )

#ifdef KWIVER_OCV_HAS_BRIEF
    , ALGORITHM( extract_descriptors, "ocv_BRIEF" )
#endif

#ifdef KWIVER_OCV_HAS_DAISY
    , ALGORITHM( extract_descriptors, "ocv_DAISY" )
#endif

#ifdef KWIVER_OCV_HAS_FREAK
    , ALGORITHM( extract_descriptors, "ocv_FREAK" )
#endif

#ifdef KWIVER_OCV_HAS_LATCH
    , ALGORITHM( extract_descriptors, "ocv_LATCH" )
#endif

#ifdef KWIVER_OCV_HAS_LUCID
    , ALGORITHM( extract_descriptors, "ocv_LUCID" )
#endif

#ifdef KWIVER_OCV_HAS_SIFT
    , ALGORITHM( extract_descriptors, "ocv_SIFT" )
#endif

#ifdef KWIVER_OCV_HAS_SURF
    , ALGORITHM( extract_descriptors, "ocv_SURF" )
#endif
  );
};

INSTANTIATE_TEST_CASE_P(
  extract_descriptors,
  algo_config,
  extract_descriptors_algorithms()
);
