#include <viame/utilities/utilities_target_clfr.h>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>

TEST( target_classification, exact_ties_use_class_names )
{
  auto dot = std::make_shared< viame::detected_object_type >(
    std::vector< std::string >{ "zebra", "ant", "high", "bee" },
    std::vector< double >{ .4, .4, .9, .4 } );
  EXPECT_EQ( viame::core::ranked_class_names( dot, 3 ),
    ( std::vector< std::string >{ "high", "ant", "bee" } ) );
  EXPECT_EQ( viame::core::ranked_class_names( dot, 0 ),
    ( std::vector< std::string >{ "high", "ant", "bee", "zebra" } ) );
  dot->set_score( "zebra", std::nextafter( .4, 1.0 ) );
  EXPECT_EQ( viame::core::ranked_class_names( dot, 2 ),
    ( std::vector< std::string >{ "high", "zebra" } ) );
}

TEST( target_classification, missing_and_nonfinite_scores )
{
  EXPECT_TRUE( viame::core::ranked_class_names( nullptr, 5 ).empty() );
  auto dot = std::make_shared< viame::detected_object_type >(
    std::vector< std::string >{ "nan", "finite", "infinity" },
    std::vector< double >{ std::numeric_limits< double >::quiet_NaN(),
                          .5, std::numeric_limits< double >::infinity() } );
  EXPECT_EQ( viame::core::ranked_class_names( dot, 20 ),
    ( std::vector< std::string >{ "infinity", "finite", "nan" } ) );
}
