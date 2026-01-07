#include <gtest/gtest.h>
#include "ocv_kmedians.h"

using namespace viame;

TEST( KMediansTest, kmedians_is_robuts_to_outliers )
{
  int max_iter_count{ 50 }, attempts{ 5 }, K{ 2 };
  double eps{ 1.0 };
  cv::Mat labels, centers;

  // Create data with 10 close data and 1 outlier
  std::vector< cv::Point2f > data_points{ { 0,   0 }, // outlier
                                          { 1.1, 1.1 },
                                          { 0.8, 0.8 },
                                          { 1.0, 1.0 },
                                          { 1.2, 1.2 },
                                          { 0.9, 0.9 },
                                          { 2.1, 2.1 },
                                          { 2.2, 2.2 },
                                          { 2.0, 2.0 },
                                          { 1.9, 1.9 },
                                          { 1.8, 1.8 } };

  cv::Mat data( data_points );

  // Expect KMedian 2 to represent the 2 clusters and not the outlier
  auto compactness = kmedians( data, (int) K, labels,
                               cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, max_iter_count, eps ),
                               attempts, cv::KMEANS_PP_CENTERS, centers );


  // Expect medians center to be close to 1.0 and 2.0
  int i_c1 = centers.at< float >( 0, 0 ) < centers.at< float >( 1, 0 ) ? 0 : 1;
  auto i_c2 = 1 - i_c1;

  double tol{ 0.1 };

  std::stringstream ss;
  ss << "Compactness : " << compactness << std::endl;;
  EXPECT_NEAR( centers.at< float >( i_c1, 0 ), 1.0, tol ) << ss.str();
  EXPECT_NEAR( centers.at< float >( i_c2, 0 ), 2.0, tol ) << ss.str();
}