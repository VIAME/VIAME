// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief tests comparing MAP-Tk lens distortion to OpenCV
 */

#include <test_eigen.h>
#include <test_random_point.h>

#include <vital/types/camera_intrinsics.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <arrows/ocv/camera_intrinsics.h>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
static void test_distortion(const Eigen::VectorXd& d)
{
  simple_camera_intrinsics K;
  K.set_dist_coeffs(d.transpose());

  cv::Mat cam_mat;
  cv::eigen2cv(K.as_matrix(), cam_mat);

  cv::Mat d_cvmat;
  cv::eigen2cv(d, d_cvmat);

  std::vector<double> cv_dist = kwiver::arrows::ocv::dist_coeffs_to_ocv(d_cvmat);

  cv::Mat dist = cv::Mat(cv_dist);

  cv::Mat tvec = cv::Mat::zeros(1,3,CV_64F);
  cv::Mat rvec = tvec;

  std::cout << "K = " << cam_mat << "\nd = " << dist <<std::endl;

  for(unsigned int i = 1; i<100; ++i)
  {
    vector_2d test_pt = kwiver::testing::random_point2d(0.5);
    // the distortion model is only valid within about a unit distance
    // from the origin in normalized coordinates
    // project distant points back onto the unit circle
    if (test_pt.norm() > 1.0)
    {
      test_pt.normalize();
    }
    vector_2d distorted_test = K.distort(test_pt);
    std::cout << "test point: "<< test_pt.transpose() << "\n"
              << "distorted: " << distorted_test.transpose() << std::endl;
    std::vector<cv::Point3d> in_pts;
    std::vector<cv::Point2d> out_pts;
    in_pts.push_back(cv::Point3d(test_pt.x(), test_pt.y(), 1.0));
    cv::projectPoints(in_pts, rvec, tvec, cam_mat, dist, out_pts);
    vector_2d cv_distorted_test(out_pts[0].x, out_pts[0].y);

    std::cout << "OpenCV distorted: " << cv_distorted_test.transpose()
              << std::endl;

    EXPECT_MATRIX_NEAR( distorted_test, cv_distorted_test, 1e-12 );
  }
}

// ----------------------------------------------------------------------------
TEST(distortion, distort_r1)
{
  Eigen::VectorXd d(1);
  d << -0.01;
  test_distortion(d);
}

// ----------------------------------------------------------------------------
TEST(distortion, distort_r2)
{
  Eigen::VectorXd d(2);
  d << -0.03, 0.007;
  test_distortion(d);
}

// ----------------------------------------------------------------------------
TEST(distortion, distort_r3)
{
  Eigen::VectorXd d(5);
  d << -0.03, 0.01, 0, 0, -0.02;
  test_distortion(d);
}

// ----------------------------------------------------------------------------
TEST(distortion, distort_tang_r3)
{
  Eigen::VectorXd d(5);
  d << -0.03, 0.01, -1e-3, 5e-4, -0.02;
  test_distortion(d);
}

// ----------------------------------------------------------------------------
TEST(distortion, distort_tang_r6)
{
  Eigen::VectorXd d(8);
  d << -0.03, 0.01, -1e-3, 5e-4, -0.02, 1e-4, -2e-3, 3e-4;
  test_distortion(d);
}
