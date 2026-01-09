/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_CANNY_EDGE_DETECTOR_H
#define VIAME_OPENCV_CANNY_EDGE_DETECTOR_H

#include "viame_opencv_export.h"
#include "ellipse_proposal.h"

#include <opencv2/core.hpp>

namespace viame {
namespace opencv {

// Configuration for Canny edge-based detector
struct VIAME_OPENCV_EXPORT CannyDetectorConfig
{
  // Canny edge detection thresholds
  double cannyLowThreshold = 18.0;
  double cannyHighThreshold = 28.0;
  int cannyApertureSize = 3;

  // Pre-blur kernel size
  int blurKernelSize = 7;

  // Minimum contour size to consider
  int minContourSize = 8;

  // Maximum number of proposals to return
  int maxProposals = 40;

  // Gaussian derivative sigma for edge computation
  float gaussianSigma = 1.5f;
};

// Internal circle representation for edge fitting
struct EdgeCircle
{
  double r, c;        // Center position
  double radius;      // Radius
  int contourSize;    // Number of points in contour
  double normalizedMSE; // Normalized mean squared error
  double coverage;    // Coverage metric

  EdgeCircle()
    : r( 0 ), c( 0 ), radius( 0 ), contourSize( 0 ),
      normalizedMSE( 0 ), coverage( 0 )
  {}
};

// Canny Edge-based Object Proposal Detector
//
// This class detects circular/elliptical objects by:
// 1. Running Canny edge detection
// 2. Linking edge pixels into contours
// 3. Fitting circles to contours using 3-point estimation
// 4. Ranking proposals by coverage/error ratio
class VIAME_OPENCV_EXPORT CannyEdgeDetector
{
public:
  CannyEdgeDetector();
  explicit CannyEdgeDetector( const CannyDetectorConfig& config );
  ~CannyEdgeDetector();

  // Set configuration
  void setConfig( const CannyDetectorConfig& config );

  // Get configuration
  CannyDetectorConfig getConfig() const;

  // Detect object proposals from edge contours
  //
  // @param grayImage Input grayscale image (8-bit or 32-bit float)
  // @param proposals Output vector of detected proposals
  // @param minRadius Minimum object radius to detect
  // @param maxRadius Maximum object radius to detect
  // @param scale Scale factor applied to output positions (default 1.0)
  // @return true on success
  bool detect( const cv::Mat& grayImage,
               ObjectProposalVector& proposals,
               float minRadius, float maxRadius,
               float scale = 1.0f );

  // Get the edge image from last detection (for visualization)
  cv::Mat getEdgeImage() const;

private:
  // Fit circle from 3 points
  bool circleFrom3Points( float r1, float c1,
                          float r2, float c2,
                          float r3, float c3,
                          EdgeCircle& circle );

  // Calculate normalized MSE of circle fit
  void calculateMSE( EdgeCircle& circle,
                     const std::vector< cv::Point >& contour );

  // Calculate coverage metric
  void calculateCoverage( EdgeCircle& circle,
                          const std::vector< cv::Point >& contour );

  // Link edge pixels into contours using flood-fill approach
  void linkEdgeContours( const cv::Mat& edges,
                         std::vector< std::vector< cv::Point > >& contours );

  CannyDetectorConfig m_config;
  cv::Mat m_lastEdgeImage;
};

// Gaussian derivative edge computation utilities
class VIAME_OPENCV_EXPORT GaussianEdges
{
public:
  // Compute vertical Gaussian derivative
  static cv::Mat gaussDerivVertical( const cv::Mat& input, double sigma );

  // Compute horizontal Gaussian derivative
  static cv::Mat gaussDerivHorizontal( const cv::Mat& input, double sigma );

  // Compute gradient magnitude from image
  static cv::Mat computeGradientMagnitude( const cv::Mat& input, double sigma );

  // Compute gradient orientation from image
  static cv::Mat computeGradientOrientation( const cv::Mat& input, double sigma );

  // Compute both magnitude and orientation
  static void computeGradient( const cv::Mat& input, double sigma,
                               cv::Mat& magnitude, cv::Mat& orientation );

private:
  static const float KERNEL_SIZE_PER_SIGMA;
};

} // namespace opencv
} // namespace viame

#endif /* VIAME_OPENCV_CANNY_EDGE_DETECTOR_H */
