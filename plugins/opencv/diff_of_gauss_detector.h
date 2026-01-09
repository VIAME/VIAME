/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_DIFF_OF_GAUSS_DETECTOR_H
#define VIAME_OPENCV_DIFF_OF_GAUSS_DETECTOR_H

#include "viame_opencv_export.h"
#include "ellipse_proposal.h"

#include <opencv2/core.hpp>

namespace viame {
namespace opencv {

// DoG detection modes
enum class DoGMode
{
  MIN_ONLY = 0,    // Detect only minima
  MAX_ONLY = 1,    // Detect only maxima
  ALL      = 2     // Detect both minima and maxima
};

// DoG detector configuration
struct VIAME_OPENCV_EXPORT DoGConfig
{
  // Upscale threshold - upscale image if min search radius is less than this
  float upscaleThreshold = 1.0f;

  // Gaussian sigma for smoothing
  float sigma = 1.6f;

  // Initial sigma of base level
  float initSigma = 0.5f;

  // Number of intervals per octave
  int intervalsPerOctave = 6;

  // Rows/columns near edge to ignore
  int scanStart = 2;

  // Maximum steps for interpolation
  int maxInterpSteps = 5;

  // Scale compensation factor
  float compensation = 1.55f;

  // Contrast threshold for filtering weak responses
  float contrastThreshold = 0.04f;

  // Curvature threshold for edge response filtering
  int curvatureThreshold = 10;
};

// Internal structure for DoG candidate during detection
struct DoGCandidate
{
  int r, c;            // Row, column in scale image
  int octv, intvl;     // Octave and interval indices
  float x, y;          // Interpolated position
  float subintvl;      // Sub-interval offset
};

// Difference of Gaussian (DoG) blob detector
//
// This class implements scale-space blob detection using Difference of Gaussian
// pyramids. It can detect both dark and bright blobs within a specified size range.
class VIAME_OPENCV_EXPORT DoGDetector
{
public:
  DoGDetector();
  explicit DoGDetector( const DoGConfig& config );
  ~DoGDetector();

  // Set configuration
  void setConfig( const DoGConfig& config );

  // Get current configuration
  DoGConfig getConfig() const;

  // Find blob candidates in an image
  //
  // @param input Input image (32-bit float, 1 or 3 channels)
  // @param candidates Output vector of detected object proposals
  // @param minRad Minimum search radius in pixels
  // @param maxRad Maximum search radius in pixels
  // @param mode Detection mode (minima, maxima, or both)
  // @return true on success
  bool detect( const cv::Mat& input,
               ObjectProposalVector& candidates,
               float minRad, float maxRad,
               DoGMode mode = DoGMode::ALL );

private:
  // Format base image for pyramid construction
  cv::Mat formatBase( const cv::Mat& img, float sigma, bool upscale, float maxRad );

  // Downsample image by half
  cv::Mat downsample( const cv::Mat& img );

  // Build Gaussian pyramid (trapezoidal structure)
  std::vector< std::vector< cv::Mat > > buildGaussPyramid(
    const cv::Mat& base, int octaves, int intervals, double sigma );

  // Build DoG pyramid from Gaussian pyramid
  std::vector< std::vector< cv::Mat > > buildDoGPyramid(
    const std::vector< std::vector< cv::Mat > >& gaussPyr, int octaves, int intervals );

  // Detect extrema in DoG pyramid
  void detectExtremum( const std::vector< std::vector< cv::Mat > >& dogPyr,
                       int octaves, int intervals,
                       ObjectProposalVector& candidates,
                       float minRad, float maxRad,
                       DoGMode mode );

  // Check if pixel is local minimum
  bool isMin( const std::vector< std::vector< cv::Mat > >& dogPyr,
              int octv, int intvl, int r, int c );

  // Check if pixel is local maximum
  bool isMax( const std::vector< std::vector< cv::Mat > >& dogPyr,
              int octv, int intvl, int r, int c );

  // Interpolate extremum location for sub-pixel accuracy
  bool interpExtremum( const std::vector< std::vector< cv::Mat > >& dogPyr,
                       int octv, int intvl, int r, int c, int intervals,
                       DoGCandidate& point );

  // Single interpolation step
  void interpStep( const std::vector< std::vector< cv::Mat > >& dogPyr,
                   int octv, int intvl, int r, int c,
                   double& xi, double& xr, double& xc );

  // Compute 3D derivative
  cv::Mat deriv3D( const std::vector< std::vector< cv::Mat > >& dogPyr,
                   int octv, int intvl, int r, int c );

  // Compute 3D Hessian
  cv::Mat hessian3D( const std::vector< std::vector< cv::Mat > >& dogPyr,
                     int octv, int intvl, int r, int c );

  // Adjust candidate positions for image scaling
  void adjustForScale( ObjectProposalVector& candidates, bool upscale, float maxRad );

  DoGConfig m_config;
};

} // namespace opencv
} // namespace viame

#endif /* VIAME_OPENCV_DIFF_OF_GAUSS_DETECTOR_H */
