/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_GABOR_FEATURES_H
#define VIAME_OPENCV_GABOR_FEATURES_H

#include "viame_opencv_export.h"
#include "ellipse_proposal.h"

#include <opencv2/core.hpp>

#include <vector>

namespace viame {
namespace opencv {

// Gabor filter parameters
struct GaborFilterParams
{
  float sigma;   // Standard deviation of Gaussian envelope
  float theta;   // Orientation in radians
  float lambda;  // Wavelength of sinusoidal factor
  float psi;     // Phase offset
  float gamma;   // Spatial aspect ratio

  GaborFilterParams( float s = 1.2f, float t = 0.0f, float l = 6.4f,
                     float p = 3.8f, float g = 1.97f )
    : sigma( s ), theta( t ), lambda( l ), psi( p ), gamma( g ) {}
};

// Gabor feature extractor
//
// This class extracts Gabor filter features from object proposals.
// Gabor filters are bandpass filters that can capture texture information
// at different orientations and scales.
class VIAME_OPENCV_EXPORT GaborFeatureExtractor
{
public:
  // Default constructor with standard filter bank
  GaborFeatureExtractor();

  // Constructor with custom filter bank
  explicit GaborFeatureExtractor( const std::vector< GaborFilterParams >& filterBank );

  ~GaborFeatureExtractor();

  // Set custom filter bank
  void setFilterBank( const std::vector< GaborFilterParams >& filterBank );

  // Get current filter bank
  std::vector< GaborFilterParams > getFilterBank() const;

  // Number of features per proposal
  int getFeatureCount() const;

  // Extract features for all proposals
  //
  // @param grayImage Input grayscale image (32-bit float)
  // @param proposals Object proposals to compute features for
  void extract( const cv::Mat& grayImage, ObjectProposalVector& proposals );

  // Extract features for a single proposal
  //
  // @param grayImage Input grayscale image (32-bit float)
  // @param proposal Single object proposal
  // @return Vector of Gabor features
  std::vector< double > extractSingle( const cv::Mat& grayImage,
                                       const ObjectProposal& proposal );

private:
  // Create a Gabor filter kernel
  cv::Mat createGaborKernel( const GaborFilterParams& params );

  // Build filter responses (cached)
  void buildFilterResponses( const cv::Mat& grayImage );

  // Sample positions relative to proposal center
  static const int NUM_SAMPLE_POINTS = 6;

  std::vector< GaborFilterParams > m_filterBank;
  std::vector< cv::Mat > m_filterResponses;
  cv::Size m_lastImageSize;
};

// Create default Gabor filter bank
VIAME_OPENCV_EXPORT std::vector< GaborFilterParams > createDefaultGaborFilterBank();

} // namespace opencv
} // namespace viame

#endif /* VIAME_OPENCV_GABOR_FEATURES_H */
