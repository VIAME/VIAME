/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_HOG_FEATURES_H
#define VIAME_OPENCV_HOG_FEATURES_H

#include "viame_opencv_export.h"
#include "ellipse_proposal.h"

#include <opencv2/core.hpp>

#include <vector>

namespace viame {
namespace opencv {

// HoG feature extractor configuration
struct HoGConfig
{
  // Minimum pixel width required for HoG calculation
  int minPixelWidth = 10;

  // Number of orientation bins per histogram
  int bins = 8;

  // Ratio to expand region around proposal
  float addRatio = 1.4f;

  // L2 normalization constant
  int normalization = 4;  // cv::NORM_L2
};

// HoG (Histogram of Oriented Gradients) feature extractor
//
// This class computes HoG features for object proposals using integral
// histograms for efficient computation across multiple scales.
class VIAME_OPENCV_EXPORT HoGFeatureExtractor
{
public:
  // Constructor
  HoGFeatureExtractor();

  // Constructor with configuration
  explicit HoGFeatureExtractor( const HoGConfig& config );

  // Destructor
  ~HoGFeatureExtractor();

  // Set configuration
  void setConfig( const HoGConfig& config );

  // Get configuration
  HoGConfig getConfig() const;

  // Initialize with grayscale image
  //
  // @param grayImage Grayscale input image (32-bit float)
  // @param minRad Minimum search radius
  // @param maxRad Maximum search radius
  // @param outputIndex Index in proposal's hogResults to store output
  void initialize( const cv::Mat& grayImage, float minRad, float maxRad, int outputIndex );

  // Extract features for all proposals
  void extract( ObjectProposalVector& proposals );

  // Extract features for a single proposal
  //
  // @param proposal Object proposal to compute features for
  // @return true if successful, false if proposal is too small
  bool extractSingle( ObjectProposal& proposal );

private:
  // Calculate integral histograms for HoG
  void calculateIntegralHoG( const cv::Mat& grayImage );

  // Calculate HoG for a window region using integral histograms
  cv::Mat calculateHoGWindow( const cv::Rect& window );

  // Calculate HoG for a block (2x2 cells)
  void calculateHoGBlock( const cv::Rect& block, cv::Mat& hogBlock );

  // Calculate HoG for a single cell
  void calculateHoGCell( const cv::Rect& cell, cv::Mat& hogCell );

  HoGConfig m_config;
  std::vector< cv::Mat > m_integrals;  // Integral histograms (9 bins)
  float m_minRad;
  float m_maxRad;
  int m_outputIndex;
  bool m_initialized;

  static const int NUM_ORIENTATION_BINS = 9;
};

} // namespace opencv
} // namespace viame

#endif /* VIAME_OPENCV_HOG_FEATURES_H */
