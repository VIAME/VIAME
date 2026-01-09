/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "hog_features.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>

namespace viame {
namespace opencv {

//------------------------------------------------------------------------------
// Constructor / Destructor
//------------------------------------------------------------------------------

HoGFeatureExtractor::HoGFeatureExtractor()
  : m_config()
  , m_minRad( 0 )
  , m_maxRad( 0 )
  , m_outputIndex( 0 )
  , m_initialized( false )
{
}

HoGFeatureExtractor::HoGFeatureExtractor( const HoGConfig& config )
  : m_config( config )
  , m_minRad( 0 )
  , m_maxRad( 0 )
  , m_outputIndex( 0 )
  , m_initialized( false )
{
}

HoGFeatureExtractor::~HoGFeatureExtractor()
{
}

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------

void HoGFeatureExtractor::setConfig( const HoGConfig& config )
{
  m_config = config;
}

HoGConfig HoGFeatureExtractor::getConfig() const
{
  return m_config;
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------

void HoGFeatureExtractor::initialize( const cv::Mat& grayImage,
                                       float minRad, float maxRad,
                                       int outputIndex )
{
  m_minRad = minRad;
  m_maxRad = maxRad;
  m_outputIndex = outputIndex;

  calculateIntegralHoG( grayImage );

  m_initialized = true;
}

//------------------------------------------------------------------------------
// Integral histogram computation
//------------------------------------------------------------------------------

void HoGFeatureExtractor::calculateIntegralHoG( const cv::Mat& grayImage )
{
  // Compute x and y gradients using Sobel
  cv::Mat xSobel, ySobel;
  cv::Sobel( grayImage, xSobel, CV_32F, 1, 0, 3 );
  cv::Sobel( grayImage, ySobel, CV_32F, 0, 1, 3 );

  // Create bin images (9 bins for 0-180 degrees)
  std::vector< cv::Mat > bins( NUM_ORIENTATION_BINS );
  for( int i = 0; i < NUM_ORIENTATION_BINS; i++ )
  {
    bins[i] = cv::Mat::zeros( grayImage.size(), CV_32FC1 );
  }

  // Fill bin images based on gradient orientation
  for( int y = 0; y < grayImage.rows; y++ )
  {
    const float* xPtr = xSobel.ptr< float >( y );
    const float* yPtr = ySobel.ptr< float >( y );

    std::vector< float* > binPtrs( NUM_ORIENTATION_BINS );
    for( int i = 0; i < NUM_ORIENTATION_BINS; i++ )
    {
      binPtrs[i] = bins[i].ptr< float >( y );
    }

    for( int x = 0; x < grayImage.cols; x++ )
    {
      // Compute gradient orientation
      float dx = xPtr[x];
      float dy = yPtr[x];

      // Avoid division by zero
      float gradient;
      if( dx == 0.0f )
      {
        gradient = ( std::atan( dy / ( dx + 0.00001f ) ) * ( 180.0f / PI ) ) + 90.0f;
      }
      else
      {
        gradient = ( std::atan( dy / dx ) * ( 180.0f / PI ) ) + 90.0f;
      }

      // Compute gradient magnitude
      float magnitude = std::sqrt( dx * dx + dy * dy );

      // Assign to appropriate bin
      int binIdx;
      if( gradient <= 20.0f )      binIdx = 0;
      else if( gradient <= 40.0f ) binIdx = 1;
      else if( gradient <= 60.0f ) binIdx = 2;
      else if( gradient <= 80.0f ) binIdx = 3;
      else if( gradient <= 100.0f ) binIdx = 4;
      else if( gradient <= 120.0f ) binIdx = 5;
      else if( gradient <= 140.0f ) binIdx = 6;
      else if( gradient <= 160.0f ) binIdx = 7;
      else                          binIdx = 8;

      binPtrs[binIdx][x] = magnitude;
    }
  }

  // Compute integral images for each bin
  m_integrals.resize( NUM_ORIENTATION_BINS );
  for( int i = 0; i < NUM_ORIENTATION_BINS; i++ )
  {
    cv::integral( bins[i], m_integrals[i], CV_64F );
  }
}

//------------------------------------------------------------------------------
// HoG computation
//------------------------------------------------------------------------------

void HoGFeatureExtractor::calculateHoGCell( const cv::Rect& cell, cv::Mat& hogCell )
{
  // Calculate bin values using integral images
  for( int i = 0; i < NUM_ORIENTATION_BINS; i++ )
  {
    double a = m_integrals[i].at< double >( cell.y, cell.x );
    double b = m_integrals[i].at< double >( cell.y + cell.height, cell.x + cell.width );
    double c = m_integrals[i].at< double >( cell.y, cell.x + cell.width );
    double d = m_integrals[i].at< double >( cell.y + cell.height, cell.x );

    hogCell.at< float >( 0, i ) = static_cast< float >( ( a + b ) - ( c + d ) );
  }
}

void HoGFeatureExtractor::calculateHoGBlock( const cv::Rect& block, cv::Mat& hogBlock )
{
  int centerX = block.x + block.width / 2;
  int centerY = block.y + block.height / 2;
  int width1 = centerX - block.x;
  int width2 = block.width - width1;
  int height1 = centerY - block.y;
  int height2 = block.height - height1;

  cv::Mat cell( 1, NUM_ORIENTATION_BINS, CV_32FC1 );

  // Top-left cell
  calculateHoGCell( cv::Rect( block.x, block.y, width1, height1 ), cell );
  for( int i = 0; i < NUM_ORIENTATION_BINS; i++ )
  {
    hogBlock.at< float >( 0, i ) = cell.at< float >( 0, i );
  }

  // Top-right cell
  calculateHoGCell( cv::Rect( centerX, block.y, width2, height1 ), cell );
  for( int i = 0; i < NUM_ORIENTATION_BINS; i++ )
  {
    hogBlock.at< float >( 0, i + 9 ) = cell.at< float >( 0, i );
  }

  // Bottom-left cell
  calculateHoGCell( cv::Rect( block.x, centerY, width1, height2 ), cell );
  for( int i = 0; i < NUM_ORIENTATION_BINS; i++ )
  {
    hogBlock.at< float >( 0, i + 18 ) = cell.at< float >( 0, i );
  }

  // Bottom-right cell
  calculateHoGCell( cv::Rect( centerX, centerY, width2, height2 ), cell );
  for( int i = 0; i < NUM_ORIENTATION_BINS; i++ )
  {
    hogBlock.at< float >( 0, i + 27 ) = cell.at< float >( 0, i );
  }

  // Normalize the block
  if( m_config.normalization != -1 )
  {
    cv::normalize( hogBlock, hogBlock, 1.0, 0.0, m_config.normalization );
  }
}

cv::Mat HoGFeatureExtractor::calculateHoGWindow( const cv::Rect& window )
{
  int bins = m_config.bins;
  int featureLength = ( bins - 1 ) * ( bins - 1 ) * 36;

  cv::Mat featureVector( 1, featureLength, CV_32FC1 );

  double cellHeight = static_cast< double >( window.height ) / bins;
  double cellWidth = static_cast< double >( window.width ) / bins;

  int imHeight = m_integrals[0].rows;
  int imWidth = m_integrals[0].cols;

  int startCol = 0;
  double blockStartY = window.y;

  for( int i = 0; i < bins - 1; i++ )
  {
    double blockStartX = window.x;

    for( int j = 0; j < bins - 1; j++ )
    {
      // Check if we have enough data
      if( blockStartX < 0 || blockStartY < 0 ||
          std::ceil( blockStartX + cellWidth * 2 ) + 1 >= imWidth ||
          std::ceil( blockStartY + cellHeight * 2 ) + 1 >= imHeight )
      {
        // Zero out this block
        for( int k = 0; k < 36; k++ )
        {
          featureVector.at< float >( 0, startCol + k ) = 0.0f;
        }
      }
      else
      {
        cv::Mat blockFeatures( 1, 36, CV_32FC1 );
        cv::Rect blockRect(
          dround( blockStartX ),
          dround( blockStartY ),
          dround( cellWidth * 2 ),
          dround( cellHeight * 2 )
        );

        calculateHoGBlock( blockRect, blockFeatures );

        for( int k = 0; k < 36; k++ )
        {
          featureVector.at< float >( 0, startCol + k ) = blockFeatures.at< float >( 0, k );
        }
      }

      startCol += 36;
      blockStartX += cellWidth;
    }

    blockStartY += cellHeight;
  }

  return featureVector;
}

//------------------------------------------------------------------------------
// Feature extraction
//------------------------------------------------------------------------------

void HoGFeatureExtractor::extract( ObjectProposalVector& proposals )
{
  if( !m_initialized )
  {
    return;
  }

  for( auto& proposal : proposals )
  {
    if( !extractSingle( *proposal ) )
    {
      proposal->isActive = false;
    }
  }
}

bool HoGFeatureExtractor::extractSingle( ObjectProposal& proposal )
{
  if( !m_initialized )
  {
    return false;
  }

  // Calculate window bounds
  double windowRadius = std::ceil( proposal.major * m_config.addRatio );
  int lowerR = static_cast< int >( proposal.r - windowRadius );
  int upperR = static_cast< int >( std::ceil( proposal.r + windowRadius ) );
  int lowerC = static_cast< int >( proposal.c - windowRadius );
  int upperC = static_cast< int >( std::ceil( proposal.c + windowRadius ) );
  int windowWidth = upperC - lowerC;
  int windowHeight = upperR - lowerR;

  // Check minimum size
  if( windowWidth < m_config.minPixelWidth )
  {
    return false;
  }

  // Calculate HoG features
  cv::Rect window( lowerC, lowerR, windowWidth, windowHeight );
  proposal.hogResults[m_outputIndex] = calculateHoGWindow( window );

  return true;
}

} // namespace opencv
} // namespace viame
