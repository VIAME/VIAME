/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "gabor_features.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>

namespace viame {
namespace opencv {

//------------------------------------------------------------------------------
// Default filter bank
//------------------------------------------------------------------------------

std::vector< GaborFilterParams > createDefaultGaborFilterBank()
{
  std::vector< GaborFilterParams > bank;

  bank.push_back( GaborFilterParams( 1.2f, 0.0f, 6.4f, 3.8f, 1.97f ) );
  bank.push_back( GaborFilterParams( 1.2f, PI / 2.0f, 6.4f, 3.8f, 1.97f ) );
  bank.push_back( GaborFilterParams( 1.2f, PI / 6.0f, 6.4f, 3.8f, 1.97f ) );
  bank.push_back( GaborFilterParams( 0.4f, 0.0f, 2.4f, 5.8f, 1.23f ) );
  bank.push_back( GaborFilterParams( 1.2f, PI / 2.0f, 7.4f, 5.8f, 2.47f ) );
  bank.push_back( GaborFilterParams( 1.8f, PI / 3.0f, 5.4f, 1.8f, 2.17f ) );

  return bank;
}

//------------------------------------------------------------------------------
// Constructor / Destructor
//------------------------------------------------------------------------------

GaborFeatureExtractor::GaborFeatureExtractor()
  : m_filterBank( createDefaultGaborFilterBank() )
  , m_lastImageSize( 0, 0 )
{
}

GaborFeatureExtractor::GaborFeatureExtractor(
  const std::vector< GaborFilterParams >& filterBank )
  : m_filterBank( filterBank )
  , m_lastImageSize( 0, 0 )
{
}

GaborFeatureExtractor::~GaborFeatureExtractor()
{
}

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------

void GaborFeatureExtractor::setFilterBank(
  const std::vector< GaborFilterParams >& filterBank )
{
  m_filterBank = filterBank;
  m_filterResponses.clear();
  m_lastImageSize = cv::Size( 0, 0 );
}

std::vector< GaborFilterParams > GaborFeatureExtractor::getFilterBank() const
{
  return m_filterBank;
}

int GaborFeatureExtractor::getFeatureCount() const
{
  return static_cast< int >( m_filterBank.size() ) * NUM_SAMPLE_POINTS;
}

//------------------------------------------------------------------------------
// Filter creation
//------------------------------------------------------------------------------

cv::Mat GaborFeatureExtractor::createGaborKernel( const GaborFilterParams& params )
{
  // Calculate bounding box based on Gabor parameters
  float sigmaX = params.sigma;
  float sigmaY = params.sigma / params.gamma;
  float sint = std::sin( params.theta );
  float cost = std::cos( params.theta );
  int nstds = 3;

  float xmax32f = std::max( std::abs( nstds * sigmaX * cost ),
                            std::abs( nstds * sigmaY * sint ) );
  int xmax = static_cast< int >( std::ceil( std::max( xmax32f, 1.0f ) ) );

  float ymax32f = std::max( std::abs( nstds * sigmaX * sint ),
                            std::abs( nstds * sigmaY * cost ) );
  int ymax = static_cast< int >( std::ceil( std::max( ymax32f, 1.0f ) ) );

  // Create kernel matrix
  int width = 2 * ymax + 1;
  int height = 2 * xmax + 1;
  int cr = ymax;
  int cc = xmax;

  cv::Mat kernel( height, width, CV_32FC1 );

  float front = 1.0f / ( PI * 2.0f * sigmaX * sigmaY );
  float sigxsq = sigmaX * sigmaX;
  float sigysq = sigmaY * sigmaY;

  for( int r = 0; r < height; r++ )
  {
    for( int c = 0; c < width; c++ )
    {
      float x = static_cast< float >( c - cc );
      float y = static_cast< float >( r - cr );
      float xTheta = x * cost + y * sint;
      float yTheta = -x * sint + y * cost;

      float value = front * std::exp( -0.5f *
        ( xTheta * xTheta / sigxsq + yTheta * yTheta / sigysq ) );
      value = value * std::cos( 2.0f * PI / params.lambda * xTheta + params.psi );

      kernel.at< float >( r, c ) = value;
    }
  }

  return kernel;
}

//------------------------------------------------------------------------------
// Filter response computation
//------------------------------------------------------------------------------

void GaborFeatureExtractor::buildFilterResponses( const cv::Mat& grayImage )
{
  // Check if we need to recompute
  if( grayImage.size() == m_lastImageSize && !m_filterResponses.empty() )
  {
    return;
  }

  m_filterResponses.clear();
  m_filterResponses.resize( m_filterBank.size() );

  for( size_t i = 0; i < m_filterBank.size(); i++ )
  {
    cv::Mat kernel = createGaborKernel( m_filterBank[i] );
    cv::Mat response;

    cv::filter2D( grayImage, response, CV_32F, kernel );
    cv::blur( response, m_filterResponses[i], cv::Size( 5, 5 ) );
  }

  m_lastImageSize = grayImage.size();
}

//------------------------------------------------------------------------------
// Feature extraction
//------------------------------------------------------------------------------

void GaborFeatureExtractor::extract( const cv::Mat& grayImage,
                                      ObjectProposalVector& proposals )
{
  if( grayImage.empty() || proposals.empty() )
  {
    return;
  }

  // Build filter responses
  buildFilterResponses( grayImage );

  int rows = grayImage.rows;
  int cols = grayImage.cols;

  // Extract features for each proposal
  for( auto& proposal : proposals )
  {
    if( !proposal->isActive )
    {
      continue;
    }

    int index = 0;

    for( size_t j = 0; j < m_filterBank.size(); j++ )
    {
      const cv::Mat& response = m_filterResponses[j];

      // Sample position 1: center
      int r = static_cast< int >( proposal->r );
      int c = static_cast< int >( proposal->c );
      if( r > 0 && c > 0 && r < rows && c < cols )
      {
        proposal->gaborFeatures[index++] = response.at< float >( r, c );
      }
      else
      {
        proposal->gaborFeatures[index++] = 0.0;
      }

      // Sample position 2: below center
      r = static_cast< int >( proposal->r + proposal->major * 0.63 );
      c = static_cast< int >( proposal->c );
      if( r > 0 && c > 0 && r < rows && c < cols )
      {
        proposal->gaborFeatures[index++] = response.at< float >( r, c );
      }
      else
      {
        proposal->gaborFeatures[index++] = 0.0;
      }

      // Sample position 3: right of center
      r = static_cast< int >( proposal->r );
      c = static_cast< int >( proposal->c + proposal->major * 0.63 );
      if( r > 0 && c > 0 && r < rows && c < cols )
      {
        proposal->gaborFeatures[index++] = response.at< float >( r, c );
      }
      else
      {
        proposal->gaborFeatures[index++] = 0.0;
      }

      // Sample position 4: left of center
      r = static_cast< int >( proposal->r );
      c = static_cast< int >( proposal->c - proposal->major * 0.63 );
      if( r > 0 && c > 0 && r < rows && c < cols )
      {
        proposal->gaborFeatures[index++] = response.at< float >( r, c );
      }
      else
      {
        proposal->gaborFeatures[index++] = 0.0;
      }

      // Sample position 5: above center
      r = static_cast< int >( proposal->r - proposal->major * 0.63 );
      c = static_cast< int >( proposal->c );
      if( r > 0 && c > 0 && r < rows && c < cols )
      {
        proposal->gaborFeatures[index++] = response.at< float >( r, c );
      }
      else
      {
        proposal->gaborFeatures[index++] = 0.0;
      }

      // Sample position 6: further below center
      r = static_cast< int >( proposal->r + proposal->major );
      c = static_cast< int >( proposal->c );
      if( r > 0 && c > 0 && r < rows && c < cols )
      {
        proposal->gaborFeatures[index++] = response.at< float >( r, c );
      }
      else
      {
        proposal->gaborFeatures[index++] = 0.0;
      }
    }
  }
}

std::vector< double > GaborFeatureExtractor::extractSingle(
  const cv::Mat& grayImage,
  const ObjectProposal& proposal )
{
  std::vector< double > features;
  features.reserve( getFeatureCount() );

  if( grayImage.empty() )
  {
    return features;
  }

  buildFilterResponses( grayImage );

  int rows = grayImage.rows;
  int cols = grayImage.cols;

  for( size_t j = 0; j < m_filterBank.size(); j++ )
  {
    const cv::Mat& response = m_filterResponses[j];

    // Sample the 6 positions
    std::vector< std::pair< int, int > > positions = {
      { static_cast< int >( proposal.r ), static_cast< int >( proposal.c ) },
      { static_cast< int >( proposal.r + proposal.major * 0.63 ), static_cast< int >( proposal.c ) },
      { static_cast< int >( proposal.r ), static_cast< int >( proposal.c + proposal.major * 0.63 ) },
      { static_cast< int >( proposal.r ), static_cast< int >( proposal.c - proposal.major * 0.63 ) },
      { static_cast< int >( proposal.r - proposal.major * 0.63 ), static_cast< int >( proposal.c ) },
      { static_cast< int >( proposal.r + proposal.major ), static_cast< int >( proposal.c ) }
    };

    for( const auto& pos : positions )
    {
      int r = pos.first;
      int c = pos.second;

      if( r > 0 && c > 0 && r < rows && c < cols )
      {
        features.push_back( response.at< float >( r, c ) );
      }
      else
      {
        features.push_back( 0.0 );
      }
    }
  }

  return features;
}

} // namespace opencv
} // namespace viame
