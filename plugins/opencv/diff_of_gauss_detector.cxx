/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "diff_of_gauss_detector.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>

namespace viame {
namespace opencv {

//------------------------------------------------------------------------------
// Constructor / Destructor
//------------------------------------------------------------------------------

DoGDetector::DoGDetector()
  : m_config()
{
}

DoGDetector::DoGDetector( const DoGConfig& config )
  : m_config( config )
{
}

DoGDetector::~DoGDetector()
{
}

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------

void DoGDetector::setConfig( const DoGConfig& config )
{
  m_config = config;
}

DoGConfig DoGDetector::getConfig() const
{
  return m_config;
}

//------------------------------------------------------------------------------
// Main detection function
//------------------------------------------------------------------------------

bool DoGDetector::detect( const cv::Mat& input,
                          ObjectProposalVector& candidates,
                          float minRad, float maxRad,
                          DoGMode mode )
{
  if( input.empty() )
  {
    return false;
  }

  // Convert to 32-bit float if needed
  cv::Mat inputFloat;
  if( input.depth() != CV_32F )
  {
    input.convertTo( inputFloat, CV_32F, 1.0 / 255.0 );
  }
  else
  {
    inputFloat = input.clone();
  }

  // Calculate pyramid characteristics
  bool upscale = ( minRad < m_config.upscaleThreshold );
  float sigma = m_config.sigma;
  int octaves = static_cast< int >( std::log( maxRad / sigma ) / std::log( 2.0f ) ) + 1;
  int intervals = m_config.intervalsPerOctave;

  // Format the base of the pyramid
  cv::Mat base = formatBase( inputFloat, sigma, upscale, maxRad );

  // Compensate scanning radii
  float scanMinRad = minRad / m_config.compensation;

  // Build Gaussian pyramid
  auto gaussPyr = buildGaussPyramid( base, octaves, intervals, sigma );

  // Build DoG pyramid
  auto dogPyr = buildDoGPyramid( gaussPyr, octaves, intervals );

  // Find candidates
  detectExtremum( dogPyr, octaves, intervals, candidates,
                  scanMinRad, maxRad, mode );

  // Adjust candidates for scale
  adjustForScale( candidates, upscale, maxRad );

  return true;
}

//------------------------------------------------------------------------------
// Pyramid construction
//------------------------------------------------------------------------------

cv::Mat DoGDetector::formatBase( const cv::Mat& img, float sigma,
                                  bool upscale, float maxRad )
{
  // Add border around the image
  int borderSize = static_cast< int >( maxRad / 2 );
  cv::Mat base;

  // Calculate median for border value
  float medianVal = quickMedian( img, 1000 );
  cv::Scalar borderVal( medianVal, medianVal, medianVal );

  cv::copyMakeBorder( img, base, borderSize, borderSize, borderSize, borderSize,
                      cv::BORDER_CONSTANT, borderVal );

  float sigDiff;

  if( upscale )
  {
    // Upscale by 2x
    sigDiff = std::sqrt( sigma * sigma -
                         m_config.initSigma * m_config.initSigma * 4 );
    cv::Mat upped;
    cv::resize( base, upped, cv::Size( base.cols * 2, base.rows * 2 ),
                0, 0, cv::INTER_CUBIC );
    cv::GaussianBlur( upped, base, cv::Size( 0, 0 ), sigDiff, sigDiff );
  }
  else
  {
    sigDiff = std::sqrt( sigma * sigma -
                         m_config.initSigma * m_config.initSigma );
    cv::GaussianBlur( base, base, cv::Size( 0, 0 ), sigDiff, sigDiff );
  }

  return base;
}

cv::Mat DoGDetector::downsample( const cv::Mat& img )
{
  cv::Mat smaller;
  cv::resize( img, smaller, cv::Size( img.cols / 2, img.rows / 2 ),
              0, 0, cv::INTER_NEAREST );
  return smaller;
}

std::vector< std::vector< cv::Mat > > DoGDetector::buildGaussPyramid(
  const cv::Mat& base, int octaves, int intervals, double sigma )
{
  // Allocate pyramid structure
  std::vector< std::vector< cv::Mat > > gaussPyr( octaves );
  for( int i = 0; i < octaves; i++ )
  {
    gaussPyr[i].resize( intervals + 3 );
  }

  // Calculate smoothing increments for each interval
  std::vector< double > sig( intervals + 3 );
  sig[0] = sigma;
  double k = std::pow( 2.0, 1.0 / intervals );

  for( int i = 1; i < intervals + 3; i++ )
  {
    double sigPrev = std::pow( k, i - 1 ) * sigma;
    double sigTotal = sigPrev * k;
    sig[i] = std::sqrt( sigTotal * sigTotal - sigPrev * sigPrev );
  }

  // Build pyramid
  for( int o = 0; o < octaves; o++ )
  {
    for( int i = 0; i < intervals + 3; i++ )
    {
      if( o == 0 && i == 0 )
      {
        gaussPyr[o][i] = base.clone();
      }
      else if( i == 0 )
      {
        // Base of new octave is halved image from end of previous octave
        gaussPyr[o][i] = downsample( gaussPyr[o - 1][intervals] );
      }
      else
      {
        // Blur current octave's last image to create next one
        cv::GaussianBlur( gaussPyr[o][i - 1], gaussPyr[o][i],
                          cv::Size( 0, 0 ), sig[i], sig[i] );
      }
    }
  }

  return gaussPyr;
}

std::vector< std::vector< cv::Mat > > DoGDetector::buildDoGPyramid(
  const std::vector< std::vector< cv::Mat > >& gaussPyr, int octaves, int intervals )
{
  std::vector< std::vector< cv::Mat > > dogPyr( octaves );

  for( int i = 0; i < octaves; i++ )
  {
    dogPyr[i].resize( intervals + 2 );
  }

  for( int o = 0; o < octaves; o++ )
  {
    for( int i = 0; i < intervals + 2; i++ )
    {
      cv::subtract( gaussPyr[o][i + 1], gaussPyr[o][i], dogPyr[o][i] );
    }
  }

  return dogPyr;
}

//------------------------------------------------------------------------------
// Extrema detection
//------------------------------------------------------------------------------

bool DoGDetector::isMax( const std::vector< std::vector< cv::Mat > >& dogPyr,
                         int octv, int intvl, int r, int c )
{
  float val = dogPyr[octv][intvl].at< float >( r, c );

  for( int i = -1; i <= 1; i++ )
  {
    for( int j = -1; j <= 1; j++ )
    {
      for( int k = -1; k <= 1; k++ )
      {
        if( val < dogPyr[octv][intvl + i].at< float >( r + j, c + k ) )
        {
          return false;
        }
      }
    }
  }

  return true;
}

bool DoGDetector::isMin( const std::vector< std::vector< cv::Mat > >& dogPyr,
                         int octv, int intvl, int r, int c )
{
  float val = dogPyr[octv][intvl].at< float >( r, c );

  for( int i = -1; i <= 1; i++ )
  {
    for( int j = -1; j <= 1; j++ )
    {
      for( int k = -1; k <= 1; k++ )
      {
        if( val > dogPyr[octv][intvl + i].at< float >( r + j, c + k ) )
        {
          return false;
        }
      }
    }
  }

  return true;
}

void DoGDetector::detectExtremum(
  const std::vector< std::vector< cv::Mat > >& dogPyr,
  int octaves, int intervals,
  ObjectProposalVector& candidates,
  float minRad, float maxRad,
  DoGMode mode )
{
  int scanStart = m_config.scanStart;

  for( int o = 0; o < octaves; o++ )
  {
    for( int i = 1; i <= intervals; i++ )
    {
      float nextScaleSigma = 2 * m_config.sigma *
        std::pow( 2.0f, o + ( i + 1.0f ) / intervals );
      float prevScaleSigma = 2 * m_config.sigma *
        std::pow( 2.0f, o + ( i - 1.0f ) / intervals );

      if( nextScaleSigma <= minRad || prevScaleSigma >= maxRad )
      {
        continue;
      }

      int rows = dogPyr[o][i].rows;
      int cols = dogPyr[o][i].cols;

      for( int r = scanStart; r < rows - scanStart; r++ )
      {
        for( int c = scanStart; c < cols - scanStart; c++ )
        {
          bool isExtremum = false;

          switch( mode )
          {
            case DoGMode::MIN_ONLY:
              isExtremum = isMin( dogPyr, o, i, r, c );
              break;
            case DoGMode::MAX_ONLY:
              isExtremum = isMax( dogPyr, o, i, r, c );
              break;
            case DoGMode::ALL:
              isExtremum = isMin( dogPyr, o, i, r, c ) ||
                           isMax( dogPyr, o, i, r, c );
              break;
          }

          if( isExtremum )
          {
            DoGCandidate point;
            if( interpExtremum( dogPyr, o, i, r, c, intervals, point ) )
            {
              auto proposal = std::make_shared< ObjectProposal >();
              proposal->r = point.y;
              proposal->c = point.x;
              proposal->major = m_config.compensation * m_config.sigma *
                std::pow( 2.0f, point.octv +
                          ( point.intvl + point.subintvl ) / intervals );
              proposal->minor = proposal->major;
              proposal->angle = 0.0;
              proposal->method = DetectionMethod::DOG;
              proposal->magnitude = dogPyr[o][i].at< float >( r, c );

              candidates.push_back( proposal );
            }
          }
        }
      }
    }
  }
}

bool DoGDetector::interpExtremum(
  const std::vector< std::vector< cv::Mat > >& dogPyr,
  int octv, int intvl, int r, int c, int intervals,
  DoGCandidate& point )
{
  double xi, xr, xc;
  int i = 0;
  int scanStart = m_config.scanStart;

  while( i < m_config.maxInterpSteps )
  {
    interpStep( dogPyr, octv, intvl, r, c, xi, xr, xc );

    if( std::abs( xi ) < 0.5 && std::abs( xr ) < 0.5 && std::abs( xc ) < 0.5 )
    {
      break;
    }

    c += static_cast< int >( std::round( xc ) );
    r += static_cast< int >( std::round( xr ) );
    intvl += static_cast< int >( std::round( xi ) );

    if( intvl < 1 ||
        intvl > intervals ||
        c < scanStart ||
        r < scanStart ||
        c >= dogPyr[octv][0].cols - scanStart ||
        r >= dogPyr[octv][0].rows - scanStart )
    {
      return false;
    }

    i++;
  }

  point.x = ( c + static_cast< float >( xc ) ) * std::pow( 2.0f, octv );
  point.y = ( r + static_cast< float >( xr ) ) * std::pow( 2.0f, octv );
  point.r = r;
  point.c = c;
  point.octv = octv;
  point.intvl = intvl;
  point.subintvl = static_cast< float >( xi );

  return true;
}

void DoGDetector::interpStep(
  const std::vector< std::vector< cv::Mat > >& dogPyr,
  int octv, int intvl, int r, int c,
  double& xi, double& xr, double& xc )
{
  cv::Mat dD = deriv3D( dogPyr, octv, intvl, r, c );
  cv::Mat H = hessian3D( dogPyr, octv, intvl, r, c );
  cv::Mat H_inv;
  cv::invert( H, H_inv, cv::DECOMP_SVD );

  cv::Mat X = -H_inv * dD;

  xc = X.at< double >( 0, 0 );
  xr = X.at< double >( 1, 0 );
  xi = X.at< double >( 2, 0 );
}

cv::Mat DoGDetector::deriv3D(
  const std::vector< std::vector< cv::Mat > >& dogPyr,
  int octv, int intvl, int r, int c )
{
  double dx = ( dogPyr[octv][intvl].at< float >( r, c + 1 ) -
                dogPyr[octv][intvl].at< float >( r, c - 1 ) ) / 2.0;
  double dy = ( dogPyr[octv][intvl].at< float >( r + 1, c ) -
                dogPyr[octv][intvl].at< float >( r - 1, c ) ) / 2.0;
  double ds = ( dogPyr[octv][intvl + 1].at< float >( r, c ) -
                dogPyr[octv][intvl - 1].at< float >( r, c ) ) / 2.0;

  cv::Mat dI( 3, 1, CV_64F );
  dI.at< double >( 0, 0 ) = dx;
  dI.at< double >( 1, 0 ) = dy;
  dI.at< double >( 2, 0 ) = ds;

  return dI;
}

cv::Mat DoGDetector::hessian3D(
  const std::vector< std::vector< cv::Mat > >& dogPyr,
  int octv, int intvl, int r, int c )
{
  double v = dogPyr[octv][intvl].at< float >( r, c );

  double dxx = dogPyr[octv][intvl].at< float >( r, c + 1 ) +
               dogPyr[octv][intvl].at< float >( r, c - 1 ) - 2 * v;

  double dyy = dogPyr[octv][intvl].at< float >( r + 1, c ) +
               dogPyr[octv][intvl].at< float >( r - 1, c ) - 2 * v;

  double dss = dogPyr[octv][intvl + 1].at< float >( r, c ) +
               dogPyr[octv][intvl - 1].at< float >( r, c ) - 2 * v;

  double dxy = ( dogPyr[octv][intvl].at< float >( r + 1, c + 1 ) -
                 dogPyr[octv][intvl].at< float >( r + 1, c - 1 ) -
                 dogPyr[octv][intvl].at< float >( r - 1, c + 1 ) +
                 dogPyr[octv][intvl].at< float >( r - 1, c - 1 ) ) / 4.0;

  double dxs = ( dogPyr[octv][intvl + 1].at< float >( r, c + 1 ) -
                 dogPyr[octv][intvl + 1].at< float >( r, c - 1 ) -
                 dogPyr[octv][intvl - 1].at< float >( r, c + 1 ) +
                 dogPyr[octv][intvl - 1].at< float >( r, c - 1 ) ) / 4.0;

  double dys = ( dogPyr[octv][intvl + 1].at< float >( r + 1, c ) -
                 dogPyr[octv][intvl + 1].at< float >( r - 1, c ) -
                 dogPyr[octv][intvl - 1].at< float >( r + 1, c ) +
                 dogPyr[octv][intvl - 1].at< float >( r - 1, c ) ) / 4.0;

  cv::Mat H( 3, 3, CV_64F );
  H.at< double >( 0, 0 ) = dxx;
  H.at< double >( 0, 1 ) = dxy;
  H.at< double >( 0, 2 ) = dxs;
  H.at< double >( 1, 0 ) = dxy;
  H.at< double >( 1, 1 ) = dyy;
  H.at< double >( 1, 2 ) = dys;
  H.at< double >( 2, 0 ) = dxs;
  H.at< double >( 2, 1 ) = dys;
  H.at< double >( 2, 2 ) = dss;

  return H;
}

void DoGDetector::adjustForScale( ObjectProposalVector& candidates,
                                   bool upscale, float maxRad )
{
  int borderOffset = static_cast< int >( maxRad / 2 );

  for( auto& p : candidates )
  {
    if( upscale )
    {
      p->r = p->r / 2.0 - borderOffset;
      p->c = p->c / 2.0 - borderOffset;
      p->major /= 2.0;
      p->minor /= 2.0;
    }
    else
    {
      p->r = p->r - borderOffset;
      p->c = p->c - borderOffset;
    }
  }
}

} // namespace opencv
} // namespace viame
